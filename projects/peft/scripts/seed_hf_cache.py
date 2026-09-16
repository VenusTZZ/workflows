#!/usr/bin/env python3
"""Warm the persistent runner caches for peft-examples.

Standalone maintenance entry point (peft-hf-cache-seed.yml); also safe
to run by hand on a coder workspace. Two phases:

1. plant: the examples load hub-id models (roberta-base /
   bert-base-uncased / bigscience/mt0-small / facebook/dinov2-base,
   some hardcoded) whose weights are xet-backed - hf-mirror 302s them
   to cas-bridge.xethub.hf.co, unreachable-ish from the runner
   cluster. Each model is downloaded from its ModelScope mirror and
   planted into the HF hub cache under the REAL latest revision, so
   from_pretrained(<hub id>) resolves fully locally (sha match ->
   snapshot hit, zero weight download).

2. prefetch: every dataset / metric the examples load, with the exact
   runtime call signature, so load_dataset at run time is a pure cache
   hit. Generous internal retries ride out bad cas-bridge windows.

Idempotent - re-run any time; an already-warm machine only pays the
modelscope hash re-validation (MODELSCOPE_ENABLE_DEFAULT_HASH_VALIDATION
re-hashes cached files and re-downloads corrupt ones). Re-run when an
upstream repo moved (sha drift -> runtime misses) or when new examples
join the supported list.
"""
from __future__ import annotations

import fcntl
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TQDM_MININTERVAL", "15")  # non-TTY logs

import requests  # noqa: E402
from huggingface_hub import try_to_load_from_cache  # noqa: E402
from modelscope import snapshot_download  # noqa: E402

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
MODEL_CACHE = Path(os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope")))
HUB_ROOT = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"

# (modelscope mirror id, huggingface hub id the example passes)
MODELS = [
    ("AI-ModelScope/roberta-base", "roberta-base"),            # adamss x2
    ("AI-ModelScope/bert-base-uncased", "bert-base-uncased"),   # no_lora
    ("bigscience/mt0-small", "bigscience/mt0-small"),           # beft (hardcoded)
    ("facebook/dinov2-base", "facebook/dinov2-base"),           # pvera (hardcoded)
]

# (args, kwargs) exactly as the examples call them at run time
DATASETS = [
    (("glue", "mrpc"), {}),   # adamss (overlay), no_lora (hardcoded task)
    (("glue", "cola"), {}),   # adamss manual (default)
    (("gtfintechlab/financial_phrasebank_sentences_allagree", "5768"), {}),  # beft
    (("beans",), {"split": "train"}),        # pvera
    (("imdb",), {"split": "train[:1%]"}),    # miss / mica
]

METRICS = [("glue", "mrpc"), ("glue", "cola")]  # adamss x2, no_lora


def attempt(fn, tries: int = 3, wait: int = 20):
    last = None
    for i in range(1, tries + 1):
        try:
            return fn(), None
        except Exception as exc:  # noqa: BLE001 - report and retry
            last = exc
            print(f"  attempt {i}/{tries} failed: {type(exc).__name__}: {exc}", flush=True)
            if i < tries:
                time.sleep(wait)
    return None, last


def plant_models() -> list[str]:
    """Plant MODELS into the HF hub cache; returns failed hub ids."""
    failed: list[str] = []
    # modelscope downloads write straight to the final path
    # (Range-resume, no temp+rename), so concurrent seeds on one
    # machine must not race. An exclusive flock serializes matrix
    # legs sharing a host cache.
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    lock = open(MODEL_CACHE / ".hf-cache-seed.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    print("acquired modelscope seed lock", flush=True)
    try:
        for ms_id, hf_id in MODELS:
            src = Path(snapshot_download(ms_id, cache_dir=str(MODEL_CACHE)))
            sha = requests.get(f"{HF_ENDPOINT}/api/models/{hf_id}", timeout=60).json()["sha"]
            repo_dir = HUB_ROOT / f"models--{hf_id.replace('/', '--')}"
            snap_dir = repo_dir / "snapshots" / sha
            snap_dir.mkdir(parents=True, exist_ok=True)
            (repo_dir / "refs").mkdir(exist_ok=True)
            # no trailing newline: hub compares this string to the
            # snapshot folder name without stripping
            (repo_dir / "refs" / "main").write_text(sha)
            for item in src.rglob("*"):
                if not item.is_file():
                    continue
                dest = snap_dir / item.relative_to(src)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists() or dest.is_symlink():
                    continue
                try:
                    dest.symlink_to(item.resolve())
                except FileExistsError:
                    pass  # sibling seed ran concurrently
            config = try_to_load_from_cache(hf_id, "config.json")
            weights = try_to_load_from_cache(hf_id, "model.safetensors")
            if config and weights:
                print(f"planted {ms_id} -> {hf_id}@{sha[:8]} (weights resolve locally)", flush=True)
            else:
                failed.append(hf_id)
                print(f"FAIL {hf_id}: config={bool(config)} weights={bool(weights)}", flush=True)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
    return failed


def prefetch_datasets() -> list[str]:
    """Prefetch DATASETS + METRICS; returns failed labels."""
    from datasets import load_dataset

    failed: list[str] = []
    for args, kwargs in DATASETS:
        ds, err = attempt(lambda a=args, k=kwargs: load_dataset(*a, **k))
        if err is None:
            splits = list(ds) if hasattr(ds, "keys") else type(ds).__name__
            print(f"prefetched {args} {kwargs}: {splits}", flush=True)
        else:
            failed.append(str(args))
            print(f"FAILED {args}", flush=True)

    import evaluate

    for repo, task in METRICS:
        _, err = attempt(lambda r=repo, t=task: evaluate.load(r, t))
        if err is None:
            print(f"prefetched evaluate metric {repo}/{task}", flush=True)
        else:
            failed.append(f"evaluate/{repo}/{task}")
            print(f"FAILED evaluate {repo}/{task}", flush=True)
    return failed


def main() -> int:
    print(f"hub cache: {HUB_ROOT}", flush=True)
    print(f"modelscope cache: {MODEL_CACHE}", flush=True)
    failed = plant_models()
    failed += prefetch_datasets()
    if failed:
        print(f"seed incomplete: {failed}", file=sys.stderr, flush=True)
        return 1
    print("cache seed complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

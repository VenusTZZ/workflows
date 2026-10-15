#!/usr/bin/env python3
"""Download models and datasets named by one upstream example.

Does not modify the example. Gated Llama and Gemma ids are planted from
ModelScope into the Hugging Face cache under the id the script writes.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download, try_to_load_from_cache
from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download

# Hugging Face id -> ModelScope id. Used when the Hub repo is gated.
GATED_MODELSCOPE = {
    "meta-llama/Meta-Llama-3-8B-Instruct": "LLM-Research/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B": "LLM-Research/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B": "LLM-Research/Meta-Llama-3.1-8B",
    "google/gemma-2-2b-it": "LLM-Research/gemma-2-2b-it",
    "google/gemma-2-9b-it": "LLM-Research/gemma-2-9b-it",
    "google/gemma-3-4b-it": "LLM-Research/gemma-3-4b-it",
}

DATASET_ALIASES = {
    "perfectblend": "mlabonne/open-perfectblend",
    "flickr30k": "lmms-lab/flickr30k",
    "flickr": "lmms-lab/flickr30k",
}

PILE_10K = "NeelNanda/pile-10k"


def configure_hub() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault(
        "MODELSCOPE_CACHE",
        str(Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "modelscope"),
    )


def plant(ms_id: str, hf_id: str) -> None:
    src = ms_snapshot_download(ms_id, ignore_file_pattern=["original/*"])
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    repo_dir = hf_home / "hub" / f"models--{hf_id.replace('/', '--')}"
    snap_id = hashlib.sha1(f"{hf_id}|modelscope".encode()).hexdigest()
    snap_dir = repo_dir / "snapshots" / snap_id
    refs_dir = repo_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text(snap_id)
    if not (snap_dir / "config.json").is_file():
        snap_dir.mkdir(parents=True, exist_ok=True)
        src_path = Path(src)
        for item in src_path.rglob("*"):
            if not item.is_file():
                continue
            rel = item.relative_to(src_path)
            if rel.parts and rel.parts[0] == "original":
                continue
            dest = snap_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() or dest.is_symlink():
                continue
            dest.symlink_to(item.resolve())
    if not (snap_dir / "config.json").is_file():
        raise SystemExit(f"planted cache missing config.json: {snap_dir}")
    cached = try_to_load_from_cache(hf_id, "config.json", cache_dir=str(hf_home / "hub"))
    if not cached:
        raise SystemExit(f"huggingface_hub cannot see planted {hf_id}")
    print(f"planted {ms_id} -> {hf_id}")


def hf_snapshot(model_id: str) -> None:
    try:
        snapshot_download(model_id)
    except Exception as exc:
        print(f"mirror download failed ({exc}); retrying huggingface.co")
        os.environ.pop("HF_ENDPOINT", None)
        snapshot_download(model_id)
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"prefetched {model_id}")


def prefetch_model(model_id: str) -> None:
    ms_id = GATED_MODELSCOPE.get(model_id)
    if ms_id is not None:
        try:
            plant(ms_id, model_id)
            return
        except Exception as exc:
            print(f"modelscope plant failed for {model_id} ({exc}); trying hub")
    if model_id.startswith("Qwen/"):
        try:
            plant(model_id, model_id)
            return
        except Exception as exc:
            print(f"modelscope plant failed for {model_id} ({exc}); trying hub")
    hf_snapshot(model_id)


def call_bodies(text: str, name: str) -> list[str]:
    key = name + "("
    bodies: list[str] = []
    start = 0
    while True:
        found = text.find(key, start)
        if found < 0:
            break
        index = found + len(key)
        depth = 1
        while index < len(text) and depth:
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
            index += 1
        bodies.append(text[found + len(key) : index - 1])
        start = index
    return bodies


def constants(text: str) -> tuple[dict[str, str], dict[str, str]]:
    strings = dict(
        re.findall(r"(?m)^([A-Z][A-Z0-9_]*)\s*=\s*[\"']([^\"']+)[\"']\s*$", text)
    )
    ints = dict(re.findall(r"(?m)^([A-Z][A-Z0-9_]*)\s*=\s*(\d+)\s*$", text))
    return strings, ints


def resolve(token: str, strings: dict[str, str], ints: dict[str, str]) -> str:
    token = token.strip()
    if len(token) >= 2 and token[0] in "\"'" and token[-1] == token[0]:
        return token[1:-1]
    if token.startswith('f"') or token.startswith("f'"):
        body = token[2:-1]

        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            if name in strings:
                return strings[name]
            if name in ints:
                return ints[name]
            raise KeyError(name)

        return re.sub(r"\{([A-Z][A-Z0-9_]*)\}", repl, body)
    if token in strings:
        return strings[token]
    raise KeyError(token)


def model_ids(text: str) -> list[str]:
    found = re.findall(
        r"(?m)^[ \t]*(?:MODEL_ID|model_id|MODEL_STUB|model_stub|MODEL)\s*=\s*[\"']([^\"']+)[\"']",
        text,
    )
    # argparse defaults, used when the script assigns model_id from args.
    found.extend(
        re.findall(r"default\s*=\s*[\"']([^\"']+/[^\"']+)[\"']", text)
    )
    ordered: list[str] = []
    for model_id in found:
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", model_id):
            continue
        if model_id not in ordered:
            ordered.append(model_id)
    return ordered


def dataset_requests(text: str) -> list[tuple[str, str | None, str | None]]:
    strings, ints = constants(text)
    requests: list[tuple[str, str | None, str | None]] = []

    def add(dataset_id: str, name: str | None, split: str | None) -> None:
        dataset_id = DATASET_ALIASES.get(dataset_id, dataset_id)
        item = (dataset_id, name, split)
        if item not in requests:
            requests.append(item)

    for body in call_bodies(text, "load_dataset"):
        parts = [part.strip() for part in body.split(",") if part.strip()]
        positional: list[str] = []
        split = None
        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                if key.strip() == "split":
                    try:
                        split = resolve(value.strip(), strings, ints)
                    except KeyError:
                        split = None
                continue
            try:
                positional.append(resolve(part, strings, ints))
            except KeyError:
                continue
        if not positional:
            continue
        name = positional[1] if len(positional) > 1 else None
        if split is None and "DATASET_SPLIT" in strings and ints.get("NUM_CALIBRATION_SAMPLES"):
            split = f"{strings['DATASET_SPLIT']}[:{ints['NUM_CALIBRATION_SAMPLES']}]"
        if split is None:
            continue
        add(positional[0], name, split)

    counts = {int(item) for item in re.findall(r"num_calibration_samples\s*=\s*(\d+)", text)}
    if "NUM_CALIBRATION_SAMPLES" in ints:
        counts.add(int(ints["NUM_CALIBRATION_SAMPLES"]))
    if not counts:
        counts.add(512)
    for body in call_bodies(text, "oneshot"):
        for match in re.finditer(r"dataset\s*=\s*([\"'][^\"']+[\"']|[A-Z][A-Z0-9_]*)", body):
            try:
                dataset_id = resolve(match.group(1), strings, ints)
            except KeyError:
                continue
            if dataset_id not in DATASET_ALIASES and "/" not in dataset_id:
                continue
            split_match = re.search(r"splits\s*=\s*([^,\n]+)", body)
            resolved_split = None
            if split_match is not None:
                try:
                    resolved_split = resolve(split_match.group(1).strip(), strings, ints)
                except KeyError:
                    resolved_split = None
            hub_id = DATASET_ALIASES.get(dataset_id, dataset_id)
            if resolved_split:
                add(dataset_id, None, resolved_split)
            elif "flickr30k" in hub_id:
                for count in sorted(counts):
                    add(dataset_id, None, f"test[:{count}]")
            else:
                for count in sorted(counts):
                    add(dataset_id, None, f"train[:{count}]")

    if "get_dataset(" in text:
        add(PILE_10K, None, "train")
    return requests


def prefetch_dataset(dataset_id: str, name: str | None, split: str | None) -> None:
    args: list[str] = [dataset_id]
    if name:
        args.append(name)
    kwargs = {"split": split} if split else {}
    try:
        load_dataset(*args, **kwargs)
    except Exception as exc:
        print(f"mirror dataset failed ({exc}); retrying huggingface.co")
        os.environ.pop("HF_ENDPOINT", None)
        load_dataset(*args, **kwargs)
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"prefetched dataset {dataset_id} {name or '-'} {split or '<all>'}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} <example.py> [--dry-run]")
    path = Path(sys.argv[1])
    dry_run = "--dry-run" in sys.argv[2:]
    text = path.read_text(encoding="utf-8")
    models = model_ids(text)
    datasets = dataset_requests(text)
    print(f"example {path}")
    print(f"models {models}")
    print(f"datasets {datasets}")
    if dry_run:
        return
    configure_hub()
    for model_id in models:
        prefetch_model(model_id)
    for dataset_id, name, split in datasets:
        prefetch_dataset(dataset_id, name, split)


if __name__ == "__main__":
    main()

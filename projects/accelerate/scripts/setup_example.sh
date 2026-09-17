#!/usr/bin/env bash
# Prepare the CI environment for one supported accelerate example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# accelerate itself is installed from TARGET_ROOT (the release checkout
# under test), so the guarded tag is exactly the code that runs.
#
# The upstream examples/requirements.txt is deliberately NOT installed
# wholesale: we install the checkout plus the minimal NLP+CV stack instead.
#
# Asset sourcing (2026-09-17, supersedes the 2026-09-16 hf-mirror Xet
# incident): every model/dataset the supported examples hardcode is
# fetched from ModelScope (China-reachable, no Xet) and planted into
# the HF hub cache layout, so from_pretrained / load_dataset at example
# runtime resolves to the planted cache and never downloads weights.
# The two assets ModelScope does not carry (pokemon-en-zh captions,
# malterei swift videos) are delivered by the cache-seed workflow into
# the same shared cache root instead — see cache-seed/accelerate/.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"
ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
ALIYUN_PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/

pip_ascend() {
  python -m pip install --extra-index-url "$ASCEND_PIP_INDEX" "$@"
}

select_pip_index() {
  # Runners live in mainland China: prefer the cluster pip cache, fall
  # back to the Aliyun mirror. The ascend index stays available via
  # PIP_EXTRA_INDEX_URL (set by the engine) for torch_npu wheels.
  if python -c "
import os
import urllib.error
import urllib.request
try:
    urllib.request.urlopen(os.environ['CLUSTER_PIP_INDEX'], timeout=3)
except urllib.error.HTTPError:
    pass
" 2>/dev/null; then
    export PIP_INDEX_URL="$CLUSTER_PIP_INDEX"
    export PIP_TRUSTED_HOST="$CLUSTER_PIP_HOST"
  else
    export PIP_INDEX_URL="$ALIYUN_PIP_INDEX"
    unset PIP_TRUSTED_HOST
  fi
  echo "pip index: $PIP_INDEX_URL"
}

ensure_torch_stack() {
  # Same torch line as accelerate quick-start (CANN 9.1.0 pairing):
  # torch 2.9.0 + torch_npu 2.9.0.post2. Reuse the image stack when it
  # already matches, otherwise install via the cluster cache + ascend
  # dual-source (peft's proven mechanism).
  if python -c "
import torch, torch_npu
raise SystemExit(
    0 if torch.__version__.startswith('2.9.0')
    and torch_npu.__version__.startswith('2.9.0') else 1)
"; then
    echo "reusing image torch stack (" \
      "$(python -c 'import torch; print(torch.__version__)'))"
    return
  fi
  echo "installing torch==2.9.0 torch_npu==2.9.0.post2"
  pip_ascend torch==2.9.0 torch_npu==2.9.0.post2
}

# ModelScope → HF hub cache plant (peft's proven mechanism):
#   snapshot_download from ModelScope (China mirror, no Xet) → cp into
#   ~/.cache/huggingface/hub/<models|datasets>--<hf_id>/snapshots/<sha>/
#   with refs/main written to the real upstream sha (queried via the HF
#   API — xet-free metadata only). from_pretrained(<hf_id>) /
#   load_dataset(<hf_id>) then resolves to the planted cache and never
#   touches weight downloads.
# cp (not symlink): symlinks into the modelscope cache break when that
#   cache is recycled by another job; after cp the HF cache is
# self-contained. Existing files are skipped, so a warm runner finishes
# in seconds.
# Pinned to 1.37.0 (same as peft): 1.40.1's "modelscope-hub>=0.4.2"
# floor dies on DEFAULT_CREDENTIALS_PATH import with the mirror-lagged
# hub 0.4.2.
ms_plant() {
  # $1 = asset group: nlp | nlp-ar | cv | infer-phi2 | infer-sd |
  #                   infer-tts | infer-llava
  python - "$1" <<'PY'
import os
import sys
import shutil
from pathlib import Path

# Non-TTY CI logs: throttle tqdm refreshes instead of disabling.
os.environ.setdefault("TQDM_MININTERVAL", "15")

from modelscope import snapshot_download

MODEL_CACHE = Path(os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope")))
HUB_ROOT = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "hub"
HF_API = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/") + "/api"

# (ms_id, hf_id, kind, allow_patterns). allow_patterns trims formats we
# never load (onnx/, duplicate .bin/.ckpt/.h5/.msgpack) so the plant
# moves the minimum bytes the example actually reads.
GROUPS = {
    # every NLP / by_feature example: bert-base-cased (hardcoded BARE id
    # in the scripts → plant to models--bert-base-cased) + glue mrpc.
    # NOTE: MS dataset downloads treat patterns as subtree roots
    # ("mrpc/*" → root /mrpc), matching peft's proven usage; the README
    # is fetched from the mirror API at load_dataset resolution (tiny,
    # xet-free — only LFS weights were ever flaky).
    "nlp": [
        ("AI-ModelScope/bert-base-cased", "bert-base-cased", "model",
         ["*.json", "model.safetensors", "vocab.txt"]),
        ("nyu-mll/glue", "nyu-mll/glue", "dataset", ["mrpc/*"]),
    ],
    # gradient_accumulation_for_autoregressive_models.py: SmolLM-360M
    # (skip the 3.9G onnx/ sidecar) + wikitext-2-v1.
    "nlp-ar": [
        ("HuggingFaceTB/SmolLM-360M", "HuggingFaceTB/SmolLM-360M", "model",
         ["*.json", "*.txt", "model.safetensors"]),
        ("Salesforce/wikitext", "Salesforce/wikitext", "dataset",
         ["wikitext-2-v1/*"]),
    ],
    # cv_example.py / complete_cv_example.py: timm/oxford-iiit-pet parquet.
    "cv": [
        ("timm/oxford-iiit-pet", "timm/oxford-iiit-pet", "dataset",
         ["data/*"]),
    ],
    # inference/distributed/phi2.py: hardcoded "microsoft/phi-2" (same
    # id exists on ModelScope).
    "infer-phi2": [
        ("microsoft/phi-2", "microsoft/phi-2", "model", None),
    ],
    # inference/distributed/stable_diffusion.py: hardcoded
    # "stable-diffusion-v1-5/stable-diffusion-v1-5"; example loads fp32
    # weights with torch_dtype=fp16 (no variant=), so the ModelScope
    # fp32 mirror suffices — safetensors only, skip .bin/.ckpt dupes.
    "infer-sd": [
        ("AI-ModelScope/stable-diffusion-v1-5",
         "stable-diffusion-v1-5/stable-diffusion-v1-5", "model",
         ["model_index.json", "*/*.json", "*/model.safetensors",
          "*/diffusion_pytorch_model.safetensors", "tokenizer/*",
          "feature_extractor/*"]),
    ],
    # inference/distributed/distributed_speech_generation.py: hardcoded
    # "facebook/mms-tts-eng" (skip duplicate pytorch_model.bin). Its
    # pokemon captions dataset is NOT on ModelScope — cache-seed
    # delivers it (preflight below).
    "infer-tts": [
        ("facebook/mms-tts-eng", "facebook/mms-tts-eng", "model",
         ["*.json", "model.safetensors"]),
    ],
    # inference/distributed/llava_next_video.py: hardcoded
    # "llava-hf/LLaVA-NeXT-Video-7B-hf" (same id on ModelScope, 14G).
    # Its malterei/LLaVA-Video-small-swift video dataset is NOT on
    # ModelScope — cache-seed delivers it (preflight below).
    "infer-llava": [
        ("llava-hf/LLaVA-NeXT-Video-7B-hf",
         "llava-hf/LLaVA-NeXT-Video-7B-hf", "model", None),
    ],
}

# Datasets that cannot come from ModelScope: delivered into the shared
# cache root by the cache-seed workflow (bundle staged from a proxied
# local HF download). Missing → warn loudly; the example itself will
# fail on the Xet flake if it runs without them.
SEED_PREFLIGHT = {
    "infer-tts": ["datasets--svjack--pokemon-blip-captions-en-zh"],
    "infer-llava": ["datasets--malterei--LLaVA-Video-small-swift"],
}


def fetch_sha(hf_id, kind):
    import requests
    url = f"{HF_API}/{kind}s/{hf_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json().get("sha") or r.json().get("oid")


def plant(ms_id, hf_id, kind, allow_patterns=None):
    src = Path(snapshot_download(
        ms_id,
        cache_dir=str(MODEL_CACHE),
        repo_type=kind,
        allow_patterns=allow_patterns,
    ))
    sha = fetch_sha(hf_id, kind)
    repo_kind = "models" if kind == "model" else "datasets"
    repo_dir = HUB_ROOT / f"{repo_kind}--{hf_id.replace('/', '--')}"
    snap_dir = repo_dir / "snapshots" / sha
    snap_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs").mkdir(exist_ok=True)
    # no trailing newline — hub compares this string to the snapshot
    # folder name without stripping
    (repo_dir / "refs" / "main").write_text(sha)
    n_bytes = 0
    for item in src.rglob("*"):
        if not item.is_file():
            continue
        dest = snap_dir / item.relative_to(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink():
            # replace a legacy symlink-based plant: links into the
            # modelscope cache break when that cache is recycled
            dest.unlink()
        elif dest.exists():
            continue  # warm runner: keep the already-planted copy
        shutil.copy2(item, dest)
        n_bytes += item.stat().st_size
    print(f"planted {hf_id}@{sha[:8]} ({n_bytes // (1024 * 1024)} MB new)",
          flush=True)


def preflight(repo_dirs):
    for repo_dir in repo_dirs:
        refs = HUB_ROOT / repo_dir / "refs" / "main"
        if not refs.is_file():
            print(f"WARN: {repo_dir} missing from shared cache root — "
                  f"dispatch the cache-seed workflow (projects=accelerate) "
                  f"before expecting this example to pass", flush=True)
        else:
            sha = refs.read_text()
            n = sum(
                1 for _ in (HUB_ROOT / repo_dir / "snapshots" / sha).rglob("*")
                if _.is_file()
            )
            print(f"seeded {repo_dir}@{sha[:8]} ({n} files)", flush=True)


group = sys.argv[1]
failures: list[str] = []
for ms_id, hf_id, kind, patterns in GROUPS[group]:
    try:
        plant(ms_id, hf_id, kind, allow_patterns=patterns)
    except Exception as exc:
        failures.append(f"{ms_id}: {type(exc).__name__}: {exc}")
        print(f"FAIL plant {ms_id}: {exc}", flush=True)
for repo_dir in SEED_PREFLIGHT.get(group, []):
    preflight([repo_dir])
if failures:
    print(f"ms_plant({group}) incomplete: {failures}", file=sys.stderr, flush=True)
    # Don't fail setup on plant errors — the example surfaces the real
    # failure when it actually loads (same policy as peft).
PY
}

# Smoke-validate the planted NLP assets by actually loading them (all
# cache hits, no network): bert-base-cased via the bare hardcoded id,
# then MRPC through datasets. This is the pre-2026-09-16 behavior,
# now backed by the ModelScope plant instead of hf-mirror downloads.
validate_nlp_assets() {
  python - <<'PY'
import os
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("bert-base-cased")
print("bert-base-cased tokenizer OK, vocab_size:", tok.vocab_size)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
print("bert-base-cased model OK, params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
from datasets import load_dataset
ds = load_dataset("nyu-mll/glue", "mrpc")
print("mrpc splits:", {k: len(v) for k, v in ds.items()})
PY
}

# Smoke-validate the SmolLM + wikitext plants (cache hits only).
validate_ar_assets() {
  python - <<'PY'
import os
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M")
AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M")
from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
print("smollm OK, wikitext-2 splits:", {k: len(v) for k, v in ds.items()})
PY
}

# Smoke-validate the planted inference models: resolve the snapshot via
# the hub cache (local_files_only) and check the weight files exist.
# A full from_pretrained here would just duplicate what the example does.
validate_infer_assets() {
  # $1 = infer group name
  python - "$1" <<'PY'
import os
import sys
from pathlib import Path
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from huggingface_hub import snapshot_download

CHECKS = {
    "infer-phi2": ("microsoft/phi-2", "model",
                   ["model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors"]),
    "infer-sd": ("stable-diffusion-v1-5/stable-diffusion-v1-5", "model",
                 ["model_index.json",
                  "unet/diffusion_pytorch_model.safetensors",
                  "vae/diffusion_pytorch_model.safetensors",
                  "text_encoder/model.safetensors",
                  "safety_checker/model.safetensors"]),
    "infer-tts": ("facebook/mms-tts-eng", "model",
                  ["model.safetensors", "vocab.json"]),
    "infer-llava": ("llava-hf/LLaVA-NeXT-Video-7B-hf", "model",
                    [f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)]),
}
repo, kind, must_have = CHECKS[sys.argv[1]]
snap = Path(snapshot_download(repo, repo_type=kind, local_files_only=True))
missing = [f for f in must_have if not (snap / f).is_file()]
if missing:
    raise SystemExit(f"{repo}: planted snapshot missing {missing}")
print(f"{repo} planted snapshot OK: {snap}")
PY
}

# Materialize the Oxford-IIT Pet Dataset jpg files used by cv_example.py
# + complete_cv_example.py. cv_example.py uses os.listdir(data_dir) +
# ".jpg" filter, so the data_dir must contain the .jpg files directly.
# The parquet now comes from the ModelScope plant (ms_plant cv); the
# materialization loop itself is unchanged.
prepare_pets_data() {
  local dst="$TARGET_ROOT/fixtures/pets/images"
  if [[ -d "$dst" ]] \
     && [[ "$(ls -A "$dst" 2>/dev/null | wc -l)" -gt 1000 ]]; then
    echo "reusing pets dataset at $dst ($(ls "$dst" | wc -l) files)"
    return
  fi
  echo "materializing Oxford-IIT Pets from planted cache to $dst"
  mkdir -p "$dst"
  python - <<PY
import os
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from datasets import load_dataset
DST = "$dst"
ds = load_dataset("timm/oxford-iiit-pet")
n = 0
for split in ("train", "test"):
    for r in ds[split]:
        path = os.path.join(DST, r["image_id"] + ".jpg")
        if not os.path.exists(path):
            img = r["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(path, "JPEG")
        n += 1
print(f"wrote {n} images to {DST}")
PY
  echo "pets dataset ready: $(ls "$dst" | wc -l) files"
}

setup_accelerate-nlp() {
  echo "installing accelerate from $TARGET_ROOT"
  # Pin torch explicitly. accelerate's setup.py requires `torchpippy>=0.2.0`,
  # which transitively pulls nvidia-cu13 metapackages and the resolver
  # then upgrades torch to 2.14.0+cu130 — breaking torch_npu 2.9.0 ABI.
  # `pip install -e` re-resolves all deps, so repeat the torch pin here
  # in the same command (constraints-npu.txt is already exported).
  python -m pip install -e "$TARGET_ROOT" "torch==2.9.0" "torch_npu==2.9.0.post2"
  # scikit-learn is needed by the `evaluate` library's glue metric (sklearn's
  # f1_score, matthews_corrcoef); not a direct dep of evaluate or transformers
  # so it must be listed explicitly. schedulefree is a pure-Python wheel
  # needed only by by_feature/schedule_free.py (tiny, kept in the base list).
  python -m pip install \
    transformers datasets evaluate safetensors scikit-learn schedulefree \
    "torch==2.9.0"
  python -c "
import torch, torch_npu
assert torch.__version__.startswith('2.9.0'), \
    f'torch drifted to {torch.__version__}'
import accelerate, transformers, datasets, evaluate, sklearn, schedulefree
print('accelerate', accelerate.__version__,
      '/ transformers', transformers.__version__,
      '/ datasets', datasets.__version__,
      '/ torch', torch.__version__,
      '/ sklearn', sklearn.__version__)
"
  # ModelScope client for the plant step (same pin + rationale as peft).
  python -m pip install "modelscope==1.37.0"
  ms_plant nlp
  validate_nlp_assets
}

setup_accelerate-nlp-ar() {
  # Autoregressive grad-accum variant = NLP base + SmolLM/wikitext.
  setup_accelerate-nlp
  ms_plant nlp-ar
  validate_ar_assets
}

setup_accelerate-cv() {
  # CV profile = NLP profile + vision deps + Pets data.
  setup_accelerate-nlp
  # torchvision ≤ 0.28.0 (matches torch 2.9.0 ABI; v0.29+ requires Stable ABI
  # symbols that torch 2.9 lacks — see `torchvision-v29-stable-abi` memory).
  python -m pip install "torchvision==0.24.0" "torch==2.9.0" timm
  python -c "
import torch, torch_npu
assert torch.__version__.startswith('2.9.0'), \
    f'torch drifted to {torch.__version__}'
import timm, torchvision
print('timm', timm.__version__,
      '/ torchvision', torchvision.__version__)
"
  ms_plant cv
  prepare_pets_data
}

# Shared pip stack for the four inference/distributed examples. Each
# example then gets its own model plant group so a job only pulls the
# weights it actually hardcodes (phi-2 5.5G / SD 5.5G / mms 145M /
# llava 14G — planting all four per job would move ~27G).
# - fire: speech_gen + llava CLI entry
# - av: llava video decode
# - diffusers: stable_diffusion pipeline
# - scipy: speech_gen wavfile output
# - torchvision: DiffusionPipeline / VitsModel import chains pull the
#   torchvision op registrations (same ABI pin as cv profile).
setup_infer_base() {
  setup_accelerate-nlp
  python -m pip install \
    fire av diffusers scipy "torchvision==0.24.0" "torch==2.9.0"
  python -c "
import torch, torch_npu
assert torch.__version__.startswith('2.9.0'), \
    f'torch drifted to {torch.__version__}'
import fire, av, diffusers, scipy, torchvision
print('av', av.__version__,
      '/ diffusers', diffusers.__version__,
      '/ torchvision', torchvision.__version__)
"
}

setup_accelerate-infer-phi2() {
  setup_infer_base
  ms_plant infer-phi2
  validate_infer_assets infer-phi2
}

setup_accelerate-infer-sd() {
  setup_infer_base
  ms_plant infer-sd
  validate_infer_assets infer-sd
}

setup_accelerate-infer-tts() {
  setup_infer_base
  ms_plant infer-tts
  validate_infer_assets infer-tts
}

setup_accelerate-infer-llava() {
  setup_infer_base
  ms_plant infer-llava
  validate_infer_assets infer-llava
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' \
    | paste -sd' ' -
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

# Xet-backed HF repos 302 to cas-bridge.xethub.hf.co which hf-mirror
# cannot proxy. Planted caches make weight downloads unnecessary, but
# keep the legacy transfer path forced for any residual metadata/file
# fetch (README misses on dataset resolution etc.).
echo "HF_HUB_DISABLE_XET=1" >> "$GITHUB_ENV"
export HF_HUB_DISABLE_XET=1

HERE=$(cd "$(dirname "$0")" && pwd)
export PIP_CONSTRAINT="$(cd "$HERE/.." && pwd)/constraints-npu.txt"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

select_pip_index
python -m pip install -U pip setuptools wheel
ensure_torch_stack

"setup_${PROFILE}"

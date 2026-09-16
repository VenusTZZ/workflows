#!/usr/bin/env bash
# Prepare the CI environment for one supported accelerate example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# accelerate itself is installed from TARGET_ROOT (the release checkout
# under test), so the guarded tag is exactly the code that runs.
#
# The upstream examples/requirements.txt is deliberately NOT installed
# wholesale: we install the checkout plus the minimal NLP+CV+inference
# stack instead. schedulefree (by_feature/schedule_free.py) joins the
# NLP profile; fire / webdataset / av / diffusers
# (inference/distributed/*) belong to the accelerate-infer profile.
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

# Pre-download NLP assets used by every NLP / by_feature supported entry.
# - bert-base-cased from HF mirror (the model all accelerate NLP
#   examples hardcode via AutoTokenizer/AutoModelForSequenceClassification.
#   `bert-base-cased` exists on HF and is reachable via hf-mirror.com from
#   China runners; pre-downloading here makes the run deterministic
#   instead of relying on per-example on-demand fetch.
# - GLUE MRPC from HF via HF_ENDPOINT=https://hf-mirror.com, cached into
#   $HF_HOME so each NLP example does not re-download.
prepare_nlp_assets() {
  python - <<'PY'
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("bert-base-cased")
print("bert-base-cased tokenizer OK, vocab_size:", tok.vocab_size)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
print("bert-base-cased model OK, params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
PY

  # Trigger MRPC download once into the shared HF cache. The by_feature
  # scripts load_dataset("nyu-mll/glue", "mrpc") and HF_ENDPOINT=
  # https://hf-mirror.com (set by the engine) is reachable from runners.
  python - <<'PY'
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from datasets import load_dataset
ds = load_dataset("nyu-mll/glue", "mrpc")
print("mrpc splits:", {k: len(v) for k, v in ds.items()})
PY

  # Extra NLP assets for the newer supported entries:
  # - HuggingFaceTB/SmolLM-360M + Salesforce/wikitext wikitext-2-v1
  #   (by_feature/gradient_accumulation_for_autoregressive_models.py;
  #   upstream switched it from GPT-2+ELI5 to this small combo)
  python - <<'PY'
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M")
AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M")
from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
print("smollm OK, wikitext-2 splits:", {k: len(v) for k, v in ds.items()})
PY
}

# Pre-download the Oxford-IIT Pet Dataset used by cv_example.py +
# complete_cv_example.py. cv_example.py uses os.listdir(data_dir) +
# ".jpg" filter, so the data_dir must contain the .jpg files directly.
# Upstream `https://www.robots.ox.ac.uk/~vgg/data/pets/...` is one-shot
# slow from China runners; use `timm/oxford-iiit-pet` on HF mirror as the
# primary source and materialize the PIL images into the format the
# upstream script expects (image_id + ".jpg" → matches the regex
# `^(.*)_\d+\.jpg$` used to extract the class label).
prepare_pets_data() {
  local dst="$TARGET_ROOT/fixtures/pets/images"
  if [[ -d "$dst" ]] \
     && [[ "$(ls -A "$dst" 2>/dev/null | wc -l)" -gt 1000 ]]; then
    echo "reusing pets dataset at $dst ($(ls "$dst" | wc -l) files)"
    return
  fi
  echo "downloading Oxford-IIT Pets via HF mirror to $dst"
  mkdir -p "$dst"
  python - <<PY
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
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
  # so it must be listed explicitly. schedulefree is pure Python (no native
  # extension) and only by_feature/schedule_free.py imports it.
  python -m pip install \
    transformers datasets evaluate safetensors scikit-learn schedulefree "torch==2.9.0"
  python -c "
import torch, torch_npu
assert torch.__version__.startswith('2.9.0'), \
    f'torch drifted to {torch.__version__}'
import accelerate, transformers, datasets, evaluate, sklearn
print('accelerate', accelerate.__version__,
      '/ transformers', transformers.__version__,
      '/ datasets', datasets.__version__,
      '/ torch', torch.__version__,
      '/ sklearn', sklearn.__version__)
"
  prepare_nlp_assets
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
  prepare_pets_data
}

# Pre-download inference assets for the accelerate-infer profile
# (examples/inference/distributed/*). snapshot_download fetches without
# instantiating, so the 16G llava weights do not spike host RAM. All via
# the HF mirror (engine sets HF_ENDPOINT; set default for local runs).
# HF_HUB_DISABLE_XET is exported above: Xet-backed repos (SD v1.5 etc.)
# cannot reconstruct chunks through the mirror (401).
prepare_infer_assets() {
  python - <<'PY'
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from huggingface_hub import snapshot_download
for repo, repo_type in [
    ("microsoft/phi-2", "model"),
    ("llava-hf/LLaVA-NeXT-Video-7B-hf", "model"),
    ("malterei/LLaVA-Video-small-swift", "dataset"),
    ("stable-diffusion-v1-5/stable-diffusion-v1-5", "model"),
    ("facebook/mms-tts-eng", "model"),
]:
    path = snapshot_download(repo, repo_type=repo_type)
    print(repo, "->", path)
from datasets import load_dataset
for name in ("svjack/pokemon-blip-captions-en-zh",):
    ds = load_dataset(name)
    print(name, "splits:", {k: len(v) for k, v in ds.items()})
PY
}

setup_accelerate-infer() {
  # Inference profile = NLP profile + generation/vision deps + models.
  # torchvision: DiffusionPipeline → transformers.AutoImageProcessor pulls
  # the torchvision op registrations; pin the same ABI-matched build as the
  # cv profile (image's default torchvision pairs with its original torch,
  # not our pinned 2.9.0). scipy: speech example's scipy.io.wavfile.
  setup_accelerate-nlp
  python -m pip install \
    fire webdataset av diffusers scipy "torchvision==0.24.0" "torch==2.9.0"
  python -c "
import torch, torch_npu
assert torch.__version__.startswith('2.9.0'), \
    f'torch drifted to {torch.__version__}'
import fire, webdataset, av, diffusers, scipy, torchvision
print('av', av.__version__,
      '/ diffusers', diffusers.__version__,
      '/ torchvision', torchvision.__version__)
"
  prepare_infer_assets
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

# Xet-backed HF repos (e.g. stable-diffusion-v1-5) reconstruct chunks
# straight from cas-server.xethub.hf.co, which hf-mirror cannot proxy
# (observed 401 Unauthorized from China runners). Force the legacy HTTP
# transfer path for the setup step and, via GITHUB_ENV, the run step.
echo "HF_HUB_DISABLE_XET=1" >> "$GITHUB_ENV"
export HF_HUB_DISABLE_XET=1

HERE=$(cd "$(dirname "$0")" && pwd)
export PIP_CONSTRAINT="$(cd "$HERE/.." && pwd)/constraints-npu.txt"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

select_pip_index
python -m pip install -U pip setuptools wheel
ensure_torch_stack

"setup_${PROFILE}"
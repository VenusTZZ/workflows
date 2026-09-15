#!/usr/bin/env bash
# Prepare the CI environment for one supported TRL example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# TRL is installed from TARGET_ROOT (the upstream checkout under test).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
FALLBACK_PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"

pip_ascend() {
  python -m pip install --extra-index-url "$ASCEND_PIP_INDEX" "$@"
}

select_pip_index() {
  # Runners live in mainland China: prefer the cluster pip cache, fall
  # back to the Tsinghua mirror. The ascend index stays available via
  # PIP_EXTRA_INDEX_URL (set by the workflow) for torch_npu wheels.
  if python -c "
import urllib.error
import urllib.request
try:
    urllib.request.urlopen('${CLUSTER_PIP_INDEX}', timeout=3)
except urllib.error.HTTPError:
    pass
" 2>/dev/null; then
    export PIP_INDEX_URL="$CLUSTER_PIP_INDEX"
    export PIP_TRUSTED_HOST="$CLUSTER_PIP_HOST"
  else
    export PIP_INDEX_URL="$FALLBACK_PIP_INDEX"
    unset PIP_TRUSTED_HOST
  fi
  echo "pip index: $PIP_INDEX_URL"
}

ensure_torch_stack() {
  # The Ascend image normally contains a compatible torch/torch_npu pair.
  # Reuse it when possible because these wheels are large.
  if python -c "import torch, torch_npu; print('found torch', torch.__version__, 'torch_npu', torch_npu.__version__)"; then
    echo "reusing image torch stack"
    return
  fi
  echo "installing torch==2.9.0 torch_npu==2.9.0.post2"
  pip_ascend torch==2.9.0 torch_npu==2.9.0.post2
}

ensure_torchvision() {
  if python -c "import torchvision; print('found torchvision', torchvision.__version__)"; then
    echo "reusing image torchvision"
    return
  fi
  TV=$(python - <<'PY'
import torch
major, minor = (int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
mapping = {(2,0):'0.15.1',(2,1):'0.16.0',(2,2):'0.17.0',(2,3):'0.18.0',
           (2,4):'0.19.0',(2,5):'0.20.0',(2,6):'0.21.0',(2,7):'0.22.0',
           (2,8):'0.22.1',(2,9):'0.24.0',(2,10):'0.24.1'}
print(mapping.get((major, minor), ''))
PY
  )
  if [ -n "$TV" ]; then
    echo "installing torchvision==$TV to match torch $(python -c 'import torch; print(torch.__version__)')"
    python -m pip install "torchvision==$TV"
  else
    echo "installing torchvision (pip resolves compatible version)"
    python -m pip install torchvision
  fi
  python -c "import torchvision; print('torchvision', torchvision.__version__)"
}

# Copy CI fixture data files into the target root so that example scripts
# can load them via a local path under $TARGET_ROOT/fixtures/. This
# decouples the example from the workflows-checkout subtree, which may
# have file-system visibility issues on self-hosted runner containers.
prepare_fixtures() {
  local src="${FIXTURE_DIR:?FIXTURE_DIR is required}"
  local dst="$TARGET_ROOT/fixtures"
  echo "preparing fixtures from $src to $dst"
  if ! ls "$src"/*.jsonl 1>/dev/null 2>&1; then
    echo "FATAL: no fixture files (*.jsonl) found in $src" >&2
    echo "the workflows checkout may be missing the fixtures directory" >&2
    exit 1
  fi
  mkdir -p "$dst"
  cp "$src"/*.jsonl "$dst/"
  local count
  count=$(ls "$dst"/*.jsonl 2>/dev/null | wc -l)
  echo "copied $count fixture file(s) to $dst"
  ls -la "$dst/"
}

setup_peft_lora() {
  # Covers the small-model LoRA examples (DPO/TPO). Installs TRL from the
  # target checkout with the peft extra; Pillow is needed by the VLM
  # processor of dpo_reduce_hallucinations. trackio/kernels from the
  # upstream dependency headers are skipped because CI passes
  # --report_to none and the default attention implementation.
  echo "installing TRL from source at $TARGET_ROOT with peft extra"
  python -m pip install -e "$TARGET_ROOT[peft]"
  python -m pip install Pillow
  python -c "import trl; print('TRL version:', trl.__version__)"

  # Pre-download example model weights from ModelScope (China-reachable)
  # because runners cannot reach HuggingFace. The returned local snapshot
  # dirs are exported as env vars for the example (overlay_args in
  # examples_manifest.yaml reference them via ${VAR}).
  python -m pip install modelscope
  python - <<'PY'
import os
# Non-TTY CI logs: throttle tqdm refreshes instead of disabling, so
# download progress is visible but not one line per MB. Tune via env.
os.environ.setdefault("TQDM_MININTERVAL", os.environ.get("TQDM_MININTERVAL", "15"))
from modelscope import snapshot_download

MODEL_CACHE = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))
mapping = {
    "DPO_MODEL_PATH": "Qwen/Qwen2.5-VL-3B-Instruct",
    "TPO_MODEL_PATH": "Qwen/Qwen3-0.6B",
}
for env_name, model_id in mapping.items():
    local = snapshot_download(model_id, cache_dir=MODEL_CACHE)
    with open(os.environ["GITHUB_ENV"], "a") as fh:
        fh.write(f"{env_name}={local}\n")
PY
}

# Shared ModelScope downloader: takes "ENV=model_id" pairs and writes each
# resolved local snapshot dir into GITHUB_ENV so overlay_args can reference it.
# Runners cannot reach HuggingFace; ModelScope is China-reachable.
# Cache lands in the shared /data/ci-cache/modelscope volume, so the second
# job that needs the same weights reuses the first download.
ms_download_models() {
  python -m pip install -q modelscope
  TQDM_MININTERVAL="${TQDM_MININTERVAL:-15}" python - "$@" <<'PY'
import os, sys
os.environ.setdefault("TQDM_MININTERVAL", os.environ.get("TQDM_MININTERVAL", "15"))
from modelscope import snapshot_download
MODEL_CACHE = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))
for pair in sys.argv[1:]:
    env_name, model_id = pair.split("=", 1)
    local = snapshot_download(model_id, cache_dir=MODEL_CACHE)
    with open(os.environ["GITHUB_ENV"], "a") as fh:
        fh.write(f"{env_name}={local}\n")
PY
}

setup_gold_distill() {
  # GOLD cross-tokenizer online logit distillation (gold_chatbot_arena):
  # a Llama-3.2-1B student learns from a Qwen2-1.5B teacher. Text-only,
  # so installs TRL[peft] and pre-downloads both weights from ModelScope.
  # The dataset trl-lib/chatbot_arena_completions is a HuggingFace id the
  # script loads itself; run_example.sh points HF at hf-mirror, so no
  # fixture is needed. No vLLM (use_vllm defaults off), pure local forward.
  echo "installing TRL from source at ${TARGET_ROOT} with peft extra"
  python -m pip install -e "${TARGET_ROOT}[peft]"
  python -c "import trl; print('TRL version:', trl.__version__)"
  ms_download_models \
    "GOLD_STUDENT_PATH=LLM-Research/Llama-3.2-1B-Instruct" \
    "GOLD_TEACHER_PATH=Qwen/Qwen2-1.5B-Instruct"
}

setup_self_distill() {
  # SSD / SDFT / SDPO self-distillation (ssd_codegen, sdft_privileged_context,
  # sdpo_math). All three accept --model_name_or_path, so they share the same
  # small Qwen2.5-0.5B-Instruct base (green = exit code, not loss/reward).
  # SSD and SDFT read local jsonl fixtures (copied by prepare_fixtures);
  # SDPO reads openai/gsm8k over hf-mirror. None import vLLM or math_verify
  # (SDPO/SSD rewards are in-script regex), so no CUDA-only deps are pulled.
  echo "installing TRL from source at ${TARGET_ROOT} with peft extra"
  python -m pip install -e "${TARGET_ROOT}[peft]"
  python -c "import trl; print('TRL version:', trl.__version__)"
  ms_download_models \
    "SSD_MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct" \
    "SDPO_MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct"
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

select_pip_index
python -m pip install -U pip setuptools wheel
ensure_torch_stack
ensure_torchvision
prepare_fixtures

"setup_${PROFILE}"

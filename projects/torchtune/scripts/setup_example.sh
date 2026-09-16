#!/usr/bin/env bash
# Prepare the CI environment for one supported torchtune example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# torchtune itself is installed from TARGET_ROOT (the release checkout
# under test), so the guarded tag is exactly the code that runs.
#
# Dependency line (2026-09-15, coder hdc-stable-npu-4 逐例验证):
#   torch 2.9.0+cpu + torch_npu 2.9.0.dev20251120 + torchao <0.16
#   (torchao 0.15.0 与 torch 2.9.0+cpu ABI 匹配) + omegaconf
#   + transformers 4.57.1 + tokenizers + safetensors + modelscope
# - transformers 4.57.1：与 torchtune v0.6.x 的 LlamaModel / Qwen2
#   forward 签名匹配；5.x 已重命名部分属性
# - torchao<0.16：quantize recipe 走 torchao.quantization.quantize_；
#   0.15.0 与 torch 2.9.0+cpu 兼容（更高版本会报 C++ ABI 不匹配）
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
  # Same torch line as torchtune quick-start (CANN 9.1.0 pairing):
  # torch 2.9.0 + torch_npu 2.9.0. Pin via Huawei ascend index.
  if python -c "
import torch, torch_npu
raise SystemExit(
    0 if torch.__version__.startswith('2.9.0')
    and torch_npu.__version__.startswith('2.9.0') else 1)
"; then
    echo "reusing image torch stack ($(python -c 'import torch; print(torch.__version__)'))"
    return
  fi
  echo "installing torch==2.9.0 torch_npu==2.9.0"
  pip_ascend torch==2.9.0 torch_npu==2.9.0
}

# Copy CI fixture data files into the target root so that example
# recipes can load them via a local path under $TARGET_ROOT/fixtures/
# (engine contract: overlay_args uses ${TARGET_ROOT}/fixtures/...;
# same decoupling from the workflows-checkout subtree as peft).
prepare_fixtures() {
  local src="${FIXTURE_DIR:?FIXTURE_DIR is required}"
  local dst="$TARGET_ROOT/fixtures"
  echo "preparing fixtures from $src to $dst"
  if ! ls "$src"/*.{json,jsonl} 1>/dev/null 2>&1; then
    echo "FATAL: no fixture files (*.json / *.jsonl) found in $src" >&2
    exit 1
  fi
  mkdir -p "$dst"
  cp "$src"/*.json "$dst/" 2>/dev/null || true
  cp "$src"/*.jsonl "$dst/" 2>/dev/null || true
  echo "copied $(ls "$dst" 2>/dev/null | wc -l) fixture file(s) to $dst"
}

setup_torchtune() {
  # torchtune from the guarded release checkout, plus the verified
  # dependency line (2026-09-15, coder hdc-stable-npu-4 逐例验证):
  #   transformers 4.57.1 + omegaconf + torchao<0.16 + tokenizers
  #   + safetensors + tqdm + pyyaml + modelscope
  # PIP_CONSTRAINT keeps CUDA metapackages out (constraints-npu.txt).
  echo "installing torchtune from $TARGET_ROOT"
  python -m pip install -e "$TARGET_ROOT"
  python -m pip install "transformers==4.57.1" "omegaconf>=2.3,<3" \
    "torchao<0.16" tokenizers safetensors tqdm pyyaml
  python -c "import torchtune, torchao, omegaconf, transformers; print('torchtune', torchtune.__version__, '/ torchao', torchao.__version__, '/ transformers', transformers.__version__)"

  # Pre-download the example model from ModelScope (China-reachable)
  # because runners cannot reach HuggingFace. The local snapshot dir
  # is exported as TT_MODEL_PATH for overlay_args to reference.
  # Pinned to the doc's verified line: modelscope>=1.38 splits the hub
  # code into modelscope-hub, and the fresh 1.40.1 wheel's loose
  # ">=0.4.2" floor breaks import when the mirror lags on hub 0.4.3.
  python -m pip install "modelscope==1.37.0"
  python - <<'PY'
import os
# Non-TTY CI logs: throttle tqdm refreshes instead of disabling.
os.environ.setdefault("TQDM_MININTERVAL", "15")
from modelscope import snapshot_download

MODEL_CACHE = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))
local = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", cache_dir=MODEL_CACHE)
with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"TT_MODEL_PATH={local}\n")
print("TT_MODEL_PATH=", local)
PY
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

HERE=$(cd "$(dirname "$0")" && pwd)
export PIP_CONSTRAINT="$(cd "$HERE/.." && pwd)/constraints-npu.txt"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

select_pip_index
python -m pip install -U pip setuptools wheel
ensure_torch_stack
prepare_fixtures

"setup_${PROFILE}"

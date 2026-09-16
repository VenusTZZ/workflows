#!/usr/bin/env bash
# Prepare the CI environment for one supported peft example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# peft itself is installed from TARGET_ROOT (the release checkout under
# test), so the guarded tag is exactly the code that runs.
#
# The upstream examples/sft/requirements.txt is deliberately NOT
# installed: it pins everything to git main (transformers/peft/trl@main
# + flash-attn + unsloth + bitsandbytes), which conflicts with testing
# a release tag and contains CUDA-only packages. We install the
# checkout plus the minimal SFTTrainer stack instead.
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
  # Same torch line as peft quick-start (CANN 9.1.0 pairing): reuse
  # the image stack when it already matches, otherwise install.
  if python -c "
import torch, torch_npu
raise SystemExit(
    0 if torch.__version__.startswith('2.9.0')
    and torch_npu.__version__.startswith('2.9.0') else 1)
"; then
    echo "reusing image torch stack ($(python -c 'import torch; print(torch.__version__)'))"
    return
  fi
  echo "installing torch==2.9.0 torch_npu==2.9.0.post2"
  pip_ascend torch==2.9.0 torch_npu==2.9.0.post2
}

# Copy CI fixture data files into the target root so that example
# scripts can load them via a local path under $TARGET_ROOT/fixtures/
# (same decoupling from the workflows-checkout subtree as trl).
prepare_fixtures() {
  local src="${FIXTURE_DIR:?FIXTURE_DIR is required}"
  local dst="$TARGET_ROOT/fixtures"
  echo "preparing fixtures from $src to $dst"
  if ! ls "$src"/*.jsonl 1>/dev/null 2>&1; then
    echo "FATAL: no fixture files (*.jsonl) found in $src" >&2
    exit 1
  fi
  mkdir -p "$dst"
  cp "$src"/*.jsonl "$dst/"
  echo "copied $(ls "$dst"/*.jsonl 2>/dev/null | wc -l) fixture file(s) to $dst"
}

setup_peft() {
  # peft from the guarded release checkout, plus the verified dependency
  # line (2026-09-15, coder npu-1 端到端验证结论):
  #   transformers 4.57.1 + datasets>=4.7.0,<6 + hub<1.0 + trl 1.12.0
  # - transformers 4.57.1: 5.x 移除 send_example_telemetry 等旧 API
  # - datasets>=4.7.0,<6: trl 1.12+ 在 wheel metadata 声明 datasets>=4.7.0
  #   (pyproject.toml 自 v1.0.0 起 commit ac5421b4 引入，datasets<4 会
  #   ResolutionImpossible)，<6 留出口避开未来 6.x breaking
  # - hub<1.0: hub 1.x 拒绝 imdb 等无命名空间数据集
  # - trl 1.12.0: trl ≥ 1.12 都默认 chunked_nll，与 peft partial lm_head
  #   冲突（纯 PyTorch patch，sft_trainer.py:1331 检测到 peft 包了 head
  #   就 raise；CUDA 同问题，NPU 是首个端到端跑这条路径的环境）。
  #   overlay 在 examples_manifest.yaml 的 miss/mica 例里显式 --loss_type nll
  #   跳过 chunked patch 走标准 cross-entropy。
  # - scikit-learn: adamss 的 ASA 回调（peft.tuners.adamss）硬性 import
  #   sklearn；evaluate.load("glue") 的 metric 模块同样要 sklearn.metrics。
  #   coder 验证机里碰巧预装，CANN 裸镜像没有（run 35045940066 实测缺失）。
  # PIP_CONSTRAINT keeps CUDA metapackages out.
  echo "installing peft from $TARGET_ROOT"
  python -m pip install -e "$TARGET_ROOT"
  python -m pip install "transformers==4.57.1" "datasets>=4.7.0,<6" \
    "huggingface_hub<1.0" "trl==1.12.0" evaluate scikit-learn \
    torchvision==0.24.0
  python -c "import peft, trl, transformers, datasets, accelerate; print('peft', peft.__version__, '/ trl', trl.__version__, '/ transformers', transformers.__version__)"

  # Pre-download the example model from ModelScope (China-reachable)
  # because runners cannot reach HuggingFace. The local snapshot dir
  # is exported as SFT_MODEL_PATH for overlay_args to reference.
  # Pinned to the doc's verified line (1.37.0): the hub code split
  # started at 1.38 and 1.40.1's "modelscope-hub>=0.4.2" floor is too
  # loose — 1.40.1 + hub 0.4.2 (mirror-lagged) dies on
  # DEFAULT_CREDENTIALS_PATH import at modelscope import time.
  python -m pip install "modelscope==1.37.0"
  python - <<'PY'
import os
# Non-TTY CI logs: throttle tqdm refreshes instead of disabling.
os.environ.setdefault("TQDM_MININTERVAL", "15")
from modelscope import snapshot_download

MODEL_CACHE = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))
local = snapshot_download("Qwen/Qwen2.5-0.5B", cache_dir=MODEL_CACHE)
with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"SFT_MODEL_PATH={local}\n")
print("SFT_MODEL_PATH=", local)
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

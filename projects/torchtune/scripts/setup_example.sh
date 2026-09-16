#!/usr/bin/env bash
# Prepare the CI environment for one supported torchtune example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# torchtune itself is installed from TARGET_ROOT (the release checkout
# under test), so the guarded tag is exactly the code that runs.
#
# Dependency line (2026-09-16, coder hdc-stable-npu-4 逐例验证):
#   torch 2.12.0+cpu + torch_npu 2.12.0 + transformers 4.57.1
#   + omegaconf + tokenizers + safetensors + modelscope
# - transformers 4.57.1：与 torchtune v0.6.x 的 LlamaModel / Qwen2
#   forward 签名匹配；5.x 已重命名部分属性
# - torchao pin 由 setup_torchtune() 内部按 checkout 探测动态决定：
#   v0.6.x 走 torchao.dtypes.nf4tensor（pin 0.13.0），main HEAD 走
#   torchao.quantization（pin 0.18.0）。两个路径互不兼容，所以 pin
#   不能写死——CI 跑 release 时 setup 选 0.13.0，跑 main 时选 0.18.0
#   （setup_example.sh 内 probe common_utils.py:19 import line）
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
  # Skip if torch is already installed (any 2.x) to avoid downgrade:
  # the previous version-pinned check forced a 2.12.0+cpu image down to
  # 2.9.0, which broke the ABI alignment and pulled torchao cpp ext
  # warnings. Trust whatever the image preinstalled; fall through to
  # install only when torch is missing entirely.
  if python -c "import torch" 2>/dev/null; then
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
  # torchtune from the guarded checkout. The torchao pin is decided
  # dynamically because the NF4Tensor import path in
  # torchtune/modules/common_utils.py:19 has moved three times:
  #
  #   torchao 0.10-0.13 : torchao.dtypes.nf4tensor.NF4Tensor  (v0.6.x)
  #   torchao 0.14-0.17 : dtypes submodule deleted; quantization
  #                       renamed NF4Tensor -> Int4Tensor. Anything
  #                       importing NF4Tensor breaks here.
  #   torchao 0.18+     : torchao.quantization.NF4Tensor
  #                       re-exposed (main HEAD post PR #2960).
  #
  # Picking the wrong line is fatal — v0.6.x + torchao 0.18 throws
  # ModuleNotFoundError on import; main + torchao 0.13 throws the
  # same. There is no single torchao that satisfies both import paths,
  # so we must probe the actual checkout before pinning.
  echo "installing torchtune from $TARGET_ROOT"
  python -m pip install -e "$TARGET_ROOT"
  python -m pip install "transformers==4.57.1" "omegaconf>=2.3,<3" \
    tokenizers safetensors tqdm pyyaml

  # Probe which NF4Tensor import path the torchtune checkout uses, then
  # install the matching torchao exact pin. Probe runs against the
  # editable-installed source, so it reflects whatever ref the engine
  # checked out (release tag OR main HEAD).
  local import_path torchao_pin
  import_path="$(python -c "
import importlib.util, pathlib
p = pathlib.Path('$TARGET_ROOT') / 'torchtune' / 'modules' / 'common_utils.py'
src = p.read_text() if p.exists() else ''
for line in src.splitlines():
    s = line.strip()
    if s.startswith('from torchao') and 'NF4Tensor' in s:
        print(s)
        break
else:
    print('NF4Tensor_NOT_IMPORTED')
")"
  echo "torchtune common_utils.py NF4Tensor import: $import_path"
  case "$import_path" in
    "from torchao.dtypes.nf4tensor import NF4Tensor")
      torchao_pin="torchao==0.13.0"
      ;;
    "from torchao.quantization import NF4Tensor")
      # 0.18.0 re-exposed NF4Tensor under torchao.quantization; main
      # HEAD depends on the exact 0.18 series (later 0.18.x keep the
      # symbol but pin to whatever the upstream test grid currently
      # passes — 0.18.0 is the first stable release with the re-add).
      torchao_pin="torchao==0.18.0"
      ;;
    "NF4Tensor_NOT_IMPORTED")
      # Newer torchtune may drop NF4Tensor entirely. Default to a
      # neutral recent torchao and let setup proceed; recipe failures
      # downstream will surface real reasons rather than setup noise.
      echo "WARN: common_utils.py does not import NF4Tensor; skipping torchao pin"
      torchao_pin=""
      ;;
    *)
      echo "FATAL: unexpected NF4Tensor import line: $import_path" >&2
      exit 1
      ;;
  esac
  if [ -n "$torchao_pin" ]; then
    echo "pinning $torchao_pin (matched import path)"
    python -m pip install "$torchao_pin"
  fi

  # importlib.metadata.version returns the real install tag for both
  # editable installs (where __version__ is empty string) and wheel
  # installs. torchtune.__version__ is "" by default in the source
  # tree; reading it directly would print a confusing blank.
  python -c "
import importlib.metadata as md
import torchao, omegaconf, transformers
import torchtune  # noqa: just to confirm the import chain
print('torchtune', md.version('torchtune'), '/ torchao', torchao.__version__, '/ transformers', transformers.__version__)
"

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
# CI runner containers start with no /root/.cache/modelscope. The
# default cache path returned by os.path.expanduser is not created by
# modelscope itself — snapshot_download fails mid-transfer when the
# ._____temp staging dir cannot be opened (FileDownloadError on the
# *.safetensors file). mkdir -p is a no-op on coder where env.sh
# already exports MODELSCOPE_CACHE to /home/coder/work/modelscope-cache.
os.makedirs(MODEL_CACHE, exist_ok=True)
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

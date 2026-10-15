#!/usr/bin/env bash
# Prepare the CI environment for one supported example.
# $1 is the manifest profile. Unknown profiles fail before any install.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

MEGATRON_LM_REF=core_v0.16.0
MINDSPEED_REF=core_r0.16.0
MCORE_BRIDGE_REF=v1.6.1
VLLM_REF=v0.23.0
ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
ASCEND_MIRROR_PIP_INDEX=https://mirrors.huaweicloud.com/ascend/repos/pypi
ASCEND_VARIANT_PIP_INDEX=https://mirrors.huaweicloud.com/ascend/repos/pypi/variant
CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"
FALLBACK_PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

is_vllm_family() {
  [[ "$PROFILE" == "vllm" || "$PROFILE" == "megatron_vllm" ]]
}

pip_ascend() {
  python -m pip install --extra-index-url "$ASCEND_PIP_INDEX" "$@"
}

pip_ascend_variant() {
  python -m pip install \
    --extra-index-url "$ASCEND_VARIANT_PIP_INDEX" \
    --extra-index-url "$ASCEND_MIRROR_PIP_INDEX" \
    "$@"
}

select_pip_index() {
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
    export PIP_INDEX_URL="$FALLBACK_PIP_INDEX"
    unset PIP_TRUSTED_HOST
  fi
  echo "pip index: $PIP_INDEX_URL"
}

ensure_torch_stack() {
  if is_vllm_family; then
    if python -c "
import importlib
wanted = {
    'torch': '2.10.0',
    'torch_npu': '2.10.0.post4',
    'torchvision': '0.25.0',
    'torchaudio': '2.10.0',
}
for name, prefix in wanted.items():
    ver = importlib.import_module(name).__version__
    print('found', name, ver)
    if not ver.startswith(prefix):
        raise SystemExit(1)
"; then
      echo "reusing image torch 2.10 stack"
      return
    fi
    echo "installing torch==2.10.0 torch-npu==2.10.0.post4 torchvision==0.25.0 torchaudio==2.10.0"
    pip_ascend_variant torch==2.10.0 torch-npu==2.10.0.post4 \
      torchvision==0.25.0 torchaudio==2.10.0
    return
  fi
  if python -c "
import torch, torch_npu
print('found torch', torch.__version__, 'torch_npu', torch_npu.__version__)
raise SystemExit(0 if torch.__version__.startswith('2.9.0') and torch_npu.__version__.startswith('2.9.0') else 1)
"; then
    echo "reusing image torch stack"
    return
  fi
  echo "installing torch==2.9.0 torch_npu==2.9.0.post2"
  pip_ascend torch==2.9.0 torch_npu==2.9.0.post2
}

# framework.txt includes gradio / fastapi / uvicorn for web-ui and deploy.
install_ms_swift() {
  python -m pip install -e "$TARGET_ROOT" --no-deps
  python - "$TARGET_ROOT" "$PROFILE" <<'PY'
import re
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
profile = sys.argv[2]
skip = {"gradio", "fastapi", "uvicorn"}
deps = []
for raw in (root / "requirements" / "framework.txt").read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or line.startswith("-"):
        continue
    name = re.split(r"[<>=!~;\[]", line, maxsplit=1)[0].strip().lower()
    if name in skip:
        print(f"skip {line}", flush=True)
        continue
    deps.append(line)
# requirements/npu.txt pins torchvision for torch 2.7.1; non-vLLM uses 0.24.0
# for torch 2.9.0. vLLM family already has torchvision 0.25.0.
# decord has no aarch64 wheel or sdist. Omni still needs the other three.
deps.extend([
    "decorator",
    "qwen_vl_utils>=0.0.14",
    "qwen_omni_utils>=0.0.9",
    "soundfile",
    "audioread",
])
if profile not in {"vllm", "megatron_vllm"}:
    deps.append("torchvision==0.24.0")
subprocess.check_call([sys.executable, "-m", "pip", "install", *deps])
PY
  python -c "import soundfile, audioread; from qwen_omni_utils.v2_5 import vision_process"
}

setup_swift() {
  :
}

setup_deepspeed() {
  python -m pip install deepspeed
}

setup_vllm() {
  python -m pip install \
    "cmake>=3.26" pyyaml nanobind ninja setuptools-rust wheel \
    "setuptools-scm>=8" "setuptools>=77,<81"
  python -m pip install math_verify
  mkdir -p "$GITHUB_WORKSPACE/deps"
  git clone --depth 1 --branch "$VLLM_REF" \
    https://github.com/vllm-project/vllm.git "$GITHUB_WORKSPACE/deps/vllm"
  VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation \
    -e "$GITHUB_WORKSPACE/deps/vllm"
  pip_ascend_variant --no-build-isolation vllm-ascend==0.23.0
}

setup_megatron() {
  mkdir -p "$GITHUB_WORKSPACE/deps"
  git clone --depth 1 --branch "$MEGATRON_LM_REF" \
    https://github.com/NVIDIA/Megatron-LM.git "$GITHUB_WORKSPACE/deps/Megatron-LM"
  git clone --depth 1 --branch "$MINDSPEED_REF" \
    https://gitcode.com/Ascend/MindSpeed.git "$GITHUB_WORKSPACE/deps/MindSpeed"
  git clone --depth 1 --branch "$MCORE_BRIDGE_REF" \
    https://github.com/modelscope/mcore-bridge.git "$GITHUB_WORKSPACE/deps/mcore-bridge"
  python -m pip install -e "$GITHUB_WORKSPACE/deps/MindSpeed"
  python -m pip install -e "$GITHUB_WORKSPACE/deps/mcore-bridge"
  if [[ "$PROFILE" == "megatron_vllm" ]]; then
    pip_ascend_variant triton-ascend==3.2.2 \
      --find-links https://repo.huaweicloud.com/ascend/repos/pypi/triton-ascend/
  else
    pip_ascend triton-ascend==3.2.1 \
      --find-links https://repo.huaweicloud.com/ascend/repos/pypi/triton-ascend/
  fi
  echo "MEGATRON_LM_PATH=$GITHUB_WORKSPACE/deps/Megatron-LM" >> "$GITHUB_ENV"
  echo "PYTHONPATH=$GITHUB_WORKSPACE/deps/Megatron-LM" >> "$GITHUB_ENV"
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python -c "import mindspeed.megatron_adaptor; from swift.megatron.init import init_megatron_env; init_megatron_env()"
}

setup_megatron_vllm() {
  setup_megatron
  setup_vllm
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

# Redirect the two caches the legacy workflow provided via container
# bind-mounts (/data/ci-cache/modelscope, /data/ci-cache/pip) onto the
# shared persistent runner cache the pool mounts at ~/.cache/huggingface
# (docs/examples-guard-engine.md). GITHUB_ENV so the run step inherits
# MODELSCOPE_CACHE too: ms-swift downloads its models at run time.
SHARED_CACHE_ROOT="${HF_HOME:-${HOME}/.cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${SHARED_CACHE_ROOT}/modelscope}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SHARED_CACHE_ROOT}/pip}"
echo "MODELSCOPE_CACHE=${MODELSCOPE_CACHE}" >> "$GITHUB_ENV"

HERE=$(cd "$(dirname "$0")" && pwd)
if is_vllm_family; then
  export PIP_CONSTRAINT="$(cd "$HERE/.." && pwd)/constraints-npu-vllm.txt"
else
  export PIP_CONSTRAINT="$(cd "$HERE/.." && pwd)/constraints-npu.txt"
fi

# Later steps are a new shell; do not rely on a previous workflow step.
source /usr/local/Ascend/ascend-toolkit/set_env.sh

select_pip_index
python -m pip install -U pip
ensure_torch_stack
install_ms_swift

"setup_${PROFILE}"

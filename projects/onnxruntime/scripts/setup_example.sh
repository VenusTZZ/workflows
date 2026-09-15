#!/usr/bin/env bash
# Prepare the CI environment for one supported example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# Do not patch files under TARGET_ROOT.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

HUAWEI_PYPI=https://repo.huaweicloud.com/repository/pypi/simple
TORCH_CPU_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cpu
TORCH_CPU_VERSION=2.9.0

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

require_example() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "example missing: $path" >&2
    exit 1
  fi
}

install_ort_cpu_stack() {
  export PIP_INDEX_URL="$HUAWEI_PYPI"
  unset PIP_EXTRA_INDEX_URL
  python3 -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-cann || true
  python3 -m pip install --index-url "$HUAWEI_PYPI" \
    onnxruntime \
    onnx \
    'numpy<2' \
    "$@"
  python3 - <<'PY'
import onnxruntime as ort
from importlib.metadata import version

providers = ort.get_available_providers()
print('onnxruntime', version('onnxruntime'))
print('onnx', version('onnx'))
print('available_providers', providers)
if 'CANNExecutionProvider' in providers:
    raise SystemExit(
        'CPU profiles must install the CPU onnxruntime wheel, not onnxruntime-cann')
if 'CPUExecutionProvider' not in providers:
    raise SystemExit('CPUExecutionProvider missing after installing onnxruntime')
PY
}

install_cpu_torch() {
  unset PIP_EXTRA_INDEX_URL
  python3 -m pip uninstall -y torch torch_npu || true
  # Aliyun find-links is torch wheels only. Install the Python
  # closure from Huawei first, including onnxscript for
  # torch.onnx.export. --no-index then keeps the torch wheel on
  # that CPU page instead of an unpinned PyPI resolve.
  python3 -m pip install --index-url "$HUAWEI_PYPI" \
    filelock \
    'networkx>=2.5.1' \
    jinja2 \
    'fsspec>=0.8.5' \
    setuptools \
    'typing-extensions>=4.10.0' \
    'sympy>=1.13.3' \
    onnxscript
  unset PIP_INDEX_URL
  python3 -m pip install --no-index -f "$TORCH_CPU_FIND_LINKS" \
    "torch==${TORCH_CPU_VERSION}"
  python3 - <<'PY'
import torch

print('torch', torch.__version__)
cuda = torch.version.cuda
if '+cu' in torch.__version__ or cuda:
    raise SystemExit(
        f'cpu-python must install a CPU torch wheel, got {torch.__version__} cuda={cuda}')
try:
    import torch_npu
except ImportError:
    print('torch_npu absent')
else:
    raise SystemExit('cpu-python must not import torch_npu')
PY
}

setup_cpu-python() {
  install_ort_cpu_stack
  install_cpu_torch
  require_example "$TARGET_ROOT/python/api/getting_started.py"
  echo "PYTHONUNBUFFERED=1" >> "$GITHUB_ENV"
}

setup_quant-cpu() {
  # Host-only static quantizer. Do not install onnxruntime-cann:
  # quantize_static creates InferenceSession internally and would
  # pick CANN when that wheel is present.
  install_ort_cpu_stack pillow
  require_example "$TARGET_ROOT/quantization/image_classification/cpu/run.py"
  echo "PYTHONUNBUFFERED=1" >> "$GITHUB_ENV"
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

"setup_${PROFILE}"

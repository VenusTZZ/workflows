#!/usr/bin/env bash
# Run one bitsandbytes example from a CI working copy of the target tree.
# Overlay CLI args come from OVERLAY_ARGS (JSON array).
set -euo pipefail

export PYTHONNOUSERSITE=1

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"

case "$EXAMPLE_REL" in
  examples/cpu/cpu_training.py)
    KIND=cpu
    ;;
  examples/compile_inference.py)
    KIND=npu_compile
    ;;
  *)
    echo "unsupported bitsandbytes guard path: $EXAMPLE_REL" >&2
    exit 1
    ;;
esac

EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"
if [[ ! -f "$EXAMPLE_PATH" ]]; then
  echo "example not found: $EXAMPLE_PATH" >&2
  exit 1
fi

mkdir -p "$CI_OUTPUT_DIR"
RUN_LOG="$CI_OUTPUT_DIR/run.log"

expand_overlay() {
  python - <<'PY'
import json
import os
import shlex

raw = os.environ.get('OVERLAY_ARGS', '').strip()
if not raw or raw in ('null', '""'):
    raise SystemExit(0)
items = json.loads(raw)
if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
    raise SystemExit('OVERLAY_ARGS must be a JSON array of strings')
tokens = []
for item in items:
    tokens.extend(shlex.split(os.path.expandvars(item), posix=True))
print(' '.join(shlex.quote(token) for token in tokens))
PY
}

eval "EXTRA_ARGS=( $(expand_overlay) )"

export PATH="/usr/local/sbin:$PATH"
# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Hugging Face Hub is not reachable from the NPU runner as huggingface.co.
# hf-mirror is the CI network path; it does not change the example script.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export HF_HOME="${HF_HOME:-${GITHUB_WORKSPACE:-$CI_OUTPUT_DIR}/hf-cache}"

echo "running upstream bitsandbytes example: $EXAMPLE_REL"
cd "$TARGET_ROOT"

if [[ "$KIND" == "npu_compile" ]]; then
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${GITHUB_WORKSPACE:-$CI_OUTPUT_DIR}/.triton}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${GITHUB_WORKSPACE:-$CI_OUTPUT_DIR}/.inductor}"
  mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  # Keep the torch_npu allocator warning visible; it is the device anchor.
  export PYTHONWARNINGS="${PYTHONWARNINGS:-default}"
fi

python "$EXAMPLE_PATH" "${EXTRA_ARGS[@]}" 2>&1 | tee "$RUN_LOG"

if [[ "$KIND" == "cpu" ]]; then
  # cpu_training.py still exits 0 when loss does not fall; it only prints a warning.
  if ! grep -q 'OK: Loss decreased as expected' "$RUN_LOG"; then
    echo "cpu_training did not report a loss decrease; exit 0 is fake green" >&2
    exit 1
  fi
elif [[ "$KIND" == "npu_compile" ]]; then
  if ! grep -q 'NPUCachingAllocator' "$RUN_LOG"; then
    echo "compile_inference did not allocate via torch_npu; exit 0 is fake green" >&2
    exit 1
  fi
  if ! grep -q 'Write me a poem about Machine Learning' "$RUN_LOG"; then
    echo "compile_inference did not print the generated prompt/output" >&2
    exit 1
  fi
fi

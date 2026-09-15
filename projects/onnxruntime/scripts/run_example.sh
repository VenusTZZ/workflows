#!/usr/bin/env bash
# Run one example from a CI working copy of the target tree.
# Overlay CLI args come from OVERLAY_ARGS (JSON array). Do not
# patch the working copy. Never git add/commit/push.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"
PROFILE="${PROFILE:?PROFILE is required}"

EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"

if [[ ! -f "$EXAMPLE_PATH" ]]; then
  echo "example not found: $EXAMPLE_PATH" >&2
  exit 1
fi

mkdir -p "$CI_OUTPUT_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

expand_overlay() {
  "$PYTHON" - <<'PY'
import json
import os
import shlex

raw = os.environ.get('OVERLAY_ARGS', '').strip()
if not raw or raw in ('null', '""'):
    raise SystemExit(0)
try:
    items = json.loads(raw)
except json.JSONDecodeError as exc:
    raise SystemExit(f'OVERLAY_ARGS is not valid JSON: {exc}') from exc
if items in (None, ''):
    raise SystemExit(0)
if not isinstance(items, list):
    raise SystemExit(
        f'OVERLAY_ARGS must be a JSON array, got {type(items).__name__}')
tokens = []
for item in items:
    if not isinstance(item, str):
        raise SystemExit(
            f'OVERLAY_ARGS items must be strings, got {type(item).__name__}')
    tokens.extend(shlex.split(os.path.expandvars(item), posix=True))
print(' '.join(shlex.quote(token) for token in tokens))
PY
}

source_cann() {
  export PATH="/usr/local/sbin:/usr/local/bin:$PATH"
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
}

assert_cpu-python() {
  if ! grep -Eq '5(\.0*)?,?[[:space:]]+7(\.0*)?,?[[:space:]]+9(\.0*)?' "$RUN_LOG"; then
    echo "getting_started did not print the Add result 5 7 9" >&2
    exit 1
  fi
}

assert_quant-cpu() {
  if ! grep -qF 'Calibrated and quantized model saved.' "$RUN_LOG"; then
    echo "quantizer did not print Calibrated and quantized model saved." >&2
    exit 1
  fi
  if [[ ! -f "$CI_OUTPUT_DIR/mobilenetv2-7.quant.onnx" ]]; then
    echo "quantized model missing: $CI_OUTPUT_DIR/mobilenetv2-7.quant.onnx" >&2
    exit 1
  fi
  if grep -qF 'CANNExecutionProvider' "$RUN_LOG"; then
    echo "cpu/ quantization log mentioned CANNExecutionProvider; this profile is host-only" >&2
    exit 1
  fi
}

eval "EXTRA_ARGS=( $(expand_overlay) )"

echo "running $EXAMPLE_PATH with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

source_cann
export CI_OUTPUT_DIR ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

RUN_LOG="$CI_OUTPUT_DIR/$(basename "$EXAMPLE_REL").log"
EXAMPLE_DIR=$(dirname "$EXAMPLE_PATH")
cd "$EXAMPLE_DIR"
"$PYTHON" "$(basename "$EXAMPLE_PATH")" "${EXTRA_ARGS[@]}" 2>&1 | tee "$RUN_LOG"

if ! declare -F "assert_${PROFILE}" >/dev/null 2>&1; then
  echo "no stdout guard for profile: ${PROFILE}" >&2
  exit 1
fi
"assert_${PROFILE}"

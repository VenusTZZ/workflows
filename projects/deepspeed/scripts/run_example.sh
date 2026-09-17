#!/usr/bin/env bash
# Run one example from a CI working copy of the examples tree.
# $1 is the manifest entry path. EXEC, when set, names the launchable file
# relative to the examples root; otherwise path itself must be launchable.
# Overlay CLI args come from OVERLAY_ARGS (JSON array). Never git add/commit/push.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"

# Examples tree root: split mode sets EXAMPLES_ROOT (the DeepSpeedExamples
# checkout); non-split it falls back to TARGET_ROOT.
EXAMPLES_ROOT="${EXAMPLES_ROOT:-$TARGET_ROOT}"

EXAMPLE_PATH="$EXAMPLES_ROOT/$EXAMPLE_REL"
if [[ ! -f "$EXAMPLE_PATH" ]]; then
  echo "example not found: $EXAMPLE_PATH" >&2
  exit 1
fi

# Resolve the launchable file: EXEC (relative to the examples root) when set,
# otherwise path itself.
if [[ -n "${EXEC:-}" ]]; then
  LAUNCH_PATH="$EXAMPLES_ROOT/$EXEC"
else
  LAUNCH_PATH="$EXAMPLE_PATH"
fi
if [[ ! -f "$LAUNCH_PATH" ]]; then
  echo "launchable file not found: $LAUNCH_PATH" >&2
  exit 1
fi

mkdir -p "$CI_OUTPUT_DIR"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

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

eval "EXTRA_ARGS=( $(expand_overlay) )"

echo "running $LAUNCH_PATH with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

cd "$(dirname "$LAUNCH_PATH")"
export CI_OUTPUT_DIR ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

case "$LAUNCH_PATH" in
  *.sh)
    bash "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    ;;
  *.py)
    "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    ;;
  *)
    echo "unsupported example type: $LAUNCH_PATH" >&2
    exit 1
    ;;
esac

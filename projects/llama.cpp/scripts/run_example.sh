#!/usr/bin/env bash
# Run one example from a CI working copy of the target tree.
# Overlay CLI args come from OVERLAY_ARGS (JSON array). Never
# git add/commit/push.
# PROFILE comes from the workflow as an environment variable.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"
EXEC_REL="${EXEC:-}"

EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"

if [[ ! -e "$EXAMPLE_PATH" ]]; then
  echo "example not found: $EXAMPLE_PATH" >&2
  exit 1
fi
if [[ -n "$EXEC_REL" ]]; then
  EXEC_PATH="$TARGET_ROOT/$EXEC_REL"
  if [[ ! -f "$EXEC_PATH" ]]; then
    echo "exec not found: $EXEC_PATH" >&2
    exit 1
  fi
elif [[ -d "$EXAMPLE_PATH" ]]; then
  echo "path is a directory; set exec on the manifest entry: $EXAMPLE_REL" >&2
  exit 1
elif [[ ! -f "$EXAMPLE_PATH" ]]; then
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

eval "EXTRA_ARGS=( $(expand_overlay) )"

if [[ -n "$EXEC_REL" ]]; then
  echo "running $EXEC_PATH for $EXAMPLE_REL with ${#EXTRA_ARGS[@]} overlay args"
else
  echo "running $EXAMPLE_PATH with ${#EXTRA_ARGS[@]} overlay args"
fi
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

cd "$TARGET_ROOT"
export CI_OUTPUT_DIR ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

assert_cann_used() {
  local run_log="$1"
  # Host-only examples (the manifest's cpu-profile entries) never
  # touch the CANN device. Keyed on the example path: the engine's run
  # step passes no PROFILE env, and the cpu set is static here.
  case "$EXAMPLE_REL" in
    examples/convert-llama2c-to-ggml|examples/gguf|examples/gguf-hash|examples/gen-docs)
      return 0
      ;;
  esac
  if ! grep -q 'CANN[0-9]' "$run_log"; then
    echo "run used no CANN device; check the container card mounts" >&2
    exit 1
  fi
}

if [[ -n "$EXEC_REL" ]]; then
  RUN_LOG="$CI_OUTPUT_DIR/$(basename "$EXEC_REL").log"
  case "$EXAMPLE_REL" in
    examples/gguf)
      "$EXEC_PATH" "$CI_OUTPUT_DIR/ci-demo.gguf" w 2>&1 | tee "$RUN_LOG"
      "$EXEC_PATH" "$CI_OUTPUT_DIR/ci-demo.gguf" r 2>&1 | tee -a "$RUN_LOG"
      ;;
    examples/simple-chat)
      printf 'Hello\n' | "$EXEC_PATH" "${EXTRA_ARGS[@]}" 2>&1 | tee "$RUN_LOG"
      ;;
    examples/retrieval)
      printf 'hello\n' | "$EXEC_PATH" "${EXTRA_ARGS[@]}" 2>&1 | tee "$RUN_LOG"
      ;;
    *)
      "$EXEC_PATH" "${EXTRA_ARGS[@]}" 2>&1 | tee "$RUN_LOG"
      ;;
  esac
  assert_cann_used "$RUN_LOG"
  exit 0
fi
case "$EXAMPLE_PATH" in
  *.sh)
    bash "$EXAMPLE_PATH" "${EXTRA_ARGS[@]}"
    ;;
  *.py)
    "$PYTHON" "$EXAMPLE_PATH" "${EXTRA_ARGS[@]}"
    ;;
  *)
    echo "unsupported example type: $EXAMPLE_PATH" >&2
    exit 1
    ;;
esac

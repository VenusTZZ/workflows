#!/usr/bin/env bash
# Run one example from a CI working copy of the target tree.
# Overlay CLI args come from OVERLAY_ARGS (JSON array). Never
# git add/commit/push.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"
EXEC_REL="${EXEC:?EXEC is required}"

EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"

if [[ ! -e "$EXAMPLE_PATH" ]]; then
  echo "example not found: $EXAMPLE_PATH" >&2
  exit 1
fi

EXEC_PATH="$TARGET_ROOT/$EXEC_REL"
if [[ ! -f "$EXEC_PATH" ]]; then
  echo "exec not found: $EXEC_PATH" >&2
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

is_whisper_ggml() {
  local magic
  magic=$(head -c 4 "$1")
  [[ "$magic" == "lmgg" || "$magic" == "ggml" ]]
}

eval "EXTRA_ARGS=( $(expand_overlay) )"

echo "running $EXEC_PATH for $EXAMPLE_REL with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

if [[ -n "${WHISPER_CI_NODE_BIN:-}" ]]; then
  export PATH="${WHISPER_CI_NODE_BIN}:$PATH"
fi

cd "$TARGET_ROOT"
export CI_OUTPUT_DIR ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

RUN_LOG="$CI_OUTPUT_DIR/$(basename "$EXEC_REL").log"

assert_cann_used() {
  local run_log="$1"
  # Host-only profiles (quantize / test-cmake / vad) never touch the
  # CANN device. Keyed on the example path - the engine's run step
  # passes no PROFILE env (and the host-only set is exactly the case
  # branches below, so no manifest duplication).
  case "$EXAMPLE_REL" in
    examples/quantize|examples/parakeet-quantize|examples/test-cmake|examples/vad-speech-segments)
      return 0
      ;;
  esac
  if ! grep -Eq 'whisper_backend_init_gpu: using CANN[0-9] backend|CANN[0-9]' \
    "$run_log"; then
    echo "run used no CANN device; check the container card mounts" >&2
    exit 1
  fi
}

# parakeet-cli and whisper-cli load the model (CANN log) then continue
# with exit 0 if the wav is missing. CANN grep alone is a false green.
assert_jfk_transcript() {
  local label="$1"
  if ! grep -Eqi 'ask not what your country|fellow Americans' "$RUN_LOG"; then
    echo "$label printed no transcription; exit 0 is not enough" >&2
    exit 1
  fi
}

assert_quantize_output() {
  if ((${#EXTRA_ARGS[@]} < 2)); then
    echo "quantize overlay must be: input output type" >&2
    exit 1
  fi
  local out="${EXTRA_ARGS[1]}"
  if [[ ! -f "$out" ]] || ! is_whisper_ggml "$out"; then
    echo "quantize did not write a ggml file: $out" >&2
    exit 1
  fi
  local size
  size=$(wc -c < "$out")
  if (( size < 1000000 )); then
    echo "quantize output is too small to be a real weight file: $out ($size bytes)" >&2
    exit 1
  fi
  if ! grep -Eq 'quantize time|whisper_model_quantize|parakeet_model_quantize' \
    "$RUN_LOG"; then
    echo "quantize log missing success marker" >&2
    exit 1
  fi
}

stop_server() {
  local p="${1:-}"
  [[ -n "$p" ]] || return 0
  kill -TERM "$p" 2>/dev/null || true
  local _
  for _ in $(seq 1 30); do
    if ! kill -0 "$p" 2>/dev/null; then
      wait "$p" 2>/dev/null || true
      return 0
    fi
    sleep 1
  done
  kill -KILL "$p" 2>/dev/null || true
  wait "$p" 2>/dev/null || true
}

run_server() {
  local port=18080
  local i
  for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
    if [[ "${EXTRA_ARGS[$i]}" == "--port" && $((i + 1)) -lt ${#EXTRA_ARGS[@]} ]]; then
      port="${EXTRA_ARGS[$((i + 1))]}"
    fi
  done

  # Redirected stdout is fully buffered. Without line buffering the
  # listen line stays invisible until the process exits (fake timeout).
  local launch=("$EXEC_PATH")
  if command -v stdbuf >/dev/null 2>&1; then
    launch=(stdbuf -oL -eL "$EXEC_PATH")
  fi
  local pid=""
  set +e
  "${launch[@]}" "${EXTRA_ARGS[@]}" >"$RUN_LOG" 2>&1 &
  pid=$!
  set -e
  # Expand pid now. A single-quoted trap would see an empty local after
  # this function returns, and cat/curl failures would leak the process.
  # shellcheck disable=SC2064
  trap "stop_server $pid" EXIT

  local ready=0
  local _
  local health
  for _ in $(seq 1 90); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "whisper-server exited before listen" >&2
      cat "$RUN_LOG" >&2
      exit 1
    fi
    if grep -q 'whisper server listening' "$RUN_LOG"; then
      ready=1
      break
    fi
    health=$(curl -sS --connect-timeout 1 --max-time 2 \
      "http://127.0.0.1:${port}/health" 2>/dev/null || true)
    if [[ "$health" == *'"status":"ok"'* ]]; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" != 1 ]]; then
    echo "whisper-server did not become ready" >&2
    exit 1
  fi

  local resp="$CI_OUTPUT_DIR/whisper-server-inference.json"
  local http=000
  local curl_ec=0
  set +e
  http=$(curl -sS -o "$resp" -w '%{http_code}' --connect-timeout 10 \
    --max-time 180 \
    -F "file=@${TARGET_ROOT}/samples/jfk.wav" \
    -F temperature=0.0 \
    "http://127.0.0.1:${port}/inference")
  curl_ec=$?
  set -e
  {
    echo "inference HTTP ${http}"
    if [[ -f "$resp" ]]; then
      cat "$resp"
    else
      echo "(no response body)"
    fi
  } | tee -a "$RUN_LOG"

  if [[ "$curl_ec" != 0 || "$http" != "200" ]]; then
    echo "whisper-server /inference failed (curl=$curl_ec http=$http)" >&2
    exit 1
  fi
  # Default JSON 200 always contains a "text" key, including "text":"".
  # Require the JFK fixture phrase, same as addon.node.
  if [[ ! -f "$resp" ]] || ! grep -Eqi 'ask not what your country' "$resp"; then
    echo "whisper-server /inference response had no JFK transcription" >&2
    exit 1
  fi
  stop_server "$pid"
  trap - EXIT
}

case "$EXAMPLE_REL" in
  examples/quantize|examples/parakeet-quantize)
    "$EXEC_PATH" "${EXTRA_ARGS[@]}" 2>&1 | tee "$RUN_LOG"
    assert_quantize_output
    assert_cann_used "$RUN_LOG"
    ;;
  examples/test-cmake)
    "$EXEC_PATH" "${EXTRA_ARGS[@]}" 2>&1 | tee "$RUN_LOG"
    if ! grep -F '[test-cmake] version:' "$RUN_LOG"; then
      echo "test-cmake did not print its version line" >&2
      exit 1
    fi
    assert_cann_used "$RUN_LOG"
    ;;
  examples/vad-speech-segments)
    "$EXEC_PATH" "${EXTRA_ARGS[@]}" 2>&1 | tee "$RUN_LOG"
    if ! grep -Eq 'Detected [1-9][0-9]* speech segments' "$RUN_LOG" \
      || ! grep -q 'Speech segment' "$RUN_LOG"; then
      echo "vad-speech-segments did not report a speech segment" >&2
      exit 1
    fi
    assert_cann_used "$RUN_LOG"
    ;;
  examples/addon.node)
    if ! command -v node >/dev/null 2>&1; then
      echo "node is not on PATH; setup_node must export it" >&2
      exit 1
    fi
    node "$TARGET_ROOT/examples/addon.node/index.js" "${EXTRA_ARGS[@]}" \
      2>&1 | tee "$RUN_LOG"
    assert_jfk_transcript "addon.node"
    assert_cann_used "$RUN_LOG"
    ;;
  examples/parakeet-cli)
    "$EXEC_PATH" "${EXTRA_ARGS[@]}" 2>&1 | tee "$RUN_LOG"
    assert_jfk_transcript "parakeet-cli"
    assert_cann_used "$RUN_LOG"
    ;;
  examples/server)
    run_server
    assert_cann_used "$RUN_LOG"
    ;;
  *)
    "$EXEC_PATH" "${EXTRA_ARGS[@]}" 2>&1 | tee "$RUN_LOG"
    assert_cann_used "$RUN_LOG"
    ;;
esac

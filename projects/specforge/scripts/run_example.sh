#!/usr/bin/env bash
# Run one specforge example from a CI working copy of the target tree.
# $1 is the manifest entry path. EXEC, when set, names the launchable
# file relative to the target root; otherwise path itself must be a
# launchable file. Overlay CLI args come from OVERLAY_ARGS (JSON array,
# possibly []). Shell and python examples that do not pass "$@" get it
# attached in this working copy only (last command line). YAML recipes
# are dispatched to `specforge train -c <path>` so the manifest entry
# can be a typed run config without a wrapper script.
# Never git add/commit/push.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"

EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"
[[ -e "$EXAMPLE_PATH" ]] || { echo "example not found: $EXAMPLE_PATH" >&2; exit 1; }

# Resolve the launchable file: EXEC (relative to the target root) when
# set, otherwise path itself.
if [[ -n "${EXEC:-}" ]]; then
  LAUNCH_PATH="$TARGET_ROOT/$EXEC"
else
  LAUNCH_PATH="$EXAMPLE_PATH"
fi
if [[ ! -f "$LAUNCH_PATH" && ! -L "$LAUNCH_PATH" ]]; then
  echo "launchable file not found: $LAUNCH_PATH (directory examples need an exec field)" >&2
  exit 1
fi

mkdir -p "$CI_OUTPUT_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c "import torch, torch_npu; print('NPU available:', torch.npu.is_available(), 'devices:', torch.npu.device_count())"

expand_overlay() {
  # The workflow serializes manifest.overlay_args as JSON. Expand each
  # item with shell quoting intact, then allow CI paths such as
  # ${CI_OUTPUT_DIR} / ${SPECFORGE_MODEL_PATH} to resolve only in this
  # job's environment.
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
    if not isinstance(item, str) or not item.strip():
        raise SystemExit('OVERLAY_ARGS items must be non-empty strings')
    tokens.extend(shlex.split(os.path.expandvars(item), posix=True))
print(' '.join(shlex.quote(token) for token in tokens))
PY
}

eval "EXTRA_ARGS=( $(expand_overlay) )"

echo "running $LAUNCH_PATH with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

ensure_passthrough() {
  # Shell examples that already forward "$@" need no patch; otherwise
  # attach it in this CI working copy only, on the last non-comment
  # line (the tail of the example's main command). YAML recipes are
  # dispatched below and never get "$@" attached — specforge reads
  # its overrides from the CLI argument list, which is what we already
  # pass through EXTRA_ARGS.
  local script="$1"
  case "$script" in
    *.yaml|*.yml) return 0 ;;
  esac
  [[ "$script" == *.sh ]] || return 0
  if grep -qE '"\$@"' "$script"; then
    echo "example already has \"\$@\"; skipping patch"
    return
  fi
  "$PYTHON" - "$script" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
for i in range(len(lines) - 1, -1, -1):
    stripped = lines[i].strip()
    if stripped and not stripped.startswith('#'):
        raw = lines[i]
        newline = ''
        if raw.endswith('\r\n'):
            newline = '\r\n'
            raw = raw[:-2]
        elif raw.endswith('\n'):
            newline = '\n'
            raw = raw[:-1]
        lines[i] = raw.rstrip() + ' "$@"' + newline
        path.write_text(''.join(lines), encoding='utf-8')
        print(f'patched {path} to pass "$@" on last command line')
        raise SystemExit(0)
raise SystemExit(f'{path}: cannot find a command line to attach "$@"')
PY
}

ensure_passthrough "$LAUNCH_PATH"

# The `external/` disaggregated recipes expect an already-running Mooncake
# master + SGLang capture server (their endpoints are hard-coded in the YAML
# as 127.0.0.1:35551 / 35880 / 30000). specforge only plays producer+consumer,
# it never spawns those (only `managed_local` does). The examples engine has a
# single "Run example" step, so we bring the two prerequisites up here first —
# mirroring Quick-start-Ascend.md's smoke-start-mooncake / smoke-start-sglang —
# then run `specforge train` against them and tear them down afterwards.
start_mooncake() {
  pkill -9 -f '^mooncake_master' 2>/dev/null || true
  MOONCAKE_RPC_PORT="${SPECFORGE_MOONCAKE_RPC_PORT:-35551}"
  MOONCAKE_HTTP_PORT="${SPECFORGE_MOONCAKE_HTTP_PORT:-35880}"
  nohup mooncake_master \
    --enable_http_metadata_server=true \
    --rpc_port="$MOONCAKE_RPC_PORT" \
    --http_metadata_server_port="$MOONCAKE_HTTP_PORT" \
    --metrics_port=35903 \
    --enable_metric_reporting=false \
    >/tmp/examples-mooncake.log 2>&1 &
  MOONCAKE_PID=$!
  for _ in $(seq 1 30); do
    if "$PYTHON" -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.5)
try:
    s.connect(('127.0.0.1', $MOONCAKE_RPC_PORT))
except Exception:
    sys.exit(1)
finally:
    s.close()
" 2>/dev/null; then
      echo "mooncake master ready (rpc $MOONCAKE_RPC_PORT, pid=$MOONCAKE_PID)"
      return 0
    fi
    sleep 1
  done
  echo "FAILED - mooncake_master did not bind $MOONCAKE_RPC_PORT in 30s" >&2
  tail -50 /tmp/examples-mooncake.log >&2
  return 1
}

start_sglang_capture() {
  pkill -9 -f '^python -m sglang\.launch_server' 2>/dev/null || true
  CAPTURE_DEVICE="${SPECFORGE_CAPTURE_DEVICE:-0}"
  SGLANG_PORT="${SPECFORGE_SGLANG_PORT:-30000}"
  SGLANG_HEALTH_TIMEOUT="${SPECFORGE_SGLANG_HEALTH_TIMEOUT:-600}"
  MOONCAKE_RPC_PORT="${SPECFORGE_MOONCAKE_RPC_PORT:-35551}"
  MOONCAKE_HTTP_PORT="${SPECFORGE_MOONCAKE_HTTP_PORT:-35880}"
  export ASCEND_RT_VISIBLE_DEVICES=$CAPTURE_DEVICE
  export MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
  export MOONCAKE_METADATA_SERVER=http://127.0.0.1:$MOONCAKE_HTTP_PORT/metadata
  export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:$MOONCAKE_RPC_PORT
  export MOONCAKE_PROTOCOL=tcp
  export MOONCAKE_GLOBAL_SEGMENT_SIZE=$((32<<30))
  ATB_LIB=/usr/local/Ascend/nnal/atb/9.0.0/atb/cxx_abi_1/lib
  if [[ -d "$ATB_LIB" ]]; then
    export LD_LIBRARY_PATH="$ATB_LIB:${LD_LIBRARY_PATH:-}"
  fi
  : "${SPECFORGE_MODEL_PATH:?SPECFORGE_MODEL_PATH is required}"
  nohup python -m sglang.launch_server \
    --model-path "$SPECFORGE_MODEL_PATH" \
    --trust-remote-code \
    --skip-tokenizer-init \
    --tp-size 1 \
    --mem-fraction-static 0.5 \
    --context-length 1024 \
    --chunked-prefill-size -1 \
    --attention-backend ascend \
    --enable-spec-capture --spec-capture-method dflash \
    --spec-capture-aux-layer-ids 1 8 15 22 29 \
    --host 127.0.0.1 --port "$SGLANG_PORT" \
    >/tmp/examples-sglang.log 2>&1 &
  SGLANG_PID=$!
  HEALTH_DEADLINE=$((SGLANG_HEALTH_TIMEOUT / 5))
  for _ in $(seq 1 "$HEALTH_DEADLINE"); do
    if curl -fsS "http://127.0.0.1:$SGLANG_PORT/health" >/dev/null 2>&1; then
      echo "sglang capture server ready (pid=$SGLANG_PID)"
      return 0
    fi
    sleep 5
  done
  echo "FAILED - SGLang capture server not healthy after ${SGLANG_HEALTH_TIMEOUT}s" >&2
  tail -50 /tmp/examples-sglang.log >&2
  return 1
}

cleanup_services() {
  pkill -9 -f '^python -m sglang\.launch_server' 2>/dev/null || true
  pkill -9 -f '^mooncake_master' 2>/dev/null || true
}

# Dispatch on file extension:
#   *.yaml / *.yml: specforge typed run configs — invoke `specforge train -c`
#     and forward EXTRA_ARGS as dotted section.field=value overrides.
#   *.sh: shell example, cd into its directory and bash it.
#   * (default): python entry point, run from target root.
case "$LAUNCH_PATH" in
  *.yaml|*.yml)
    cd "$TARGET_ROOT"
    rm -rf outputs/qwen3.5-4b-dflash-npu-online
    start_mooncake
    start_sglang_capture
    trap cleanup_services EXIT
    ASCEND_RT_VISIBLE_DEVICES="${SPECFORGE_TRAINER_DEVICE:-1}" \
    HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=7200 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
      specforge train -c "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    ;;
  *.sh)
    cd "$(dirname "$LAUNCH_PATH")"
    bash "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    ;;
  *)
    cd "$TARGET_ROOT"
    "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    ;;
esac

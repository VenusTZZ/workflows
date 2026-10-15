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
PROFILE="${PROFILE:-}"

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

source_cann() {
  export PATH="/usr/local/sbin:/usr/local/bin:$PATH"
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
}

stop_pid() {
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

parse_listen_endpoint() {
  local log="$1"
  "$PYTHON" - "$log" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding='utf-8', errors='replace').read()
matches = re.findall(r'listening on (\S+:\d+)', text)
if not matches:
    raise SystemExit(1)
print(matches[-1])
PY
}

eval "EXTRA_ARGS=( $(expand_overlay) )"

echo "running $EXEC_PATH for $EXAMPLE_REL with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

case "$PROFILE" in
  ascend-direct|ascend-direct-http|host-tcp|host-oneshot|ascend-hccl) ;;
  *)
    echo "unknown profile for run: ${PROFILE:-empty} (supported: ascend-direct ascend-direct-http host-tcp host-oneshot ascend-hccl)" >&2
    exit 2
    ;;
esac

source_cann
export GLOG_logtostderr=1
export GLOG_alsologtostderr=1
cd "$TARGET_ROOT"

launch=("$EXEC_PATH")
if command -v stdbuf >/dev/null 2>&1; then
  launch=(stdbuf -oL -eL "$EXEC_PATH")
fi

CLEANUP_PIDS=()
remember_pid() {
  CLEANUP_PIDS+=("$1")
}
on_exit() {
  local i
  for ((i=${#CLEANUP_PIDS[@]}-1; i>=0; i--)); do
    stop_pid "${CLEANUP_PIDS[i]}"
  done
  if [[ -n "${META_LOG:-}" && -f "$META_LOG" ]]; then
    echo '===== metadata ====='
    cat "$META_LOG" || true
  fi
}
trap on_exit EXIT

# HTTP metadata does not rewrite the segment name to the dynamic RPC port.
# P2PHANDSHAKE does. Callers pass the id the initiator must open.
TARGET_NAME="127.0.0.1:12345"
INITIATOR_NAME="127.0.0.1:12346"

wait_for_endpoint() {
  local pid="$1"
  local log="$2"
  local endpoint=""
  local _
  for _ in $(seq 1 60); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "target exited before listen" >&2
      cat "$log" >&2
      return 1
    fi
    if endpoint=$(parse_listen_endpoint "$log"); then
      printf '%s\n' "$endpoint"
      return 0
    fi
    sleep 1
  done
  echo "target did not print a listening endpoint" >&2
  cat "$log" >&2
  return 1
}

assert_ascend_direct() {
  local combined="$1"
  local init_log="$2"
  local init_ec="$3"
  if [[ "$init_ec" != 0 ]]; then
    echo "initiator exited $init_ec" >&2
    exit "$init_ec"
  fi
  if grep -q 'Failed to install Ascend transport' "$combined"; then
    echo "Ascend Direct transport failed to install" >&2
    exit 1
  fi
  if grep -qE 'getTransferStatus FAILED|Sync data transfer timeout' "$init_log"; then
    echo "initiator reported a failed or timed-out transfer" >&2
    exit 1
  fi
  if ! grep -q 'Success to initialize adxl engine' "$combined"; then
    echo "ADXL engine did not initialize; check /etc/hccn.conf and card mounts" >&2
    exit 1
  fi
  if ! grep -q 'Test completed:' "$init_log"; then
    echo "initiator log missing Test completed" >&2
    exit 1
  fi
  # Example registerLocalMemory location is "npu:<logicid>". Keep mem type:device
  # as an alternate library phrasing seen on the same path.
  if ! grep -Eq 'npu:[0-9]+|mem type:device' "$combined"; then
    echo "run used no NPU device buffer; check card mounts and USE_ASCEND_DIRECT" >&2
    exit 1
  fi
}

run_ascend_direct() {
  local metadata="$1"
  local segment_from_listen="$2"
  local target_log="$CI_OUTPUT_DIR/target.log"
  local init_log="$CI_OUTPUT_DIR/initiator.log"
  local combined
  combined="$CI_OUTPUT_DIR/$(basename "$EXEC_PATH").log"
  local target_pid=""
  local endpoint=""
  local init_ec=0
  local put_ready=0

  set +e
  "${launch[@]}" \
    --mode=target \
    --device_logicid=0 \
    --local_server_name="$TARGET_NAME" \
    --metadata_server="$metadata" \
    "${EXTRA_ARGS[@]}" >"$target_log" 2>&1 &
  target_pid=$!
  set -e
  remember_pid "$target_pid"

  if ! endpoint=$(wait_for_endpoint "$target_pid" "$target_log"); then
    exit 1
  fi
  echo "target listening on $endpoint"
  if [[ "$segment_from_listen" != 1 ]]; then
    endpoint="$TARGET_NAME"
    echo "segment id is target local_server_name $endpoint"
    put_ready=0
    for _ in $(seq 1 30); do
      if [[ -f "${META_LOG:-}" ]] && grep -q 'PUT /metadata' "$META_LOG"; then
        put_ready=1
        break
      fi
      if ! kill -0 "$target_pid" 2>/dev/null; then
        echo "target exited before HTTP PUT" >&2
        cat "$target_log" >&2
        exit 1
      fi
      sleep 1
    done
    if [[ "$put_ready" != 1 ]]; then
      echo "no HTTP PUT before initiator start" >&2
      exit 1
    fi
  fi

  set +e
  "${launch[@]}" \
    --mode=initiator \
    --device_logicid=1 \
    --local_server_name="$INITIATOR_NAME" \
    --metadata_server="$metadata" \
    --segment_id="$endpoint" \
    --operation=write \
    "${EXTRA_ARGS[@]}" >"$init_log" 2>&1
  init_ec=$?
  set -e
  cat "$init_log"

  {
    echo '===== target ====='
    cat "$target_log"
    echo '===== initiator ====='
    cat "$init_log"
  } >"$combined"

  assert_ascend_direct "$combined" "$init_log" "$init_ec"
}

run_ascend_direct_http() {
  local meta_script="$TARGET_ROOT/mooncake-transfer-engine/example/http-metadata-server-python/bootstrap_server.py"
  local meta_pid=""
  local metadata="http://127.0.0.1:8080/metadata"
  META_LOG="$CI_OUTPUT_DIR/metadata.log"

  if [[ ! -f "$meta_script" ]]; then
    echo "metadata server script not found: $meta_script" >&2
    exit 1
  fi

  # The upstream server and HTTPStoragePlugin stay silent on a successful
  # PUT/GET. Turn on aiohttp.access so the run log can prove HTTP was used.
  # The file itself is still executed as __main__.
  PYTHONUNBUFFERED=1 "$PYTHON" - "$meta_script" >"$META_LOG" 2>&1 <<'PY' &
import logging
import runpy
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s %(levelname)s %(message)s',
)
runpy.run_path(sys.argv[1], run_name='__main__')
PY
  meta_pid=$!
  remember_pid "$meta_pid"

  local ready=0
  local _
  for _ in $(seq 1 30); do
    if ! kill -0 "$meta_pid" 2>/dev/null; then
      echo "metadata server exited before port 8080 was ready" >&2
      cat "$META_LOG" >&2
      exit 1
    fi
    if "$PYTHON" -c 'import socket,sys; s=socket.create_connection((sys.argv[1], int(sys.argv[2])), 2); s.close()' 127.0.0.1 8080 2>/dev/null; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" != 1 ]]; then
    echo "metadata server did not accept connections on 8080" >&2
    exit 1
  fi

  # segment_from_listen=0: HTTP keeps the name passed to --local_server_name.
  run_ascend_direct "$metadata" 0

  if ! grep -q 'PUT /metadata' "$META_LOG"; then
    echo "metadata access log has no HTTP PUT; transfer did not use the HTTP metadata server" >&2
    exit 1
  fi
}

parse_hccl_devices() {
  local raw="${ASCEND_RT_VISIBLE_DEVICES:-}"
  local mapped=""
  if [[ -z "$raw" ]]; then
    echo "ASCEND_RT_VISIBLE_DEVICES is empty; ascend-hccl needs two cards" >&2
    exit 1
  fi
  IFS=',' read -r -a HCCL_LOGIC <<< "$raw"
  if ((${#HCCL_LOGIC[@]} < 2)); then
    echo "ascend-hccl needs two devices in ASCEND_RT_VISIBLE_DEVICES, got: $raw" >&2
    exit 1
  fi
  # aclrtSetDevice uses the remapped logical ids 0 and 1. Physical ids for
  # hccn.conf and the npu_ suffix come from npu-smi when that mapping exists.
  HCCL_TARGET_LOGIC=0
  HCCL_INIT_LOGIC=1
  HCCL_TARGET_PHY="${HCCL_LOGIC[0]}"
  HCCL_INIT_PHY="${HCCL_LOGIC[1]}"
  if mapped=$("$PYTHON" - <<'PY'
import subprocess
import sys

try:
    text = subprocess.check_output(
        ['npu-smi', 'info', '-m'],
        text=True,
        stderr=subprocess.STDOUT,
    )
except (OSError, subprocess.CalledProcessError) as exc:
    raise SystemExit(f'npu-smi mapping unavailable: {exc}') from exc

logic_to_phy = {}
for line in text.splitlines():
    parts = line.split()
    if len(parts) < 3:
        continue
    npu, chip, logic = parts[0], parts[1], parts[2]
    if not npu.isdigit() or not chip.isdigit() or not logic.isdigit():
        continue
    if chip != '0':
        continue
    logic_to_phy[int(logic)] = int(npu)
if 0 not in logic_to_phy or 1 not in logic_to_phy:
    raise SystemExit('npu-smi mapping missing logic 0 or 1')
print(f'{logic_to_phy[0]},{logic_to_phy[1]}')
PY
  ); then
    IFS=',' read -r HCCL_TARGET_PHY HCCL_INIT_PHY <<< "$mapped"
    echo "HCCL phy ids from npu-smi: target=$HCCL_TARGET_PHY initiator=$HCCL_INIT_PHY"
  else
    echo "HCCL phy ids from ASCEND_RT_VISIBLE_DEVICES: target=$HCCL_TARGET_PHY initiator=$HCCL_INIT_PHY"
  fi
}

assert_ascend_hccl() {
  local combined="$1"
  local init_log="$2"
  local init_ec="$3"
  local name
  name=$(basename "$EXAMPLE_REL")
  if [[ "$init_ec" != 0 ]]; then
    echo "initiator exited $init_ec" >&2
    exit "$init_ec"
  fi
  if grep -q 'Failed to install Ascend transport' "$combined"; then
    echo "HCCL Ascend transport failed to install" >&2
    exit 1
  fi
  if grep -qE 'getTransferStatus FAILED|Sync data transfer timeout|Hccl transport failed|nicServerSocket_ Listen failed' "$combined"; then
    echo "initiator reported a failed or timed-out HCCL transfer" >&2
    exit 1
  fi
  "$PYTHON" - "$init_log" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding='utf-8', errors='replace').read()
matches = re.findall(
    r'local devicePhyId:\s*(\d+),\s*target devicePhyId:\s*(\d+)',
    text,
)
if not matches:
    raise SystemExit(
        'initiator log missing HCCL batch anchor '
        '(local devicePhyId / target devicePhyId)')
for local_id, target_id in matches:
    if local_id == target_id:
        raise SystemExit(
            f'HCCL batch used the same devicePhyId on both ends: {local_id}')
print(
    f'HCCL batch anchor ok, local={matches[-1][0]} target={matches[-1][1]}')
PY
  case "$name" in
    transfer_engine_ascend_one_sided.cpp)
      if ! grep -q 'The First Time Send OK' "$init_log"; then
        echo "initiator log missing The First Time Send OK" >&2
        exit 1
      fi
      if ! grep -q 'The Second Time Send OK' "$init_log"; then
        echo "initiator log missing The Second Time Send OK" >&2
        exit 1
      fi
      if ! grep -q 'Test completed:' "$init_log"; then
        echo "initiator log missing Test completed" >&2
        exit 1
      fi
      ;;
    transfer_engine_ascend_perf.cpp)
      if ! grep -q 'Test completed:' "$init_log"; then
        echo "initiator log missing Test completed" >&2
        exit 1
      fi
      ;;
    *)
      echo "ascend-hccl has no success rule for $name" >&2
      exit 2
      ;;
  esac
}

run_ascend_hccl() {
  parse_hccl_devices
  export ASCEND_TRANSPORT_PRINT=1
  unset MC_FORCE_TCP || true

  local so_dir="$TARGET_ROOT/build-ascend-hccl/mooncake-transfer-engine/src/transport/ascend_transport/hccl_transport/ascend_transport_c"
  if [[ -d "$so_dir" ]]; then
    export LD_LIBRARY_PATH="$so_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi

  local target_log="$CI_OUTPUT_DIR/target.log"
  local init_log="$CI_OUTPUT_DIR/initiator.log"
  local combined
  combined="$CI_OUTPUT_DIR/$(basename "$EXEC_PATH").log"
  local target_pid=""
  local endpoint=""
  local init_ec=0
  echo "HCCL target logicid=$HCCL_TARGET_LOGIC phyid=$HCCL_TARGET_PHY initiator logicid=$HCCL_INIT_LOGIC phyid=$HCCL_INIT_PHY"

  set +e
  "${launch[@]}" \
    --mode=target \
    --device_logicid="$HCCL_TARGET_LOGIC" \
    --device_phyid="$HCCL_TARGET_PHY" \
    --local_server_name="$TARGET_NAME" \
    --metadata_server=P2PHANDSHAKE \
    "${EXTRA_ARGS[@]}" >"$target_log" 2>&1 &
  target_pid=$!
  set -e
  remember_pid "$target_pid"

  if ! endpoint=$(wait_for_endpoint "$target_pid" "$target_log"); then
    exit 1
  fi
  echo "target listening on $endpoint"

  set +e
  "${launch[@]}" \
    --mode=initiator \
    --device_logicid="$HCCL_INIT_LOGIC" \
    --device_phyid="$HCCL_INIT_PHY" \
    --local_server_name="$INITIATOR_NAME" \
    --metadata_server=P2PHANDSHAKE \
    --segment_id="$endpoint" \
    --operation=write \
    "${EXTRA_ARGS[@]}" >"$init_log" 2>&1
  init_ec=$?
  set -e
  cat "$init_log"

  {
    echo '===== target ====='
    cat "$target_log"
    echo '===== initiator ====='
    cat "$init_log"
  } >"$combined"

  assert_ascend_hccl "$combined" "$init_log" "$init_ec"
}

assert_host_tcp() {
  local init_log="$1"
  local init_ec="$2"
  local name
  name=$(basename "$EXAMPLE_REL")
  if [[ "$init_ec" != 0 ]]; then
    echo "initiator exited $init_ec" >&2
    exit "$init_ec"
  fi
  case "$name" in
    transfer_engine_validator.cpp)
      if ! grep -q 'Data validation passed' "$init_log"; then
        echo "initiator log missing Data validation passed" >&2
        exit 1
      fi
      if ! grep -q 'Test completed:' "$init_log"; then
        echo "initiator log missing Test completed" >&2
        exit 1
      fi
      ;;
    transfer_engine_bench.cpp|transfer_engine_bench_with_notify.cpp)
      if grep -w -q 'FAILED' "$init_log"; then
        echo "initiator log contains FAILED" >&2
        exit 1
      fi
      if ! grep -q 'Test completed:' "$init_log"; then
        echo "initiator log missing Test completed" >&2
        exit 1
      fi
      ;;
    *)
      echo "host-tcp has no success rule for $name" >&2
      exit 2
      ;;
  esac
}

run_host_tcp() {
  local target_log="$CI_OUTPUT_DIR/target.log"
  local init_log="$CI_OUTPUT_DIR/initiator.log"
  local combined
  combined="$CI_OUTPUT_DIR/$(basename "$EXEC_PATH").log"
  local target_pid=""
  local endpoint=""
  local init_ec=0

  export MC_FORCE_TCP=1

  set +e
  "${launch[@]}" \
    --mode=target \
    --local_server_name="$TARGET_NAME" \
    --metadata_server=P2PHANDSHAKE \
    "${EXTRA_ARGS[@]}" >"$target_log" 2>&1 &
  target_pid=$!
  set -e
  remember_pid "$target_pid"

  if ! endpoint=$(wait_for_endpoint "$target_pid" "$target_log"); then
    exit 1
  fi
  echo "target listening on $endpoint"

  # bench_with_notify on current upstream main calls initiatorWorker on the
  # main thread before clearing `running`, so that process never returns.
  # Bound the wait so the job fails instead of sitting until timeout_minutes.
  set +e
  if command -v timeout >/dev/null 2>&1; then
    timeout --signal=TERM 180 \
      "${launch[@]}" \
      --mode=initiator \
      --local_server_name="$INITIATOR_NAME" \
      --metadata_server=P2PHANDSHAKE \
      --segment_id="$endpoint" \
      "${EXTRA_ARGS[@]}" >"$init_log" 2>&1
    init_ec=$?
  else
    "${launch[@]}" \
      --mode=initiator \
      --local_server_name="$INITIATOR_NAME" \
      --metadata_server=P2PHANDSHAKE \
      --segment_id="$endpoint" \
      "${EXTRA_ARGS[@]}" >"$init_log" 2>&1
    init_ec=$?
  fi
  set -e
  cat "$init_log"
  if [[ "$init_ec" == 124 ]]; then
    echo "initiator did not exit within 180s" >&2
    exit 1
  fi

  {
    echo '===== target ====='
    cat "$target_log"
    echo '===== initiator ====='
    cat "$init_log"
  } >"$combined"

  assert_host_tcp "$init_log" "$init_ec"
}

run_host_oneshot() {
  local stdout_log="$CI_OUTPUT_DIR/stdout.log"
  local stderr_log="$CI_OUTPUT_DIR/stderr.log"
  local init_ec=0

  set +e
  "${launch[@]}" "${EXTRA_ARGS[@]}" >"$stdout_log" 2>"$stderr_log"
  init_ec=$?
  set -e
  cat "$stderr_log" >&2 || true
  cat "$stdout_log"

  if [[ "$init_ec" != 0 ]]; then
    echo "show_link exited $init_ec" >&2
    exit "$init_ec"
  fi
  "$PYTHON" - "$stdout_log" <<'PY'
import json
import sys

text = open(sys.argv[1], encoding='utf-8', errors='replace').read()
try:
    data = json.loads(text)
except json.JSONDecodeError as exc:
    raise SystemExit(f'show_link stdout is not JSON: {exc}') from exc
if not isinstance(data, dict) or 'local_nics' not in data:
    raise SystemExit('show_link JSON is missing top-level local_nics')
print('show_link json ok, local_nics present')
PY
}

case "$PROFILE" in
  ascend-direct)
    run_ascend_direct P2PHANDSHAKE 1
    ;;
  ascend-direct-http)
    run_ascend_direct_http
    ;;
  host-tcp)
    run_host_tcp
    ;;
  host-oneshot)
    run_host_oneshot
    ;;
  ascend-hccl)
    run_ascend_hccl
    ;;
esac

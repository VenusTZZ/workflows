#!/usr/bin/env bash
# Run one supported ROLL example from a CI working copy of the target tree.
# $1 is the manifest entry path (a .yaml config relative to the target
# root). EXEC names the launchable file relative to the target root.
# Overlay CLI args come from OVERLAY_ARGS (a JSON array). Never git
# add/commit/push anything in the target tree.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"

EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"
[[ -e "$EXAMPLE_PATH" ]] || { echo "example config not found: $EXAMPLE_PATH" >&2; exit 1; }

if [[ -z "${EXEC:-}" ]]; then
  echo "ROLL examples are launched through a start_*_pipeline.py; EXEC is required" >&2
  exit 2
fi
LAUNCH_PATH="$TARGET_ROOT/$EXEC"
if [[ ! -f "$LAUNCH_PATH" ]]; then
  echo "launcher not found: $LAUNCH_PATH" >&2
  exit 1
fi
if [[ "$EXAMPLE_REL" != examples/*.yaml ]]; then
  echo "example path must be an examples/*.yaml config, got: $EXAMPLE_REL" >&2
  exit 2
fi

for path in "$EXAMPLE_PATH" "$LAUNCH_PATH"; do
  case "$(realpath "$path")" in
    "$(realpath "$TARGET_ROOT")"/*) ;;
    *)
      echo "refusing to run a path outside the target checkout: $path" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$CI_OUTPUT_DIR"

cleanup_ray() {
  local status=$?
  echo "cleaning up local Ray cluster (status $status)"
  ray stop --force >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup_ray EXIT

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi

# Single-node Ray contract for the thin engine. ROLL starts Ray itself and
# derives HCCL ranks from the per-worker ASCEND_RT_VISIBLE_DEVICES.
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=6379
export DASHBOARD_PORT=8265
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=0
export RAY_DEDUP_LOGS=0
export PYTHONPATH="$TARGET_ROOT:${PYTHONPATH:-}"
export MODEL_DOWNLOAD_TYPE="${MODEL_DOWNLOAD_TYPE:-MODELSCOPE}"
export USE_MODELSCOPE="1"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="1"
export TQDM_MININTERVAL="15"

echo "=== diagnostics ==="
df -h /dev/shm
nproc
python - <<'PY'
import os, sys, pathlib
import torch, torch_npu
import roll
target = pathlib.Path(os.environ["TARGET_ROOT"]).resolve()
source = pathlib.Path(roll.__path__[0]).resolve()
print("roll import path:", source)
print("NPU available:", torch.npu.is_available(), "devices:", torch.npu.device_count())
PY

expand_overlay() {
  python - <<'PY'
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
  printf 'overlay arg: %q
' "${EXTRA_ARGS[@]}"
fi

cd "$TARGET_ROOT"
python "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"

#!/usr/bin/env bash
# Run one supported slime example.
#
# $1 is the manifest entry path. The upstream launchers hardcode paths,
# models and GPU layouts without "$@" passthrough, so execution goes
# through projects/slime/scripts/ci_train_driver.py, which re-assembles
# the same train.py / train_async.py invocation with CI-sized parameters
# (recipe: the fork's NPU CI tests). Never git add/commit/push here.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"
EXAMPLES_ROOT="${EXAMPLES_ROOT:-$TARGET_ROOT}"
SLIME_FORK_ROOT="${SLIME_FORK_ROOT:?SLIME_FORK_ROOT was not exported by setup}"

EXAMPLE_PATH="$EXAMPLES_ROOT/$EXAMPLE_REL"
if [[ ! -f "$EXAMPLE_PATH" ]]; then
  echo "example not found: $EXAMPLE_PATH" >&2
  exit 1
fi

mkdir -p "$CI_OUTPUT_DIR"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
# shellcheck disable=SC1091
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true

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

echo "running $EXAMPLE_REL with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

entry_key="$EXAMPLE_REL"

require_visible_devices() {
  local required="$1"
  local default_devices="$2"
  local -a devices

  if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    export ASCEND_RT_VISIBLE_DEVICES="$default_devices"
  fi
  IFS=',' read -r -a devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
  if ((${#devices[@]} < required)); then
    echo "insufficient NPU devices for $entry_key: required=$required visible=$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
  fi
  echo "NPU devices for $entry_key: required=$required visible=$ASCEND_RT_VISIBLE_DEVICES"
}

# Environment contract copied from the fork's ascend launchers and NPU CI
# tests (scripts/ascend_script/run-qwen3-8B-npu.sh, tests/tests_npu/):
# Ray must not rewrite the visible-device mask, HCCL needs its port range
# and the NPU allocator keeps expandable_segments (no vLLM CaMemAllocator
# in this stack, unlike projects/roll).
require_visible_devices 4 '0,1,2,3'
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
# Offline wandb: get_default_wandb_args() stays quiet without
# WANDB_API_KEY, and WANDB_MODE=offline guarantees no network call even
# if one is set.
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

# run-id derived ray dashboard port avoids collisions between parallel
# matrix legs that share a runner.
export RAY_DASHBOARD_PORT=$((8265 + GITHUB_RUN_ID % 100))

cd "$SLIME_FORK_ROOT"

$PYTHON "$GITHUB_WORKSPACE/workflows/projects/slime/scripts/ci_train_driver.py" \
  --example "$entry_key" \
  --fork-root "$SLIME_FORK_ROOT" \
  --ci-output-dir "$CI_OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"


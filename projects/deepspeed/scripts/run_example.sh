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
if [[ ! -e "$EXAMPLE_PATH" ]]; then
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
export CI_OUTPUT_DIR
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

entry_key="${EXEC:-$EXAMPLE_REL}"

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

first_visible_devices() {
  local count="$1"
  local -a devices
  IFS=',' read -r -a devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
  local selected="${devices[0]}"
  local index
  for ((index = 1; index < count; index++)); do
    selected+=",${devices[index]}"
  done
  printf '%s\n' "$selected"
}

master_port_for() {
  local offset="$1"
  local run_id="${GITHUB_RUN_ID:-0}"
  printf '%s\n' "$((20000 + run_id % 20000 + offset))"
}

run_cifar_moe() {
  require_visible_devices 2 '0,1'
  local devices
  devices="$(first_visible_devices 2)"
  echo "reproducing upstream run_ds_moe.sh on devices $devices"
  echo "disabling optional torch.compile for DeepSpeed MoE helpers; eager MoE/EP execution is preserved"
  TORCH_COMPILE_DISABLE=1 \
  ASCEND_RT_VISIBLE_DEVICES="$devices" \
  deepspeed \
    --master_port "$(master_port_for 2)" \
    --num_nodes 1 \
    --num_gpus 2 \
    --bind_cores_to_rank \
    "$EXAMPLES_ROOT/training/cifar/cifar10_deepspeed.py" \
    --deepspeed \
    --moe \
    --ep-world-size 2 \
    --num-experts 2 \
    --top-k 1 \
    --noisy-gate-policy RSample \
    --moe-param-group \
    "${EXTRA_ARGS[@]}"
}

run_autotp_equivalence() {
  require_visible_devices 4 '0,1,2,3'
  : "${QWEN3_06B_PATH:?QWEN3_06B_PATH was not exported by setup}"

  local example_dir="$EXAMPLES_ROOT/training/autotp_equivalence"
  local metrics_dir="$CI_OUTPUT_DIR/autotp-equivalence"
  mkdir -p "$metrics_dir"

  local size devices
  for size in 1 3 4; do
    devices="$(first_visible_devices "$size")"
    echo "running AutoTP=$size on ASCEND_RT_VISIBLE_DEVICES=$devices"
    ASCEND_RT_VISIBLE_DEVICES="$devices" deepspeed \
      --master_port "$(master_port_for "$((10 + size))")" \
      --num_gpus "$size" \
      "$LAUNCH_PATH" \
      --deepspeed_config "$example_dir/configs/autotp${size}.json" \
      --metrics_file "$metrics_dir/autotp${size}.jsonl" \
      "${EXTRA_ARGS[@]}"
  done

  "$PYTHON" "$example_dir/compare_loss.py" \
    "$metrics_dir/autotp1.jsonl" "$metrics_dir/autotp3.jsonl" \
    --print-every 1
  "$PYTHON" "$example_dir/compare_loss.py" \
    "$metrics_dir/autotp1.jsonl" "$metrics_dir/autotp4.jsonl" \
    --print-every 1
}

case "$entry_key" in
  training/cifar/run_ds_moe.sh)
    run_cifar_moe
    exit 0
    ;;
  training/autotp_equivalence/train.py)
    run_autotp_equivalence
    exit 0
    ;;
  *)
    require_visible_devices 1 '0'
    ;;
esac

case "$LAUNCH_PATH" in
  *.sh)
    bash "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    ;;
  *.py)
    # DeepSpeed-Chat reads torch.distributed.get_rank() even when local_rank
    # keeps its default -1, so it still needs the launcher to initialize a
    # one-rank process group. Its upstream training_scripts wrappers hardcode
    # large models/configs and do not pass arbitrary overlay args through;
    # invoke the same launcher directly with our CI-sized manifest arguments.
    case "${EXEC:-}" in
      applications/DeepSpeed-Chat/training/*/main.py)
        deepspeed --master_port "$(master_port_for 1)" --num_gpus 1 \
          "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
        ;;
      *)
        "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
        ;;
    esac
    ;;
  *)
    echo "unsupported example type: $LAUNCH_PATH" >&2
    exit 1
    ;;
esac

#!/usr/bin/env bash
# Run one example from a CI working copy of the examples tree.
# $1 is the manifest entry path. EXEC, when set, names the launchable file
# relative to the examples root; otherwise path itself must be launchable.
# Overlay CLI args come from OVERLAY_ARGS (JSON array). Never git add/commit/push.
set -euo pipefail
PROJECT_SCRIPT_DIR="${DEEPSPEED_PROJECT_SCRIPTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

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

require_overlay_path() {
  local flag="$1"
  local kind="$2"
  local index value=''
  for ((index = 0; index < ${#EXTRA_ARGS[@]}; index++)); do
    if [[ "${EXTRA_ARGS[index]}" == "$flag" ]]; then
      value="${EXTRA_ARGS[index + 1]:-}"
      break
    fi
  done
  if [[ -z "$value" ]] ||
     { [[ "$kind" == directory ]] && [[ ! -d "$value" ]]; } ||
     { [[ "$kind" == file ]] && [[ ! -f "$value" ]]; }; then
    echo "missing local $kind for $entry_key: $flag=${value:-<unset>}" >&2
    exit 1
  fi
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

run_cifar_prmoe() {
  require_visible_devices 2 '0,1'
  local devices
  devices="$(first_visible_devices 2)"
  echo "reproducing upstream run_ds_prmoe.sh on devices $devices: EP=2, experts=2/4, residual MoE"
  echo "disabling optional torch.compile for DeepSpeed MoE helpers; eager PR-MoE/EP execution is preserved"
  TORCH_COMPILE_DISABLE=1 \
  ASCEND_RT_VISIBLE_DEVICES="$devices" \
  deepspeed \
    --master_port "$(master_port_for 3)" \
    --num_nodes 1 \
    --num_gpus 2 \
    "$EXAMPLES_ROOT/training/cifar/cifar10_deepspeed.py" \
    --deepspeed \
    --moe \
    --ep-world-size 2 \
    --num-experts 2 4 \
    --top-k 1 \
    --mlp-type residual \
    --noisy-gate-policy RSample \
    --moe-param-group \
    "${EXTRA_ARGS[@]}"
}

run_hf_autotp() {
  require_visible_devices 8 '0,1,2,3,4,5,6,7'
  require_overlay_path --model_name_or_path directory
  require_overlay_path --data_path file
  local devices config_path
  devices="$(first_visible_devices 8)"
  config_path="$CI_OUTPUT_DIR/hf-autotp-config.json"
  "$PYTHON" - "$EXAMPLES_ROOT/training/tensor_parallel/hf_integration/configs/ds_config_temp.json" "$config_path" <<'PY'
import json
import sys
from pathlib import Path

template = Path(sys.argv[1]).read_text()
config = json.loads(template.replace('${zero_stage}', '0').replace('${autotp_size}', '8'))
if config['zero_optimization']['stage'] != 0 or config['tensor_parallel']['autotp_size'] != 8:
    raise SystemExit('HF AutoTP CI requires ZeRO=0 and AutoTP=8')
Path(sys.argv[2]).write_text(json.dumps(config, indent=2) + '\n')
PY
  # The upstream example caches tokenized data in its working directory and
  # unconditionally saves the final model. Keep all generated files in CI output.
  if [[ ! -s "$CI_OUTPUT_DIR/hf-autotp-work/dataset_dict.pkl" ]]; then
    echo "HF AutoTP shared tokenization cache missing; run ds_hf_autotp setup first" >&2
    exit 1
  fi
  mkdir -p "$CI_OUTPUT_DIR/hf-autotp-work"
  cd "$CI_OUTPUT_DIR/hf-autotp-work"
  echo "running HF Trainer AutoTP=8 on devices $devices with config $config_path"
  ASCEND_RT_VISIBLE_DEVICES="$devices" WANDB_MODE=disabled deepspeed \
    --master_port "$(master_port_for 28)" \
    --num_gpus 8 \
    --no_local_rank \
    "$LAUNCH_PATH" \
    --deepspeed "$config_path" \
    "${EXTRA_ARGS[@]}"
}

overlay_value() {
  local flag="$1" index
  for ((index = 0; index < ${#EXTRA_ARGS[@]}; index++)); do
    if [[ "${EXTRA_ARGS[index]}" == "$flag" ]]; then
      printf '%s\n' "${EXTRA_ARGS[index + 1]:-}"
      return 0
    fi
  done
  return 1
}

run_chat_prompt_eval() {
  require_visible_devices 1 '0'
  require_overlay_path --model_name_or_path_baseline directory
  local devices baseline checkpoint finetune
  devices="$(first_visible_devices 1)"
  baseline="$(overlay_value --model_name_or_path_baseline)"
  checkpoint="$CI_OUTPUT_DIR/prompt-eval-sft"
  finetune="$(overlay_value --model_name_or_path_finetune)"
  if [[ "$finetune" != "$checkpoint" ]]; then
    echo "prompt eval requires the same-job SFT checkpoint: expected=$checkpoint actual=$finetune" >&2
    exit 1
  fi
  if [[ -e "$checkpoint/config.json" ]] || [[ -e "$checkpoint/pytorch_model.bin" ]]; then
    echo "prompt eval checkpoint already exists; refusing to reuse stale training output: $checkpoint" >&2
    exit 1
  fi
  echo "training a genuine SFT checkpoint before baseline/finetuned prompt evaluation"
  ASCEND_RT_VISIBLE_DEVICES="$devices" deepspeed \
    --master_port "$(master_port_for 51)" --num_gpus 1 \
    "$EXAMPLES_ROOT/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/main.py" \
    --model_name_or_path "$baseline" --data_path local/jsonfile \
    --num_train_epochs 1 --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 --max_seq_len 128 \
    --zero_stage 2 --dtype bf16 --print_loss --deepspeed \
    --data_output_path "$CI_OUTPUT_DIR/prompt-eval-data" \
    --output_dir "$checkpoint"
  # save_hf_format in the unchanged SFT entry writes these exact two files.
  if [[ ! -s "$checkpoint/config.json" ]] || [[ ! -s "$checkpoint/pytorch_model.bin" ]]; then
    echo "SFT did not produce a loadable checkpoint for prompt eval: $checkpoint" >&2
    exit 1
  fi
  echo "evaluating baseline against the newly trained SFT checkpoint on NPU $devices"
  ASCEND_RT_VISIBLE_DEVICES="$devices" "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
}

run_chat_reward_eval() {
  require_visible_devices 1 '0'
  require_overlay_path --model_name_or_path directory
  local devices
  devices="$(first_visible_devices 1)"
  echo "running upstream reward-head initialization/forward smoke on NPU $devices; not trained-head restoration"
  ASCEND_RT_VISIBLE_DEVICES="$devices" "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" \
    | tee "$CI_OUTPUT_DIR/reward-eval.log"
  "$PYTHON" - "$CI_OUTPUT_DIR/reward-eval.log" "$CI_OUTPUT_DIR/reward-eval-scores.json" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

scores = []
for line in Path(sys.argv[1]).read_text().splitlines():
    match = re.fullmatch(r'\s*(good_ans|bad_ans) score:\s*(\S+)\s*', line)
    if match:
        value = float(match.group(2))
        if not math.isfinite(value):
            raise SystemExit('reward evaluation produced a non-finite score')
        scores.append({'kind': match.group(1), 'score': value})
if [score['kind'] for score in scores] != ['good_ans', 'bad_ans'] * 2:
    raise SystemExit('reward evaluation must produce four scores for the two upstream preference pairs')
# An initialized reward head has no quality ordering guarantee; finite forward
# results, not good>bad, are the smoke acceptance criterion.
Path(sys.argv[2]).write_text(json.dumps(scores, indent=2, allow_nan=False) + '\n')
print('validated four finite reward scores (initialized-head forward smoke)')
PY
}

run_hf_bench_length() {
  require_visible_devices 8 '0,1,2,3,4,5,6,7'
  require_overlay_path --model_name_or_path directory
  require_overlay_path --data_path file
  local devices config_path
  devices="$(first_visible_devices 8)"
  config_path="$CI_OUTPUT_DIR/hf-bench-length-config.json"
  if [[ "$(overlay_value --model_max_length)" != 128 ]]; then
    echo "HF fixed-length CI recipe requires --model_max_length 128" >&2
    exit 1
  fi
  if [[ ! -s "$CI_OUTPUT_DIR/hf-bench-length-work/dataset_dict128.pkl" ]]; then
    echo "HF fixed-length shared tokenization cache missing; run ds_hf_bench_length setup first" >&2
    exit 1
  fi
  "$PYTHON" - "$EXAMPLES_ROOT/training/tensor_parallel/hf_integration/configs/ds_config_temp.json" "$config_path" <<'PY'
import json
import sys
from pathlib import Path

template = Path(sys.argv[1]).read_text()
config = json.loads(template.replace('${zero_stage}', '0').replace('${autotp_size}', '8'))
if config['zero_optimization']['stage'] != 0 or config['tensor_parallel']['autotp_size'] != 8:
    raise SystemExit('HF fixed-length CI requires ZeRO=0 and AutoTP=8')
Path(sys.argv[2]).write_text(json.dumps(config, indent=2) + '\n')
PY
  cd "$CI_OUTPUT_DIR/hf-bench-length-work"
  echo "running fixed-length HF Trainer AutoTP=8 on devices $devices"
  ASCEND_RT_VISIBLE_DEVICES="$devices" WANDB_MODE=disabled deepspeed \
    --master_port "$(master_port_for 61)" --num_gpus 8 --no_local_rank \
    "$LAUNCH_PATH" --deepspeed "$config_path" "${EXTRA_ARGS[@]}"
}

run_superoffload() {
  require_visible_devices 2 '0,1'
  require_overlay_path --model_name directory
  require_overlay_path --dataset_name directory
  local devices config_path
  devices="$(first_visible_devices 2)"
  config_path="$CI_OUTPUT_DIR/superoffload-config.json"
  "$PYTHON" - "$config_path" <<'PY'
import json
import sys
from pathlib import Path

# The unchanged entry creates DeepSpeedCPUAdam itself (LR=0.001); do not
# provide a duplicate optimizer or claim that the logging-only --lr sets it.
config = {
    'train_batch_size': 2,
    'train_micro_batch_size_per_gpu': 1,
    'gradient_accumulation_steps': 1,
    'bf16': {'enabled': True},
    'zero_optimization': {
        'stage': 3,
        'overlap_comm': False,
        'contiguous_gradients': True,
        'reduce_bucket_size': 65536,
        'stage3_prefetch_bucket_size': 65536,
        'stage3_param_persistence_threshold': 4096,
        'offload_param': {'device': 'cpu', 'pin_memory': False},
        'offload_optimizer': {'device': 'cpu', 'pin_memory': False},
    },
    'steps_per_print': 1,
    'wall_clock_breakdown': True,
}
Path(sys.argv[1]).write_text(json.dumps(config, indent=2) + '\n')
PY
  mkdir -p "$CI_OUTPUT_DIR/superoffload-work"
  cd "$CI_OUTPUT_DIR/superoffload-work"
  echo "running SuperOffload entry / ZeRO-Offload smoke on devices $devices; native super_offload is not enabled"
  ASCEND_RT_VISIBLE_DEVICES="$devices" WANDB_MODE=disabled deepspeed \
    --master_port "$(master_port_for 71)" --num_gpus 2 \
    "$LAUNCH_PATH" --deepspeed --deepspeed_config "$config_path" "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$CI_OUTPUT_DIR/superoffload.log"
  "$PYTHON" - "$CI_OUTPUT_DIR/superoffload.log" "$CI_OUTPUT_DIR/superoffload-losses.json" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

rank_zero_losses = {}
for line in Path(sys.argv[1]).read_text().splitlines():
    match = re.search(r'Step\s+(\d+)\s*\|\s*Loss:\s*(\S+)\s*\|', line)
    if not match:
        continue
    step, loss = int(match.group(1)), float(match.group(2))
    if not math.isfinite(loss):
        raise SystemExit(f'SuperOffload entry produced a non-finite loss at step {step}')
    # Upstream adds the timestamped handler only on rank 0. Its root logging
    # handler can repeat messages on other ranks, so tolerate duplicates and
    # check every reported loss, but require rank-0 evidence for all 3 steps.
    if ' - finetune_zero3 - INFO - ' in line:
        rank_zero_losses[step] = loss
if sorted(rank_zero_losses) != [1, 2, 3]:
    raise SystemExit('SuperOffload entry must report rank-0 finite losses for steps 1, 2 and 3')
report = [{'step': step, 'loss': rank_zero_losses[step]} for step in sorted(rank_zero_losses)]
Path(sys.argv[2]).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print('validated three finite SuperOffload entry train-step losses (ZeRO-Offload smoke)')
PY
}

run_variable_batch() {
  require_visible_devices 1 '0'
  local devices
  devices="$(first_visible_devices 1)"
  mkdir -p "$CI_OUTPUT_DIR/variable-batch-work"
  cd "$CI_OUTPUT_DIR/variable-batch-work"
  echo "running upstream dynamic sequence packing and LR scaling on device $devices"
  ASCEND_RT_VISIBLE_DEVICES="$devices" deepspeed \
    --master_port "$(master_port_for 31)" \
    --num_gpus 1 \
    --no_local_rank \
    "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
}

run_zenflow() {
  require_visible_devices 2 '0,1'
  local devices
  devices="$(first_visible_devices 2)"
  mkdir -p "$CI_OUTPUT_DIR/zenflow-work"
  cd "$CI_OUTPUT_DIR/zenflow-work"
  echo "running two-rank ZenFlow optimizer-offload smoke on devices $devices"
  ASCEND_RT_VISIBLE_DEVICES="$devices" deepspeed \
    --master_port "$(master_port_for 42)" \
    --num_gpus 2 \
    "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
}

run_rac_prune() {
  require_visible_devices 1 '0'
  require_overlay_path --model directory
  require_overlay_path --dataset file
  local devices
  devices="$(first_visible_devices 1)"
  echo "running upstream calibration-based pruning on NPU $devices"
  # Import the vendor backend normally before executing the unchanged entry.
  # runpy needs the sibling rac package on sys.path, as ordinary python does.
  ASCEND_RT_VISIBLE_DEVICES="$devices" "$PYTHON" -c '
import os
import runpy
import sys
import torch_npu
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(sys.argv[0]))
runpy.run_path(sys.argv[0], run_name="__main__")
' "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
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

source "$PROJECT_SCRIPT_DIR/run_expansion_examples.sh"
case "$entry_key" in
  benchmarks/pin_memory/model_tensor_offload/bench.py|benchmarks/pin_memory/activation_offload/bench.py|benchmarks/pin_memory/h2d_d2h/bench.py|training/DeepSpeed-ZenFlow/finetuning/finetune_llama.py|training/opsd/main.py|training/opsd/test_student_autotp_zero3.py|training/opsd/test_teacher_autotp_zero3.py)
    run_expansion_example
    exit 0
    ;;
esac

source "$PROJECT_SCRIPT_DIR/run_inference_examples.sh"
if is_new_inference_entry; then
  run_new_inference
  exit 0
fi

source "$PROJECT_SCRIPT_DIR/run_training_expansion.sh"
if is_training_expansion_entry; then
  run_training_expansion_executor
  exit 0
fi

case "$entry_key" in
  training/cifar/run_ds_moe.sh)
    run_cifar_moe
    exit 0
    ;;
  training/cifar/run_ds_prmoe.sh)
    run_cifar_prmoe
    exit 0
    ;;
  training/tensor_parallel/hf_integration/train.py)
    run_hf_autotp
    exit 0
    ;;
  applications/DeepSpeed-Chat/training/step1_supervised_finetuning/prompt_eval.py)
    run_chat_prompt_eval
    exit 0
    ;;
  applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/rw_eval.py)
    run_chat_reward_eval
    exit 0
    ;;
  training/tensor_parallel/hf_integration/train_bench_length.py)
    run_hf_bench_length
    exit 0
    ;;
  training/DeepSpeed-SuperOffload/finetune_zero3.py)
    run_superoffload
    exit 0
    ;;
  training/data_efficiency/variable_batch_size_and_lr/variable_batch_size_and_lr_example.py)
    run_variable_batch
    exit 0
    ;;
  training/DeepSpeed-ZenFlow/benchmark/zf_benchmark.py)
    run_zenflow
    exit 0
    ;;
  compression/reasoning_aware_compression/prune.py)
    run_rac_prune
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

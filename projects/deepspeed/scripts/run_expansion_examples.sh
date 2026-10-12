#!/usr/bin/env bash
# Project launch recipes only; all upstream entry points execute unchanged.

run_pin_memory_example() {
  require_visible_devices 1 '0'
  local experiment="$1" devices log
  devices="$(first_visible_devices 1)"
  log="$CI_OUTPUT_DIR/$experiment.log"
  # Limits are per-process: setup's soft limit does not carry to this step.
  local memlock_hard
  memlock_hard="$(ulimit -Hl)"
  ulimit -Sl "$memlock_hard"
  # Upstream drivers own the arm subprocesses and allocate fresh rendezvous
  # ports. Do not wrap the two-arm driver in a multi-rank launcher.
  ASCEND_RT_VISIBLE_DEVICES="$devices" "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" 2>&1 | tee "$log"
  "$PYTHON" "$PROJECT_SCRIPT_DIR/validate_expansion.py" \
    "$experiment" "$log" "$CI_OUTPUT_DIR/$experiment-results.json"
}

run_zenflow_finetune() {
  require_visible_devices 1 '0'
  require_overlay_path --model_name directory
  : "${ZENFLOW_WORK_DIR:?ZENFLOW_WORK_DIR was not exported by setup}"
  [[ -s "$ZENFLOW_WORK_DIR/tatsu-lab/alpaca/train.parquet" ]] || {
    echo 'ZenFlow local Alpaca parquet fixture is missing' >&2; exit 1;
  }
  local config="$CI_OUTPUT_DIR/zenflow-finetune-config.json" devices
  devices="$(first_visible_devices 1)"
  "$PYTHON" - "$config" <<'PY'
import json
from pathlib import Path
import sys
config = {
    "train_batch_size": 1, "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1, "gradient_clipping": 1.0,
    "bf16": {"enabled": True}, "zero_allow_untested_optimizer": True,
    "zero_optimization": {"stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": False},
        "zenflow": {"topk_ratio": 0.1, "update_interval": 2,
                    "full_warm_up_rounds": 0, "overlap_step": False}},
    "optimizer": {"type": "AdamW", "params": {"lr": 2e-5,
        "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01}},
}
Path(sys.argv[1]).write_text(json.dumps(config, indent=2) + '\n')
PY
  cd "$ZENFLOW_WORK_DIR"
  # Native datasets local-directory resolution of the original fixed ID.
  # Offline mode makes any accidental fallback to the Hub fail explicitly.
  ASCEND_RT_VISIBLE_DEVICES="$devices" HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    deepspeed --master_port "$(master_port_for 81)" --num_gpus 1 \
    "$LAUNCH_PATH" --deepspeed --deepspeed_config "$config" "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$CI_OUTPUT_DIR/zenflow-finetune.log"
  "$PYTHON" "$PROJECT_SCRIPT_DIR/validate_expansion.py" zenflow_finetune \
    "$CI_OUTPUT_DIR/zenflow-finetune.log" "$CI_OUTPUT_DIR/zenflow-finetune-results.json" \
    "$(overlay_value --output_dir)"
}

run_opsd() {
  require_visible_devices 1 '0'
  : "${QWEN25_STUDENT_PATH:?QWEN25_STUDENT_PATH was not exported by setup}"
  : "${QWEN25_TEACHER_PATH:?QWEN25_TEACHER_PATH was not exported by setup}"
  : "${OPSD_CI_PATH:?OPSD_CI_PATH was not exported by setup}"
  [[ -d "$QWEN25_STUDENT_PATH" && -d "$QWEN25_TEACHER_PATH" && -s "$OPSD_CI_PATH" ]] || {
    echo 'OPSD requires two local ModelScope models and the local prompt fixture' >&2; exit 1;
  }
  local config="$CI_OUTPUT_DIR/opsd-config.json" devices
  devices="$(first_visible_devices 1)"
  "$PYTHON" - "$config" <<'PY'
import json
import os
from pathlib import Path
import sys
output = Path(os.environ['CI_OUTPUT_DIR'])
ds_config = output / 'opsd-student-config.json'
ds_config.write_text(json.dumps({
    'train_micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': 1,
    'bf16': {'enabled': True}, 'zero_optimization': {'stage': 0},
    'optimizer': {'type': 'AdamW', 'params': {'lr': 1e-6}},
    'gradient_clipping': 1.0,
}, indent=2) + '\n')
cfg = {
    'student': {'model_name_or_path': os.environ['QWEN25_STUDENT_PATH'], 'dtype': 'bfloat16'},
    'teacher': {'model_name_or_path': os.environ['QWEN25_TEACHER_PATH'], 'dtype': 'bfloat16',
                'offload_to_cpu': True, 'autotp_size': 1},
    'rollout': {'engine': 'hybrid_engine', 'max_prompt_length': 512,
                'max_response_length': 8, 'temperature': 0.0, 'n_samples_per_prompt': 1},
    'distillation': {'loss_type': 'reverse_kl', 'temperature': 1.0, 'chunk_size': 8},
    'training': {'micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': 1,
                 'learning_rate': 1e-6, 'num_train_epochs': 1, 'max_steps': 3,
                 'logging_steps': 1, 'save_steps': 500, 'save_dir': str(output / 'opsd-checkpoints')},
    'data': {'path': os.environ['OPSD_CI_PATH'], 'prompt_field': 'prompt', 'shuffle': False},
    'deepspeed_config': str(ds_config),
}
Path(sys.argv[1]).write_text(json.dumps(cfg, indent=2) + '\n')
PY
  # Default rollout delegates to the original student module.generate(),
  # then the original trainer does teacher-cache + streamed KL backward.
  ASCEND_RT_VISIBLE_DEVICES="$devices" HF_HUB_OFFLINE=1 \
    deepspeed --master_port "$(master_port_for 91)" --num_gpus 1 \
    "$LAUNCH_PATH" --config "$config" "${EXTRA_ARGS[@]}" 2>&1 | tee "$CI_OUTPUT_DIR/opsd.log"
  "$PYTHON" "$PROJECT_SCRIPT_DIR/validate_expansion.py" opsd \
    "$CI_OUTPUT_DIR/opsd.log" "$CI_OUTPUT_DIR/opsd-results.json"
}

run_opsd_isolated() {
  local role="$1" devices log
  require_visible_devices 2 '0,1'
  devices="$(first_visible_devices 2)"
  : "${QWEN25_STUDENT_PATH:?QWEN25_STUDENT_PATH was not exported by setup}"
  : "${QWEN25_TEACHER_PATH:?QWEN25_TEACHER_PATH was not exported by setup}"
  [[ -d "$QWEN25_STUDENT_PATH" && -d "$QWEN25_TEACHER_PATH" ]] || {
    echo 'isolated OPSD requires local ModelScope model paths' >&2; exit 1;
  }
  log="$CI_OUTPUT_DIR/opsd-$role.log"
  # The upstream ENV knobs are the interface; no model/source alias is needed.
  # Student's optional CPU optimizer offload is disabled because its explicitly
  # selected torch AdamW is not the ZeRO CPUAdam recipe; TP=2 + ZeRO-3 remain.
  # Legacy mem= output uses uninitialized CUDA stats (zero), not NPU memory.
  ASCEND_RT_VISIBLE_DEVICES="$devices" AUTOTP=2 \
    OFFLOAD="$([[ "$role" == teacher ]] && printf 1 || printf 0)" \
    STUDENT_MODEL="$QWEN25_STUDENT_PATH" TEACHER_MODEL="$QWEN25_TEACHER_PATH" \
    deepspeed --master_port "$(master_port_for "$([[ "$role" == teacher ]] && printf 93 || printf 92)")" \
    --num_gpus 2 --no_local_rank "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" 2>&1 | tee "$log"
  "$PYTHON" "$PROJECT_SCRIPT_DIR/validate_expansion.py" "opsd_$role" "$log" \
    "$CI_OUTPUT_DIR/opsd-$role-results.json"
}

run_expansion_example() {
  case "$entry_key" in
    benchmarks/pin_memory/model_tensor_offload/bench.py) run_pin_memory_example model_tensor_offload ;;
    benchmarks/pin_memory/activation_offload/bench.py) run_pin_memory_example activation_offload ;;
    benchmarks/pin_memory/h2d_d2h/bench.py) run_pin_memory_example h2d_d2h ;;
    training/DeepSpeed-ZenFlow/finetuning/finetune_llama.py) run_zenflow_finetune ;;
    training/opsd/main.py) run_opsd ;;
    training/opsd/test_student_autotp_zero3.py) run_opsd_isolated student ;;
    training/opsd/test_teacher_autotp_zero3.py) run_opsd_isolated teacher ;;
    *) return 1 ;;
  esac
  # Called outside a conditional so errexit/pipefail remain active.
}

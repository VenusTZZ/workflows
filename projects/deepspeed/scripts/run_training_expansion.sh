#!/usr/bin/env bash
# Sourced after the project runner has expanded overlay_args and device helpers.

is_training_expansion_entry() {
  case "$entry_key" in
    training/deepspeed_finetune_demo/finetune_llama.py|training/stable_diffusion/train_sd_distil_lora.py|training/opsd/benchmarks/bench_decode_1p1r.py) return 0 ;;
    *) return 1 ;;
  esac
}

run_training_expansion_executor() {
  case "$entry_key" in
    training/deepspeed_finetune_demo/finetune_llama.py) run_finetune_demo_ci ;;
    training/stable_diffusion/train_sd_distil_lora.py) run_sd_distil_ci ;;
    training/opsd/benchmarks/bench_decode_1p1r.py) run_opsd_decode_ci ;;
    *) echo "unknown training expansion entry: $entry_key" >&2; return 1 ;;
  esac
}

run_finetune_demo_ci() {
  require_visible_devices 2 '0,1'
  require_overlay_path --model_name directory
  require_overlay_path --dataset_name directory
  local devices config log
  devices="$(first_visible_devices 2)"
  config="$CI_OUTPUT_DIR/finetune-demo-config.json"
  log="$CI_OUTPUT_DIR/finetune-demo.log"
  "$PYTHON" - "$config" <<'PY'
import json
from pathlib import Path
import sys

config = {
    'train_batch_size': 2, 'train_micro_batch_size_per_gpu': 1,
    'gradient_accumulation_steps': 1, 'bf16': {'enabled': True},
    'zero_optimization': {'stage': 2},
    'optimizer': {'type': 'AdamW', 'params': {'lr': 2e-5, 'torch_adam': True}},
    'gradient_clipping': 1.0, 'steps_per_print': 1,
}
Path(sys.argv[1]).write_text(json.dumps(config, indent=2) + '\n')
PY
  mkdir -p "$CI_OUTPUT_DIR/finetune-demo-work"
  cd "$CI_OUTPUT_DIR/finetune-demo-work"
  echo "running native SmolLM2/Llama finetune recipe: two-rank DP/ZeRO-2, three updates"
  ASCEND_RT_VISIBLE_DEVICES="$devices" WANDB_MODE=disabled deepspeed \
    --master_port "$(master_port_for 111)" --num_gpus 2 \
    "$LAUNCH_PATH" --deepspeed --deepspeed_config "$config" "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$log"
  "$PYTHON" - "$log" "$CI_OUTPUT_DIR/finetune-demo-losses.json" <<'PY'
import json
import math
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text()
if not re.search(r'First batch valid shifted labels: [1-9]\d*, finite params: True, finite logits: True,.*finite loss: True', text):
    raise SystemExit('finetune demo did not prove valid labels and finite first-batch tensors')
losses = {}
for match in re.finditer(r'Step\s+(\d+), Loss:\s*([^,\s]+)', text):
    step, value = int(match[1]), float(match[2])
    if not math.isfinite(value):
        raise SystemExit(f'finetune demo produced nonfinite loss at step {step}')
    losses[step] = value
if sorted(losses) != [1, 2, 3] or 'Training complete!' not in text:
    raise SystemExit('finetune demo must finish exactly three finite updates')
Path(sys.argv[2]).write_text(json.dumps(losses, allow_nan=False) + '\n')
print('validated three finite finetune demo updates')
PY
}

run_sd_distil_ci() {
  require_visible_devices 2 '0,1'
  require_overlay_path --pretrained_model_name_or_path directory
  local devices config output log
  devices="$(first_visible_devices 2)"
  config="$CI_OUTPUT_DIR/sd-distil-config.json"
  log="$CI_OUTPUT_DIR/sd-distil.log"
  output="$(overlay_value --output_dir)"
  [[ "$output" == "$CI_OUTPUT_DIR/sd-distil" ]] || {
    echo "SD distillation must save its fresh pipeline under CI output" >&2; return 1;
  }
  [[ ! -e "$output/model_index.json" ]] || {
    echo "SD distillation refuses to reuse a stale output pipeline" >&2; return 1;
  }
  [[ -s "$CI_OUTPUT_DIR/sd-distil-work/poloclub/diffusiondb/train.parquet" ]] || {
    echo "SD local eight-image dataset missing; run ds_sd_distil setup first" >&2; return 1;
  }
  [[ "$(overlay_value --mixed_precision)" == no ]] || {
    echo "SD unwrapped teacher requires the FP32 CI recipe (--mixed_precision no)" >&2; return 1;
  }
  "$PYTHON" - "$config" <<'PY'
import json
from pathlib import Path
import sys

config = {
    'train_batch_size': 2, 'train_micro_batch_size_per_gpu': 1,
    'gradient_accumulation_steps': 1, 'zero_optimization': {'stage': 2},
    'fp16': {'enabled': False}, 'bf16': {'enabled': False},
    'gradient_clipping': 1.0,
}
Path(sys.argv[1]).write_text(json.dumps(config, indent=2) + '\n')
PY
  cd "$CI_OUTPUT_DIR/sd-distil-work"
  echo "running teacher-CFG full-UNet distillation using native local images, FP32 and two-rank DeepSpeed"
  ASCEND_RT_VISIBLE_DEVICES="$devices" WANDB_MODE=disabled TQDM_MININTERVAL=0 \
  accelerate launch --use_deepspeed --num_processes 2 --num_machines 1 \
    --mixed_precision no --deepspeed_config_file "$config" \
    --main_process_port "$(master_port_for 121)" \
    "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" 2>&1 | tee "$log"
  "$PYTHON" - "$log" "$output" "$CI_OUTPUT_DIR/sd-distil-losses.json" <<'PY'
import json
import math
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text()
losses = {}
# set_postfix(refresh=True) in the original entry prints the current loss
# after each progress update. Parse CR-separated tqdm frames, not wall-clock speed.
for frame in re.split(r'[\r\n]', text):
    match = re.search(r'\b([123])/3\b.*?\bloss=([^,\]\s]+)', frame)
    if match:
        step, value = int(match[1]), float(match[2])
        if not math.isfinite(value):
            raise SystemExit(f'SD distillation nonfinite loss at step {step}')
        losses[step] = value
if sorted(losses) != [1, 2, 3]:
    raise SystemExit('SD distillation must display three finite train-step losses')
output = Path(sys.argv[2])
for name in ('model_index.json', 'unet/config.json', 'unet/diffusion_pytorch_model.safetensors'):
    file = output / name
    if not file.is_file() or not file.stat().st_size:
        raise SystemExit(f'SD distillation failed to save a fresh trained pipeline: {file}')
Path(sys.argv[3]).write_text(json.dumps(losses, allow_nan=False) + '\n')
print('validated three finite SD distillation updates and trained pipeline save')
PY
}

run_opsd_decode_ci() {
  require_visible_devices 1 '0'
  require_overlay_path --model directory
  local devices log
  devices="$(first_visible_devices 1)"
  log="$CI_OUTPUT_DIR/opsd-decode.log"
  mkdir -p "$CI_OUTPUT_DIR/opsd-decode-work"
  cd "$CI_OUTPUT_DIR/opsd-decode-work"
  # Normal backend import and runpy preserve every original API and function.
  ASCEND_RT_VISIBLE_DEVICES="$devices" "$PYTHON" -c '
import os, runpy, sys
import torch, torch_npu
from deepspeed.accelerator import get_accelerator
index = get_accelerator().current_device()
assert torch.empty(1, device="cpu").to(index).device.type == "npu"
assert torch.randint(10, 1000, (1, 2), device=index).device.type == "npu"
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(sys.argv[0]))
runpy.run_path(sys.argv[0], run_name="__main__")
' "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" 2>&1 | tee "$log"
  "$PYTHON" - "$log" "$CI_OUTPUT_DIR/opsd-decode-timings.json" <<'PY'
import json
import math
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text()
timings = {}
for label in ('Raw decode loop', 'HybridEngine rollout'):
    match = re.search(re.escape(label) + r':\s*([^\s]+)\s+ms', text)
    if not match:
        raise SystemExit(f'OPSD decode missing completed timing: {label}')
    value = float(match[1])
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(f'OPSD decode invalid timing: {label}={value}')
    timings[label] = value
Path(sys.argv[2]).write_text(json.dumps(timings, allow_nan=False) + '\n')
print('validated raw sampling/decode and HybridEngine rollout completion; no speed threshold')
PY
}

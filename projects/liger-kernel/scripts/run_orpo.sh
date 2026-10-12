#!/usr/bin/env bash
# Called only by the project runner. The original example is not edited.
set -euo pipefail
: "${LIGER_ORPO_WORK:?LIGER_ORPO_WORK was not exported by setup}"
[[ -f "$LIGER_ORPO_WORK/meta-llama/Llama-3.2-1B-Instruct/config.json" &&
   -s "$LIGER_ORPO_WORK/trl-lib/tldr-preference/train.parquet" ]] || {
  echo 'ORPO local model/preferences are missing' >&2; exit 1;
}
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export TORCH_COMPILE_DISABLE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LIGER_ORPO_SOURCE="$EXAMPLES_ROOT/examples/alignment/run_orpo.py"
"$PYTHON" - <<'PY'
import os
from pathlib import Path
import yaml

config = yaml.safe_load((Path(os.environ['EXAMPLES_ROOT']) / 'examples/alignment/accelerate_config.yaml').read_text())
config['num_processes'] = 2
config['mixed_precision'] = 'bf16'
config['fsdp_config']['fsdp_transformer_layer_cls_to_wrap'] = 'LlamaDecoderLayer'
output = Path(os.environ['CI_OUTPUT_DIR'])
(output / 'orpo-accelerate.yaml').write_text(yaml.safe_dump(config))
(output / 'orpo-entry.py').write_text('''import json
import math
import os
from pathlib import Path
import runpy
import torch
import torch_npu
from torch.distributed.fsdp import FullyShardedDataParallel

if not torch.npu.is_available() or torch.npu.device_count() < 2:
    raise SystemExit("ORPO requires two actual NPU devices")
torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
state = runpy.run_path(os.environ["LIGER_ORPO_SOURCE"], run_name="__main__")
trainer = state["trainer"]
if trainer.state.global_step != 100:
    raise SystemExit("original ORPO example must complete all 100 steps")
if not isinstance(trainer.model_wrapped, FullyShardedDataParallel):
    raise SystemExit("ORPO did not use its required FSDP wrapper")
losses = [float(row[key]) for row in trainer.state.log_history
          for key in ("loss", "train_loss") if key in row]
if not losses or not all(math.isfinite(loss) for loss in losses):
    raise SystemExit("ORPO must report finite train losses")
if trainer.args.device.type != "npu" or trainer.args.world_size != 2:
    raise SystemExit("ORPO did not train on two NPU ranks")
if trainer.is_world_process_zero():
    report = {"global_step": 100, "world_size": 2, "device": "npu",
              "fsdp": True, "losses": losses, "compiled": False}
    (Path(os.environ["CI_OUTPUT_DIR"]) / "orpo-results.json").write_text(json.dumps(report))
    print("NPU example passed: ORPO, 100 steps, two-rank FSDP, finite losses")
''')
PY
cd "$LIGER_ORPO_WORK"
"$PYTHON" -m accelerate.commands.launch \
  --config_file "$CI_OUTPUT_DIR/orpo-accelerate.yaml" \
  --main_process_port "$((20000 + ${GITHUB_RUN_ID:-0} % 20000))" \
  "$CI_OUTPUT_DIR/orpo-entry.py"

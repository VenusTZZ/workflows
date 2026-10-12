#!/usr/bin/env bash
# Prepare the CI environment for one supported AReaL example.
# $1 is the manifest profile. Unknown profiles fail before any install.
#
# CI base image is ghcr.io/hwvanici/areal_npu, which already has the full stack:
# CANN, torch, vLLM-Ascend, Megatron, MindSpeed, huggingface_hub etc.
# We should NOT reinstall these; just install the target AReaL source and assets.
# Exception (section 1b): the vllm/vllm-ascend pair is swapped to the 0.22.1
# line to match the guarded v2.x code's vllm layout.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

SUPPORTED_PROFILES="areal-vlm-grpo areal-vlm-mt-grpo areal-vlm-sft areal-clevr-sft areal-tir-grpo areal-scaffold-grpo areal-agents-grpo areal-math-grpo areal-math-sft areal-math-aime areal-math-boba areal-countdown-grpo areal-align"
AIME_PREP=0
BOBA_PREP=0
COUNTDOWN_PREP=0
HHRLHF_PREP=0
TORL_PREP=0
SCAFFOLD_PREP=0
CLEVR_SUBSET_PREP=0
HERMES_PREP=0
SWE_PREP=0
case "$PROFILE" in
  areal-vlm-grpo) MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct" ;;
  areal-vlm-mt-grpo) MODEL_ID="Qwen/Qwen3-VL-2B-Instruct" ;;
  areal-vlm-sft) MODEL_ID="Qwen/Qwen3-VL-2B-Instruct" ;;
  # clevr_count_70k_sft.py: see section 10 (local subset; the full 70k train
  # split blows the data worker when the SFT loader materializes pixel_values).
  areal-clevr-sft) MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"; CLEVR_SUBSET_PREP=1 ;;
  # tir/train_tir.py: see section 8 (fixture staging; the loader's own
  # GitHub download is unreachable from the CI job containers).
  areal-tir-grpo) MODEL_ID="Qwen/Qwen2.5-Math-1.5B"; TORL_PREP=1 ;;
  # gsm8k_rlvr_scaffolding.py: 1.5B model + section 9 (model alias symlink).
  areal-scaffold-grpo) MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"; SCAFFOLD_PREP=1 ;;
  areal-agents-grpo) MODEL_ID="Qwen/Qwen2-1.5B-Instruct" ;;
  # gsm8k_rl.py and gsm8k_eval.py share the same model.
  areal-math-grpo) MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct" ;;
  areal-math-sft) MODEL_ID="Qwen/Qwen3-1.7B" ;;
  areal-math-aime) MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"; AIME_PREP=1 ;;
  areal-math-boba) MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"; BOBA_PREP=1 ;;
  areal-countdown-grpo) MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"; COUNTDOWN_PREP=1 ;;
  # hermes/train.py: online RL; agent runtime package installed in
  # section 1c, orchestration in run_example.sh's hermes branch.
  areal-hermes-grpo) MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"; HERMES_PREP=1 ;;
  # swe/train_swe_rl.py: agent runtime (external AReaL-SWEAgent + aenv),
  # config and fixture are all built in section 11; plain foreground run.
  areal-swe-rl) MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"; SWE_PREP=1 ;;
  # hhrlhf_dpo.py: DPO pipeline is model-agnostic; upstream yaml's 7B swapped
  # for 1.5B (7B fp32 init OOMs the 32GB cards in the pool, 15GB per run).
  areal-align) MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"; HHRLHF_PREP=1 ;;
  *)
    echo "unknown profile: ${PROFILE} (supported: ${SUPPORTED_PROFILES})" >&2
    exit 1
    ;;
esac

if [[ -z "${TARGET_ROOT:-}" || -z "${GITHUB_WORKSPACE:-}" || -z "${GITHUB_ENV:-}" ]]; then
  echo "TARGET_ROOT / GITHUB_WORKSPACE / GITHUB_ENV must be set" >&2
  exit 2
fi

# -------------------------------------------------------
# 1. Install the AReaL source under test.
# The official image ships its own copy of AReaL, but we need to use
# the checked-out TARGET_ROOT version because it may contain fixes.
# -------------------------------------------------------
command -v uv >/dev/null 2>&1 || pip install -q uv
uv pip install --no-deps -e "$TARGET_ROOT" --system

# -------------------------------------------------------
# 1b. vLLM stack: swap the image's 0.23 line for 0.22.1.
# The image ships the ascend-v1.0.5 branch stack (vllm/vllm-ascend
# 0.23.0). The guarded v2.x code imports the pre-0.23 vllm layout
# (vllm.entrypoints.openai.utils / vllm.entrypoints.utils exist
# through 0.22.1, removed in 0.23) - without this swap every
# rollout-backed entry dies on
#   ModuleNotFoundError: No module named 'vllm.entrypoints.openai.utils'
#
# Recipe per vllm-ascend's own Dockerfile at v0.22.1rc1:
#   - vllm-ascend 0.22.1rc1 PyPI wheel with --no-deps: its pins would
#     re-resolve the frozen image stack (torch_npu 2.10.0.post2 etc);
#   - vllm 0.22.1 from SOURCE with VLLM_TARGET_DEVICE=empty (pure
#     Python, no compiled kernels - the ascend plugin supplies the
#     device layer). NEVER install vllm from PyPI here: the wheel is
#     built against torch==2.11.0, incompatible with the image's
#     torch 2.10.0 and areal's torch<2.11 pin. --no-deps keeps the
#     image stack untouched, --no-build-isolation lets setup.py see
#     the system torch;
#   - triton-ascend stays as the image's 3.2.2 (same 3.2.x generation
#     as this line's 3.2.1);
#   - the source build additionally needs setuptools-rust/setuptools-scm
#     in the system env: --no-build-isolation skips the declared
#     build-system requires, but setup.py imports both unconditionally.
#     The optional rust extension is skipped when cargo is absent (same
#     as vllm-ascend's own image, which ships no rust toolchain).
# Applies to ALL profiles: fsdp-only entries (gsm8k_sft) double as
# the control that the swap does not disturb the vllm-free path.
# TODO(vllm-0.23): drop this block once upstream ships a v2.x NPU
# image or vllm-ascend catches up with mainline's target layout.
# -------------------------------------------------------
VLLM_SRC="$GITHUB_WORKSPACE/vllm-0.22.1-src"
git clone --depth 1 --branch v0.22.1 https://github.com/vllm-project/vllm.git "$VLLM_SRC"
uv pip install --system --no-deps \
  --index-url https://mirrors.aliyun.com/pypi/simple \
  vllm-ascend==0.22.1rc1
uv pip install --system \
  --index-url https://mirrors.aliyun.com/pypi/simple \
  "setuptools-rust>=1.9.0" "setuptools-scm>=8.0"
VLLM_TARGET_DEVICE=empty uv pip install --system --no-deps --no-build-isolation \
  -e "$VLLM_SRC"
python -c "import torch, torch_npu, vllm, vllm_ascend; from vllm.entrypoints.openai.utils import validate_json_request; print(f'vllm {vllm.__version__} + vllm-ascend 0.22.1rc1 layout OK; torch {torch.__version__}, torch_npu {torch_npu.__version__}')"

# -------------------------------------------------------
# 1c. hermes profile: the agent runtime package. The model is the
# standard shared-cache Qwen2.5-1.5B-Instruct (hermes-agent itself is
# model-agnostic - it drives whatever OpenAI-compatible endpoint the
# rollout engine serves); the Agent Service imports
# examples.hermes.hermes.HermesAgent from the target checkout.
# Index: tsinghua (not aliyun) - aliyun's PEP 658 .metadata sidecar for
# hermes-agent's python-dotenv dep 404s intermittently; tuna mirrors the
# metadata reliably (same mirror as transformers-quick-start.yml).
# -------------------------------------------------------
if [[ "$HERMES_PREP" == 1 ]]; then
  uv pip install --system \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    hermes-agent==0.19.0
fi

# -------------------------------------------------------
# 1d. NPU portability patch: lazy MegatronEngine import.
# areal/v2/training_service/worker/awex.py:_create_training_adapter
# imports areal.engine.megatron_engine (and the megatron weight-update
# adapter) unconditionally, so even FSDP-only runs load megatron.bridge ->
# peft/lora_layers.py -> `import transformer_engine`, a CUDA-only package
# with no Ascend wheel (same class as bitsandbytes on aarch64). Defer both
# imports until the engine is known to be Megatron - behaviour-preserving;
# the editable install reads the patched file live.
# TODO(npu-upstream): drop once upstream guards the import.
# -------------------------------------------------------
python3 - <<'PY'
import os

path = os.path.join(
    os.environ["TARGET_ROOT"],
    "areal", "v2", "training_service", "worker", "awex.py",
)
with open(path, encoding="utf-8") as fh:
    text = fh.read()

old = '''    from areal.engine.fsdp_engine import FSDPEngine
    from areal.engine.megatron_engine import MegatronEngine
    from areal.v2.weight_update.awex.fsdp_adapter import AwexFSDPAdapter
    from areal.v2.weight_update.awex.megatron_adapter import (
        AwexMegatronAdapter,
    )

    if isinstance(engine, FSDPEngine):
        return AwexFSDPAdapter(engine)
'''
new = '''    from areal.engine.fsdp_engine import FSDPEngine
    from areal.v2.weight_update.awex.fsdp_adapter import AwexFSDPAdapter

    if isinstance(engine, FSDPEngine):
        return AwexFSDPAdapter(engine)

    # NPU: defer the Megatron imports - megatron.bridge drags the CUDA-only
    # transformer_engine, which has no Ascend wheel.
    from areal.engine.megatron_engine import MegatronEngine
    from areal.v2.weight_update.awex.megatron_adapter import (
        AwexMegatronAdapter,
    )
'''
assert text.count(old) == 1, "awex.py layout changed; update the NPU patch"
with open(path, "w", encoding="utf-8") as fh:
    fh.write(text.replace(old, new))
print("patched awex.py: MegatronEngine import is now lazy", flush=True)
PY

# -------------------------------------------------------
# 2. Runtime Environment setup.
# -------------------------------------------------------
echo "PYTHONPATH=/areal-workspace/MindSpeed:/areal-workspace/Megatron-Bridge/src:${PYTHONPATH:-}" >> "$GITHUB_ENV"
echo "HCCL_IF_BASE_PORT=63000" >> "$GITHUB_ENV"
echo "HCCL_NPU_SOCKET_PORT_RANGE=62100-62350" >> "$GITHUB_ENV"
# Rank-0 weight broadcast (memory_efficient_load) makes rank 1 wait inside
# the HCCL collective while rank 0 slowly CPU-loads the full weights (~4min
# for 7B on CI disk); the default connect/exec timeouts (~2min) kill the
# link first (EI0006 "Getting socket times out"). Same values as the proven
# gsm8k_grpo_npu.yaml scheduling_spec env suite.
echo "HCCL_CONNECT_TIMEOUT=7200" >> "$GITHUB_ENV"
echo "HCCL_EXEC_TIMEOUT=14400" >> "$GITHUB_ENV"
echo "ACL_DEVICE_SYNC_TIMEOUT=14400" >> "$GITHUB_ENV"
echo "TASK_QUEUE_ENABLE=1" >> "$GITHUB_ENV"
echo "OMP_NUM_THREADS=1" >> "$GITHUB_ENV"
echo "WANDB_MODE=disabled" >> "$GITHUB_ENV"
# swanlab resolves the run mode from SWANLAB_MODE only: SwanLabRun.__init__
# does `self.__mode = get_mode()` (swanlab/env.py), which reads that env and
# defaults to "cloud"; the mode= kwarg AReaL passes to swanlab.init() is not
# threaded into the run. AReaL calls swanlab.init() unconditionally
# (areal/utils/stats_logger.py:91) with the config default mode="disabled", so
# without this env the run resolves to cloud -> FileUploadManager(mode="cloud")
# -> get_client() -> ValueError "client object is not initialized". Every
# trainer builds StatsLogger, so this applies to all profiles.
echo "SWANLAB_MODE=disabled" >> "$GITHUB_ENV"
echo "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True" >> "$GITHUB_ENV"
echo "USE_OPTIMIZED_MODEL=0" >> "$GITHUB_ENV"
echo "AREAL_ALLOW_DEFAULT_ADMIN_KEY=1" >> "$GITHUB_ENV"

# -------------------------------------------------------
# 3. Pre-download Model & Dataset (using image's native tools)
# -------------------------------------------------------
# No local fixtures. The dataset is fetched online at train time
# (`load_dataset`, inside the data service); the model is resolved from the
# shared runner cache below. Export the Hub mirror through GITHUB_ENV so the
# run-example step inherits it too.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "HF_ENDPOINT=${HF_ENDPOINT}" >> "$GITHUB_ENV"
export AREAL_MODEL_ID="$MODEL_ID"

# The model comes from the shared runner cache. The pool keeps a persistent
# volume at ~/.cache/huggingface (see cache-seed/README.md and
# docs/examples-guard-engine.md), and cache-seed/areal/ms_seeds.yaml plants
# every profile model into it from ModelScope (HF hub-cache layout,
# refs/main = real HF sha). setup only resolves the snapshot path from
# refs/main, so a warm pool is zero-network, zero-download - same pattern as
# peft/torchtune.
#
# Cold-cache self-heal (sequence: cache-seed not dispatched yet, or a model
# added since): pull once with huggingface_hub - no local_dir, so it lands in
# the shared hub-cache layout and every later run is a hit. If HF/hf-mirror is
# down too, fall back to ModelScope into the per-run workspace, which is only a
# last resort: it is discarded with the workspace, so the durable fix is
# dispatching the cache-seed workflow (projects=areal).
#
# The resolved path is ALSO captured into the AREAL_MODEL_PATH shell variable
# (tee keeps the diagnostics streaming into the job log): later sections of
# this script - countdown's tokenizer (section 6) and the scaffolding model
# alias (section 9) - consume it and must not guess the pre-cache workspace
# layout, which no longer exists in the shared-cache flow.
AREAL_MODEL_PATH=$(python3 <<'PY' | tee /dev/stderr | sed -n 's/^AREAL_MODEL_PATH=//p' | tail -n 1
import os
import subprocess
import sys

model_id = os.environ["AREAL_MODEL_ID"]
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
repo_dir = os.path.join(hf_home, "hub", "models--" + model_id.replace("/", "--"))
workspace_dir = os.path.join(
    os.environ["GITHUB_WORKSPACE"], "areal_models", model_id.split("/")[-1]
)


def cached_snapshot():
    refs = os.path.join(repo_dir, "refs", "main")
    if not os.path.isfile(refs):
        return None
    with open(refs) as fh:
        sha = fh.read().strip()
    snap = os.path.join(repo_dir, "snapshots", sha)
    if os.path.isdir(snap) and os.listdir(snap):
        return snap
    return None


snapshot = cached_snapshot()
if snapshot is not None:
    print(f"{model_id}: shared-cache hit -> {snapshot}", flush=True)
else:
    print(
        f"{model_id}: not in the shared cache ({repo_dir}); filling it via HF",
        flush=True,
    )
    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(model_id)
        print(f"filled shared cache: {snapshot}", flush=True)
    except Exception as e:
        print(f"HF fill failed ({type(e).__name__}: {e}); trying ModelScope", flush=True)
        snapshot = None

    if snapshot is None:
        try:
            from modelscope import snapshot_download as ms_snapshot
        except ImportError:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "modelscope==1.37.0"],
                check=True,
            )
            from modelscope import snapshot_download as ms_snapshot
        os.makedirs(workspace_dir, exist_ok=True)
        ms_snapshot(model_id, local_dir=workspace_dir)
        snapshot = workspace_dir
        print(f"downloaded to ephemeral {snapshot}", flush=True)

with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"AREAL_MODEL_PATH={snapshot}\n")
print(f"AREAL_MODEL_PATH={snapshot}", flush=True)
PY
)
[[ -n "${AREAL_MODEL_PATH}" ]] || {
  echo "model resolution printed no AREAL_MODEL_PATH line" >&2
  exit 2
}

# -------------------------------------------------------
# 4. AIME dataset + derived config (areal-math-aime).
# -------------------------------------------------------
# areal/dataset/aime.py loads <path>/aime_train.parquet and
# <path>/aime_test.parquet from a LOCAL directory (upstream ships no aime
# config; the loader takes no HF repo id). Convert the public MathArena
# parquets (same flow as examples/distillation/mopd/README.md), then derive
# the GRPO config from gsm8k_grpo_npu.yaml: point the datasets at the local
# dir and set rollout.agent to null (aime_rl.py runs the standard
# RLVRWorkflow, not an agent workflow). The generated config is exposed to
# overlay args via AREAL_AIME_CONFIG.
if [[ "$AIME_PREP" == 1 ]]; then
python3 <<'PY'
import os

import pandas as pd
import yaml
from huggingface_hub import hf_hub_download

workspace = os.environ["GITHUB_WORKSPACE"]
target_root = os.environ["TARGET_ROOT"]
aime_dir = os.path.join(workspace, "areal_data", "aime")
os.makedirs(aime_dir, exist_ok=True)

# hf_hub_download honors HF_ENDPOINT (mirror) set in section 3.
for repo, dst in [
    ("MathArena/aime_2025", "aime_train.parquet"),
    ("MathArena/aime_2026", "aime_test.parquet"),
]:
    src = hf_hub_download(
        repo_id=repo, repo_type="dataset",
        filename="data/train-00000-of-00001.parquet",
    )
    df = pd.read_parquet(src).rename(columns={"problem": "question"})
    df = df[["question", "answer"]]
    df.to_parquet(os.path.join(aime_dir, dst))
    print(f"prepared {dst}: {len(df)} rows", flush=True)

base = os.path.join(target_root, "examples", "math", "gsm8k_grpo_npu.yaml")
with open(base, encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)
cfg["experiment_name"] = "aime-grpo"
cfg["train_dataset"]["path"] = aime_dir
cfg["valid_dataset"]["path"] = aime_dir
config_path = os.path.join(aime_dir, "aime_grpo_npu.yaml")
with open(config_path, "w", encoding="utf-8") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)

with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"AREAL_AIME_CONFIG={config_path}\n")
print(f"generated {config_path}", flush=True)
PY
fi

# -------------------------------------------------------
# 5. BoBa dataset (areal-math-boba).
# -------------------------------------------------------
# boba_grpo.py loads train_dataset.path as a LOCAL jsonl file
# (load_dataset("json", data_files=path)); the upstream yaml ships the HF
# repo id, which would fail with FileNotFoundError. Pre-download the public
# AReaL-boba-106k.jsonl and expose its local path via AREAL_BOBA_DATA.
if [[ "$BOBA_PREP" == 1 ]]; then
python3 <<'PY'
import os
import shutil

from huggingface_hub import hf_hub_download

src = hf_hub_download(
    repo_id="inclusionAI/AReaL-boba-Data", repo_type="dataset",
    filename="AReaL-boba-106k.jsonl",
)
dst = os.path.join(
    os.environ["GITHUB_WORKSPACE"], "areal_data", "boba", "AReaL-boba-106k.jsonl"
)
os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copyfile(src, dst)
with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"AREAL_BOBA_DATA={dst}\n")
print(f"downloaded boba dataset to {dst}", flush=True)
PY
fi

# -------------------------------------------------------
# 6. Countdown dataset (areal-countdown-grpo).
# -------------------------------------------------------
# countdown.py writes ./data/countdown/qwen/*.jsonl relative to CWD; run it
# from TARGET_ROOT so the generated files land where train_config.yaml's
# default train_dataset.path=data/countdown/qwen/train_e.jsonl expects them
# (run_example.sh also cds to TARGET_ROOT). The default generation size is
# 500k samples; 32/8 is plenty for the 1-step smoke. The tokenizer comes
# from the model predownloaded in section 3, so generation is offline.
if [[ "$COUNTDOWN_PREP" == 1 ]]; then
  MODEL_DIR="$AREAL_MODEL_PATH"
  mkdir -p "$TARGET_ROOT/data/countdown/qwen"
  (cd "$TARGET_ROOT" && python3 examples/countdown/countdown.py \
    --num_samples 32 --eval_size 8 --tokenizer_path "$MODEL_DIR")
fi

# -------------------------------------------------------
# 7. HH-RLHF dataset (areal-align: hhrlhf_dpo.py / hhrlhf_rw.py).
# -------------------------------------------------------
# Anthropic/hh-rlhf is public (NOT gated), but its jsonl.gz files live in
# per-subset subdirectories (harmless-base/, helpful-base/, ...), so the
# upstream loader's load_dataset(path, split=...) without a config name
# fails with a multi-config error. Work around it boba-style: download the
# harmless-base subset online (via HF_ENDPOINT mirror) into a flat local
# dir; load_dataset(<dir>, split="train"/"test") then infers splits from
# the file names. Exposed to overlay args via AREAL_HHRLHF_DATA.
# NOTE: the dir is deliberately named "hh-rlhf" — areal's dataset dispatch
# matches the literal substring "hh-rlhf" in the path; a dir named
# "hhrlhf" falls through to the load_from_disk fallback and errors out.
if [[ "$HHRLHF_PREP" == 1 ]]; then
python3 <<'PY'
import os
import shutil

from huggingface_hub import hf_hub_download

dst_dir = os.path.join(os.environ["GITHUB_WORKSPACE"], "areal_data", "hh-rlhf")
os.makedirs(dst_dir, exist_ok=True)
for filename in ("train.jsonl.gz", "test.jsonl.gz"):
    src = hf_hub_download(
        repo_id="Anthropic/hh-rlhf", repo_type="dataset",
        filename=f"harmless-base/{filename}",
    )
    shutil.copyfile(src, os.path.join(dst_dir, filename))
with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"AREAL_HHRLHF_DATA={dst_dir}\n")
print(f"prepared hh-rlhf harmless-base at {dst_dir}", flush=True)
PY
fi

# -------------------------------------------------------
# 8. ToRL dataset (areal-tir-grpo): fixture staging.
# -------------------------------------------------------
# The torl_data loader downloads its parquets from github.com at load time
# (areal/dataset/torl_data.py), but the CI job containers cannot reach
# github.com (mainland network, no proxy; the runner agent pulls code on the
# host side). No HF/ModelScope mirror of ToRL exists (ModelScope 404;
# upstream ships GitHub only), so the two tiny parquets are vendored as
# fixtures - the documented last resort. prepare_torl_data() skips its
# download once /tmp/areal/torl_data/_SUCCESS exists, so staging the files
# plus the flag here makes the loader fully offline. The data service worker
# runs in the same container, so /tmp is shared with the run step.
# Fixture pin (GAIR-NLP/ToRL @ main): train.parquet blob 57bd7d5d (6035244
# bytes), test.parquet blob e8ca20fd (51798 bytes).
if [[ "$TORL_PREP" == 1 ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  TORL_FIXTURE="$SCRIPT_DIR/../fixtures"
  for f in train.parquet test.parquet; do
    [[ -s "$TORL_FIXTURE/$f" ]] || {
      echo "missing fixture: $TORL_FIXTURE/$f (see section 8 comment for source)" >&2
      exit 2
    }
  done
  mkdir -p /tmp/areal/torl_data
  cp "$TORL_FIXTURE/train.parquet" "$TORL_FIXTURE/test.parquet" /tmp/areal/torl_data/
  : > /tmp/areal/torl_data/_SUCCESS
  echo "staged torl fixtures at /tmp/areal/torl_data (loader download skipped)"
fi

# -------------------------------------------------------
# 9. Scaffolding model alias (areal-scaffold-grpo).
# -------------------------------------------------------
# examples/scaffolding hardcodes model="default" in its OpenAI-compatible
# worker (workflow.py:102, a TRT-LLM convention), while AReaL's vLLM server
# serves under the model path - requests 404 and every rollout comes back
# empty, which then trips the "total loss weight must be positive" assert.
# vLLMConfig has no served_model_name passthrough (strict dataclass, unknown
# keys are rejected at the structured merge), so serve the model under the
# literal name "default" instead: vLLM uses the raw --model string as the
# served model name, so a relative "default" path resolving to the model dir
# makes the name match. The server subprocess inherits the run CWD
# (TARGET_ROOT), hence the symlink location.
if [[ "$SCAFFOLD_PREP" == 1 ]]; then
  MODEL_DIR="$AREAL_MODEL_PATH"
  [[ -d "$MODEL_DIR" ]] || { echo "model dir missing: $MODEL_DIR" >&2; exit 2; }
  ln -sfn "$MODEL_DIR" "$TARGET_ROOT/default"
  echo "linked $TARGET_ROOT/default -> $MODEL_DIR (scaffolding model alias)"
fi

# -------------------------------------------------------
# 10. CLEVR subset (areal-clevr-sft).
# -------------------------------------------------------
# The clevr SFT loader materializes per-sample pixel_values across the whole
# split (clevr_count_70k.py:104-127), unlike the RL loader which only keeps
# JPEG bytes, so the 70k train split blows the data worker (host OOM / tens
# of GB of arrow cache). The example hardcodes split="train" and
# get_custom_dataset ignores train_dataset.split when the script passes a
# non-None split (dataset/__init__.py:322,328), and the loader takes no extra
# kwargs - so subset via path instead: train_dataset.path points at a dir
# whose name contains "clevr_count_70k" (dispatch substring) holding
# train.parquet (load_dataset infers the split from the file name). The
# absolute slice train[:200] downloads only the first shard (~0.5GB of the
# 10.4GB), and the parquet keeps the HF feature metadata so images stay PIL.
if [[ "$CLEVR_SUBSET_PREP" == 1 ]]; then
python3 <<'PY'
import os

from datasets import load_dataset

dst = os.path.join(os.environ["GITHUB_WORKSPACE"], "areal_data", "clevr_count_70k")
os.makedirs(dst, exist_ok=True)
ds = load_dataset("BUAADreamer/clevr_count_70k", split="train[:200]")
ds.to_parquet(os.path.join(dst, "train.parquet"))
with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"AREAL_CLEVR_SUBSET_DIR={dst}\n")
print(f"prepared clevr subset ({len(ds)} rows) at {dst}", flush=True)
PY
fi

# -------------------------------------------------------
# 11. SWE-bench agent RL (areal-swe-rl).
# -------------------------------------------------------
# train_swe_rl.py has no in-tree config or dataset; build all pieces:
#  a) clone the external AReaL-SWEAgent checkout (the workflow imports
#     `aweagent.lifecycle` from econfig.agent_root) and install its
#     requirements.txt (aenvironment sandbox SDK + openai-agents stack);
#  b) author a minimal agent config: the shipped default
#     (1_0_0/min-swe-agent-train-top1) uses step_limit=200 with 16K
#     completion tokens and the swebench eval env - unusable in CI. Derive
#     it with tiny limits and the inline eval_script reward path;
#  c) author the PPO config (derived from gsm8k_grpo_npu.yaml) + a
#     self-contained fixture JSONL (instance_id / problem_statement /
#     eval_script; no repo clone, no network), and export
#     AWEAGENT_ROOT / AREAL_SWE_CONFIG for the overlay.
# The persistent-bash env starts with `cd /testbed`, so prepare /testbed as
# an empty git repo in the job container (first-run risk: the aenvironment
# persistent-bash env must execute the shell locally, not in a remote FaaS
# sandbox).
if [[ "$SWE_PREP" == 1 ]]; then
# AREAL_MODEL_PATH is a non-exported shell variable from section 3; export it
# for this heredoc only so os.environ sees it (it is already in GITHUB_ENV for
# the run step).
AREAL_MODEL_PATH="$AREAL_MODEL_PATH" python3 <<'PY'
import json
import os
import subprocess

import yaml

workspace = os.environ["GITHUB_WORKSPACE"]
target_root = os.environ["TARGET_ROOT"]
aweagent_root = os.path.join(workspace, "AReaL-SWEAgent")

# a) agent runtime checkout + its requirements. NOTE: the SDK is imported as
# `from aenv import Environment`, but the PyPI package is `aenvironment` (the
# package and module names differ) - installing PyPI `aenv` grabs an
# unrelated same-named project and the import dies with
# "cannot import name 'Environment' from 'aenv' (unknown location)". Follow
# the upstream mechanism (train_swe_rl._install_aweagent_deps_on_ray_nodes
# runs `uv pip install -r requirements.txt`) and install the pinned set:
# aenvironment + openai/openai-agents/tenacity/loguru. Tuna index (aliyun's
# PEP 658 metadata sidecars 404 intermittently).
# Those requirements can pull a NEWER starlette whose Router dropped the
# `on_startup` kwarg; vllm 0.22.1's `APIRouter(on_startup=...)` then dies on
# import in the SAME python env (areal_vllm_server imports
# vllm.entrypoints.openai.api_server -> ... -> `router = APIRouter()`),
# even though the image's bundled starlette works. Pin fastapi/starlette to
# the image's current versions via a constraint file.
import importlib.metadata as importlib_metadata

constraint_path = os.path.join(workspace, "swe-web-constraints.txt")
with open(constraint_path, "w", encoding="utf-8") as fh:
    for pkg in ("fastapi", "starlette"):
        try:
            fh.write(f"{pkg}=={importlib_metadata.version(pkg)}\n")
        except importlib_metadata.PackageNotFoundError:
            pass

if not os.path.isdir(aweagent_root):
    subprocess.run(
        [
            "git", "clone", "--depth", "1",
            "https://github.com/areal-project/AReaL-SWEAgent.git", aweagent_root,
        ],
        check=True,
    )
subprocess.run(
    [
        "uv", "pip", "install", "--system",
        "--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple",
        "--constraint", constraint_path,
        "-r", os.path.join(aweagent_root, "requirements.txt"),
    ],
    check=True,
)

# b) minimal agent config derived from the shipped Qwen SWE-RL config
src_cfg = os.path.join(
    aweagent_root, "aweagent", "configs", "1_0_0", "min-swe-agent-train-top1.yaml"
)
with open(src_cfg, encoding="utf-8") as fh:
    agent_cfg = yaml.safe_load(fh)
agent_cfg.update(
    max_completion_tokens=256,
    step_limit=6,
    no_tool_call_limit=3,
    multiple_tools_limit=3,
    rl_test=False,  # inline eval_script reward, not the swebench eval env
)
dst_cfg = os.path.join(aweagent_root, "aweagent", "configs", "ci_swe_smoke.yaml")
with open(dst_cfg, "w", encoding="utf-8") as fh:
    yaml.safe_dump(agent_cfg, fh, sort_keys=False, allow_unicode=True)

# c) self-contained fixture + /testbed workspace + PPO config
data_dir = os.path.join(workspace, "areal_data", "swe")
os.makedirs(data_dir, exist_ok=True)
fixture = os.path.join(data_dir, "ci_swe_smoke.jsonl")
with open(fixture, "w", encoding="utf-8") as fh:
    fh.write(
        json.dumps(
            {
                "instance_id": "ci-swe-smoke-1",
                "problem_statement": (
                    "Create a file `hello.py` in the current working directory "
                    "that defines a function `add(a, b)` returning the sum of a "
                    "and b."
                ),
                "eval_script": (
                    "cd /testbed && python -c \"import hello; "
                    "assert hello.add(2, 3) == 5\""
                ),
            }
        )
        + "\n"
    )

os.makedirs("/testbed", exist_ok=True)
subprocess.run(["git", "init", "-q", "/testbed"], check=True)
subprocess.run(
    ["git", "-C", "/testbed", "config", "user.email", "sweagent@test.com"], check=True
)
subprocess.run(
    ["git", "-C", "/testbed", "config", "user.name", "sweagent"], check=True
)

base = os.path.join(target_root, "examples", "math", "gsm8k_grpo_npu.yaml")
with open(base, encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)
cfg["experiment_name"] = "swe-rl-smoke"
cfg["scheduler"]["type"] = "local"
# Keep the base's rollout.agent (mode: inline) - the field is a non-Optional
# AgentConfig in the schema, and the SWE workflow is passed explicitly to
# trainer.train(), so the built-in agent config just needs to stay valid.
cfg["train_dataset"].update(path=fixture, type="rl", batch_size=1)
cfg["valid_dataset"].update(path=fixture, type="rl", batch_size=1)
cfg["econfig"] = {
    "dataset_path": data_dir,
    "agent_type": "swe",
    "agent_config": "ci_swe_smoke",
    "agent_root": aweagent_root,
    "llm_model": os.environ["AREAL_MODEL_PATH"],
    "timeout": 600,
}
cfg_path = os.path.join(data_dir, "swe_ppo_npu.yaml")
with open(cfg_path, "w", encoding="utf-8") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)

with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"AWEAGENT_ROOT={aweagent_root}\n")
    fh.write(f"AREAL_SWE_CONFIG={cfg_path}\n")
print(f"prepared AReaL-SWEAgent at {aweagent_root}", flush=True)
print(f"generated {cfg_path} and {fixture}", flush=True)
PY
fi

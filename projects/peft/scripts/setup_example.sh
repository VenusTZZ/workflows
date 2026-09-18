#!/usr/bin/env bash
# Prepare the CI environment for one supported peft example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# peft itself is installed from TARGET_ROOT (the release checkout under
# test), so the guarded tag is exactly the code that runs.
#
# The upstream examples/sft/requirements.txt is deliberately NOT
# installed: it pins everything to git main (transformers/peft/trl@main
# + flash-attn + unsloth + bitsandbytes), which conflicts with testing
# a release tag and contains CUDA-only packages. We install the
# checkout plus the minimal SFTTrainer stack instead.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"
ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
ALIYUN_PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/

pip_ascend() {
  python -m pip install --extra-index-url "$ASCEND_PIP_INDEX" "$@"
}

select_pip_index() {
  # Runners live in mainland China: prefer the cluster pip cache, fall
  # back to the Aliyun mirror. The ascend index stays available via
  # PIP_EXTRA_INDEX_URL (set by the engine) for torch_npu wheels.
  if python -c "
import os
import urllib.error
import urllib.request
try:
    urllib.request.urlopen(os.environ['CLUSTER_PIP_INDEX'], timeout=3)
except urllib.error.HTTPError:
    pass
" 2>/dev/null; then
    export PIP_INDEX_URL="$CLUSTER_PIP_INDEX"
    export PIP_TRUSTED_HOST="$CLUSTER_PIP_HOST"
  else
    export PIP_INDEX_URL="$ALIYUN_PIP_INDEX"
    unset PIP_TRUSTED_HOST
  fi
  echo "pip index: $PIP_INDEX_URL"
}

ensure_torch_stack() {
  # Same torch line as peft quick-start (CANN 9.1.0 pairing): reuse
  # the image stack when it already matches, otherwise install.
  if python -c "
import torch, torch_npu
raise SystemExit(
    0 if torch.__version__.startswith('2.9.0')
    and torch_npu.__version__.startswith('2.9.0') else 1)
"; then
    echo "reusing image torch stack ($(python -c 'import torch; print(torch.__version__)'))"
    return
  fi
  echo "installing torch==2.9.0 torch_npu==2.9.0.post2"
  pip_ascend torch==2.9.0 torch_npu==2.9.0.post2
}

# Copy CI fixture data into the target root so that example scripts can
# load them via a local path under $TARGET_ROOT/fixtures/ (same
# decoupling from the workflows-checkout subtree as trl).
#
# Both flavors are supported, all top-level entries in $FIXTURE_DIR:
#   *.jsonl    → single-file fixtures (e.g. ci_sft_8.jsonl)
#   */         → directory fixtures (e.g. ci_alpaca_10/ with train.jsonl,
#                ci_corda_8/ with train.jsonl + test.jsonl)
# Examples that need a directory data_path (corda, glora, hira, olora,
# waveft) fail with FileNotFoundError if directory fixtures are not
# copied here — root cause of CI failures in run 35222723526.
prepare_fixtures() {
  local src="${FIXTURE_DIR:?FIXTURE_DIR is required}"
  local dst="$TARGET_ROOT/fixtures"
  echo "preparing fixtures from $src to $dst"
  if [[ ! -d "$src" ]]; then
    echo "FATAL: fixture source dir not found: $src" >&2
    exit 1
  fi
  shopt -s nullglob
  # Top-level *.jsonl files
  local n_files=0
  for f in "$src"/*.jsonl; do
    mkdir -p "$dst"
    cp "$f" "$dst/"
    n_files=$((n_files + 1))
  done
  # Top-level subdirectories (e.g. ci_alpaca_10/, ci_corda_8/)
  local n_dirs=0
  for d in "$src"/*/; do
    [[ -d "$d" ]] || continue
    mkdir -p "$dst"
    cp -r "$d" "$dst/"
    n_dirs=$((n_dirs + 1))
  done
  shopt -u nullglob
  if (( n_files == 0 && n_dirs == 0 )); then
    echo "FATAL: no fixture files (*.jsonl) or subdirs found in $src" >&2
    exit 1
  fi
  echo "copied $n_files fixture file(s) and $n_dirs fixture dir(s) to $dst"
}

setup_peft() {
  # peft from the guarded release checkout, plus the verified dependency
  # line (2026-09-15, coder npu-1 端到端验证结论):
  #   transformers 4.57.1 + datasets>=4.7.0,<6 + hub<1.0 + trl 1.12.0
  # - transformers 4.57.1: 5.x 移除 send_example_telemetry 等旧 API
  # - datasets>=4.7.0,<6: trl 1.12+ 在 wheel metadata 声明 datasets>=4.7.0
  #   (pyproject.toml 自 v1.0.0 起 commit ac5421b4 引入，datasets<4 会
  #   ResolutionImpossible)，<6 留出口避开未来 6.x breaking
  # - hub<1.0: hub 1.x 拒绝 imdb 等无命名空间数据集
  # - trl 1.12.0: trl ≥ 1.12 默认 chunked_nll 走 _patch_chunked_ce_lm_head
  #   (sft_trainer.py:233)。该 patch 第 383 行 (1.13 在 386) 走
  #   inspect.signature(original_forward.__func__)，假设 forward 是普通
  #   method；但 miss/mica 用 device_map="auto" 加载，accelerate
  #   .add_hook_to_module 把 forward 包成 functools.partial，没 __func__
  #   → AttributeError（纯 PyTorch + accelerate，硬件无关，CUDA 同问题）。
  #   overlay 在 examples_manifest.yaml 的 miss/mica 例里显式 --loss_type nll
  #   跳过 chunked patch 走标准 cross-entropy。
  #   注：sft_trainer.py:1331 isinstance(BaseTunerLayer) guard 在这两例不
  #   会触发（target_modules 不含 lm_head），guard 放过后 patch 内部才崩。
  # - scikit-learn: adamss 的 ASA 回调（peft.tuners.adamss）硬性 import
  #   sklearn；evaluate.load("glue") 的 metric 模块同样要 sklearn.metrics。
  #   coder 验证机里碰巧预装，CANN 裸镜像没有（run 35045940066 实测缺失）。
  # PIP_CONSTRAINT keeps CUDA metapackages out.
  echo "installing peft from $TARGET_ROOT"
  python -m pip install -e "$TARGET_ROOT"
  python -m pip install "transformers==4.57.1" "datasets>=4.7.0,<6" \
    "huggingface_hub<1.0" "trl==1.12.0" evaluate scikit-learn \
    torchvision==0.24.0
  python -c "import peft, trl, transformers, datasets, accelerate; print('peft', peft.__version__, '/ trl', trl.__version__, '/ transformers', transformers.__version__)"

  # Resolve seeded asset paths for overlay_args. The shared cache root
  # is populated by the cache-seed workflow (spec:
  # cache-seed/peft/ms_seeds.yaml — ModelScope download → HF hub cache
  # layout, refs/main = real upstream sha; this plant used to live here
  # in setup, moved 2026-09-17 so the seed workflow is the single
  # writer). Nothing downloads in the example jobs anymore.
  # Hardcoded hub ids (mt0-small / dinov2-base / glue) resolve through
  # the same seeded cache at example runtime; missing env paths are a
  # hard error — every example that uses them fails without them.
  python - <<'PY'
import os
from pathlib import Path

HUB_ROOT = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "hub"

# (hf_id, env var) — consumed by manifest overlay_args
#   ${SFT_MODEL_PATH}          sft / miss / mica / supertuning
#   ${ROBERTA_BASE_PATH}       adamss ×2
#   ${BERT_BASE_UNCASED_PATH}  sequence_classification
TO_ENV = [
    ("Qwen/Qwen2.5-0.5B", "SFT_MODEL_PATH"),
    ("roberta-base", "ROBERTA_BASE_PATH"),
    ("bert-base-uncased", "BERT_BASE_UNCASED_PATH"),
]

for hf_id, var in TO_ENV:
    repo_dir = HUB_ROOT / f"models--{hf_id.replace('/', '--')}"
    refs = repo_dir / "refs" / "main"
    if not refs.is_file():
        raise SystemExit(
            f"{hf_id} missing from shared cache root — dispatch the "
            f"cache-seed workflow (spec: cache-seed/peft/ms_seeds.yaml)")
    sha = refs.read_text().strip()
    snap = repo_dir / "snapshots" / sha
    if not snap.is_dir() or not any(snap.iterdir()):
        raise SystemExit(f"{hf_id}: refs/main -> {sha[:8]} has no snapshot files")
    with open(os.environ["GITHUB_ENV"], "a") as fh:
        fh.write(f"{var}={snap}\n")
    print(f"{var}={snap}", flush=True)
PY
}

setup_peft_dreambooth() {
  # SD dreambooth 例（lora/oft/deft/hra/stable_diffusion ×5）：SFT 栈之外
  # 还要 diffusers + tensorboard。run 35303810367 实测缺 diffusers 直接
  # import 崩（train_dreambooth.py:14）；--report_to tensorboard 是
  # accelerate log_with，需要 tensorboard 包。
  # diffusers 不 pin：requirements 对 transformers>=4.x / hub>=0.30 的约束
  # 均被 setup_peft 的 pinned 线满足，pip 不会动已装版本；具体版本线以
  # 首轮 CI 绿后为准再固定。
  setup_peft
  echo "installing dreambooth stack (diffusers + tensorboard)"
  python -m pip install diffusers tensorboard
  python -c "import diffusers, tensorboard; print('diffusers', diffusers.__version__)"

  # SD v1.5 由 cache-seed 投递（与 accelerate 共享同一缓存卷，2026-09-17
  # 已 plant；peft 的 ms_seeds.yaml 同步声明，冷缓存时 peft 自己 dispatch
  # 也能补）。resolve refs/main 得 ${SD_MODEL_PATH}，只影响本 profile——
  # 非 SD 例的 setup 不做这个校验，缺资产不拦其它例。
  python - <<'PY'
import os
from pathlib import Path

HUB_ROOT = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "hub"
hf_id, var = "stable-diffusion-v1-5/stable-diffusion-v1-5", "SD_MODEL_PATH"

repo_dir = HUB_ROOT / f"models--{hf_id.replace('/', '--')}"
refs = repo_dir / "refs" / "main"
if not refs.is_file():
    raise SystemExit(
        f"{hf_id} missing from shared cache root — dispatch the "
        f"cache-seed workflow (spec: cache-seed/peft/ms_seeds.yaml)")
sha = refs.read_text().strip()
snap = repo_dir / "snapshots" / sha
if not snap.is_dir() or not any(snap.iterdir()):
    raise SystemExit(f"{hf_id}: refs/main -> {sha[:8]} has no snapshot files")
with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"{var}={snap}\n")
print(f"{var}={snap}", flush=True)
PY
}

setup_peft_ds() {
  # deepspeed 多卡 sft 例（run_peft_deepspeed.sh /
  # run_peft_qlora_deepspeed_stage3.sh）：setup_peft 之上装 deepspeed。
  # 0.19.7 的 npu accelerator 自动识别 Ascend + HCCL，coder npu-5
  # 2026-09-18 实跑 ZeRO-3 2 卡 exit 0（loss 4.64）。DS_BUILD_OPS=0
  # 跳过 CPU op 内核编译：ZeRO-3 bf16 训练路径不需要编译 op，且
  # 编译耗时 + 裸镜像缺编译链，CI 不划算。
  setup_peft
  echo "installing deepspeed for ZeRO-3 sft entries"
  DS_BUILD_OPS=0 python -m pip install deepspeed==0.19.7
  python -c "import deepspeed; from deepspeed.accelerator import get_accelerator; print('deepspeed', deepspeed.__version__, 'accelerator', get_accelerator()._name)"
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

HERE=$(cd "$(dirname "$0")" && pwd)
export PIP_CONSTRAINT="$(cd "$HERE/.." && pwd)/constraints-npu.txt"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

select_pip_index
python -m pip install -U pip setuptools wheel
ensure_torch_stack
prepare_fixtures

"setup_${PROFILE}"

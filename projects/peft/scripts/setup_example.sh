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

# Copy CI fixture data files into the target root so that example
# scripts can load them via a local path under $TARGET_ROOT/fixtures/
# (same decoupling from the workflows-checkout subtree as trl).
prepare_fixtures() {
  local src="${FIXTURE_DIR:?FIXTURE_DIR is required}"
  local dst="$TARGET_ROOT/fixtures"
  echo "preparing fixtures from $src to $dst"
  if ! ls "$src"/*.jsonl 1>/dev/null 2>&1; then
    echo "FATAL: no fixture files (*.jsonl) found in $src" >&2
    exit 1
  fi
  mkdir -p "$dst"
  cp "$src"/*.jsonl "$dst/"
  echo "copied $(ls "$dst"/*.jsonl 2>/dev/null | wc -l) fixture file(s) to $dst"
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
  # - trl 1.12.0: trl ≥ 1.12 都默认 chunked_nll，与 peft partial lm_head
  #   冲突（纯 PyTorch patch，sft_trainer.py:1331 检测到 peft 包了 head
  #   就 raise；CUDA 同问题，NPU 是首个端到端跑这条路径的环境）。
  #   overlay 在 examples_manifest.yaml 的 miss/mica 例里显式 --loss_type nll
  #   跳过 chunked patch 走标准 cross-entropy。
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

  # Pre-download all example content from ModelScope (China-reachable)
  # because runners cannot reliably reach HuggingFace (xet-backed files
  # 302 to cas-bridge.xethub.hf.co, intermittently unreachable).
  #
  # Strategy:
  #   - Modelscope snapshot_download (China mirror) → plant to
  #     ~/.cache/huggingface/hub/ via symlinks, with refs/main written
  #     to the real upstream sha (queried via HF API which is xet-free).
  #     This way from_pretrained(<hf_id>) / load_dataset(<hf_id>, ...)
  #     resolves to the planted cache and never touches the network.
  #   - Qwen2.5-0.5B also gets SFT_MODEL_PATH exposed for overlay_args.
  #
  # Pinned to 1.37.0: the hub code split started at 1.38 and 1.40.1's
  # "modelscope-hub>=0.4.2" floor is too loose — 1.40.1 + hub 0.4.2
  # (mirror-lagged) dies on DEFAULT_CREDENTIALS_PATH import at
  # modelscope import time.
  python -m pip install "modelscope==1.37.0"
  python - <<'PY'
import os, sys, shutil
from pathlib import Path

# Non-TTY CI logs: throttle tqdm refreshes instead of disabling.
os.environ.setdefault("TQDM_MININTERVAL", "15")

import requests
from modelscope import snapshot_download

MODEL_CACHE = Path(os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope")))
HUB_ROOT = Path(os.path.expanduser("~/.cache/huggingface/hub"))
HF_API = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/") + "/api"

# (1) snapshot_download → env var: 例里 overlay 传 ${VAR}，
#     from_pretrained(<local_path>) 直接吃本地，无需 plant
TO_ENV = [
    ("Qwen/Qwen2.5-0.5B",               "SFT_MODEL_PATH"),         # sft/miss/mica/supertuning
    ("AI-ModelScope/roberta-base",      "ROBERTA_BASE_PATH"),      # adamss x2（overlay 注入）
    ("AI-ModelScope/bert-base-uncased", "BERT_BASE_UNCASED_PATH"), # sequence_classification
]

# (2) snapshot_download + cp plant (model): 例里硬编码 hub_id（beft/pvera），
#     from_pretrained(<hub_id>) 必须命中本地 cache，否则会去打 xet
#     走 cp 而非 symlink：symlink 指向 modelscope cache，后者被 pod 回收
#     / 别的 job 清掉就会断链；cp 后 HF cache 自包含，与 modelscope 无关
TO_PLANT_MODEL = [
    ("bigscience/mt0-small",  "bigscience/mt0-small"),   # beft_finetuning.py 硬编码
    ("facebook/dinov2-base",  "facebook/dinov2-base"),   # pvera/...py 硬编码
]

# (3) snapshot_download + cp plant (dataset): adamss/no_lora 用 glue mrpc，
#     adamss_manual 默认 cola。allow_patterns 只下要的 config，避免下完整
#     140MB 的 9 个 config。imdb 不在这里——modelscope 没 parquet 数据，
#     走 cache-seed/peft（beans、financial_phrasebank 也走那条路）。
TO_PLANT_DATASET = [
    # (ms_id, hf_id, allow_patterns)
    ("nyu-mll/glue", "nyu-mll/glue", ["mrpc/*", "cola/*"]),
]


def fetch_sha(hf_id, kind):
    url = f"{HF_API}/{kind}s/{hf_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json().get("sha") or r.json().get("oid")


def plant(ms_id, hf_id, kind, allow_patterns=None):
    src = Path(snapshot_download(
        ms_id,
        cache_dir=str(MODEL_CACHE),
        repo_type=kind,
        allow_patterns=allow_patterns,
    ))
    sha = fetch_sha(hf_id, kind)
    repo_kind = "models" if kind == "model" else "datasets"
    repo_dir = HUB_ROOT / f"{repo_kind}--{hf_id.replace('/', '--')}"
    snap_dir = repo_dir / "snapshots" / sha
    snap_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs").mkdir(exist_ok=True)
    # no trailing newline — hub compares this string to the snapshot
    # folder name without stripping
    (repo_dir / "refs" / "main").write_text(sha)
    n_bytes = 0
    for item in src.rglob("*"):
        if not item.is_file():
            continue
        dest = snap_dir / item.relative_to(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink():
            # 替换之前 symlink-based plant 留下的链（避免断链风险）
            dest.unlink()
        elif dest.exists():
            continue  # 已有真实文件，跳过
        try:
            shutil.copy2(item, dest)
            n_bytes += item.stat().st_size
        except FileExistsError:
            pass
    print(f"planted (cp) {hf_id}@{sha[:8]} ({n_bytes // (1024 * 1024)} MB)",
          flush=True)


failures: list[str] = []

# (1) snapshot → env var
for ms_id, var_name in TO_ENV:
    try:
        local = Path(snapshot_download(ms_id, cache_dir=str(MODEL_CACHE)))
        with open(os.environ["GITHUB_ENV"], "a") as fh:
            fh.write(f"{var_name}={local}\n")
        print(f"{var_name}={local}", flush=True)
    except Exception as exc:
        failures.append(f"{ms_id} (env): {type(exc).__name__}: {exc}")
        print(f"FAIL {ms_id} (env): {exc}", flush=True)

# (2) snapshot + plant model（脚本硬编码 hub_id）
for ms_id, hf_id in TO_PLANT_MODEL:
    try:
        plant(ms_id, hf_id, "model")
    except Exception as exc:
        failures.append(f"{ms_id} (plant model): {type(exc).__name__}: {exc}")
        print(f"FAIL {ms_id} (plant model): {exc}", flush=True)

# (3) snapshot + plant dataset（脚本硬编码 dataset 名）
for ms_id, hf_id, patterns in TO_PLANT_DATASET:
    try:
        plant(ms_id, hf_id, "dataset", allow_patterns=patterns)
    except Exception as exc:
        failures.append(f"{ms_id} (plant dataset): {type(exc).__name__}: {exc}")
        print(f"FAIL {ms_id} (plant dataset): {exc}", flush=True)

if failures:
    print(f"setup_peft incomplete: {failures}", file=sys.stderr, flush=True)
    # Don't fail setup on plant errors — examples that need planted
    # content will surface the real failure when they actually load.
PY
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

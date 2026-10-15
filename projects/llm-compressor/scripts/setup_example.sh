#!/usr/bin/env bash
# Prepare the CI environment for one supported llm-compressor example.
# $1 is the manifest profile. $2 is the example path relative to the
# target repo (the engine passes it as the EXAMPLE_PATH env instead;
# either form works). Unknown profiles fail before any install.
set -euo pipefail

export PYTHONNOUSERSITE=1

if [[ $# -lt 1 ]] || [[ $# -gt 2 ]]; then
  echo "usage: $0 <profile> [example-relpath]" >&2
  exit 2
fi

PROFILE="$1"
EXAMPLE_REL="${2:-${EXAMPLE_PATH:?EXAMPLE_PATH env or \$2 is required}}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
ASCEND_PIP_VARIANT=https://mirrors.huaweicloud.com/ascend/repos/pypi/variant
FALLBACK_PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"

# CANN 9.1.0 官方推荐组合是 torch 2.12.0 + torch_npu 2.12.0。
# PyPI 上的 Linux torch==2.12.0 轮子依赖 cuda-toolkit；CPU 轮子只在
# download.pytorch.org，版本号带 +cpu。后面的 pip 必须钉住这颗轮子，
# 否则解析器会换成 CUDA 构建。
write_torch_constraint() {
  if [[ -z "${TORCH_CONSTRAINT:-}" ]]; then
    TORCH_CONSTRAINT=$(mktemp)
    export TORCH_CONSTRAINT
  fi
  printf '%s\n' 'torch==2.12.0+cpu' 'cuda-toolkit<0' > "$TORCH_CONSTRAINT"
}

pip_locked() {
  write_torch_constraint
  python -m pip install \
    --extra-index-url "$ASCEND_PIP_VARIANT" \
    --extra-index-url "$ASCEND_PIP_INDEX" \
    -c "$TORCH_CONSTRAINT" \
    "$@"
}

select_pip_index() {
  if python -c "
import urllib.error
import urllib.request
try:
    urllib.request.urlopen('${CLUSTER_PIP_INDEX}', timeout=3)
except urllib.error.HTTPError:
    pass
" 2>/dev/null; then
    export PIP_INDEX_URL="$CLUSTER_PIP_INDEX"
    export PIP_TRUSTED_HOST="$CLUSTER_PIP_HOST"
  else
    export PIP_INDEX_URL="$FALLBACK_PIP_INDEX"
    unset PIP_TRUSTED_HOST
  fi
  echo "pip index: $PIP_INDEX_URL"
}

ensure_torch_npu_stack() {
  write_torch_constraint
  if python -c "
import torch, torch_npu
print('found torch', torch.__version__, 'torch_npu', torch_npu.__version__)
raise SystemExit(
    0 if torch.__version__.startswith('2.12.0')
    and torch_npu.__version__.startswith('2.12.0')
    else 1)
"; then
    echo "reusing torch 2.12 / torch_npu 2.12 stack"
    return
  fi
  echo "installing torch==2.12.0+cpu (direct URL) + torch_npu==2.12.0"
  CP_ABI=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
  python -m pip install --no-deps \
    "https://download.pytorch.org/whl/cpu/torch-2.12.0%2Bcpu-${CP_ABI}-${CP_ABI}-manylinux_2_28_aarch64.whl"
  python -m pip install -c "$TORCH_CONSTRAINT" \
    'filelock' 'typing-extensions>=4.10.0' 'setuptools<82' \
    'sympy>=1.13.3' 'networkx>=2.5.1' 'jinja2' 'fsspec>=0.8.5' \
    numpy pyyaml
  pip_locked torch_npu==2.12.0
}

install_llmcompressor() {
  # modelscope pinned to the last pre-hub-split line: >=1.38 depends on
  # the separate modelscope-hub package, and 1.40.1's loose ">=0.4.2"
  # floor breaks import when the pip mirror lags on hub 0.4.3. The
  # plant_hf_from_modelscope import path (modelscope.hub.snapshot_download)
  # exists in 1.37.0.
  pip_locked "modelscope==1.37.0" huggingface_hub
  pip_locked -e "$TARGET_ROOT"
}

prefetch_hf_model() {
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export HF_HUB_DISABLE_XET=1
  python - <<'PY'
import os

from huggingface_hub import snapshot_download

model_id = os.environ['PREFETCH_HF_ID']
try:
    snapshot_download(model_id)
except Exception as exc:
    print(f'mirror download failed ({exc}); retrying huggingface.co')
    os.environ.pop('HF_ENDPOINT', None)
    snapshot_download(model_id)
print(f'prefetched {model_id}')
PY
}

plant_hf_from_modelscope() {
  export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/modelscope}"
  python - <<'PY'
import hashlib
import os
from pathlib import Path

from huggingface_hub import try_to_load_from_cache
from modelscope.hub.snapshot_download import snapshot_download

ms_id = os.environ['PREFETCH_MS_ID']
hf_id = os.environ['PREFETCH_HF_ID']
src = snapshot_download(ms_id, ignore_file_pattern=['original/*'])
hf_home = Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface'))
repo_dir = hf_home / 'hub' / f"models--{hf_id.replace('/', '--')}"
# huggingface_hub only resolves refs that look like git commit SHAs.
snap_id = hashlib.sha1(f'{hf_id}|modelscope'.encode()).hexdigest()
snap_dir = repo_dir / 'snapshots' / snap_id
refs_dir = repo_dir / 'refs'
refs_dir.mkdir(parents=True, exist_ok=True)
# huggingface_hub compares this string to the snapshot folder name
# without stripping; a trailing newline makes the cache miss.
(refs_dir / 'main').write_text(snap_id)
if (snap_dir / 'config.json').is_file():
    print(f'reusing planted {hf_id} at {snap_dir}')
else:
    snap_dir.mkdir(parents=True, exist_ok=True)
    src_path = Path(src)
    for item in src_path.rglob('*'):
        if not item.is_file():
            continue
        rel = item.relative_to(src_path)
        if rel.parts and rel.parts[0] == 'original':
            continue
        dest = snap_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            continue
        dest.symlink_to(item.resolve())
    print(f'planted {ms_id} -> {hf_id} at {snap_dir}')
if not (snap_dir / 'config.json').is_file():
    raise SystemExit(f'planted cache missing config.json: {snap_dir}')
cached = try_to_load_from_cache(
    hf_id, 'config.json', cache_dir=str(hf_home / 'hub')
)
if not cached:
    raise SystemExit(
        f'huggingface_hub cannot see planted {hf_id} '
        f'(refs/main={snap_id})'
    )
print(f'hub cache hit {hf_id} config.json -> {cached}')
PY
}

prefetch_dataset() {
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export HF_HUB_DISABLE_XET=1
  python - <<'PY'
import os

from datasets import load_dataset

dataset_id = os.environ['PREFETCH_DATASET_ID']
# Empty split means the same call oneshot makes: load the whole DatasetDict.
split = os.environ.get('PREFETCH_DATASET_SPLIT') or None
kwargs = {'split': split} if split else {}
try:
    load_dataset(dataset_id, **kwargs)
except Exception as exc:
    print(f'mirror dataset failed ({exc}); retrying huggingface.co')
    os.environ.pop('HF_ENDPOINT', None)
    load_dataset(dataset_id, **kwargs)
print(f'prefetched dataset {dataset_id} {split or "<all splits>"}')
PY
}

# from_pretrained(gated_or_planted_id) still hits the Hub for the real
# commit SHA unless offline. The planted snapshot name is not that SHA,
# so leave this until after every prefetch that needs the network.
emit_offline_hub() {
  # Datasets stay online. Calibration splits are often computed at
  # runtime, and a wrong offline cache miss is guard noise. Model ids
  # stay offline so a planted gated snapshot is not re-checked on the Hub.
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    {
      echo 'HF_HUB_OFFLINE=1'
      echo 'TRANSFORMERS_OFFLINE=1'
    } >> "$GITHUB_ENV"
  fi
}

setup_npu_inference() {
  export PATH="/usr/local/sbin:$PATH"
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  select_pip_index
  python -m pip install -U pip 'setuptools<82' wheel
  ensure_torch_npu_stack
  python -c "
import torch, torch_npu
assert torch.npu.is_available()
print(torch.__version__, torch_npu.__version__, torch.npu.device_count())
"
  install_llmcompressor
  python -c "
import llmcompressor
import torch
import torch_npu
assert torch.__version__.startswith('2.12.0'), torch.__version__
assert torch_npu.__version__.startswith('2.12.0'), torch_npu.__version__
print('llmcompressor', llmcompressor.__version__)
"
  PREFETCH_HF_ID=nm-testing/tinyllama-fp8-dynamic-compressed prefetch_hf_model
}

install_npu_stack() {
  export PATH="/usr/local/sbin:$PATH"
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  select_pip_index
  python -m pip install -U pip 'setuptools<82' wheel
  ensure_torch_npu_stack
  python -c "
import torch, torch_npu
assert torch.npu.is_available()
print(torch.__version__, torch_npu.__version__, torch.npu.device_count())
"
  install_llmcompressor
  python -c "
import llmcompressor
import torch
import torch_npu
assert torch.__version__.startswith('2.12.0'), torch.__version__
assert torch_npu.__version__.startswith('2.12.0'), torch_npu.__version__
assert torch.npu.is_available()
print('llmcompressor', llmcompressor.__version__)
"
}

install_example_extras() {
  local example="$TARGET_ROOT/$EXAMPLE_REL"
  if grep -q 'qwen_vl_utils' "$example"; then
    echo "installing qwen_vl_utils for ${EXAMPLE_REL}"
    pip_locked qwen_vl_utils
    python -c "
import torch
assert torch.__version__.startswith('2.12.0'), torch.__version__
import qwen_vl_utils
print('qwen_vl_utils ok', torch.__version__)
"
  fi
}

prefetch_for_example() {
  if [[ ! -f "$TARGET_ROOT/$EXAMPLE_REL" ]]; then
    echo "example not found: $TARGET_ROOT/$EXAMPLE_REL" >&2
    exit 1
  fi
  python "$SCRIPT_DIR/prefetch_example_assets.py" "$TARGET_ROOT/$EXAMPLE_REL"
  emit_offline_hub
}

setup_npu_oneshot() {
  install_npu_stack
  install_example_extras
  prefetch_for_example
}

setup_npu_ddp() {
  install_npu_stack
  install_example_extras
  prefetch_for_example
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"

"setup_${PROFILE}"

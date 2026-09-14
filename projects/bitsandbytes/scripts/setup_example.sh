#!/usr/bin/env bash
# Prepare the CI environment for one supported bitsandbytes example.
# $1 is the manifest profile. Unknown profiles fail before any install.
set -euo pipefail

export PYTHONNOUSERSITE=1

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
ASCEND_VARIANT_INDEX=https://mirrors.huaweicloud.com/ascend/repos/pypi/variant
TRITON_ASCEND_LINKS=https://repo.huaweicloud.com/ascend/repos/pypi/triton-ascend/
FALLBACK_PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"

pip_ascend() {
  python -m pip install --extra-index-url "$ASCEND_PIP_INDEX" "$@"
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

ensure_torch() {
  if python -c "
import torch
print('found torch', torch.__version__)
raise SystemExit(0 if torch.__version__.startswith('2.9.0') else 1)
"; then
    echo "reusing torch"
    return
  fi
  echo "installing torch==2.9.0 numpy"
  pip_ascend torch==2.9.0 numpy
}

assert_bnb_native_lib() {
  python - <<'PY'
import bitsandbytes
import bitsandbytes.cextension as ce
name = type(ce.lib).__name__
print('bitsandbytes', bitsandbytes.__version__)
print('BNB_BACKEND', ce.BNB_BACKEND)
print('lib', name)
assert name != 'ErrorHandlerMockBNBNativeLibrary', (
    'native library missing; CPU .so did not load'
)
PY
}

plant_hf_from_modelscope() {
  export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${HF_HOME:?HF_HOME is required}/modelscope}"
  python - <<'PY'
import hashlib
import os
from pathlib import Path

from huggingface_hub import try_to_load_from_cache
from modelscope.hub.snapshot_download import snapshot_download

ms_id = os.environ['PREFETCH_MS_ID']
hf_id = os.environ['PREFETCH_HF_ID']
src = snapshot_download(ms_id, ignore_file_pattern=['original/*'])
hf_home = Path(os.environ['HF_HOME'])
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

emit_offline_hub() {
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    {
      echo 'HF_HUB_OFFLINE=1'
      echo 'TRANSFORMERS_OFFLINE=1'
    } >> "$GITHUB_ENV"
  fi
}

emit_hf_home() {
  export HF_HOME="${HF_HOME:-${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}/hf-cache}"
  mkdir -p "$HF_HOME"
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    echo "HF_HOME=${HF_HOME}" >> "$GITHUB_ENV"
  fi
  echo "HF_HOME=${HF_HOME}"
}

install_triton_ascend() {
  # Community CUDA Triton has no Ascend backend. triton-ascend 3.2.2
  # Provides the `triton` module; its metadata also Requires-Dist
  # triton==3.5.0, which must not be installed or it overwrites the fork.
  python -m pip uninstall -y triton >/dev/null 2>&1 || true
  python -m pip install --no-deps --force-reinstall \
    --find-links "$TRITON_ASCEND_LINKS" \
    --extra-index-url "$ASCEND_PIP_INDEX" \
    triton-ascend==3.2.2
  python -m pip install pybind11
}

setup_cpu() {
  select_pip_index
  python -m pip install -U pip setuptools wheel
  ensure_torch
  python -m pip install -e "$TARGET_ROOT" -v
  python -m pip install transformers datasets sentencepiece protobuf
  assert_bnb_native_lib
}

setup_npu() {
  export PATH="/usr/local/sbin:$PATH"
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  select_pip_index
  python -m pip install -U pip setuptools wheel
  echo "installing torch==2.10.0 torch-npu==2.10.0.post4"
  python -m pip install \
    --extra-index-url "$ASCEND_VARIANT_INDEX" \
    --extra-index-url "$ASCEND_PIP_INDEX" \
    --find-links "$TRITON_ASCEND_LINKS" \
    torch==2.10.0 torch-npu==2.10.0.post4 numpy pyyaml
  python -m pip install -e "$TARGET_ROOT"
  python -m pip install transformers accelerate huggingface_hub modelscope
  install_triton_ascend
  python - <<'PY'
from importlib.metadata import version
import torch
import torch_npu
import triton
assert torch.__version__.startswith('2.10.0'), torch.__version__
assert torch_npu.__version__.startswith('2.10.0'), torch_npu.__version__
assert torch.npu.is_available(), 'torch.npu.is_available() is False'
assert version('triton-ascend') == '3.2.2', version('triton-ascend')
print('torch', torch.__version__)
print('torch_npu', torch_npu.__version__)
print('npu_count', torch.npu.device_count())
print('triton-ascend', version('triton-ascend'))
print('triton_file', triton.__file__)
PY
  assert_bnb_native_lib
  emit_hf_home
  export HF_HUB_DISABLE_XET=1
  PREFETCH_MS_ID=LLM-Research/gemma-2-2b-it \
    PREFETCH_HF_ID=google/gemma-2-2b-it \
    plant_hf_from_modelscope
  emit_offline_hub
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

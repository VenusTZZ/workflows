#!/usr/bin/env bash
# Prepare the CI environment for one supported slime example.
# $1 is the manifest profile. Unknown profiles fail before any install.
#
# The guarded upstream (THUDM/slime) has no Ascend support; the NPU
# adaptation lives in the gitcode fork Ascend/slime-ascend (main branch,
# no releases). This script clones that fork into $DEPS_ROOT and installs
# the full NPU stack following the fork's own recipes
# (scripts/ascend_script/quick_install.sh + docker/npu_docker/v0.3.0/
# Dockerfile.910b.ubuntu22.04.cann90.latest + the py3.12 component table
# in docs/ascend_tutorial/get_started/quick_start.md). The fork tree is
# both the installed package and the execution root for run_example.sh.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

# ----- pinned component versions (fork Dockerfile ARGs + quick_install.sh) -----
readonly SGLANG_REF=v0.5.13
readonly MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087
readonly MBRIDGE_COMMIT=89eb10887887bc74853f89a4de258c0702932a1c
readonly MEGATRON_ADAPTOR_COMMIT=f707a3b6
readonly TRANSFORMER_ENGINE_NPU_COMMIT=47d60449
readonly SGL_KERNEL_NPU_VERSION=2026.08.21
readonly SGL_KERNEL_NPU_URL="https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGL_KERNEL_NPU_VERSION}/sgl-kernel-npu-${SGL_KERNEL_NPU_VERSION}-torch2.10.0-py312-cann9.1.0-910b-aarch64.zip"
readonly SLIME_FORK_URL=https://gitcode.com/Ascend/slime-ascend.git
readonly SGLANG_GITCODE_URL=https://gitcode.com/gh_mirrors/sg/sglang.git
readonly MEGATRON_GITCODE_URL=https://gitcode.com/gh_mirrors/me/Megatron-LM.git
readonly MEGATRON_GITHUB_URL=https://github.com/NVIDIA/Megatron-LM.git

ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
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

ensure_torch_stack() {
  # The fork pins torch 2.10.0 / torch_npu 2.10.0 / torchvision 0.25.0 on
  # CANN 9.1.0 (py3.12 component table). Install them ahead of sglang's
  # extras so the extras' torch dependency resolves to the pinned line.
  if python -c "
import torch, torch_npu
print('found torch', torch.__version__, 'torch_npu', torch_npu.__version__)
raise SystemExit(0 if torch.__version__.startswith('2.10.0') and torch_npu.__version__.startswith('2.10.0') else 1)
"; then
    echo "reusing installed torch stack"
  else
    echo "installing torch==2.10.0 torch_npu==2.10.0 torchvision==0.25.0"
    pip_ascend torch==2.10.0 torch_npu==2.10.0 torchvision==0.25.0
  fi
}

check_npu_devices() {
  local required="$1"
  local count
  count=$(python -c 'import torch_npu; print(torch_npu.npu.device_count())')
  if ((count < required)); then
    echo "insufficient NPU devices: required=${required} visible=${count}" >&2
    exit 1
  fi
  echo "NPU devices visible: ${count} (required ${required})"
}

git_clone() {
  # git_clone <primary_url> <fallback_url|''> <dest> [extra git args...]
  local primary="$1" fallback="$2" dest="$3"
  shift 3
  if git clone "$@" "$primary" "$dest" 2>&1 | sed 's/^/  /'; then
    return 0
  fi
  rm -rf "$dest"
  if [[ -n "$fallback" ]]; then
    echo "clone from $primary failed, retrying with $fallback"
    git clone "$@" "$fallback" "$dest"
    return 0
  fi
  echo "clone from $primary failed and no fallback is configured" >&2
  return 1
}

append_github_env() {
  printf '%s\n' "$1" >> "$GITHUB_ENV"
}

# ----- step 1: the Ascend fork (installed package + execution root) -----
clone_slime_fork() {
  git_clone "$SLIME_FORK_URL" '' "$SLIME_FORK_ROOT" --depth 1
  local sha
  sha=$(git -C "$SLIME_FORK_ROOT" rev-parse HEAD)
  echo "slime-ascend fork HEAD: $sha"
  append_github_env "SLIME_FORK_HEAD_SHA=$sha"
  append_github_env "SLIME_FORK_ROOT=$SLIME_FORK_ROOT"
}

# ----- step 2: sglang from source with the fork's NPU pyproject -----
install_sglang_source() {
  local dest="$DEPS_ROOT/sglang"
  git_clone "https://github.com/sgl-project/sglang.git" "$SGLANG_GITCODE_URL" "$dest" --depth 1 --branch "$SGLANG_REF"
  mv "$dest/python/pyproject.toml" "$dest/python/pyproject.toml.backup"
  mv "$dest/python/pyproject_npu.toml" "$dest/python/pyproject.toml"
  python -m pip install -e "$dest/python[all_npu]"
  # sglang's python/ package dir must be importable by the launcher.
  append_github_env "PYTHONPATH=$dest/python:${SLIME_FORK_ROOT}:\${PYTHONPATH}"
}

# ----- step 3: prebuilt NPU kernel wheels (torch_memory_saver / sgl_kernel_npu / deep_ep) -----
install_sgl_kernel_npu() {
  local bundle="$DEPS_ROOT/sgl-kernel-npu.zip"
  curl -L --fail --retry 3 --retry-delay 5 --connect-timeout 30 -o "$bundle" "$SGL_KERNEL_NPU_URL"
  unzip -q -o "$bundle" -d "$DEPS_ROOT/sgl-kernel-npu"
  python -m pip install \
    "$DEPS_ROOT"/sgl-kernel-npu/torch_memory_saver-*-cp312-cp312-linux_aarch64.whl \
    "$DEPS_ROOT"/sgl-kernel-npu/sgl_kernel_npu-*-cp312-cp312-linux_aarch64.whl \
    "$DEPS_ROOT"/sgl-kernel-npu/deep_ep-*-cp312-cp312-linux_aarch64.whl
  # deep_ep's C++ extension ships beside the wheel inside the bundle
  # (the fork's quick_install.sh links it into site-packages the same way).
  local site_dir
  site_dir=$(python -c 'import site; print(site.getsitepackages()[0])')
  ln -sf "$DEPS_ROOT/sgl-kernel-npu/lib/deep_ep_cpp.cpython-312-aarch64-linux-gnu.so" \
    "$site_dir/deep_ep_cpp.cpython-312-aarch64-linux-gnu.so"
  python -c 'import deep_ep; print("deep_ep ok:", deep_ep.__path__)'
}

# ----- step 4: mbridge / Megatron-Bridge / Megatron-LM + Ascend adaptors -----
install_megatron_stack() {
  local dest

  dest="$DEPS_ROOT/mbridge"
  git_clone "https://github.com/ISEEKYAN/mbridge.git" '' "$dest"
  git -C "$dest" checkout "$MBRIDGE_COMMIT"
  python -m pip install -e "$dest"

  dest="$DEPS_ROOT/Megatron-Bridge"
  # dev_rl is a moving branch: a shallow clone cannot resolve a commit on
  # it, so clone the branch history here (patches below need a git repo).
  git_clone "https://github.com/fzyzcjy/Megatron-Bridge.git" '' "$dest" --branch dev_rl
  python -m pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

  dest="$DEPS_ROOT/Megatron-LM"
  git_clone "$MEGATRON_GITHUB_URL" "$MEGATRON_GITCODE_URL" "$dest" --recursive
  git -C "$dest" checkout "$MEGATRON_COMMIT"
  python -m pip install -e "$dest"

  dest="$DEPS_ROOT/MegatronAdaptor"
  git_clone "https://gitcode.com/Ascend/MegatronAdaptor.git" '' "$dest"
  git -C "$dest" checkout "$MEGATRON_ADAPTOR_COMMIT"
  python -m pip install -e "$dest"

  dest="$DEPS_ROOT/TransformerEngineNPU"
  git_clone "https://gitcode.com/Ascend/TransformerEngineNPU.git" '' "$dest"
  git -C "$dest" checkout "$TRANSFORMER_ENGINE_NPU_COMMIT"
  python -m pip install -e "$dest"
}

# ----- step 5: triton-ascend replaces the CUDA triton -----
install_triton_ascend() {
  python -m pip uninstall -y triton triton-ascend opencv-python 2>/dev/null || true
  python -m pip install triton-ascend==3.2.1 \
    --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ \
    --trusted-host triton-ascend.osinfra.cn --no-cache-dir
}

# ----- step 6: slime itself + the fork's NPU patch series -----
install_slime_editable() {
  python -m pip install -e "$SLIME_FORK_ROOT"
}

apply_npu_patches() {
  local patch_root="$SLIME_FORK_ROOT/docker/npu_patch/v0.3.0"
  local repo patches
  for repo in sglang Megatron-LM MegatronAdaptor TransformerEngineNPU Megatron-Bridge mbridge; do
    patches="$patch_root/${repo}"
    [[ -d "$patches" ]] || { echo "no NPU patches for $repo, skipping"; continue; }
    echo "applying $(ls "$patches" | wc -l) NPU patches to $repo"
    git -C "$DEPS_ROOT/$repo" am --whitespace=fix "$patches"/*
  done
}

# Shared runtime self-check: the installed slime must resolve to the fork.
verify_installed_runtime() {
  python - <<'PY'
import os
from pathlib import Path

import sglang
import slime
import torch
import torch_npu

fork_root = Path(os.environ["SLIME_FORK_ROOT"]).resolve()
slime_file = Path(slime.__file__).resolve()
print("runtime torch:", torch.__version__)
print("runtime torch_npu:", torch_npu.__version__)
print("runtime sglang:", sglang.__version__)
print("runtime slime:", slime_file)
slime_file.relative_to(fork_root)
print("slime resolves inside fork tree", fork_root)
PY
}

# ----- profile: slime_fully_async -----
# Mirrors the fork's verified NPU nightly config
# tests/tests_npu/nightly_CI/test_qwen2.5_0.5B_fully_async_short_npu.py:
# HF weights + a torch_dist ref checkpoint converted with the fork's own
# tools/convert_hf_to_torch_dist.py (4 procs; conversion is ray-free).
setup_slime_fully_async() {
  check_npu_devices 4
  python -m pip install -q "modelscope==1.37.0"
  local model_dir="$DEPS_ROOT/weights/Qwen2.5-0.5B-Instruct"
  TQDM_MININTERVAL=15 python - <<'PY'
import os
from modelscope import snapshot_download
local = snapshot_download(
    "Qwen/Qwen2.5-0.5B-Instruct",
    cache_dir=os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope")),
)
print("model snapshot:", local)
with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"SLIME_MODEL_PATH={local}\n")
PY
  local torch_dist="$DEPS_ROOT/weights-MA/Qwen2.5-0.5B-Instruct_torch_dist"
  mkdir -p "$DEPS_ROOT/weights-MA"
  if [[ ! -d "$torch_dist" ]]; then
    echo "converting HF checkpoint to torch_dist (4 procs)"
    (
      cd "$SLIME_FORK_ROOT"
      # shellcheck disable=SC1091
      source scripts/models/qwen2.5-0.5B.sh
      export PYTHONPATH="$DEPS_ROOT/Megatron-LM:$DEPS_ROOT/Megatron-Bridge/src:$PYTHONPATH"
      # MODEL_ARGS comes from scripts/models/qwen2.5-0.5B.sh (the same
      # contract as the fork's command_utils.convert_checkpoint).
      # shellcheck disable=SC2086
      torchrun --nproc-per-node 4 \
        tools/convert_hf_to_torch_dist.py \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint "$model_dir" \
        --save "$torch_dist"
    )
  else
    echo "torch_dist checkpoint already present: $torch_dist"
  fi
  append_github_env "SLIME_TORCH_DIST_PATH=$torch_dist"
  append_github_env "SLIME_FIXTURE_JSONL=$FIXTURE_DIR/ci_dapo_16.jsonl"
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
EXAMPLES_ROOT="${EXAMPLES_ROOT:-$TARGET_ROOT}"
FIXTURE_DIR="${FIXTURE_DIR:-$GITHUB_WORKSPACE/workflows/projects/slime/fixtures}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"
DEPS_ROOT="$GITHUB_WORKSPACE/deps"
SLIME_FORK_ROOT="$DEPS_ROOT/slime-ascend"
export SLIME_FORK_ROOT
mkdir -p "$DEPS_ROOT"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
# nnal/atb ships with the CANN toolkit image; tolerate images without it.
# shellcheck disable=SC1091
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true

select_pip_index
python -m pip install -U pip setuptools wheel

ensure_torch_stack
clone_slime_fork
install_sglang_source
install_sgl_kernel_npu
install_megatron_stack
install_triton_ascend
install_slime_editable
apply_npu_patches

"setup_${PROFILE}"
verify_installed_runtime


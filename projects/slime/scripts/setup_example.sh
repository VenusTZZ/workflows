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
# Pinned release-asset digest from the GitHub API. A resumed transfer is
# accepted only when the complete archive matches this value.
readonly SGL_KERNEL_NPU_SHA256=8a12a3b861ea7ae331a1640ad5140203cebc0edae9964b002d33a1b02be462d1
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

append_project_env() {
  local key="${1%%=*}" value="${1#*=}"
  append_github_env "$1"
  printf 'export %s=%q\n' "$key" "$value" >> "$SLIME_PROJECT_ENV"
}

# ----- step 1: the Ascend fork (installed package + execution root) -----
clone_slime_fork() {
  git_clone "$SLIME_FORK_URL" '' "$SLIME_FORK_ROOT" --depth 1
  local sha
  sha=$(git -C "$SLIME_FORK_ROOT" rev-parse HEAD)
  echo "slime-ascend fork HEAD: $sha"
  append_project_env "SLIME_FORK_HEAD_SHA=$sha"
  append_project_env "SLIME_FORK_ROOT=$SLIME_FORK_ROOT"
}

# ----- step 2: sglang from source with the fork's NPU pyproject -----
install_sglang_source() {
  local dest="$DEPS_ROOT/sglang"
  git_clone "https://github.com/sgl-project/sglang.git" "$SGLANG_GITCODE_URL" "$dest" --depth 1 --branch "$SGLANG_REF"
  mv "$dest/python/pyproject.toml" "$dest/python/pyproject.toml.backup"
  mv "$dest/python/pyproject_npu.toml" "$dest/python/pyproject.toml"
  python -m pip install -e "$dest/python[all_npu]"
  # sglang's python/ package dir must be importable by the launcher.
  append_project_env "PYTHONPATH=$dest/python:${SLIME_FORK_ROOT}:$DEPS_ROOT/Megatron-LM:$DEPS_ROOT/Megatron-Bridge/src:${PYTHONPATH:-}"
}

# ----- step 3: prebuilt NPU kernel wheels (torch_memory_saver / sgl_kernel_npu / deep_ep) -----
install_sgl_kernel_npu() {
  local bundle="$DEPS_ROOT/sgl-kernel-npu.zip"
  local cache_root="${SHARED_CACHE_ROOT:-${HOME:?HOME is required}/.cache/huggingface}"
  local cached_bundle="$cache_root/third_party/slime/${SGL_KERNEL_NPU_URL##*/}"
  local attempt actual_sha curl_status
  if [[ -f "$cached_bundle" ]]; then
    if actual_sha=$(sha256sum "$cached_bundle" 2>/dev/null) && \
       [[ "${actual_sha%% *}" == "$SGL_KERNEL_NPU_SHA256" ]]; then
      bundle="$cached_bundle"
      echo "sgl-kernel-npu shared cache hit: $bundle (sha256 ok)"
    else
      echo "warning: sgl-kernel-npu shared cache checksum mismatch: $cached_bundle" >&2
    fi
  fi
  if [[ "$bundle" != "$cached_bundle" ]]; then
    echo "warning: sgl-kernel-npu shared cache unavailable; downloading directly. Seed cache-seed/slime first." >&2
    # On #17 each 300-second attempt downloaded only part of this 12.7 MB
    # asset, then restarted at byte zero. Keep job-local partial bytes and
    # resume after a broken connection; never modify the shared cache here.
    for attempt in 1 2 3 4 5 6; do
      echo "sgl-kernel-npu download attempt $attempt/6 (bytes present: $(stat -c %s "$bundle" 2>/dev/null || echo 0))"
      if curl --location --fail --silent --show-error --continue-at - \
        --connect-timeout 30 --speed-limit 1024 --speed-time 120 \
        --output "$bundle" "$SGL_KERNEL_NPU_URL"; then
        actual_sha=$(sha256sum "$bundle")
        actual_sha=${actual_sha%% *}
        if [[ "$actual_sha" == "$SGL_KERNEL_NPU_SHA256" ]]; then
          echo "sgl-kernel-npu direct download sha256 verified: $actual_sha"
          break
        fi
        echo "sgl-kernel-npu checksum mismatch; restarting download" >&2
        : > "$bundle"
      else
        curl_status=$?
        # Exit 33 means the server rejected Range; a fresh request is the
        # only valid fallback. Other transfer errors leave resumable bytes.
        if ((curl_status == 33)); then
          echo "sgl-kernel-npu server refused resume; restarting download" >&2
          : > "$bundle"
        fi
      fi
      if ((attempt == 6)); then
        echo "sgl-kernel-npu download failed after $attempt attempts" >&2
        return 1
      fi
      sleep 10
    done
  fi
  # The CANN image does not ship unzip; use the stdlib zipfile module
  # (also gives us explicit overwrite semantics).
  python - "$bundle" "$DEPS_ROOT/sgl-kernel-npu" <<'PY'
import sys
import zipfile

bundle, dest = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(bundle) as zf:
    zf.extractall(dest)
print("extracted", len(zf.infolist()), "entries ->", dest)
PY
  python -m pip install \
    "$DEPS_ROOT"/sgl-kernel-npu/torch_memory_saver-*-cp312-cp312-linux_aarch64.whl \
    "$DEPS_ROOT"/sgl-kernel-npu/sgl_kernel_npu-*-cp312-cp312-linux_aarch64.whl \
    "$DEPS_ROOT"/sgl-kernel-npu/deep_ep-*-cp312-cp312-linux_aarch64.whl
  # deep_ep's C++ extension ships inside the deep_ep package dir but is
  # imported as a top-level module, so the fork's Dockerfile links it at
  # the site-packages root. The glob has to expand *inside* site-packages:
  # a relative glob is resolved against the CWD, so running this from
  # anywhere else silently creates a dangling link literally named
  # "deep_ep_cpp*.so" instead of the module (the run #5 failure).
  local site_dir
  site_dir=$(python -c 'import site; print(site.getsitepackages()[0])')
  (
    cd "$site_dir"
    ln -sf deep_ep/deep_ep_cpp*.so .
  )
  # deep_ep_cpp.so links against libtorch_npu.so, so import torch_npu
  # first the way the training runtime does before checking deep_ep.
  python -c 'import torch_npu, deep_ep; print("deep_ep ok:", deep_ep.__path__)'
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
  # git am records a committer, so an identity must exist or git aborts
  # with "Committer identity unknown". The fork's Dockerfile sets the same
  # placeholder via git config; env vars are used here so the step does not
  # depend on $HOME being writable inside the container.
  export GIT_AUTHOR_NAME=temp GIT_AUTHOR_EMAIL=temp@example.com
  export GIT_COMMITTER_NAME=temp GIT_COMMITTER_EMAIL=temp@example.com
  local repo patch_dir patches
  local -a patch_specs=(
    "sglang:sglang"
    "Megatron-LM:megatron"
    "TransformerEngineNPU:transformer_engine_npu"
    "Megatron-Bridge:megatron-bridge"
    "mbridge:mbridge"
  )
  for patch_spec in "${patch_specs[@]}"; do
    repo="${patch_spec%%:*}"
    patch_dir="${patch_spec#*:}"
    patches="$patch_root/$patch_dir"
    if [[ ! -d "$patches" ]]; then
      echo "required NPU patch directory missing: $patches" >&2
      return 1
    fi
    shopt -s nullglob
    local -a patch_files=("$patches"/*)
    shopt -u nullglob
    if ((${#patch_files[@]} == 0)); then
      echo "required NPU patch directory is empty: $patches" >&2
      return 1
    fi
    if [[ ! -d "$DEPS_ROOT/$repo/.git" ]]; then
      echo "patch target repository missing: $DEPS_ROOT/$repo" >&2
      return 1
    fi
    echo "applying ${#patch_files[@]} NPU patches from $patch_dir to $repo"
    git -C "$DEPS_ROOT/$repo" am --whitespace=fix "${patch_files[@]}"
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

# ----- shared Qwen2.5-0.5B assets for fully async and OPD -----
# Both fork NPU recipes use the same HF weights and torch_dist checkpoint.
# Conversion uses the fork's own tool (4 procs; conversion is ray-free).
prepare_qwen25_assets() {
  python -m pip install -q "modelscope==1.37.0"
  # snapshot_download() returns the real cache path (under the runner's
  # persistent ModelScope cache), and that value is what the converter
  # needs: --hf-checkpoint must be an existing directory, because
  # transformers treats a non-existent path as a Hub repo id and dies with
  # HFValidationError. Hand the resolved path to the shell via a file.
  local model_path_file="$DEPS_ROOT/model_path.txt"
  TQDM_MININTERVAL=15 python - "$model_path_file" "${SLIME_RETOOL_MODEL_REPO:-Qwen/Qwen3-4B-Instruct-2507}" <<'PY'
import os
import sys
from modelscope import snapshot_download
model_path_file = sys.argv[1]
local = snapshot_download(
    "Qwen/Qwen2.5-0.5B-Instruct",
    cache_dir=os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope")),
)
print("model snapshot:", local)
with open(model_path_file, "w") as fh:
    fh.write(local + "\n")
PY
  local model_dir
  model_dir=$(cat "$model_path_file")
  if [[ ! -d "$model_dir" ]]; then
    echo "model snapshot dir missing: $model_dir" >&2
    exit 1
  fi
  echo "using HF checkpoint: $model_dir"
  append_project_env "SLIME_MODEL_PATH=$model_dir"
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
      # Transformers 5.8.x emits one compatibility warning per lazy alias
      # lookup; thousands of aliases multiplied by four torchrun ranks made
      # setup logs exceed 12 MB. Keep errors visible while suppressing that
      # repetitive library warning during conversion.
      export TRANSFORMERS_VERBOSITY=error
      torchrun --nproc-per-node 4 \
        tools/convert_hf_to_torch_dist.py \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint "$model_dir" \
        --save "$torch_dist"
    )
  else
    echo "torch_dist checkpoint already present: $torch_dist"
  fi
  append_project_env "SLIME_TORCH_DIST_PATH=$torch_dist"
  append_project_env "SLIME_FIXTURE_JSONL=$FIXTURE_DIR/ci_dapo_16.jsonl"
}

setup_slime_fully_async() {
  check_npu_devices 4
  prepare_qwen25_assets
}

setup_slime_opd() {
  check_npu_devices 8
  prepare_qwen25_assets
}

setup_slime_retool() {
  check_npu_devices "${SLIME_RETOOL_REQUIRED_GPUS:-8}"
  # The fork's ReTool modules import these at module load; its launcher
  # assumes they are preinstalled in the upstream container image.
  python -m pip install -q "modelscope==1.37.0" jinja2 psutil
  python -c 'import jinja2, psutil; print("ReTool deps:", jinja2.__version__, psutil.__version__)'

  local model_path_file="$DEPS_ROOT/retool_model_path.txt"
  TQDM_MININTERVAL=15 python - "$model_path_file" <<'PY'
import os
import sys
from modelscope import snapshot_download
local = snapshot_download(
    sys.argv[2],
    cache_dir=os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope")),
)
print("ReTool model snapshot:", local)
with open(sys.argv[1], "w") as output:
    output.write(local + "\n")
PY
  local model_dir
  model_dir=$(cat "$model_path_file")
  if [[ ! -d "$model_dir" ]]; then
    echo "ReTool model snapshot dir missing: $model_dir" >&2
    exit 1
  fi
  append_project_env "SLIME_MODEL_PATH=$model_dir"

  local model_type="${SLIME_RETOOL_MODEL_TYPE:-qwen3-4B-Instruct-2507}"
  local checkpoint_name="${model_type/qwen3-/Qwen3-}"
  local torch_dist="$DEPS_ROOT/weights-MA/${checkpoint_name}_torch_dist"
  mkdir -p "$DEPS_ROOT/weights-MA"
  if [[ ! -d "$torch_dist" ]]; then
    echo "converting Qwen3-4B-Instruct-2507 to torch_dist (4 procs)"
    (
      cd "$SLIME_FORK_ROOT"
      # shellcheck disable=SC1091
      source "scripts/models/${model_type}.sh"
      export PYTHONPATH="$DEPS_ROOT/Megatron-LM:$DEPS_ROOT/Megatron-Bridge/src:$PYTHONPATH"
      export TRANSFORMERS_VERBOSITY=error
      # shellcheck disable=SC2086
      torchrun --nproc-per-node "${SLIME_RETOOL_REQUIRED_GPUS:-4}" \
        tools/convert_hf_to_torch_dist.py \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint "$model_dir" \
        --save "$torch_dist"
    )
  else
    echo "torch_dist checkpoint already present: $torch_dist"
  fi
  append_project_env "SLIME_TORCH_DIST_PATH=$torch_dist"
  append_project_env "SLIME_FIXTURE_JSONL=$FIXTURE_DIR/ci_retool_math_8.jsonl"
}

setup_slime_retool_sft() {
  # Upstream debug-train-only skips SGLang; two actor NPUs suffice for TP2.
  SLIME_RETOOL_REQUIRED_GPUS=2 setup_slime_retool
  append_project_env "SLIME_SFT_FIXTURE_JSONL=$FIXTURE_DIR/ci_retool_sft_8.jsonl"
  python "$GITHUB_WORKSPACE/workflows/projects/slime/scripts/validate_sft_fixture.py" \
    "$FIXTURE_DIR/ci_retool_sft_8.jsonl" "$(cat "$DEPS_ROOT/retool_model_path.txt")"
}

setup_slime_opd_megatron() {
  setup_slime_opd
  local checkpoint="$DEPS_ROOT/weights-MA/Qwen2.5-0.5B-Instruct_torch_dist"
  if [[ ! -f "$checkpoint/latest_checkpointed_iteration.txt" ]]; then
    echo "Megatron teacher checkpoint is incomplete: $checkpoint" >&2
    exit 1
  fi
}

setup_slime_mis() {
  setup_slime_retool
  if [[ ! -f "$SLIME_FORK_ROOT/examples/train_infer_mismatch_helper/mis.yaml" ]]; then
    echo "MIS requires the fork's native correction configuration" >&2
    exit 1
  fi
}

setup_slime_multi_agent() {
  # Native agent_system requires </think> boundaries to execute rewrite
  # and selector stages; use a thinking Qwen3 checkpoint, not 2507 Instruct.
  SLIME_RETOOL_MODEL_REPO=Qwen/Qwen3-4B \
    SLIME_RETOOL_MODEL_TYPE=qwen3-4B setup_slime_retool
}

setup_slime_multi_task() {
  setup_slime_opd
  python "$GITHUB_WORKSPACE/workflows/projects/slime/scripts/prepare_eval_config.py" \
    "$FIXTURE_DIR" "$DEPS_ROOT/multi-task-ci.yaml"
  append_project_env "SLIME_EVAL_CONFIG=$DEPS_ROOT/multi-task-ci.yaml"
}

setup_slime_strands() {
  setup_slime_retool
  # Constrain installed stack versions while resolving the example's own
  # requirements: tool scaffolding must not replace torch/torch_npu/SGLang.
  local constraints="$DEPS_ROOT/strands-stack-constraints.txt"
  python - "$constraints" <<'PY'
from importlib.metadata import version, PackageNotFoundError
import sys
with open(sys.argv[1], "w") as output:
    for name in ("torch", "torch-npu", "transformers", "sglang", "ray", "numpy", "tokenizers"):
        try:
            output.write(f"{name}=={version(name)}\n")
        except PackageNotFoundError:
            pass
PY
  python -m pip install -c "$constraints" 'strands-sglang==0.3.2' camel-ai
  python - <<'PY'
from camel.interpreters import SubprocessInterpreter
from strands import Agent, tool
from strands_sglang import SGLangModel, ToolLimiter, get_client_from_slime_args
from strands_sglang.tool_parsers import HermesToolParser
result = SubprocessInterpreter(require_confirm=False, print_stdout=False,
    print_stderr=False, execution_timeout=5.0).run("print(2 + 3)", "python")
if "5" not in str(result):
    raise RuntimeError(f"Strands subprocess tool failed: {result}")
print("Strands native tool imports and local Python execution ready")
PY
}

setup_slime_search() {
  setup_slime_opd
  # The upstream CLI supports BM25 without the dense encoder's .cuda().
  # Pyserini 0.25 uses Java 11 and bundles its Anserini JAR; no wiki index
  # or external search API is needed for this local eight-document corpus.
  if ! command -v java >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openjdk-11-jdk-headless
  fi
  java -version
  local constraints="$DEPS_ROOT/search-stack-constraints.txt"
  python - "$constraints" <<'PY'
from importlib.metadata import version, PackageNotFoundError
import sys
with open(sys.argv[1], "w") as output:
    for name in ("torch", "torch-npu", "transformers", "sglang", "ray", "numpy", "tokenizers"):
        try:
            output.write(f"{name}=={version(name)}\n")
        except PackageNotFoundError:
            pass
PY
  python -m pip install -c "$constraints" 'pyserini==0.25.0' faiss-cpu
  python -c 'import faiss; from pyserini.search.lucene import LuceneSearcher; print("native CPU BM25 dependencies ready")'
  local index="$DEPS_ROOT/search-ci-index"
  python -m pyserini.index.lucene --collection JsonCollection \
    --input "$FIXTURE_DIR/search_corpus" --index "$index" \
    --generator DefaultLuceneDocumentGenerator --threads 1 \
    --storePositions --storeDocvectors --storeRaw
  python - "$index" <<'PY'
from pyserini.search.lucene import LuceneSearcher
import sys
searcher = LuceneSearcher(sys.argv[1])
hits = searcher.search("France capital", k=3)
if not hits or "Paris" not in searcher.doc(hits[0].docid).raw():
    raise RuntimeError("CI BM25 index did not retrieve the fixture fact")
print("native BM25 fixture retrieval passed")
PY
  append_project_env "SLIME_SEARCH_INDEX=$index"
  append_project_env "SLIME_FIXTURE_JSONL=$FIXTURE_DIR/ci_search_8.jsonl"
}

setup_slime_geo3k() {
  check_npu_devices 4
  python -m pip install -q 'modelscope==1.37.0' Pillow
  # Image processing uses the torch/torchvision stack already installed
  # above. Do not let this helper package resolve a different torch build.
  python -m pip install -q --no-deps 'qwen-vl-utils==0.0.14'
  local model_path_file="$DEPS_ROOT/geo3k_model_path.txt"
  TQDM_MININTERVAL=15 python - "$model_path_file" <<'PY'
from modelscope import snapshot_download
from pathlib import Path
import os, sys
local = snapshot_download("Qwen/Qwen3-VL-2B-Instruct",
    cache_dir=os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope")))
Path(sys.argv[1]).write_text(str(local) + "\n")
print("Geo3K ModelScope checkpoint:", local)
PY
  local model_dir
  model_dir=$(cat "$model_path_file")
  if [[ ! -f "$model_dir/config.json" ]]; then
    echo "Geo3K model checkpoint is incomplete: $model_dir" >&2
    exit 1
  fi
  append_project_env "SLIME_MODEL_PATH=$model_dir"
  local data_dir="$DEPS_ROOT/geo3k-ci"
  python "$GITHUB_WORKSPACE/workflows/projects/slime/scripts/prepare_geo_fixture.py" \
    "$FIXTURE_DIR/ci_geo_angles_8.json" "$data_dir"
  append_project_env "SLIME_FIXTURE_JSONL=$data_dir/geo3k-ci.jsonl"
  # Native checkpoint._load_checkpoint_hf supports --load HF + bridge.
  # The text-only torch_dist converter is deliberately not used here.
  python - "$model_dir" "$data_dir/geo3k-ci.jsonl" <<'PY'
import sys
from slime.utils.processing_utils import load_tokenizer, load_processor
from slime.utils.data import Dataset
from transformers import AutoConfig
model, fixture = sys.argv[1:]
config = AutoConfig.from_pretrained(model, trust_remote_code=True)
if config.model_type != "qwen3_vl":
    raise RuntimeError(f"Geo3K requires a vision-language model, got {config.model_type}")
tokenizer = load_tokenizer(model, trust_remote_code=True)
processor = load_processor(model, trust_remote_code=True)
if processor is None:
    raise RuntimeError("Geo3K native multimodal processor is unavailable")
dataset = Dataset(fixture, tokenizer, processor, 1024,
    prompt_key="problem", label_key="answer", multimodal_keys={"image": "images"}, apply_chat_template=True)
if len(dataset.samples) != 8:
    raise RuntimeError("Geo3K fixture was unexpectedly filtered")
for sample in dataset.samples:
    if not sample.multimodal_inputs or not sample.multimodal_inputs.get("images"):
        raise RuntimeError("Geo3K native Dataset lost fixture images")
    output = processor(text=sample.prompt, **sample.multimodal_inputs)
    if not all(key in output and output[key].numel() for key in ("pixel_values", "image_grid_thw")):
        raise RuntimeError("Geo3K native processor produced no visual tensors")
print("Geo3K native dataset: eight images with pixel_values and image_grid_thw")
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
EXAMPLES_ROOT="${EXAMPLES_ROOT:-$TARGET_ROOT}"
FIXTURE_DIR="${FIXTURE_DIR:-$GITHUB_WORKSPACE/workflows/projects/slime/fixtures}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"
DEPS_ROOT="$GITHUB_WORKSPACE/deps"
SLIME_PROJECT_ENV="$DEPS_ROOT/slime-example.env"
SLIME_FORK_ROOT="$DEPS_ROOT/slime-ascend"
export SLIME_FORK_ROOT
mkdir -p "$DEPS_ROOT"
: > "$SLIME_PROJECT_ENV"

# Keep the verbosity setting for the separate run step as well. This only
# changes Transformers' logger level; shell errors and other libraries stay
# visible.
append_project_env "TRANSFORMERS_VERBOSITY=error"

# Vendor CANN/ATB env scripts assume a login shell and reference optional
# variables (e.g. $ZSH_VERSION) without ${VAR:-} guards. Under this
# project's `set -u` they die with "unbound variable"; relax strict mode
# only while sourcing vendor code, then restore it (same pattern as
# projects/roll after its CI hit the identical silent failure).
source_vendor_env() {
  local vendor_file="$1"
  if [[ ! -f "$vendor_file" ]]; then
    echo "vendor env script not found, skipping: $vendor_file"
    return 0
  fi
  set +eu
  # shellcheck disable=SC1090
  source "$vendor_file"
  set -eu
}

source_vendor_env /usr/local/Ascend/ascend-toolkit/set_env.sh
source_vendor_env /usr/local/Ascend/nnal/atb/set_env.sh

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

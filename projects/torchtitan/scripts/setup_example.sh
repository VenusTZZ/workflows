#!/usr/bin/env bash
# Prepare the CI environment for one supported torchtitan example.
# $1 is the manifest profile. Unknown profiles fail before any install.
#
# No-patch policy (2026-09-17): we install the upstream v0.3.0 checkout
# as-is and let any NPU-stack bug surface in CI. If a supported example
# fails because of a torch_npu / triton-ascend / CANN issue, the entry
# stays in supported but its run fails — that's the signal upstream
# needs. The seven sed patches previously applied here (and the
# TORCH_NPU_DEVICE_CAPABILITY=9.0 env var in each launcher) are all
# reverted in this commit. Quick-start-Ascend.md and the case doc still
# describe the patches as historical record of what would be needed.
#
# What we DO install: CANN 9.1.0 + torch 2.12.0+cpu + torch_npu 2.12.0
# + triton-ascend 3.5.0+dev20260701 + the v0.3.0 release checkout +
# pyproject deps. Multi-card overlay_args still pass
# --parallelism.spmd-backend full_dtensor (limitation 4 is a CLI switch,
# not a sed patch). Limitation 6 (8B scale) is not patched — the
# supported entries below cap at debugmodel (6 M params).
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
  # Reuse the image torch stack when it already matches the verified
  # line (torch 2.12.0 + torch_npu 2.12.0 + CANN 9.1.0 - per
  # projects/torchtitan/docs/Quick-start-Ascend.md and the case doc
  # docs/case-torchtitan-v0.3.0-torch2.12-npu.md). Otherwise install
  # torch + torch_npu via separate indices.
  #
  # Why not `pip install -i aliyun torch==2.12.0`: aliyun (and the
  # cluster pip cache, both PyPI mirrors) only host the CUDA torch
  # wheel, whose METADATA declares
  # `Requires-Dist: cuda-toolkit==13.0.2`. That conflicts with
  # constraints-npu.txt's `cuda-toolkit<0` and the resolver dies with
  # ResolutionImpossible. The CPU variant is `torch==2.12.0+cpu`
  # (PEP 440 local-version label) and is published ONLY at
  # https://download.pytorch.org/whl/cpu/ - never on PyPI / aliyun /
  # huaweicloud / cluster cache. So pip must fetch the CPU wheel
  # directly from pytorch.org instead of going through any index.
  # We pin the cp312 / manylinux_2_28 / aarch64 filename because the
  # CI image is cann:9.1.0-...-py3.12 on ubuntu22.04 arm64 (verified
  # in projects/torchtitan/tests/test_quick_start_ascend.py +
  # examples_manifest.yaml image field).
  if python -c "
import torch, torch_npu
raise SystemExit(
    0 if torch.__version__.startswith('2.12.0')
    and torch_npu.__version__.startswith('2.12.0') else 1)
" 2>/dev/null; then
    echo "reusing image torch stack ($(python -c 'import torch; print(torch.__version__)'))"
  else
    echo "installing torch==2.12.0+cpu (direct URL from pytorch.org) + torch_npu==2.12.0"
    # Direct URL: pip downloads the specific +cpu wheel file and
    # resolves its (pure-Python) deps from whichever index is active.
    # No cuda-toolkit appears in the dependency set so the constraint
    # file never triggers. Coder verified the URL returns 200 with
    # Content-Length 150023160 (~143 MB) - the file exists.
    pip install --no-deps \
      'https://download.pytorch.org/whl/cpu/torch-2.12.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl'
    # Pull torch's pure-Python deps (filelock, typing-extensions,
    # networkx, sympy, jinja2, fsspec) from aliyun so the import
    # chain `import torch` -> `filelock` -> ... succeeds. Confirmed
    # via `unzip -p ... torch-2.12.0+cpu.dist-info/METADATA |
    # grep Requires-Dist` on 2026-09-16: the +cpu wheel declares
    # none of these as Requires-Dist on cuda-toolkit / nvidia-*,
    # so the resolver is safe under constraints-npu.txt.
    pip install -i "$ALIYUN_PIP_INDEX" \
      'filelock' 'typing-extensions>=4.10.0' 'setuptools<82' \
      'sympy>=1.13.3' 'networkx>=2.5.1' 'jinja2' 'fsspec>=0.8.5'
    pip_ascend torch_npu==2.12.0
  fi
}

ensure_triton_ascend() {
  # triton-ascend provides the only Ascend backend for triton. The
  # community `triton` is blocked via constraints-npu.txt so this
  # never collides. We force --no-deps because the wheel declares
  # triton==3.5.0 (community) as a dep; allowing it to install would
  # clobber the triton/ directory into a half-forked half-community
  # mix. The one dep actually imported at runtime is pybind11 (used
  # by triton-ascend's driver to JIT-compile kernel extensions), so
  # install it explicitly. See
  # docs/case-torchtitan-v0.3.0-torch2.12-npu.md §2.2 layer 3-4.
  if python -c "
from importlib.metadata import version
raise SystemExit(0 if version('triton-ascend').startswith('3.5.0') else 1)
" 2>/dev/null; then
    echo "reusing triton-ascend $(python -c 'from importlib.metadata import version; print(version(\"triton-ascend\"))')"
  else
    echo "installing triton-ascend==3.5.0+dev20260701 (--no-deps) + pybind11"
    pip install --no-deps --extra-index-url \
      https://repo.huaweicloud.com/ascend/repos/pypi/nightly \
      triton-ascend==3.5.0+dev20260701
    pip install pybind11
  fi
}

setup_torchtitan() {
  echo "installing torchtitan from $TARGET_ROOT (release checkout, as-is)"
  # Install the checked-out release (so the guarded tag is exactly the
  # code that runs) plus the v0.3.0 pyproject dependencies. The full
  # deps are large (torchdata / datasets / tensorboard / wandb / tyro /
  # tokenizers / safetensors / einops / pillow / spmd_types); we
  # install them all because multiple components (dataloader,
  # tokenizer, optimizer sharding, spmd_types backend) require them.
  # PIP_CONSTRAINT keeps the CUDA metapackages listed in
  # constraints-npu.txt out.
  python -m pip install -e "$TARGET_ROOT"
  python -m pip install -i "$ALIYUN_PIP_INDEX" \
    "tyro>=1.0.5" "tokenizers>=0.15.0" safetensors einops pillow \
    "torchdata>=0.8.0" "datasets>=3.6.0,<4.8.0" tensorboard wandb \
    "spmd_types==0.2.3"
  python -c "import torchtitan; print('torchtitan', torchtitan.__version__)"
  write_launchers
}

# Write thin .sh launchers into $TARGET_ROOT/scripts/ so the manifest
# entries can target them by path. Each launcher takes CLI args via
# "$@" (preserved verbatim from the engine's OVERLAY_ARGS expansion),
# cd's into the target root, then execs torchrun/-m torchtitan.train.
# The launchers exist only because torchtitan upstream does not ship
# any .sh entry points; running `python -m torchtitan.train` directly
# from the manifest would not let us invoke torchrun (1-card test
# could use python, but 2-card cannot). Writing them in setup keeps
# the run_example.sh logic unchanged from peft.
write_launchers() {
  mkdir -p "$TARGET_ROOT/scripts"
  cat > "$TARGET_ROOT/scripts/run_llama3_debugmodel_2card.sh" <<'LAUNCHER'
#!/usr/bin/env bash
# Launcher for the torchtitan llama3_debugmodel smoke, 2 ranks.
# Adds --parallelism.spmd-backend full_dtensor to overlay_args
# (Quick-start §"限制四": default spmd_types backend needs torch >=2.13)
# and --training.dtype bfloat16 to verify mixed precision on HCCL
# all-reduce. setup_example.sh installs upstream v0.3.0 as-is per the
# no-patch policy.
set -euo pipefail
cd "${TARGET_ROOT:?TARGET_ROOT is required}"
exec torchrun --nproc_per_node=2 \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    --local-ranks-filter 0 \
    --tee 3 \
    -m torchtitan.train \
    "$@"
LAUNCHER
  chmod +x "$TARGET_ROOT/scripts/run_llama3_debugmodel_2card.sh"

  cat > "$TARGET_ROOT/scripts/run_llama3_debugmodel_ce_loss_1card.sh" <<'LAUNCHER'
#!/usr/bin/env bash
# Launcher for the llama3_debugmodel_ce_loss variant. Same as the base
# 1-card launcher; the registry entry uses
# --config llama3_debugmodel_ce_loss. Keeping a separate launcher so
# the manifest path documents the config switch. setup_example.sh
# installs upstream v0.3.0 as-is per the no-patch policy.
set -euo pipefail
cd "${TARGET_ROOT:?TARGET_ROOT is required}"
exec torchrun --nproc_per_node=1 \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    -m torchtitan.train \
    "$@"
LAUNCHER
  chmod +x "$TARGET_ROOT/scripts/run_llama3_debugmodel_ce_loss_1card.sh"

  cat > "$TARGET_ROOT/scripts/run_sft_debugmodel_1card.sh" <<'LAUNCHER'
#!/usr/bin/env bash
# Launcher for the sft_debugmodel example (torchtitan SFT, 1-card).
# sft_debugmodel is a config in torchtitan/models/llama3/config_registry.py:349
# (not a separate module), so the entry is the same
# -m torchtitan.train with --module llama3 --config sft_debugmodel.
# setup_example.sh installs upstream v0.3.0 as-is per the no-patch
# policy.
set -euo pipefail
cd "${TARGET_ROOT:?TARGET_ROOT is required}"
exec torchrun --nproc_per_node=1 \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    -m torchtitan.train \
    "$@"
LAUNCHER
  chmod +x "$TARGET_ROOT/scripts/run_sft_debugmodel_1card.sh"

  echo "wrote launchers to $TARGET_ROOT/scripts/"
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
ensure_triton_ascend

"setup_${PROFILE}"

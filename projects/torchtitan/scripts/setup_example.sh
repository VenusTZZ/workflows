#!/usr/bin/env bash
# Prepare the CI environment for one supported torchtitan example.
# $1 is the manifest profile. Unknown profiles fail before any install.
#
# This script installs the same torchtitan + CANN stack the Quick-start
# guard covers (CANN 9.1.0 + torch 2.12.0 + torch_npu 2.12.0 +
# triton-ascend 3.5.0+dev20260701 + the v0.3.0 release checkout + the
# minimal pyproject deps). Then it applies seven compatibility sed
# patches documented in projects/torchtitan/docs/Quick-start-Ascend.md
# §"兼容性补丁" + the sft_debugmodel-specific 6+7 added after end-to-end
# verification on 2026-09-15. All seven are required for any supported
# example to reach step 1 on this NPU stack.
#
# The patches:
#   1. attention backend: flex -> sdpa (limitation 1, compiler wall)
#   2. ComplexRoPE -> CosSinRoPE + scaling="llama" -> "none"
#      (limitation 2, aclnnIndex complex64 not implemented)
#   3. register ScaledDotProductAttention in config_utils (carries patch 1)
#   4. ChunkedLossWrapper -> CrossEntropyLoss
#      (limitation 3, backward NPU meta-tensor leak)
#   5. set_pg_timeouts torch.distributed.set_timeout -> instance set_timeout
#      (limitation 5, torch 2.13+ module-level API not in 2.12)
#   6. drop torch 2.13-only separate_full_blocks= kwarg in
#      _create_flex_attention_mask (decoder.py:274). Required by any
#      config that hard-codes attn_backend="flex" (e.g. sft_debugmodel
#      before patch 7). Without this, create_block_mask raises
#      TypeError. Added 2026-09-15 after sft_debugmodel verification
#      on hdc-stable-npu-4.
#   7. flip sft_debugmodel's hard-coded attn_backend="flex" to "sdpa"
#      (config_registry.py:349). Patch 6 makes create_block_mask succeed,
#      but the actual flex_attention kernel compile then fails with
#      `ImportError: cannot import name 'triton_key' from
#      'triton.compiler.compiler'` — torch_npu's inductor backend
#      expects a private triton API that the community triton 3.5.0
#      wheel (pinned by triton-ascend) doesn't expose. Switching to sdpa
#      drops document masking but keeps the SFT pipeline (ChatDataLoader,
#      CrossEntropyLoss, Trainer) intact. Added 2026-09-15 after
#      sft_debugmodel verification on hdc-stable-npu-4.
# Limitation 4 (spmd_types default backend needing torch >=2.13) is not
# a sed - it is a CLI switch (--parallelism.spmd-backend full_dtensor)
# in multi-card overlay_args. Limitation 6 (8B scale) is not patched -
# the supported entries below cap at debugmodel (6 M params).
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

# Apply the six compatibility sed patches from
# docs/Quick-start-Ascend.md §"兼容性补丁" to the release checkout.
# Idempotent: each sed has a guard pattern, re-running is a no-op.
apply_compat_patches() {
  cd "$TARGET_ROOT"

  # Patch 1+2: ComplexRoPE -> CosSinRoPE, scaling="llama" -> "none",
  # default attn_backend "flex" -> "sdpa". The first line is a
  # one-line sed with two substitutions (add import + change default).
  # Patch is guarded by `s/^    ComplexRoPE,$/` (only matches the
  # original list entry, not the newly added CosSinRoPE line).
  if ! grep -q '^    CosSinRoPE,$' torchtitan/models/llama3/__init__.py; then
    sed -i 's/^    ComplexRoPE,$/    ComplexRoPE,\n    CosSinRoPE,/; s/ComplexRoPE\.Config(/CosSinRoPE.Config(/; s/scaling="llama",/scaling="none",/' torchtitan/models/llama3/__init__.py
    echo "patched ComplexRoPE -> CosSinRoPE in torchtitan/models/llama3/__init__.py"
  fi
  if ! grep -q 'attn_backend: str = "sdpa",' torchtitan/models/llama3/__init__.py; then
    sed -i 's/attn_backend: str = "flex",/attn_backend: str = "sdpa",/' torchtitan/models/llama3/__init__.py
    echo "patched attn_backend default flex -> sdpa in torchtitan/models/llama3/__init__.py"
  fi

  # Patch 3: register ScaledDotProductAttention so the patched
  # attn_backend="sdpa" actually resolves to a Module.Config.
  if ! grep -q 'ScaledDotProductAttention,$' torchtitan/models/common/config_utils.py; then
    sed -i 's/    VarlenAttention,$/    VarlenAttention,\n    ScaledDotProductAttention,/' torchtitan/models/common/config_utils.py
    # Add an `elif backend == "sdpa":` arm that returns
    # ScaledDotProductAttention.Config(). The guard matches the
    # original line so re-running is a no-op.
    if ! grep -q 'sdpa_banned' torchtitan/models/common/config_utils.py; then
      sed -i 's|    elif backend == "sdpa":|    elif backend == "sdpa":\n        return ScaledDotProductAttention.Config()\n    elif backend == "sdpa_banned":|' torchtitan/models/common/config_utils.py
    fi
    echo "patched config_utils to register ScaledDotProductAttention"
  fi

  # Patch 4: ChunkedLossWrapper -> CrossEntropyLoss. The wrapper
  # does backward-inside-forward which leaks meta tensors on NPU
  # (case doc §2.5); CE loss is mathematically equivalent for the
  # smoke. The sed `/start/,/end/c\` matches the FIRST occurrence in
  # the file (sed's default for non-numeric addresses) - this is the
  # llama3_debugmodel config at line 37; later occurrences at lines
  # 187/260/312/361 belong to other configs and must stay intact.
  #
  # Guard pattern MUST distinguish unpatched from patched state.
  # The previous guard `grep -q 'global_vocab_size=decoder_vocab_size'`
  # matched the unpatched file too (line 39 is
  # `loss_fn=CrossEntropyLoss.Config(global_vocab_size=..., ...)`
  # inside the unpatched ChunkedLossWrapper), so the patch silently
  # no-op'd on every run. Caught only when running setup_example.sh
  # end-to-end on 2026-09-16 - the original 4 supported verifications
  # applied patches by hand and never exercised this codepath.
  # The post-patch signature at 8-space indent
  # `        loss=CrossEntropyLoss.Config(` does not exist in
  # upstream v0.3.0 (which has ChunkedLossWrapper.Config at that
  # indent; the only top-level CrossEntropyLoss is at 4-space indent
  # inside `config.loss = CrossEntropyLoss.Config(` at line 178).
  if ! grep -q '^        loss=CrossEntropyLoss.Config($' torchtitan/models/llama3/config_registry.py; then
    sed -i '/^        loss=ChunkedLossWrapper.Config($/,/^        ),$/c\        loss=CrossEntropyLoss.Config(\n            global_vocab_size=decoder_vocab_size(model_spec),\n        ),' torchtitan/models/llama3/config_registry.py
    echo "patched ChunkedLossWrapper -> CrossEntropyLoss in torchtitan/models/llama3/config_registry.py"
  fi

  # Patch 5: torch.distributed.set_timeout (module-level, torch 2.13+)
  # -> ProcessGroup.set_timeout (instance method, torch 2.12). Without
  # this fix the trainer hits an AttributeError after step 1's
  # set_pg_timeouts call.
  if ! grep -q 'ProcessGroup.set_timeout' torchtitan/distributed/utils.py; then
    sed -i 's|        torch.distributed.set_timeout(timeout, group)|        (group if group is not None else torch.distributed.distributed_c10d._get_default_group()).set_timeout(timeout)|' torchtitan/distributed/utils.py
    echo "patched torch.distributed.set_timeout -> ProcessGroup.set_timeout in torchtitan/distributed/utils.py"
  fi

  # Patch 6: drop the torch 2.13-only `separate_full_blocks=` kwarg in
  # _create_flex_attention_mask. Any model that explicitly selects
  # attn_backend="flex" (e.g. sft_debugmodel hardcodes this at
  # torchtitan/models/llama3/config_registry.py:349) goes through
  # create_block_mask(separate_full_blocks=...) which raises
  # TypeError on torch 2.12. The kwarg only affects the
  # batch-invariance optimization (separating fully-unmasked blocks
  # from partial blocks); removing it falls back to torch 2.12's
  # default. The trainer never enables batch invariance on this NPU
  # stack, so the optimization was off in practice anyway. Idempotent:
  # guard matches the original line, not a subsequent blank line.
  if grep -q '^            separate_full_blocks=not is_in_batch_invariant_mode(),$' torchtitan/models/common/decoder.py; then
    sed -i '/^            separate_full_blocks=not is_in_batch_invariant_mode(),$/d' torchtitan/models/common/decoder.py
    echo "patched separate_full_blocks kwarg removed in torchtitan/models/common/decoder.py"
  fi

  # Patch 7: sft_debugmodel (torchtitan/models/llama3/config_registry.py
  # ~line 349) hardcodes attn_backend="flex". On the NPU stack
  # flex_attention goes through torch.compile -> torch_npu._inductor
  # which calls `from triton.compiler.compiler import triton_key` (an
  # unstable community-triton API not exposed in the 3.5.0 wheel pinned
  # by our triton-ascend pin). The compile fails with ImportError before
  # any kernel is built. Switch the SFT smoke to sdpa (causal-only): SFT
  # pipeline, ChatDataLoader and CrossEntropyLoss are the things under
  # test, document masking is orthogonal. The flex path can be
  # re-enabled later by upgrading triton or torch_npu.
  #
  # Guard pattern fix (2026-09-16): the original guard
  # `'^        model_spec = model_registry(...)' ` used 8-space indent
  # but the upstream line at v0.3.0 config_registry.py:350 is at 4-space
  # indent (inside `def sft_debugmodel()`'s body). The guard never
  # matched, so patch 7 was silently no-op'd on every run - same class
  # of bug as patch 4 (see above). Caught by re-running setup_example.sh
  # end-to-end on a fresh checkout.
  if grep -q '^    model_spec = model_registry("debugmodel", attn_backend="flex")$' torchtitan/models/llama3/config_registry.py; then
    sed -i 's|^    model_spec = model_registry("debugmodel", attn_backend="flex")$|    model_spec = model_registry("debugmodel", attn_backend="sdpa")|' torchtitan/models/llama3/config_registry.py
    echo "patched sft_debugmodel attn_backend flex -> sdpa in torchtitan/models/llama3/config_registry.py"
  fi
}

setup_torchtitan() {
  echo "installing torchtitan from $TARGET_ROOT (release checkout)"
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
  apply_compat_patches
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
  cat > "$TARGET_ROOT/scripts/run_llama3_debugmodel_1card.sh" <<'LAUNCHER'
#!/usr/bin/env bash
# Launcher for the torchtitan llama3_debugmodel smoke, single rank.
# Real HCCL backend (comm.mode default) is selected in overlay_args;
# 1-rank self-barriers, no actual cross-card traffic. Patches 1-5 from
# projects/torchtitan/docs/Quick-start-Ascend.md §"兼容性补丁" are
# applied by setup_example.sh; this launcher only injects torchrun.
# TORCH_NPU_DEVICE_CAPABILITY=9.0 makes torch_npu's shim
# `torch.cuda.get_device_capability` return (9, 0) for every device
# (including the freshly-built rng_state tensor in DTensor's
# OffsetBasedRNGTracker whose `.device` attribute is unset at this
# point). Without the env var, c10d's `broadcast` path computes
# `torch.cuda.get_device_capability(tensor.device)[0] >= 9`, hits
# `None[0]`, and aborts the very first `init_weights` step.
set -euo pipefail
export TORCH_NPU_DEVICE_CAPABILITY=9.0
cd "${TARGET_ROOT:?TARGET_ROOT is required}"
exec torchrun --nproc_per_node=1 \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    -m torchtitan.train \
    "$@"
LAUNCHER
  chmod +x "$TARGET_ROOT/scripts/run_llama3_debugmodel_1card.sh"

  cat > "$TARGET_ROOT/scripts/run_llama3_debugmodel_2card.sh" <<'LAUNCHER'
#!/usr/bin/env bash
# Launcher for the torchtitan llama3_debugmodel smoke, 2 ranks.
# Adds --parallelism.spmd-backend full_dtensor to overlay_args
# (Quick-start §"限制四": default spmd_types backend needs torch >=2.13)
# and --training.dtype bfloat16 to verify mixed precision on HCCL
# all-reduce. Patches 1-5 applied by setup_example.sh.
# TORCH_NPU_DEVICE_CAPABILITY=9.0 — see 1-card launcher for the
# DTensor / c10d broadcast reasoning.
set -euo pipefail
export TORCH_NPU_DEVICE_CAPABILITY=9.0
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
# --config llama3_debugmodel_ce_loss which is the same CrossEntropyLoss
# wiring the patch sets up for llama3_debugmodel. Keeping a separate
# launcher so the manifest path documents the config switch.
# TORCH_NPU_DEVICE_CAPABILITY=9.0 — see 1-card launcher for the
# DTensor / c10d broadcast reasoning.
set -euo pipefail
export TORCH_NPU_DEVICE_CAPABILITY=9.0
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
# Patches 1-5 cover it identically to llama3_debugmodel because
# both use the llama3 model + CE loss.
# TORCH_NPU_DEVICE_CAPABILITY=9.0 — see 1-card launcher for the
# DTensor / c10d broadcast reasoning.
set -euo pipefail
export TORCH_NPU_DEVICE_CAPABILITY=9.0
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

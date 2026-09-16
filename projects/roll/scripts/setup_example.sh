#!/usr/bin/env bash
# Prepare the CI environment for one supported ROLL example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# ROLL itself is installed from TARGET_ROOT (the upstream checkout under test),
# replacing the image's preinstalled copy without touching the pinned
# torch / torch_npu / vLLM / vLLM-Ascend / triton-ascend stack.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

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

# The engine no longer sources CANN for project jobs; bring it into this
# step and persist the runtime variables that ROLL's system_envs does not
# cover so the Run step inherits them.
prepare_ascend_env() {
  if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
  fi
  if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/nnal/atb/set_env.sh
  fi
  # Canonical ROLL NPU knobs (ascend_npu_env_config.md), tightened to smoke
  # timeouts. VLLM_ASCEND_ENABLE_NZ stays off for vLLM stability.
  export HCCL_NPU_SOCKET_PORT_RANGE="auto"
  export HCCL_CONNECT_TIMEOUT="600"
  export HCCL_EXEC_TIMEOUT="600"
  export HCCL_DETERMINISTIC="false"
  export HCCL_OP_EXPANSION_MODE="AIV"
  export HCCL_WHITELIST_DISABLE="1"
  export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
  export TASK_QUEUE_ENABLE="2"
  export COMBINED_ENABLE="1"
  export VLLM_ASCEND_ENABLE_NZ="0"
  export ACL_OP_COMPILER_CACHE_MODE="enable"
  export ACL_OP_COMPILER_CACHE_DIR="/tmp/npu_cache"
  export MODEL_DOWNLOAD_TYPE="MODELSCOPE"
  export USE_MODELSCOPE="1"
  for VAR in HCCL_NPU_SOCKET_PORT_RANGE HCCL_CONNECT_TIMEOUT HCCL_EXEC_TIMEOUT              HCCL_DETERMINISTIC HCCL_OP_EXPANSION_MODE HCCL_WHITELIST_DISABLE              PYTORCH_NPU_ALLOC_CONF TASK_QUEUE_ENABLE COMBINED_ENABLE              VLLM_ASCEND_ENABLE_NZ ACL_OP_COMPILER_CACHE_MODE              ACL_OP_COMPILER_CACHE_DIR MODEL_DOWNLOAD_TYPE USE_MODELSCOPE; do
    echo "${VAR}=${!VAR}" >> "$GITHUB_ENV"
  done
}

ensure_roll_installed() {
  local target_root="${TARGET_ROOT:?TARGET_ROOT is required}"
  # Both the current main (pyproject.toml without a pip build-system and a
  # legacy setup.py fallback) and older refs like v0.3.0 (setup.py only)
  # install cleanly via setuptools; --no-build-isolation keeps the in-image
  # setuptools path while --no-deps protects the pinned torch/npu stack.
  echo "=> installing ROLL from ${target_root} (editable, no deps, no isolation)"
  python -m pip install -q --no-deps --no-build-isolation -e "$target_root"
  python - "$target_root" <<'PY'
import os, sys, pathlib
import roll
target = pathlib.Path(sys.argv[1]).resolve()
source = pathlib.Path(roll.__path__[0]).resolve()
try:
    source.relative_to(target)
except ValueError:
    raise SystemExit(
        f"roll resolves to {source}, expected a path inside {target}; "
        "a stale image install has taken precedence over TARGET_ROOT"
    )
print(f"roll import path ok: {source}")
PY
}

ensure_agentic_deps() {
  if python -c "import gym, gymnasium, gem, gym_sokoban" 2>/dev/null; then
    echo "reusing image agentic env deps"
    return
  fi
  echo "=> installing agentic env deps (gem-llm / gym_sokoban / gymnasium)"
  python -m pip install "gem-llm==0.0.4" "gym_sokoban" "gymnasium[toy-text]"
  python -c "import gym, gymnasium, gem, gym_sokoban; print('agentic env deps ok')"
}

ms_download_model() {
  local model_id="${1:?model id required}"
  if python -c "import modelscope" 2>/dev/null; then
    echo "reusing image modelscope"
  else
    echo "=> installing modelscope"
    python -m pip install -q "modelscope==1.37.0"
  fi
  echo "=> downloading ${model_id} from ModelScope"
  MODEL_ID="$model_id" python - <<'PY'
import os
from modelscope import snapshot_download
TQDM_MININTERVAL = os.environ.get("TQDM_MININTERVAL", "15")
model_id = os.environ["MODEL_ID"]
cache = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))
local = snapshot_download(model_id, cache_dir=cache)
with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as handle:
    handle.write(f"ROLL_MODEL_PATH={local}
")
print(f"ROLL_MODEL_PATH={local}")
PY
}

prepare_ci_configs() {
  local src="${PROJECT_ROOT:?PROJECT_ROOT is required}/configs"
  local dst="$TARGET_ROOT/examples/ci_roll"
  echo "preparing CI configs: $src -> $dst"
  mkdir -p "$dst"
  cp "$src"/ci_agentic_train.yaml "$src"/ci_agentic_rollout.yaml "$src"/ci_rlvr.yaml "$dst/"
  ls -la "$dst/"
}

check_npu_devices() {
  local minimum="$1"
  local devices
  devices=$(python - <<'PY'
import torch, torch_npu
print(int(torch.npu.device_count()))
PY
)
  echo "NPU available: $(python -c "import torch, torch_npu; print(torch.npu.is_available())"), devices: $devices"
  if [[ "$devices" -lt "$minimum" ]]; then
    echo "FATAL: profile $PROFILE needs at least $minimum NPUs, runner exposes $devices" >&2
    exit 1
  fi
}

prepare_ascend_env
select_pip_index

python -c "import torch, torch_npu, vllm, vllm_ascend; print('torch', torch.__version__, 'torch_npu', torch_npu.__version__, 'vllm', vllm.__version__)"

ensure_roll_installed
ms_download_model "Qwen/Qwen2.5-0.5B-Instruct"
prepare_ci_configs

case "$PROFILE" in
  agentic_train_npu)
    echo "profile: agentic_train_npu (2 NPUs: FSDP2 train + vLLM rollouts)"
    ensure_agentic_deps
    check_npu_devices 2
    ;;
  agentic_rollout_npu)
    echo "profile: agentic_rollout_npu (1 NPU: vLLM rollouts only)"
    ensure_agentic_deps
    check_npu_devices 1
    ;;
  rlvr_npu)
    echo "profile: rlvr_npu (4 NPUs: FSDP2 train(2) + vLLM(1) + reference(1))"
    check_npu_devices 4
    ;;
  *)
    echo "FATAL: unknown profile '$PROFILE'" >&2
    exit 2
    ;;
esac

echo "setup complete for profile $PROFILE"

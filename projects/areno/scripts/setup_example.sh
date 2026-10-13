#!/usr/bin/env bash
# Prepare the CI environment for one supported AReno example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# AReno installs from the target checkout (TARGET_ROOT) - the engine's
# monitored ref, i.e. the latest upstream release. v0.9.0 ships the NPU
# backend, so no main switch is needed. The source build compiles
# areno.accel._areno_accel_npu (Ascend C + CMake, no Python fallback).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
FALLBACK_PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"

source_vendor_env() {
  # CANN/ATB env scripts assume a login shell and reference optional
  # variables without ${VAR:-} guards; relax strict mode while sourcing
  # vendor code, then restore it.
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

select_pip_index() {
  # Runners live in mainland China: prefer the cluster pip cache, fall
  # back to the Tsinghua mirror. The ascend index stays available via
  # PIP_EXTRA_INDEX_URL (set by the workflow) for torch_npu wheels.
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

ensure_build_tools() {
  # Building the Ascend C extension needs cmake and binutils (nm).
  command -v cmake >/dev/null 2>&1 || {
    apt-get update -qq >/dev/null
    apt-get install -y -qq cmake binutils >/dev/null
  }
  command -v cmake && command -v nm
}

ensure_torch_stack() {
  # The CANN 9.0.0 image ships no torch, so install the pinned pair. Fetch
  # torch from the Ascend/Huawei index instead of the upstream +cpu
  # find-links mirror: the Ascend repo hosts the torch build paired with
  # torch_npu and the Huawei CDN is fast from the mainland runners - the same
  # approach as trl / accelerate / dgl - avoiding a slow single-connection
  # 146 MB aliyun download.
  if python -c "import torch, torch_npu; print('found torch', torch.__version__, 'torch_npu', torch_npu.__version__)"; then
    echo "reusing image torch stack"
    return
  fi
  echo "installing torch==2.10.0 + torch_npu==2.10.0.post2 from the Ascend repo"
  if ! python -m pip install --extra-index-url "$ASCEND_PIP_INDEX" \
      torch==2.10.0 torch_npu==2.10.0.post2; then
    echo "Ascend index lacked the pair; falling back to the upstream +cpu find-links"
    python -m pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.10.0
    python -m pip install --extra-index-url "$ASCEND_PIP_INDEX" torch_npu==2.10.0.post2
  fi
  python -c "import torch, torch_npu; print('torch', torch.__version__, 'torch_npu', torch_npu.__version__)"
}

install_areno() {
  source_vendor_env /usr/local/Ascend/ascend-toolkit/set_env.sh
  # Build AReno + its Ascend C extension, but skip the two heavy optional deps
  # the CI never uses: flash-attn-npu (only for --attn-backend flash; the legs
  # pin native) and flash-linear-attention (only for KDA / linear-attention
  # models; the legs use dense Qwen3, and areno routes NPU lightning attn to
  # its own accel.npu.seg_la). Building both from source (kernel compile + git
  # clone) is the slowest part of setup, so install AReno with --no-deps and
  # then install the remaining npu.txt deps from wheels.
  python -m pip install -e "$TARGET_ROOT" --no-build-isolation --no-deps
  local deps
  deps=$(grep -vE '^[[:space:]]*(#|flash-attn-npu|flash-linear-attention)' \
    "$TARGET_ROOT/requirements/npu.txt")
  # shellcheck disable=SC2086
  python -m pip install $deps
  python - <<'PY'
import importlib.util as u
if u.find_spec('areno.accel._areno_accel_npu') is None:
    raise SystemExit(
        'areno.accel._areno_accel_npu MISSING - installer did not take the '
        'NPU branch; check that torch_npu is installed before AReno')
print('areno NPU extension ok')
PY
}

install_pi_toolchain() {
  # pi (pi.dev) is an external Node.js CLI that the pi example drives through
  # its own environment knobs (ARENO_PI_EXECUTABLE / ARENO_PI_PYTHON /
  # ARENO_PI_RESULT_DIR). Install everything from mainland-reachable mirrors:
  #   - Node >= 22.19 from the npmmirror node binaries (runner is aarch64);
  #   - pi from the npmmirror npm registry (package
  #     @earendil-works/pi-coding-agent);
  #   - fd / ripgrep via apt (pi's bash tools use them);
  #   - the SWE-smith task set generated from ModelScope.
  # Paths/executables are exported via GITHUB_ENV so run_example.sh sees them.
  local node_ver="${ARENO_PI_NODE_VERSION:-22.19.0}"
  if ! command -v node >/dev/null 2>&1; then
    echo "installing Node ${node_ver} (aarch64) from npmmirror"
    curl -fsSL "https://cdn.npmmirror.com/binaries/node/v${node_ver}/node-v${node_ver}-linux-arm64.tar.gz" \
      | tar -xz -C /usr/local --strip-components=1
  fi
  node --version
  npm --version

  echo "installing pi from the npmmirror npm registry"
  npm install -g --ignore-scripts --registry=https://registry.npmmirror.com \
    @earendil-works/pi-coding-agent
  local pi_bin
  pi_bin="$(command -v pi)"
  "$pi_bin" --version

  # fd / ripgrep for pi's bash tools.
  command -v rg >/dev/null 2>&1 || {
    apt-get update -qq >/dev/null
    apt-get install -y -qq fd-find ripgrep >/dev/null
  }
  command -v fd >/dev/null 2>&1 || ln -sf /usr/bin/fdfind /usr/local/bin/fd
  command -v fd && command -v rg

  # The example's task Python (pi's bash tools + the controller's grader) just
  # needs pytest, already present in the main env; the upstream README's
  # separate python3.12 venv is not required here.
  local task_py
  task_py="$(command -v python)"
  # Dataset-prep deps (modelscope + pyarrow) for generate_dataset.py.
  python -m pip install -q -r "$TARGET_ROOT/examples/agentic/pi/requirements.txt"

  local data="$GITHUB_WORKSPACE/pi-swesmith.jsonl"
  local results="$GITHUB_WORKSPACE/pi-results"
  mkdir -p "$results"
  echo "generating SWE-smith tasks from ModelScope -> $data"
  python "$TARGET_ROOT/examples/agentic/pi/generate_dataset.py" \
    --python "$task_py" --output "$data"

  {
    echo "ARENO_PI_EXECUTABLE=$pi_bin"
    echo "ARENO_PI_PYTHON=$task_py"
    echo "ARENO_PI_RESULT_DIR=$results"
    echo "ARENO_PI_DATA=$data"
  } >> "$GITHUB_ENV"
}

# Known profiles. Every current profile shares the same preparation (the AReno
# NPU source build below); a profile that needs extra dependencies would branch
# on PROFILE here, before the shared steps.
SUPPORTED_PROFILES="bash_game_npu codebreaker_npu coding_npu duelgrid_npu shopping_npu terminal_hacking_npu tictactoe_npu tictactoe_no_tool_npu math_npu alpaca_npu pi_npu"
case "$PROFILE" in
  bash_game_npu | codebreaker_npu | coding_npu | duelgrid_npu | shopping_npu | terminal_hacking_npu | tictactoe_npu | tictactoe_no_tool_npu | math_npu | alpaca_npu | pi_npu) ;;
  *)
    echo "unknown profile: ${PROFILE} (supported: ${SUPPORTED_PROFILES})" >&2
    exit 1
    ;;
esac

select_pip_index
python -m pip install -U pip 'setuptools>=69' wheel
ensure_build_tools
ensure_torch_stack
install_areno
# pytest backs the coding example's run_command test execution: its tasks run
# `python -m pytest ...` inside the per-task workspace during rollout.
python -m pip install -q pytest
# All three supported examples run agentic rollouts: their run_agent.py drives
# an OpenAI-compatible client against the local rollout proxy. AReno's NPU
# requirements list `openai`, but the base image can already satisfy `openai`
# without pulling its `httpx` dependency, so install both explicitly.
python -m pip install -q openai httpx

# Per-profile extra toolchain (external deps an example needs on top of AReno).
case "$PROFILE" in
  pi_npu) install_pi_toolchain ;;
esac

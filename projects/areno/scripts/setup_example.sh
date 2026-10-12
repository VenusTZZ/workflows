#!/usr/bin/env bash
# Prepare the CI environment for one supported AReno example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# AReno installs from the target checkout (TARGET_ROOT). The engine checks
# out the monitored ref (latest release); because v0.0.8 predates the NPU
# backend, the checkout is switched to upstream main when the NPU sources
# are absent - temporary handling, same as areno-quick-start.yml. The
# source build compiles areno.accel._areno_accel_npu (Ascend C + CMake, no
# Python fallback).
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

use_npu_capable_source() {
  # v0.0.8 (the guarded release) has no NPU backend; switch the workspace
  # copy to upstream main so the source build can produce the NPU
  # extension. Manual dispatch with target_ref=main is a no-op here.
  if [[ ! -d "$TARGET_ROOT/areno/accel/csrc/npu" ]]; then
    echo "NPU sources absent at $(git -C "$TARGET_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown); switching to upstream main"
    git -C "$TARGET_ROOT" fetch --depth 1 origin main
    git -C "$TARGET_ROOT" checkout --detach FETCH_HEAD
  fi
  echo "building from $(git -C "$TARGET_ROOT" rev-parse --short HEAD)"
}

install_areno() {
  source_vendor_env /usr/local/Ascend/ascend-toolkit/set_env.sh
  python -m pip install -e "$TARGET_ROOT" --no-build-isolation
  python - <<'PY'
import importlib.util as u
if u.find_spec('areno.accel._areno_accel_npu') is None:
    raise SystemExit(
        'areno.accel._areno_accel_npu MISSING - installer did not take the '
        'NPU branch; check that torch_npu is installed before AReno')
print('areno NPU extension ok')
PY
}

# Known profiles. Every current profile shares the same preparation (the AReno
# NPU source build below); a profile that needs extra dependencies would branch
# on PROFILE here, before the shared steps.
SUPPORTED_PROFILES="bash_game_npu codebreaker_npu"
case "$PROFILE" in
  bash_game_npu | codebreaker_npu) ;;
  *)
    echo "unknown profile: ${PROFILE} (supported: ${SUPPORTED_PROFILES})" >&2
    exit 1
    ;;
esac

select_pip_index
python -m pip install -U pip 'setuptools>=69' wheel
ensure_build_tools
ensure_torch_stack
use_npu_capable_source
install_areno

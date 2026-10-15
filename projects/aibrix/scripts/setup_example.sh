#!/usr/bin/env bash
# Prepare the CI environment for one supported aibrix example.
# $1 is the manifest profile. Unknown profiles fail before any install.
set -euo pipefail

export PYTHONNOUSERSITE=1

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

FALLBACK_PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"

select_pip_index() {
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
    export PIP_INDEX_URL="$FALLBACK_PIP_INDEX"
    unset PIP_TRUSTED_HOST
  fi
  echo "pip index: $PIP_INDEX_URL"
}

setup_local_gateway() {
  export PATH="/usr/local/sbin:/usr/local/bin:${PATH}"
  set +u
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  # ATB env: the legacy workflow bind-mounted /usr/local/Ascend/nnal
  # with an atb/latest/atb layout; the CANN image ships atb directly
  # (same layout as the other engine-backed projects - opencv / slime /
  # speculators / roll source /usr/local/Ascend/nnal/atb/set_env.sh).
  # Accept both layouts, fail loudly when neither exists (vllm-ascend
  # needs the atb libs at runtime).
  if [[ -f /usr/local/Ascend/nnal/atb/latest/atb/set_env.sh ]]; then
    source /usr/local/Ascend/nnal/atb/latest/atb/set_env.sh
  elif [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
  else
    echo "ATB env script not found under /usr/local/Ascend/nnal" >&2
    exit 1
  fi
  set -euo pipefail

  if ! command -v ss >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y iproute2
  fi
  command -v setsid >/dev/null
  command -v ss >/dev/null

  # Tools live on the shared persistent runner cache (the pool mounts
  # one volume at ~/.cache/huggingface; see docs/examples-guard-engine.md).
  # The legacy default /root/.cache/cosdt-ci-test/aibrix existed only via
  # a container bind-mount the engine does not provide. First engine run
  # pays the Go/Envoy/vllm downloads + builds; later runs hit the warm
  # cache. MODELSCOPE_CACHE lands vllm's model download (VLLM_USE_MODEL-
  # SCOPE=True in run_example.sh) on the same volume - same pattern as
  # projects/bitsandbytes.
  SHARED_CACHE_ROOT="${HF_HOME:-${HOME}/.cache/huggingface}"
  TOOLS="${AIBRIX_TOOLS_DIR:-${SHARED_CACHE_ROOT}/aibrix/tools}"
  export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${SHARED_CACHE_ROOT}/modelscope}"
  mkdir -p "${TOOLS}/toolchain" "${TOOLS}/bin" "${TOOLS}/src" "${MODELSCOPE_CACHE}"
  GH_PROXY=https://gh-proxy.test.osinfra.cn
  if [[ ! -x "${TOOLS}/toolchain/go/bin/go" ]]; then
    curl -fL --connect-timeout 20 --retry 5 --retry-delay 3 --max-time 180 \
      -o "${TOOLS}/go.tar.gz" \
      https://mirrors.aliyun.com/golang/go1.22.6.linux-arm64.tar.gz
    echo 'c15fa895341b8eaf7f219fada25c36a610eb042985dc1a912410c1c90098eaf2  '"${TOOLS}/go.tar.gz" | sha256sum -c
    tar -C "${TOOLS}/toolchain" -xzf "${TOOLS}/go.tar.gz"
  fi
  if [[ ! -x "${TOOLS}/bin/envoy" ]]; then
    envoy_url='https://github.com/envoyproxy/envoy/releases/download/v1.39.0/envoy-1.39.0-linux-aarch_64'
    envoy_ok=0
    for src in "${GH_PROXY}/${envoy_url}" "$envoy_url"; do
      if curl -fL --connect-timeout 20 --retry 8 --retry-all-errors --retry-delay 3 --max-time 300 \
        -C - -o "${TOOLS}/bin/envoy.part" "$src" \
        && echo 'ee53a4f5375566f15944dc9cb03afb1fc228df38f61737c677f139213215afcf  '"${TOOLS}/bin/envoy.part" | sha256sum -c; then
        envoy_ok=1
        break
      fi
      rm -f "${TOOLS}/bin/envoy.part"
    done
    if [[ "$envoy_ok" -ne 1 ]]; then
      echo "failed to download Envoy 1.39.0" >&2
      exit 1
    fi
    mv "${TOOLS}/bin/envoy.part" "${TOOLS}/bin/envoy"
    chmod +x "${TOOLS}/bin/envoy"
  fi
  export PATH="${TOOLS}/bin:${TOOLS}/toolchain/go/bin:${PATH}"
  export GOPROXY=https://goproxy.cn,direct
  export GOPATH="${TOOLS}/gopath"
  export GOCACHE="${TOOLS}/gocache"
  mkdir -p "${GOPATH}" "${GOCACHE}"

  if [[ -n "${GITHUB_ENV:-}" ]]; then
    {
      echo "PATH=${PATH}"
      echo "GOPROXY=${GOPROXY}"
      echo "GOPATH=${GOPATH}"
      echo "GOCACHE=${GOCACHE}"
      echo "AIBRIX_TOOLS_DIR=${TOOLS}"
      echo "MODELSCOPE_CACHE=${MODELSCOPE_CACHE}"
    } >> "${GITHUB_ENV}"
  fi

  mkdir -p "${TARGET_ROOT}/bin"
  if [[ ! -x "${TARGET_ROOT}/bin/gateway-plugins" ]]; then
    (
      cd "${TARGET_ROOT}"
      CGO_ENABLED=0 go build -tags=nozmq -o bin/gateway-plugins cmd/plugins/main.go
    )
  fi
  test -x "${TARGET_ROOT}/bin/gateway-plugins"

  select_pip_index
  export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
  export PIP_RETRIES="${PIP_RETRIES:-5}"
  python -m pip install -U pip
  python -m pip install \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    --find-links https://repo.huaweicloud.com/ascend/repos/pypi/triton-ascend/ \
    torch==2.10.0 torch-npu==2.10.0.post4 torchvision==0.25.0 torchaudio==2.10.0 triton-ascend==3.2.2
  python -m pip install 'cmake>=3.26' nanobind ninja setuptools-rust wheel 'setuptools-scm>=8' 'setuptools>=77,<81'
  if [[ ! -d "${TOOLS}/src/vllm/.git" ]]; then
    vllm_origin='https://github.com/vllm-project/vllm.git'
    vllm_ok=0
    for src in "${GH_PROXY}/${vllm_origin}" "$vllm_origin"; do
      for _ in 1 2 3; do
        if GIT_TERMINAL_PROMPT=0 GIT_HTTP_VERSION=HTTP/1.1 \
          git clone --depth 1 --branch v0.23.0 "$src" "${TOOLS}/src/vllm"; then
          vllm_ok=1
          break
        fi
        rm -rf "${TOOLS}/src/vllm"
        sleep 5
      done
      if [[ "$vllm_ok" -eq 1 ]]; then
        break
      fi
    done
    if [[ "$vllm_ok" -ne 1 ]]; then
      echo "failed to clone vllm v0.23.0" >&2
      exit 1
    fi
  fi
  export VLLM_TARGET_DEVICE=empty
  python -m pip install --no-build-isolation -e "${TOOLS}/src/vllm"
  python -m pip install grpcio-tools
  export CMAKE_PREFIX_PATH="$(python -m nanobind --cmake_dir)${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  export Python_EXECUTABLE="$(command -v python)"
  export PYTHON_EXECUTABLE="${Python_EXECUTABLE}"
  python -m pip install --no-build-isolation \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    vllm-ascend==0.23.0
  python -m pip install modelscope==1.31.0
  python -c "import importlib.metadata as m
for n in ['torch', 'torch-npu', 'vllm', 'vllm-ascend', 'modelscope']:
    print(n, m.version(n))"
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"

"setup_${PROFILE}"

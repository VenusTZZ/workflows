#!/usr/bin/env bash
# Prepare the CI environment for one supported example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# EXEC comes from the workflow as an environment variable.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"
SUPPORTED_PROFILES="ascend-direct ascend-direct-http host-tcp host-oneshot ascend-hccl"

profile_ok=0
for name in $SUPPORTED_PROFILES; do
  if [[ "$PROFILE" == "$name" ]]; then
    profile_ok=1
  fi
done
if [[ "$profile_ok" != 1 ]]; then
  echo "unknown profile: $PROFILE (supported: $SUPPORTED_PROFILES)" >&2
  exit 2
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
EXEC_REL="${EXEC:?EXEC is required}"

case "$PROFILE" in
  ascend-hccl)
    BUILD_DIR="build-ascend-hccl"
    ;;
  *)
    BUILD_DIR="build"
    ;;
esac

if [[ "$EXEC_REL" != "$BUILD_DIR/"* ]]; then
  echo "setup: EXEC=$EXEC_REL does not start with $BUILD_DIR/ for profile $PROFILE" >&2
  exit 1
fi

case "$PROFILE" in
  ascend-direct|ascend-direct-http|ascend-hccl)
    if [[ ! -f /etc/hccn.conf ]]; then
      echo "setup failed: /etc/hccn.conf is missing" >&2
      echo "HCCL and Ascend Direct read device NIC IPs from this file." >&2
      echo "The NPU driver writes it on the host. In a container, bind-mount the host file." >&2
      exit 1
    fi
    ;;
esac

DEPS=(
  build-essential
  cmake
  git
  pkg-config
  libgoogle-glog-dev
  libgflags-dev
  libibverbs-dev
  libjsoncpp-dev
  libnuma-dev
  libyaml-cpp-dev
  libssl-dev
  libcurl4-openssl-dev
)

if [[ "$PROFILE" == "ascend-hccl" ]]; then
  if dpkg -s libopenmpi-dev >/dev/null 2>&1 || dpkg -s openmpi-bin >/dev/null 2>&1; then
    echo "setup: openmpi packages are present; upstream Ascend Transport docs warn this conflicts with mpich" >&2
  fi
  DEPS+=(mpich libmpich-dev)
fi

source_cann() {
  export PATH="/usr/local/sbin:/usr/local/bin:$PATH"
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
}

missing_debs() {
  local pkg
  for pkg in "${DEPS[@]}"; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
      printf '%s\n' "$pkg"
    fi
  done
}

install_debs() {
  local missing
  missing=$(missing_debs)
  if [[ -z "$missing" ]]; then
    echo "setup: build packages already installed"
    return
  fi
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  # shellcheck disable=SC2086
  if ! apt-get install -y --no-install-recommends $missing; then
    echo "setup failed: could not install: $missing" >&2
    echo "This is a guard setup failure, not an example failure." >&2
    exit 1
  fi
}

init_pybind11() {
  if [[ -f "$TARGET_ROOT/extern/pybind11/CMakeLists.txt" ]]; then
    echo "setup: pybind11 submodule already present"
    return
  fi
  if git -C "$TARGET_ROOT" submodule update --init --depth 1 extern/pybind11; then
    echo "setup: initialized extern/pybind11"
    return
  fi
  echo "setup: submodule update failed; cloning pybind11 via ghfast.top" >&2
  local expect
  expect=$(git -C "$TARGET_ROOT" ls-tree HEAD extern/pybind11 | awk '{print $3}')
  if [[ ! "$expect" =~ ^[0-9a-f]{40}$ ]]; then
    echo "setup: cannot read pybind11 gitlink SHA from target tree" >&2
    exit 1
  fi
  rm -rf "$TARGET_ROOT/extern/pybind11"
  mkdir -p "$TARGET_ROOT/extern/pybind11"
  git -C "$TARGET_ROOT/extern/pybind11" init
  git -C "$TARGET_ROOT/extern/pybind11" remote add origin \
    https://ghfast.top/https://github.com/pybind/pybind11.git
  git -C "$TARGET_ROOT/extern/pybind11" fetch --depth 1 origin "$expect"
  git -C "$TARGET_ROOT/extern/pybind11" checkout --detach FETCH_HEAD
  local got
  got=$(git -C "$TARGET_ROOT/extern/pybind11" rev-parse HEAD)
  if [[ "$got" != "$expect" ]]; then
    echo "setup: pybind11 SHA mismatch: got $got want $expect" >&2
    exit 1
  fi
  echo "setup: cloned extern/pybind11 @$got via ghfast.top"
}

ensure_aiohttp() {
  if python3 -c 'import aiohttp' >/dev/null 2>&1; then
    echo "setup: aiohttp already importable"
    return
  fi
  if [[ -z "${PIP_EXTRA_INDEX_URL:-}" ]]; then
    echo "setup failed: aiohttp is missing and PIP_EXTRA_INDEX_URL is empty" >&2
    echo "This is a guard setup failure, not an example failure." >&2
    exit 1
  fi
  if ! python3 -m pip install --extra-index-url "$PIP_EXTRA_INDEX_URL" aiohttp; then
    echo "setup failed: could not install aiohttp" >&2
    echo "This is a guard setup failure, not an example failure." >&2
    exit 1
  fi
  echo "setup: installed aiohttp"
}

# CANN 9.1 public include/hccl is not enough for USE_ASCEND. Internal
# HCCL headers live under aicpu_kfc/pub_inc and pkg_inc. Upstream CMake
# still points at experiment/hccl, which this toolkit does not ship.
export_hccl_cpath() {
  local home cpu pub_inc pkg_inc inc test_src d
  home=$(readlink -f "${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}")
  cpu=$(uname -m)
  pub_inc="$home/${cpu}-linux/asc/impl/adv_api/detail/hccl/cc/src/aicpu_kfc/pub_inc"
  pkg_inc="$home/${cpu}-linux/pkg_inc"
  inc="$home/${cpu}-linux/include"
  test_src="$home/tools/hccl_test/common/src"
  if [[ ! -f "$pub_inc/adapter_hccp_common.h" ]]; then
    echo "setup failed: adapter_hccp_common.h not found at $pub_inc" >&2
    echo "USE_ASCEND needs CANN internal HCCL headers, not only include/hccl." >&2
    echo "This is a guard setup failure, not an example failure." >&2
    exit 1
  fi
  CPATH="$pub_inc:$pub_inc/new:$pkg_inc:$inc:$test_src"
  for d in "$pkg_inc"/* "$inc"/experiment/* "$inc"/hccl; do
    if [[ -d "$d" ]]; then
      CPATH="$CPATH:$d"
    fi
  done
  export CPATH
  echo "setup: HCCL CPATH uses $pub_inc"
}

configure_and_build() {
  source_cann
  local cmake_args=(
    -S "$TARGET_ROOT"
    -B "$TARGET_ROOT/$BUILD_DIR"
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_EXAMPLES=ON
    -DBUILD_UNIT_TESTS=OFF
    -DWITH_STORE=OFF
    -DWITH_STORE_RUST=OFF
    -DWITH_EP=OFF
    -DWITH_P2P_STORE=OFF
    -DUSE_ETCD=OFF
    -DUSE_REDIS=OFF
  )
  if [[ "$PROFILE" == "ascend-hccl" ]]; then
    export_hccl_cpath
    # CANN 9.1 SalGetBareTgid takes s32*, upstream still casts to uint32_t*.
    cmake_args+=(-DUSE_ASCEND=ON -DCMAKE_CXX_FLAGS="-fpermissive")
  else
    cmake_args+=(-DUSE_ASCEND_DIRECT=ON)
  fi
  cmake "${cmake_args[@]}"
  cmake --build "$TARGET_ROOT/$BUILD_DIR" \
    --target "$(basename "$EXEC_REL")" \
    -j "$(nproc)"
}

assert_exec() {
  local path="$TARGET_ROOT/$EXEC_REL"
  if [[ ! -x "$path" ]]; then
    echo "expected executable missing: $path" >&2
    find "$TARGET_ROOT/$BUILD_DIR" -name "$(basename "$EXEC_REL")" -print >&2 || true
    exit 1
  fi
  echo "setup: built $path"
}

install_debs
init_pybind11
if [[ "$PROFILE" == "ascend-direct-http" ]]; then
  ensure_aiohttp
fi
configure_and_build
assert_exec

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

# Cached assets (models, Node tarball, cmake-js headers) live on the
# shared persistent runner cache - the pool mounts one volume at
# ~/.cache/huggingface (docs/examples-guard-engine.md). The legacy
# default /root/.cache/cosdt-ci-test/whisper.cpp existed only via a
# container bind-mount the engine does not provide.
CACHE_ROOT="${WHISPER_CI_CACHE_ROOT:-${HF_HOME:-${HOME}/.cache/huggingface}/whisper.cpp}"

TINY_EN_DEST=${CACHE_ROOT}/ggml-tiny.en.bin
TINY_EN_SHA256=921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f
TINY_EN_URLS=(
  https://www.modelscope.cn/models/cjc1887415157/whisper.cpp/resolve/master/ggml-tiny.en.bin
  https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin
)

PARAKEET_F16_DEST=${CACHE_ROOT}/ggml-parakeet-tdt-0.6b-v3-f16.bin
PARAKEET_F16_SHA256=833bffc9513b2cae867ee9e51633cfd11e4d51aaa5597c8ac02159385a2b426f
PARAKEET_F16_URLS=(
  https://www.modelscope.cn/models/ggml-org/parakeet-GGUF/resolve/master/ggml-parakeet-tdt-0.6b-v3-f16.bin
  https://hf-mirror.com/ggml-org/parakeet-GGUF/resolve/main/ggml-parakeet-tdt-0.6b-v3-f16.bin
)

VAD_DEST=${CACHE_ROOT}/ggml-silero-v6.2.0.bin
VAD_SHA256=2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987
VAD_URLS=(
  https://www.modelscope.cn/models/ggml-org/whisper-vad/resolve/master/ggml-silero-v6.2.0.bin
  https://hf-mirror.com/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin
)

NODE_VERSION=v20.18.2
NODE_ARCH=linux-arm64
NODE_NAME="node-${NODE_VERSION}-${NODE_ARCH}"
NODE_DEST="${CACHE_ROOT}/${NODE_NAME}"
NODE_TARBALL="${NODE_DEST}.tar.xz"
NODE_URL="https://mirrors.huaweicloud.com/nodejs/${NODE_VERSION}/${NODE_NAME}.tar.xz"
NODE_TARBALL_SHA256=5c1437aa16e7e6a2e0687a42c4d3f0a8f8a2039cda8880cb3be8cd983aeefb44
NPM_REGISTRY=https://repo.huaweicloud.com/repository/npm/
NVM_NODEJS_ORG_MIRROR=https://mirrors.huaweicloud.com/nodejs
CMAKE_JS_CACHE=${CACHE_ROOT}/cmake-js
CMAKE_JS_RUNTIME="${HOME}/.cmake-js/node-arm64/${NODE_VERSION}"

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    python3 -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"
  fi
}

file_complete() {
  local path="$1"
  local expected_sha="$2"
  [[ -f "$path" ]] || return 1
  [[ "$(file_sha256 "$path")" == "$expected_sha" ]]
}

source_cann() {
  export PATH="/usr/local/sbin:$PATH"
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
}

assert_cann_lib() {
  local root="${1:-$TARGET_ROOT/build}"
  if [[ -z "$(find "$root" -name 'libggml-cann.*' -print -quit)" ]]; then
    echo "build produced no CANN backend library under $root" >&2
    exit 1
  fi
}

download_to_part() {
  local url="$1"
  local part="$2"
  shift 2
  curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 "$@" \
    "$url" -o "$part"
}

# Resume-then-scratch GET into dest, trying each URL in order.
# Usage: fetch_pinned <dest> <sha256> <url> [url...]
# Caller holds the lock if needed.
fetch_pinned() {
  local dest="$1"
  local expected_sha="$2"
  shift 2
  local part="$dest.part"
  if file_complete "$dest" "$expected_sha"; then
    echo "reusing $dest"
    return
  fi
  if [[ -f "$dest" ]]; then
    echo "cached $dest is corrupt or the wrong object; re-downloading" >&2
    rm -f "$dest"
  fi
  local url
  for url in "$@"; do
    echo "fetching $dest from $url"
    if download_to_part "$url" "$part" -C - \
      && file_complete "$part" "$expected_sha"; then
      mv -f "$part" "$dest"
      return
    fi
    echo "resume from $url produced no valid file; downloading from scratch" >&2
    rm -f "$part"
    if download_to_part "$url" "$part" \
      && file_complete "$part" "$expected_sha"; then
      mv -f "$part" "$dest"
      return
    fi
    echo "failed to fetch a valid file from $url" >&2
    rm -f "$part"
  done
  echo "all download sources failed for $dest" >&2
  exit 1
}

fetch_ggml() {
  local dest="$1"
  local expected_sha="$2"
  local env_name="$3"
  shift 3
  mkdir -p "$(dirname "$dest")"
  local lock="$dest.lock"
  (
    flock 9
    fetch_pinned "$dest" "$expected_sha" "$@"
    echo "${env_name}=${dest}" >> "$GITHUB_ENV"
    echo "model ready: $dest"
  ) 9>"$lock"
}

fetch_tiny_en() {
  fetch_ggml "$TINY_EN_DEST" "$TINY_EN_SHA256" WHISPER_CI_MODEL \
    "${TINY_EN_URLS[@]}"
}

fetch_parakeet_f16() {
  fetch_ggml "$PARAKEET_F16_DEST" "$PARAKEET_F16_SHA256" PARAKEET_CI_MODEL \
    "${PARAKEET_F16_URLS[@]}"
}

fetch_vad() {
  fetch_ggml "$VAD_DEST" "$VAD_SHA256" WHISPER_CI_VAD_MODEL \
    "${VAD_URLS[@]}"
}

configure_cann_build() {
  source_cann
  cmake -S "$TARGET_ROOT" -B "$TARGET_ROOT/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CANN=on
}

build_exec_target() {
  cmake --build "$TARGET_ROOT/build" --target "$(basename "$EXEC")" -j "$(nproc)"
}

node_complete() {
  [[ -x "$NODE_DEST/bin/node" ]] || return 1
  [[ "$("$NODE_DEST/bin/node" --version)" == "$NODE_VERSION" ]]
}

fetch_node_tarball() {
  fetch_pinned "$NODE_TARBALL" "$NODE_TARBALL_SHA256" "$NODE_URL"
}

restore_cmake_js_runtime() {
  local dest="$CMAKE_JS_RUNTIME"
  local cache="$CMAKE_JS_CACHE/node-arm64/${NODE_VERSION}"
  if [[ -s "$dest/include/node/node.h" ]]; then
    echo "cmake-js headers already at $dest"
    return
  fi
  if [[ -s "$cache/include/node/node.h" ]]; then
    mkdir -p "$(dirname "$dest")"
    rm -rf "$dest"
    cp -a "$cache" "$dest"
    echo "restored cmake-js headers from $cache"
  fi
}

stage_cmake_js_runtime() {
  local dest="$CMAKE_JS_RUNTIME"
  local cache="$CMAKE_JS_CACHE/node-arm64/${NODE_VERSION}"
  if [[ ! -s "$dest/include/node/node.h" ]] \
      && [[ -s "$NODE_DEST/include/node/node.h" ]]; then
    mkdir -p "$dest"
    rm -rf "$dest/include"
    cp -a "$NODE_DEST/include" "$dest/"
    echo "staged cmake-js headers from $NODE_DEST"
  fi
  if [[ -s "$dest/include/node/node.h" ]] \
      && [[ ! -s "$cache/include/node/node.h" ]]; then
    mkdir -p "$(dirname "$cache")"
    rm -rf "${cache}.part"
    cp -a "$dest" "${cache}.part"
    mv "${cache}.part" "$cache"
    echo "persisted cmake-js headers to $cache"
  fi
}

extract_node() {
  local unpack
  unpack="$(mktemp -d "${TMPDIR:-/tmp}/whisper-node.XXXXXX")"
  if ! tar -xJf "$NODE_TARBALL" -C "$unpack"; then
    echo "failed to extract $NODE_TARBALL" >&2
    rm -rf "$unpack" "$NODE_TARBALL"
    exit 1
  fi
  if [[ ! -x "$unpack/$NODE_NAME/bin/node" ]]; then
    echo "Node binary missing after extract: $unpack/$NODE_NAME/bin/node" >&2
    rm -rf "$unpack" "$NODE_TARBALL"
    exit 1
  fi
  if [[ "$("$unpack/$NODE_NAME/bin/node" --version)" != "$NODE_VERSION" ]]; then
    echo "extracted Node version is not $NODE_VERSION" >&2
    rm -rf "$unpack" "$NODE_TARBALL"
    exit 1
  fi
  rm -rf "$NODE_DEST"
  mv "$unpack/$NODE_NAME" "$NODE_DEST"
  rmdir "$unpack"
}

ensure_node() {
  if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    echo "using existing node $(command -v node)"
    return
  fi
  mkdir -p "$(dirname "$NODE_DEST")"
  (
    flock 9
    if node_complete; then
      echo "reusing $NODE_DEST/bin/node"
    else
      fetch_node_tarball
      extract_node
    fi
  ) 9>"${NODE_DEST}.lock"
  if ! node_complete; then
    echo "Node binary missing after extract: $NODE_DEST/bin/node" >&2
    exit 1
  fi
  export PATH="$NODE_DEST/bin:$PATH"
  echo "WHISPER_CI_NODE_BIN=$NODE_DEST/bin" >> "$GITHUB_ENV"
  if [[ -n "${GITHUB_PATH:-}" ]]; then
    echo "$NODE_DEST/bin" >> "$GITHUB_PATH"
  fi
  echo "using cached node $NODE_DEST/bin/node"
}

patch_addon_prints() {
  local index_js="$TARGET_ROOT/examples/addon.node/index.js"
  if [[ ! -f "$index_js" ]]; then
    echo "addon.node index.js missing: $index_js" >&2
    exit 1
  fi
  python3 - "$index_js" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = "  no_prints: true,"
new = "  no_prints: false,"
if old not in text:
    raise SystemExit(
        "working-copy patch failed: index.js has no 'no_prints: true,'")
path.write_text(text.replace(old, new, 1))
print(f"patched {path}: no_prints false so CANN logs stay visible")
PY
}

setup_cann() {
  configure_cann_build
  build_exec_target
  assert_cann_lib
  fetch_tiny_en
}

setup_host() {
  configure_cann_build
  build_exec_target
  assert_cann_lib
  case "${EXAMPLE_PATH:-}" in
    examples/quantize)
      fetch_tiny_en
      ;;
    examples/parakeet-quantize)
      fetch_parakeet_f16
      ;;
    *)
      echo "setup_host: unexpected EXAMPLE_PATH=${EXAMPLE_PATH:-}" >&2
      exit 1
      ;;
  esac
}

setup_vad() {
  configure_cann_build
  build_exec_target
  assert_cann_lib
  fetch_vad
}

setup_parakeet() {
  configure_cann_build
  build_exec_target
  assert_cann_lib
  fetch_parakeet_f16
}

setup_cmake-pkg() {
  source_cann
  local pkg_dir="$TARGET_ROOT/examples/test-cmake"
  local build_dir="$pkg_dir/whisper-build-install"
  local prefix="$pkg_dir/install"
  cmake -S "$TARGET_ROOT" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_EXAMPLES=OFF \
    -DWHISPER_BUILD_SERVER=OFF \
    -DGGML_CANN=on \
    -DCMAKE_INSTALL_PREFIX="$prefix"
  cmake --build "$build_dir" -j "$(nproc)"
  cmake --install "$build_dir"
  assert_cann_lib "$prefix"
  cmake -S "$pkg_dir" -B "$pkg_dir/build" \
    -DCMAKE_PREFIX_PATH="$prefix"
  cmake --build "$pkg_dir/build" -j "$(nproc)"
  local bin="$pkg_dir/build/test-cmake"
  if [[ ! -f "$bin" ]]; then
    echo "cmake-pkg binary not found: $bin" >&2
    exit 1
  fi
  local libdir
  libdir=$(find "$prefix" -name 'libwhisper.so*' -printf '%h\n' -quit)
  if [[ -z "$libdir" ]]; then
    echo "installed libwhisper.so missing under $prefix" >&2
    exit 1
  fi
  echo "LD_LIBRARY_PATH=${libdir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" >> "$GITHUB_ENV"
}

setup_node() {
  source_cann
  restore_cmake_js_runtime
  ensure_node
  stage_cmake_js_runtime
  fetch_tiny_en
  patch_addon_prints
  npm install --prefix "$TARGET_ROOT/examples/addon.node" \
    --registry "$NPM_REGISTRY"
  export NVM_NODEJS_ORG_MIRROR
  (
    cd "$TARGET_ROOT"
    npx --prefix "$TARGET_ROOT/examples/addon.node" cmake-js compile \
      -d "$TARGET_ROOT" \
      -T addon.node \
      -B Release \
      --CDGGML_CANN=on
  )
  if [[ ! -f "$TARGET_ROOT/build/Release/addon.node" ]]; then
    echo "cmake-js did not write $TARGET_ROOT/build/Release/addon.node" >&2
    exit 1
  fi
  assert_cann_lib "$TARGET_ROOT/build"
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"
EXEC="${EXEC:?EXEC is required}"

"setup_${PROFILE}"

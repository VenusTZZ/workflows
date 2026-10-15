#!/usr/bin/env bash
# Build opencv + opencv_contrib with WITH_CANN=ON for the runner.
# Idempotent — re-runs are no-ops if the build is already installed
# at /usr/local/opencv-cann.
#
# This is the build half of the setup. The 5 source patches are the
# same ones Quick-start-Ascend.md §"打 5 个源码补丁" applies; see
# that section for the per-patch rationale. We replicate them here so
# the example-guard setup is self-contained.
#
# $1 is the upstream ref (tag/branch/SHA). Defaults to 5.0.0.
#
# Environment: expects CANN env to be sourced (the caller
# setup_example.sh does this) so cmake's WITH_CANN can find the
# toolkit.

set -euo pipefail

UPSTREAM_REF="${1:-5.0.0}"
WORK="${WORK:-/home/coder/work}"
INSTALL_PREFIX=/usr/local/opencv-cann

mkdir -p "$WORK"
cd "$WORK"

# 1) Clone (idempotent) with retry. CI runners hit transient GitHub
#    HTTP 500s on the opencv_contrib clone (observed 2026-09-18 on
#    linux-aarch64-a2-1: "RPC failed; HTTP 500 ... error reading
#    section header 'shallow-info'" left opencv_contrib/ with a broken
#    .git and no working tree, which propagated to the patch script
#    as "FileNotFoundError: opencv_contrib/modules/cannops/src/
#    cann_call.cpp" and then to cmake configure as "Configuring
#    incomplete, errors occurred!" — surfaced too far downstream to
#    diagnose). After each clone, verify a sentinel file exists; if
#    not, nuke .git and retry up to 3 times before giving up.
clone_repo() {
    local url="$1" dir="$2"
    local tries=0 max=3
    while (( tries < max )); do
        tries=$((tries + 1))
        if [[ ! -d "$dir/.git" ]]; then
            echo "build: cloning $url (try $tries/$max)"
            if git clone --depth 1 --branch "$UPSTREAM_REF" "$url" "$dir"; then
                :
            else
                rm -rf "$dir"
            fi
        fi
        # Sentinel files prove the working tree actually arrived; a
        # half-finished --depth 1 clone passes `[[ -d $dir/.git ]]`
        # but is missing everything else.
        local sentinel="$dir/.git/HEAD"
        if [[ -f "$sentinel" ]] && [[ -d "$dir/modules" ]]; then
            echo "build: $dir clone OK (sentinel: $sentinel)"
            return 0
        fi
        echo "build: $dir clone incomplete (sentinel: $sentinel), retrying" >&2
        rm -rf "$dir"
    done
    echo "build: $dir clone FAILED after $max tries" >&2
    return 1
}

clone_repo https://github.com/opencv/opencv.git          opencv
clone_repo https://github.com/opencv/opencv_contrib.git  opencv_contrib

# 2) Apply the CANN source patches (each idempotent — checks before
# patching). Moved to a standalone script so the patch set is reviewable
# as a unit; it runs from $WORK and expects ./opencv + ./opencv_contrib.
python3 "$(dirname "$0")/patch_opencv_cann.py"

# 3) Symlink CANN libs into the locations OpenCVFindCANN.cmake probes
ln -sfn /usr/local/Ascend/cann-9.1.0/aarch64-linux   /usr/local/Ascend/cann-9.1.0/acllib
ln -sfn /usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64 /usr/local/Ascend/cann-9.1.0/lib64
ln -sfn /usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64 /usr/local/Ascend/cann-9.1.0/compiler/lib64

# 4) cmake configure + build + install
cd "$WORK/opencv"
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
      -DWITH_CANN=ON \
      -DBUILD_opencv_world=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_TESTS=ON \
      -DOPENCV_BUILD_TEST_MODULES_LIST=cannops \
      -DINSTALL_TESTS=ON \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_LIST=core,imgproc,imgcodecs,videoio,dnn,python3,cannops,ts \
      -DSOC_VERSION=ascend910b1 \
      -DBUILD_opencv_python3=ON \
      -DBUILD_opencv_python_bindings_generator=ON \
      -DPYTHON_INCLUDE_DIR=/usr/local/python3.12.13/include/python3.12 \
      -DPYTHON_LIBRARY=/usr/local/python3.12.13/lib/libpython3.12.so \
      -DOPENCV_ENABLE_NONFREE=OFF \
      -DOPENCV_DOWNLOAD_MIRROR_ID=gitcode \
      -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      .. > cmake.log 2>&1
CMAKE_RC=$?
if [[ $CMAKE_RC -ne 0 ]]; then
    echo "cmake configure FAILED rc=$CMAKE_RC (tail cmake.log)" >&2
    tail -50 cmake.log >&2
    exit $CMAKE_RC
fi

cmake --build . --target install --parallel 2 > build.log 2>&1
BUILD_RC=$?
if [[ $BUILD_RC -ne 0 ]]; then
    echo "build FAILED rc=$BUILD_RC (tail build.log)" >&2
    tail -100 build.log >&2
    exit $BUILD_RC
fi

echo "build DONE: $($INSTALL_PREFIX/bin/opencv_version)"

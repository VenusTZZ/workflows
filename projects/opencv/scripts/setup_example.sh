#!/usr/bin/env bash
# Prepare the CI environment for one opencv example.
# $1 is the manifest profile. Unknown profiles fail before any install.
#
# Profile "opencv" (the only one we ship):
#   1. Source CANN env (toolkit + nnal/atb; cannops' custom kernel
#      launch path links against ATB, so the opencv_test_cannops gtest
#      binary needs both).
#   2. Run a source build of opencv + opencv_contrib with WITH_CANN=ON
#      if /usr/local/opencv-cann/bin/opencv_version is missing. The
#      build is heavy (~50-70 min at -j2) and is fully idempotent; a
#      pre-built install is left in place across example runs because
#      the engine's run-example job reuses the same self-hosted
#      runner (linux-aarch64-a2-1) and the install path
#      /usr/local/opencv-cann lives in the same image overlay.
#      The build is the same 5-patch sequence the Quick-start-Ascend
#      doc runs (sources: modules/dnn/src/op_cann.{hpp,cpp} + 23 layer
#      TUs, opencv_contrib/modules/cannops/{src,include}/*, ACL D2H
#      clamp); see docs/Quick-start-Ascend.md §"打 5 个源码补丁" for
#      the rationale of each patch (CANN 9.1.0 + aarch64
#      incompatibilities + driver 25.5.x host-buffer alignment).
#   3. Prepend the source-built cv2 site-packages to PYTHONPATH so
#      example scripts import cv2 with DNN_BACKEND_CANN compiled in.
#   4. Write the resolved fixtures (baboon.jpg copied from the
#      upstream checkout into $TARGET_ROOT/fixtures/ for examples that
#      want a stable input path; the 13.9MB mobilenetv2-12.onnx comes
#      from projects/opencv/fixtures/ via FIXTURE_DIR).

set -uo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <profile>" >&2
    exit 2
fi

PROFILE="$1"

case "$PROFILE" in
    opencv) ;;
    *)
        echo "unknown profile: $PROFILE (only 'opencv' is wired)" >&2
        exit 2
        ;;
esac

# 1) CANN env — sourced WITHOUT set -u (nnal/atb set_env.sh references
# $ZSH_VERSION which is unset under bash + set -u -> unbound variable
# exit; hdc env.sh sources atb so we have to be careful about order).
unset ZSH_VERSION
set +u
source /home/coder/.hdc/env.sh 2>/dev/null || true
set -u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# nnal/atb is required by opencv_test_cannops (ATB libs in LD_LIBRARY_PATH)
[[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]] && source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/sbin:$PATH

# 2) Build opencv-cann if not already installed.
OPENCV_INSTALL=/usr/local/opencv-cann
UPSTREAM_REF="${UPSTREAM_REF:-5.0.0}"  # set by the engine's monitor job

if [[ -x "$OPENCV_INSTALL/bin/opencv_version" ]]; then
    echo "setup: reusing pre-built opencv-cann ($($OPENCV_INSTALL/bin/opencv_version))"
else
    echo "setup: building opencv-cann from source (UPSTREAM_REF=$UPSTREAM_REF, ~50-70 min at -j2)"
    bash "$(dirname "$0")/build_opencv_cann.sh" "$UPSTREAM_REF"
fi

# 3) PYTHONPATH for the source-built cv2. PREPEND, not setdefault:
# the image's set_env.sh exports PYTHONPATH including CANN Python
# bits (TBE/ACL); setdefault would silently keep the image value and
# Python would import the pip wheel.
PP="$OPENCV_INSTALL/lib/python3.12/site-packages"
if [[ ":${PYTHONPATH:-}:" != *":$PP:"* ]]; then
    export PYTHONPATH="$PP:${PYTHONPATH:-}"
fi
echo "setup: PYTHONPATH -> $PYTHONPATH"

# 4) Materialise fixtures under $TARGET_ROOT/fixtures/ so example
# scripts can reference them via a path that survives checkout
# re-shuffles.
FIXTURE_DIR="${FIXTURE_DIR:?FIXTURE_DIR is required}"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
mkdir -p "$TARGET_ROOT/fixtures"
# baboon.jpg comes from the upstream checkout, not the workflows
# fixtures (the workflows copy of baboon is in docs/images/, but
# example scripts default to upstream's samples/data/ which is the
# upstream-canonical location).
if [[ ! -f "$TARGET_ROOT/fixtures/baboon.jpg" ]] \
   && [[ -f "$TARGET_ROOT/samples/data/baboon.jpg" ]]; then
    cp "$TARGET_ROOT/samples/data/baboon.jpg" "$TARGET_ROOT/fixtures/baboon.jpg"
fi
# mobilenetv2-12.onnx: 13.9MB, shipped in projects/opencv/fixtures
# to avoid depending on github.com raw from CI.
if [[ ! -f "$TARGET_ROOT/fixtures/mobilenetv2-12.onnx" ]] \
   && [[ -f "$FIXTURE_DIR/mobilenetv2-12.onnx" ]]; then
    cp "$FIXTURE_DIR/mobilenetv2-12.onnx" "$TARGET_ROOT/fixtures/mobilenetv2-12.onnx"
fi
ls -la "$TARGET_ROOT/fixtures"

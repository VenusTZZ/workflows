#!/usr/bin/env bash
# Prepare the CI environment for one opencv example.
# $1 is the manifest profile. Unknown profiles fail before any install.
#
# Profile "opencv" (the only one we ship):
#   1. Source CANN env (toolkit + nnal/atb; the ATB libs are kept in
#      LD_LIBRARY_PATH for forward-compat with CANN-backend legs — the
#      opencv_test_cannops gtest example that originally required them
#      is retired, but the sourced env is harmless and the CANN build
#      still needs the toolkit).
#   2. Run a source build of opencv + opencv_contrib with WITH_CANN=ON
#      if /usr/local/opencv-cann/bin/opencv_version is missing. The
#      build is heavy (~23 min at -j2) and is fully idempotent; a
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
#   5. Run the 5 doc-snippet algorithm sanity checks on the source
#      cv2 (imread / cvtColor / resize / draw / video, plus a
#      DNN_BACKEND_CANN != 0 probe). Fail-fast: a broken source
#      build surfaces here as a shape / dtype mismatch instead of
#      waiting for the NPU leg to mis-pick CPU silently.

set -euo pipefail

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
# nnal/atb: kept for CANN-backend legs (see header note 1). Wrap in
# set +u because atb's set_env.sh references $ZSH_VERSION which is
# unset under bash + set -u -> unbound variable exit (empirically
# reproduced on a2-1 runner with CANN 9.1.0; matches torchtune's
# run_example.sh workaround). Toggle set -u in place because env vars
# set inside a subshell don't propagate to the parent.
unset ZSH_VERSION
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    set +u
    source /usr/local/Ascend/nnal/atb/set_env.sh
    set -u
fi
export PATH=/usr/local/sbin:$PATH

# 2) Build opencv-cann if not already installed.
OPENCV_INSTALL=/usr/local/opencv-cann
UPSTREAM_REF="${UPSTREAM_REF:-5.0.0}"  # set by the engine's monitor job

if [[ -x "$OPENCV_INSTALL/bin/opencv_version" ]]; then
    echo "setup: reusing pre-built opencv-cann ($($OPENCV_INSTALL/bin/opencv_version))"
else
    echo "setup: building opencv-cann from source (UPSTREAM_REF=$UPSTREAM_REF, ~23 min at -j2)"
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

# 5) Algorithm sanity check on the source-built cv2. The Quick-start
# §9a-9e examples (imread / cvtColor / resize / draw / video) used to
# be 5 separate supported entries; they're not NPU-specific so they
# don't belong in the manifest, but we still want to confirm the
# source cv2 actually runs the canonical doc snippets — a broken
# source build (wrong opencv_extra, missing libjpeg-turbo, etc.)
# surfaces as a shape / dtype mismatch here, NOT at the DNN
# inference leg. Fail-fast keeps the failure cause close to the
# cause-of-the-build.
#
# One python heredoc instead of 5 subshells: each leg needs cv2
# import + fixture path resolution, sharing them costs ~10ms but
# keeps the failure output contiguous (no interleaved stderr from
# 5 spawns). Exit code propagates via set -e; if python3 raises
# AssertionError the traceback goes to stderr and the script exits
# non-zero automatically.
python3 - "$TARGET_ROOT/fixtures/baboon.jpg" <<'PY'
import sys, os, cv2, numpy as np
img_path = sys.argv[1]
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
assert img is not None and img.shape == (512, 512, 3) and img.dtype == np.uint8, \
    f"imread: got shape={None if img is None else img.shape} dtype={img.dtype}"
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
m = float(gray.mean())
assert 128.0 < m < 131.0, f"cvtColor: baboon grayscale mean out of band ({m:.2f})"
small = cv2.resize(img, (200, 50), interpolation=cv2.INTER_AREA)
assert small.shape == (50, 200, 3), f"resize: got shape={small.shape}"
draw = img.copy()
cv2.rectangle(draw, (10, 10), (100, 100), (0, 255, 0), 2)
cv2.putText(draw, "Hello OpenCV", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
draw_out = "/tmp/opencv_sanity_draw.png"
assert cv2.imwrite(draw_out, draw), "imwrite draw output failed"
assert os.path.getsize(draw_out) > 0, "draw output file is empty"
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
vid = cv2.VideoWriter("/tmp/opencv_sanity_video.avi", fourcc, 10.0, (320, 320))
assert vid.isOpened(), "VideoWriter MJPG 320x320 @10fps refused to open"
for _ in range(3):
    vid.write(np.zeros((320, 320, 3), dtype=np.uint8))
vid.release()
assert os.path.getsize("/tmp/opencv_sanity_video.avi") > 0, "VideoWriter produced empty file"
# DNN backend probe (same semantics as the Quick-start version probe:
# a source-vs-wheel cv2 swap shows up here as DNN_BACKEND_CANN == 0
# before any leg pays the GE-compile tax).
assert cv2.dnn.DNN_BACKEND_CANN != 0, \
    f"cv2.dnn.DNN_BACKEND_CANN == 0 — source build didn't link CANN backend"
print(f"setup: cv2 sanity OK (imread={img.shape} gray_mean={m:.2f} "
      f"resize={small.shape} draw={os.path.getsize(draw_out)}B "
      f"video={os.path.getsize('/tmp/opencv_sanity_video.avi')}B "
      f"DNN_BACKEND_CANN={cv2.dnn.DNN_BACKEND_CANN})")
PY
echo "setup: PYTHONPATH at sanity time -> $PYTHONPATH"

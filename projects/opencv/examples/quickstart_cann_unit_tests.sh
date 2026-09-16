#!/usr/bin/env bash
# Quickstart CANN unit tests: run opencv_test_cannops (53 expected
# green of 78) on the source-built OpenCV with CANN backend enabled.
#
# Corresponds to the ``opencv-cann-run-tests`` block in
# ``docs/Quick-start-Ascend.md`` (CANN test section).
#
# 78 gtests total; 25 are filtered out as known-flaky / known-broken
# on CANN 9.1.0 / 910B (see comments inline + ``unsupported`` section
# of projects/opencv/examples_manifest.yaml). The filter is the
# negation of the gtest names we expect to pass, so the test set is
# explicit and editable.
#
# Overlay args (all optional):
#   --output PATH   log destination (default: /tmp/cannops_gtest.log)
#   --bin    PATH   opencv_test_cannops binary
#                   (default: /usr/local/opencv-cann/bin/opencv_test_cannops)
#   --keep-broken   if passed, also include the 25 known-broken tests
#                   (useful for upstream triage; do NOT use in CI)
#
# The script forwards any extra "$@" to gtest as-is.

set -uo pipefail

# Defaults match the doc; overlay_args can override.
OUTPUT=/tmp/cannops_gtest.log
BIN=/usr/local/opencv-cann/bin/opencv_test_cannops
KEEP_BROKEN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output) OUTPUT="$2"; shift 2 ;;
        --bin)    BIN="$2"; shift 2 ;;
        --keep-broken) KEEP_BROKEN=1; shift ;;
        *) break ;;  # remaining args -> gtest
    esac
done

# Source CANN env (gtest binary needs LD_LIBRARY_PATH for libascend_hal,
# libge_runner, etc. — none of these are in the default loader path).
# nnal/atb set_env.sh references $ZSH_VERSION which is unset under bash
# + set -u -> unbound variable exit. Same workaround as setup_example.sh
# / run_example.sh: drop set -u around the source.
# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# + nnal/atb (cannops' custom kernel launch path links against ATB).
unset ZSH_VERSION
set +u
# shellcheck disable=SC1091
[[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]] && source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: $BIN not found or not executable" >&2
    exit 1
fi

# Exclusion list = the 25 tests we KNOW fail on CANN 9.1.0 / 910B
# (4 known-bug categories, see manifest unsupported section for the
# breakdown and reasoning).
EXCLUDE_FILTER=(
    # Resize: ResizeArea GE shape infer fails on 910B
    'CORE.RESIZE'
    # Resize: ResizeArea NEW kernel output numerically close but flaky
    'CORE.RESIZE_NEW'
    # Resize: CropResize off by 2 vs CPU ref
    'CORE.CROP_RESIZE'
    # Resize: CropResizeMakeBorder tolerance 1e-10 too tight on 910B4
    'CORE.CROP_RESIZE_MAKE_BORDER'
    # CVT_COLOR XYZ/YCrCb/YUV: 18 tests share the same AscendC kernel
    # that has an out-of-bounds UB read on non-32F input (cvtColor
    # truncates intermediate twice; the kernel is the bug).
    'CVT_COLOR.RGB2XYZ' 'CVT_COLOR.BGR2XYZ'
    'CVT_COLOR.XYZ2BGR' 'CVT_COLOR.XYZ2RGB'
    'CVT_COLOR.XYZ2BGR_DC4' 'CVT_COLOR.XYZ2RGB_DC4'
    'CVT_COLOR.BGR2YCrCb' 'CVT_COLOR.RGB2YCrCb'
    'CVT_COLOR.YCrCb2BGR' 'CVT_COLOR.YCrCb2RGB'
    'CVT_COLOR.YCrCb2BGR_DC4' 'CVT_COLOR.YCrCb2RGB_DC4'
    'CVT_COLOR.BGR2YUV' 'CVT_COLOR.RGB2YUV'
    'CVT_COLOR.YUV2BGR' 'CVT_COLOR.YUV2RGB'
    'CVT_COLOR.YUV2BGR_DC4' 'CVT_COLOR.YUV2RGB_DC4'
    # Threshold family: same AscendC kernel, same UB-OOB
    'ELEMENTWISE_OP.MAT_THRESHOLD'
    'ELEMENTWISE_OP.MAT_THRESHOLD_ASCENDC'
    'ASCENDC_KERNEL.THRESHOLD'
)

# Build the gtest --gtest_filter negation string. Format:
#   --gtest_filter=-A:B:C:...
NEG="--gtest_filter=-$(IFS=:; echo "${EXCLUDE_FILTER[*]}")"
if [[ "$KEEP_BROKEN" -eq 1 ]]; then
    NEG=""  # include everything
fi

"$BIN" --gtest_color=no $NEG "$@" > "$OUTPUT" 2>&1
RC=$?
# Always show the gtest tail (last 25 lines incl. the PASSED/FAILED
# summary) regardless of outcome — saves a follow-up ssh.
tail -n 25 "$OUTPUT"
if [[ $RC -ne 0 ]]; then
    echo '--- per-test failure summary:'
    grep -E '^\[ RUN|unknown file: Failure|C\+\+ exception|op\[[A-Za-z0-9_]+\]|E[0-9]{5}' "$OUTPUT" | head -80
fi
exit $RC

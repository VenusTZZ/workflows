#!/usr/bin/env python3
"""Quickstart version probe: print cv2 version + CANN backend / NPU target enums.

Corresponds to the ``opencv-py-version`` block in
``docs/Quick-start-Ascend.md`` (DNN section header).

The whole point of this probe is to confirm the source-built cv2 (with
CANN backend) is the one Python is actually loading, NOT the pip
opencv-python wheel. The two checks that matter:
    1. cv2 is importable from the source-built site-packages (set by
       PYTHONPATH via the engine's prepare_environment).
    2. DNN_BACKEND_CANN and DNN_TARGET_NPU enum values are non-zero
       (i.e. compiled in; the pip wheel returns 0 for both).

Overlay args: none — this is a pure probe.
"""

from __future__ import annotations

import cv2  # noqa: E402


def main() -> int:
    print(f'cv2 version: {cv2.__version__}')
    print(f'CANN backend available: {cv2.dnn.DNN_BACKEND_CANN}')
    print(f'NPU target available: {cv2.dnn.DNN_TARGET_NPU}')
    # Sanity guard: if both enums are zero, the wheel was loaded instead
    # of the source build. Surface as a hard failure so the engine
    # doesn't pass a false-green.
    if cv2.dnn.DNN_BACKEND_CANN == 0 or cv2.dnn.DNN_TARGET_NPU == 0:
        print('ERROR: DNN BACKEND_CANN / TARGET_NPU not compiled in; '
              'cv2 likely came from the pip wheel, not the source build',
              flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

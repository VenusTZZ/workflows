#!/usr/bin/env python3
"""Quickstart 9a: read a BGR image from disk and re-encode it as PNG.

Corresponds to the ``quickstart-imread-imwrite`` block in
``docs/Quick-start-Ascend.md`` (9a).

The output is a single line of ``shape`` / ``dtype`` plus the encoded
PNG size in bytes; the engine only checks the exit code, but a
deterministic line is useful when humans re-run the script.

Overlay args (parsed via argparse, all optional):
    --image PATH    input image (default: baboon.jpg from opencv samples)
    --output PATH   output PNG (default: /tmp/opencv_quickstart.png)
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--image',
        default='opencv/samples/data/baboon.jpg',
        help='input image path (relative to cwd, which is the opencv checkout)',
    )
    parser.add_argument(
        '--output',
        default='/tmp/opencv_quickstart.png',
        help='output PNG path',
    )
    args = parser.parse_args()

    import cv2  # noqa: E402  (import after arg parse so --help works without cv2)

    img = cv2.imread(args.image)
    if img is None:
        print(f'ERROR: failed to read image: {args.image}', file=sys.stderr)
        return 1
    if not cv2.imwrite(args.output, img):
        print(f'ERROR: failed to write image: {args.output}', file=sys.stderr)
        return 1
    print(f'imwrite ok: True shape: {img.shape} dtype: {img.dtype}')
    print(f'output size: {os.path.getsize(args.output)} bytes')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

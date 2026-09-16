#!/usr/bin/env python3
"""Quickstart 9c: resize image to (W=50, H=200) with INTER_AREA.

Corresponds to the ``quickstart-resize`` block in
``docs/Quick-start-Ascend.md`` (9c).

Note: ``cv2.resize(img, (W, H), ...)`` swaps W/H into (cols, rows); the
output shape is therefore (H, W, C) = (200, 50, 3) by design.

Overlay args (all optional):
    --image PATH   input image (default: baboon.jpg)
    --output PATH  output PNG (default: /tmp/quickstart-resize.png)
    --width  INT   target width  (default: 50)
    --height INT   target height (default: 200)
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='opencv/samples/data/baboon.jpg')
    parser.add_argument('--output', default='/tmp/quickstart-resize.png')
    parser.add_argument('--width', type=int, default=50)
    parser.add_argument('--height', type=int, default=200)
    args = parser.parse_args()

    import cv2  # noqa: E402

    img = cv2.imread(args.image)
    if img is None:
        print(f'ERROR: failed to read: {args.image}', file=sys.stderr)
        return 1
    resized = cv2.resize(img, (args.width, args.height),
                         interpolation=cv2.INTER_AREA)
    print(f'resized shape: {resized.shape}')
    if not cv2.imwrite(args.output, resized):
        print(f'ERROR: failed to write: {args.output}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

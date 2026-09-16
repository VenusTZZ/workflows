#!/usr/bin/env python3
"""Quickstart 9b: BGR -> grayscale conversion, write to PNG.

Corresponds to the ``quickstart-cvtcolor`` block in
``docs/Quick-start-Ascend.md`` (9b).

Output: ``gray shape: (H, W) mean: <f>`` (the ``mean`` is what the doc's
``#test-result`` block fuzzy-matches; the baboon mean hovers around
129.7/255 across encoder variants).

Overlay args (all optional):
    --image PATH   input image (default: baboon.jpg from opencv samples)
    --output PATH  output PNG (default: /tmp/quickstart-gray.png)
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='opencv/samples/data/baboon.jpg')
    parser.add_argument('--output', default='/tmp/quickstart-gray.png')
    args = parser.parse_args()

    import cv2  # noqa: E402

    bgr = cv2.imread(args.image)
    if bgr is None:
        print(f'ERROR: failed to read: {args.image}', file=sys.stderr)
        return 1
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    print(f'gray shape: {gray.shape} mean: {float(gray.mean()):.3f}')
    if not cv2.imwrite(args.output, gray):
        print(f'ERROR: failed to write: {args.output}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

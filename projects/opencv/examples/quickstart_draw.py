#!/usr/bin/env python3
"""Quickstart 9d: draw a green rectangle and white text on the image.

Corresponds to the ``quickstart-draw`` block in
``docs/Quick-start-Ascend.md`` (9d).

Overlay args (all optional):
    --image PATH   input image (default: baboon.jpg)
    --output PATH  output PNG (default: /tmp/quickstart-draw.png)
    --text  STR    text to render (default: "Hello OpenCV")
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='opencv/samples/data/baboon.jpg')
    parser.add_argument('--output', default='/tmp/quickstart-draw.png')
    parser.add_argument('--text', default='Hello OpenCV')
    args = parser.parse_args()

    import cv2  # noqa: E402

    img = cv2.imread(args.image)
    if img is None:
        print(f'ERROR: failed to read: {args.image}', file=sys.stderr)
        return 1
    # (x1, y1) is top-left, (x2, y2) is bottom-right. baboon is 512x512.
    cv2.rectangle(img, (10, 10), (500, 500), (0, 255, 0), thickness=2)
    cv2.putText(img, args.text, (60, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    print(f'rect+text drawn, image mean: {float(img.mean()):.3f}')
    if not cv2.imwrite(args.output, img):
        print(f'ERROR: failed to write: {args.output}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

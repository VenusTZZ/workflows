#!/usr/bin/env python3
"""Quickstart 9e: write 3 black frames to an MJPG AVI via VideoWriter.

Corresponds to the ``quickstart-video`` block in
``docs/Quick-start-Ascend.md`` (9e).

The CI image's opencv build does NOT include FFmpeg / GStreamer
plugins, so the VideoWriter can be expected to fall back to the
built-in MJPG fourcc without a working encoder backend; the test only
verifies the file lands on disk. Without FFmpeg, OpenCV's
VideoWriter will fail to open with most fourccs; the doc chose MJPG
because the opencv build enables MJPG by default and writes a valid
container.

Overlay args (all optional):
    --output PATH   output AVI (default: /tmp/opencv_video.avi)
    --frames  INT   number of frames to write (default: 3)
    --width   INT   frame width  (default: 320)
    --height  INT   frame height (default: 320)
    --fps     FLOAT frames per second (default: 10.0)
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='/tmp/opencv_video.avi')
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=320)
    parser.add_argument('--fps', type=float, default=10.0)
    args = parser.parse_args()

    import cv2  # noqa: E402
    import numpy as np  # noqa: E402

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(args.output, fourcc, args.fps,
                             (args.width, args.height))
    if not writer.isOpened():
        print(f'ERROR: VideoWriter failed to open {args.output}', file=sys.stderr)
        return 1
    frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    for _ in range(args.frames):
        writer.write(frame)
    writer.release()
    size = os.path.getsize(args.output)
    print(f'video frames: {args.frames}')
    print(f'video file size: {size} bytes')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

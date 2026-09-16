#!/usr/bin/env python3
"""Quickstart DNN inference: MobileNetV2 ONNX forward on Ascend NPU via CANN.

Corresponds to the ``opencv-cann-infer`` block in
``docs/Quick-start-Ascend.md`` (DNN section).

The script:
  1. Loads MobileNetV2 (ONNX, 1000-class ImageNet classifier).
  2. Reads baboon.jpg, BGR->RGB, /255, normalises by ImageNet mean/std,
     packs into 1xCxHxW float32 blob.
  3. Calls ``net.forward()`` with the classic DNN engine forced
     (``OPENCV_FORCE_DNN_ENGINE=1``); on the source build, this routes
     through ``switchToCannBackend`` -> GE graph compile -> NPU
     execution.
  4. Computes softmax of the logits, prints top-1 class index + score.
  5. Writes a 1-line result to ``/tmp/dnn_result.txt`` via ``os.write``
     (bypasses Python's buffered stdout because GE may fork subprocesses
     that flush partial C++ log lines into our stdout).
  6. Calls ``os._exit(0)`` to bypass the C-runtime atexit chain — the
     driver 25.5.x + CANN 9.1.0 mismatch corrupts the heap on shutdown
     (the DVPP atexit handler frees a bad block -> SIGABRT 134); the
     inference result itself is correct, this is purely a clean-exit
     workaround.

Overlay args (all optional):
    --model PATH  ONNX model path. If unset and the bundled fixture
                  exists at ../../../fixtures/mobilenetv2-12.onnx
                  (relative to this script), the fixture is copied to
                  /tmp; otherwise we attempt a 5-retry download from
                  onnx/models (the GitHub raw host is unreliable on
                  CN networks).
    --image PATH  input image (default: baboon.jpg from opencv samples)
    --output PATH output PNG with the bounding box drawn
                  (default: /tmp/dnn-result.png)
    --result PATH result txt output (default: /tmp/dnn_result.txt)
    --engine  STR 'classic' (default; forces classic DNN engine to
                        route through CANN backend) or 'auto'
                        (5.0.0 default; routes through onnx_importer2
                        which does NOT support the CANN backend yet).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.request


def _ensure_model(model_path: str, fixture_path: str) -> None:
    """Materialise model_path, preferring the bundled fixture."""
    if os.path.exists(model_path):
        return
    if os.path.exists(fixture_path):
        shutil.copy(fixture_path, model_path)
        return
    # Fallback: download with retries (GitHub raw is flaky on CN networks)
    url = ('https://github.com/onnx/models/raw/main/'
           'validated/vision/classification/mobilenet/'
           'model/mobilenetv2-12.onnx')
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(url, model_path)
            return
        except OSError:
            if os.path.exists(model_path):
                os.unlink(model_path)
            if attempt == 4:
                raise
            time.sleep(10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/tmp/mobilenetv2-12.onnx')
    parser.add_argument('--image', default='opencv/samples/data/baboon.jpg')
    parser.add_argument('--output', default='/tmp/dnn-result.png')
    parser.add_argument('--result', default='/tmp/dnn_result.txt')
    parser.add_argument('--engine', default='classic',
                        choices=['classic', 'auto'])
    args = parser.parse_args()

    # Force the classic DNN engine: 5.0.0's default onnx_importer2
    # silently ignores setPreferableBackend (only logs WARN), so the
    # network would fall back to CPU. Classic engine actually routes
    # through switchToCannBackend -> GE -> NPU.
    if args.engine == 'classic':
        os.environ['OPENCV_FORCE_DNN_ENGINE'] = '1'

    # Resolve fixture path relative to this script:
    #   examples/quickstart_dnn_inference.py
    #   ../../../fixtures/mobilenetv2-12.onnx
    fixture = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', '..', 'fixtures', 'mobilenetv2-12.onnx'))
    _ensure_model(args.model, fixture)

    import cv2  # noqa: E402
    import numpy as np  # noqa: E402

    net = cv2.dnn.readNetFromONNX(args.model)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CANN)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_NPU)

    image = cv2.imread(args.image)
    if image is None:
        print(f'ERROR: failed to read: {args.image}', file=sys.stderr)
        return 1
    x = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0
    x = x[..., ::-1]  # BGR -> RGB
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    blob = np.ascontiguousarray(x.transpose(2, 0, 1))[np.newaxis]

    net.setInput(blob)
    out = net.forward()

    # mobilenetv2-12 outputs logits; softmax manually
    logits = out.reshape(-1)
    e = np.exp(logits - logits.max())
    prob = e / e.sum()
    top_idx = int(np.argmax(prob))
    top_score = float(np.max(prob))

    img = image.copy()
    cv2.rectangle(img, (10, 10), (500, 500), (0, 255, 0), thickness=2)
    label = f'top-1: #{top_idx}  {top_score:.3f}'
    cv2.putText(img, label, (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(args.output, img)

    # Write result via os.write to avoid GE's subprocess stdout flush
    # interleaving with our print lines.
    result = (
        f'model bytes: {os.path.getsize(args.model)}\n'
        f'output shape: {out.shape}\n'
        f'output dtype: {out.dtype}\n'
        f'top class index: {top_idx}\n'
        f'top class score: {top_score}\n'
    )
    fd = os.open(args.result, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.write(fd, result.encode())
    os.close(fd)
    print(result.strip())

    # Bypass C runtime atexit chain (driver 25.5.x + CANN 9.1.0 heap
    # corruption in DVPP destructor). Inference is already correct.
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    raise SystemExit(main())

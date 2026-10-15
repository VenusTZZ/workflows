#!/usr/bin/env python3
"""One-leg NPU DNN CANN inference smoke — 超分辨率（非分类 layer 覆盖）。

覆盖 OpenCV 在 NPU 上的第一条通路里「分类之外」的 layer 子集。super-
resolution-10（SubPixel CNN，onnx/models validated）用 Conv + ReLU +
Reshape + Transpose（PixelShuffle）做 3× 上采样，无 Shape/Gather/Cast、
单输出、静态 H/W（input 1x1x224x224 → output 1x1x672x672），所以能在
CANN 后端整图上 NPU。

为什么选它而不是 fcn-resnet50（分割）：
  - fcn-resnet50 的 H/W 是动态（(0,3,0,0)/(0,21,0,0)）+ 双输出（out/aux），
    GE 编译后 createOutputDataset 的 aclrtMalloc 报 ret=100000；
  - 本模型的 batch 虽也是 0（动态），但 H/W 静态、单输出——与
    mobilenetv2-12（同样 batch=0 但静态 H/W，能跑通）一致。

模型：super-resolution-10.onnx（240KB，入仓 fixtures/）。灰度单通道输入，
224x224，3× 上采样。不 patch 上游：纯 numpy + cv2 + argparse。
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True,
                        help='Path to super-resolution-10.onnx')
    parser.add_argument('--input', required=True,
                        help='Path to input image (e.g. baboon.jpg)')
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f'test_npu_dnn_superres: model not found: {args.model}',
              file=sys.stderr)
        return 2
    if not os.path.isfile(args.input):
        print(f'test_npu_dnn_superres: input not found: {args.input}',
              file=sys.stderr)
        return 2

    import numpy as np
    import cv2

    # 5.0.0 默认 ENGINE_AUTO 会选新引擎（不支持非 CPU 后端）；强制 classic。
    os.environ.setdefault('OPENCV_FORCE_DNN_ENGINE', '1')

    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f'test_npu_dnn_superres: cv2.imread returned None for '
              f'{args.input}', file=sys.stderr)
        return 2

    # 输入契约：单通道灰度 224x224，归一化到 [0,1]。
    gray = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(gray[np.newaxis, np.newaxis, :, :])  # 1x1x224x224

    # NPU 推理
    net = cv2.dnn.readNetFromONNX(args.model, cv2.dnn.ENGINE_CLASSIC)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CANN)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_NPU)
    net.setInput(blob)
    out = net.forward()
    if out is None or out.size == 0:
        print('test_npu_dnn_superres: empty forward output', file=sys.stderr)
        return 3

    print(f'test_npu_dnn_superres: model bytes: {os.path.getsize(args.model)}')
    print(f'test_npu_dnn_superres: output shape: {out.shape} dtype: {out.dtype}')
    o = out[0, 0]
    print(f'test_npu_dnn_superres: range [%.3f, %.3f] mean=%.3f '
          f'std=%.4f' % (float(o.min()), float(o.max()), float(o.mean()),
                        float(o.std())))

    # 非退化性：3× 上采样的输出必须有非平凡方差。
    if float(o.std()) < 1e-4:
        print('test_npu_dnn_superres: FAIL degenerate output '
              '(near-constant)', file=sys.stderr)
        return 4

    # 与 CPU 参考（DNN_BACKEND_OPENCV）对齐。NPU 上的 conv 累加顺序/精度与
    # CPU 不同，实测 max|NPU-CPU| ~0.001（输出范围 ~1.4 的 ~0.08%），属
    # float32 精度噪声而非错误；阈值放宽到 0.01 留足余量。
    ref = cv2.dnn.readNetFromONNX(args.model, cv2.dnn.ENGINE_CLASSIC)
    ref.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    ref.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    ref.setInput(blob)
    ref_out = ref.forward()
    diff = np.abs(out - ref_out).max()
    print(f'test_npu_dnn_superres: max|NPU-CPU| = {float(diff):.6f}')
    if float(diff) > 1e-2:
        print(f'test_npu_dnn_superres: FAIL NPU/CPU mismatch '
              f'(max diff {float(diff):.6f})', file=sys.stderr)
        return 4

    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    sys.exit(main())
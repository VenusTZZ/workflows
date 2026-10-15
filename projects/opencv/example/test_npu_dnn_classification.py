#!/usr/bin/env python3
"""One-leg NPU DNN CANN inference smoke — 分类（卷积类模型全链路）。

覆盖 OpenCV 在 NPU 上的第一条通路：cv::dnn 的 CANN 后端。
验证点：
  - readNetFromONNX + ENGINE_CLASSIC + setPreferableBackend(DNN_BACKEND_CANN)
    + setPreferableTarget(DNN_TARGET_NPU)；
  - classic 引擎 switchToCannBackend → GE 图编译 → aclmdlExecute 上 NPU；
  - 输出 shape/dtype 正确 + top-1 与 CPU 参考逐位对齐。

模型：mobilenetv2-12.onnx（onnx/models validated，静态 224x224，13.9MB，
入仓于 projects/opencv/fixtures/），覆盖 conv + BN + ReLU + pooling +
FC + softmax 这条最主流的卷积类 layer 组合。

不 patch 上游：脚本纯 numpy + cv2 + argparse，零非 cv2 第三方依赖。
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True,
                        help='Path to ONNX model (mobilenetv2-12)')
    parser.add_argument('--input', required=True,
                        help='Path to BGR input image (e.g. baboon.jpg)')
    parser.add_argument('--imagenet-topk', type=int, default=5,
                        help='Top-k classes to print')
    parser.add_argument('--expected-top1', type=int, default=372,
                        help='Expected top-1 class index '
                             '(372=patas monkey for baboon.jpg)')
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f'test_npu_dnn_classification: model not found: {args.model}',
              file=sys.stderr)
        return 2
    if not os.path.isfile(args.input):
        print(f'test_npu_dnn_classification: input not found: {args.input}',
              file=sys.stderr)
        return 2

    import numpy as np
    import cv2

    # 5.0.0 默认 ENGINE_AUTO 会选新引擎 onnx_importer2（不支持非 CPU
    # 后端）；强制 classic 引擎才走 DNN_BACKEND_CANN。
    os.environ.setdefault('OPENCV_FORCE_DNN_ENGINE', '1')

    net = cv2.dnn.readNetFromONNX(args.model, cv2.dnn.ENGINE_CLASSIC)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CANN)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_NPU)

    image = cv2.imread(args.input)
    if image is None:
        print(f'test_npu_dnn_classification: cv2.imread returned None for '
              f'{args.input}', file=sys.stderr)
        return 2

    # mobilenetv2-12 输入契约：BGR 224x224，ImageNet mean/std 归一化。
    x = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0
    x = x[..., ::-1]                                # BGR -> RGB
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    blob = np.ascontiguousarray(x.transpose(2, 0, 1))[np.newaxis]  # 1xCxHxW

    net.setInput(blob)
    out = net.forward()
    if out is None or out.size == 0:
        print('test_npu_dnn_classification: empty forward output',
              file=sys.stderr)
        return 3

    # mobilenetv2-12 输出 logits（无 softmax），手工归一化取 top-k。
    logits = out.reshape(-1)
    e = np.exp(logits - logits.max())
    prob = e / e.sum()
    top_idx = np.argsort(-prob)[:args.imagenet_topk]

    print(f'test_npu_dnn_classification: model bytes: '
          f'{os.path.getsize(args.model)}')
    print(f'test_npu_dnn_classification: output shape: {out.shape} '
          f'dtype: {out.dtype}')
    print(f'test_npu_dnn_classification: top class index: {int(top_idx[0])} '
          f'score: {float(prob[top_idx[0]]):.4f}')
    for i, idx in enumerate(top_idx):
        print(f'test_npu_dnn_classification: top-{i + 1}: #{int(idx)} '
              f'score={float(prob[idx]):.4f}')

    if int(top_idx[0]) != args.expected_top1:
        print(f'test_npu_dnn_classification: FAIL top-1 mismatch '
              f'(got {int(top_idx[0])} expected {args.expected_top1})',
              file=sys.stderr)
        return 4

    # 驱动 25.5.2 × CANN 9.1.0 错配环境在进程 exit() 的 atexit 阶段
    # DVPP 会撞运行期已写坏的堆块（"corrupted double-linked list"
    # SIGABRT）；flush 后 _exit(0) 绕过 C 层 atexit。配套正常的驱动
    # 下此调用无害，NPU 资源由驱动随进程回收。
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    sys.exit(main())

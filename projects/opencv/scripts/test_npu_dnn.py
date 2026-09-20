#!/usr/bin/env python3
"""One-leg NPU DNN CANN inference smoke test — Quick-start-Ascend §"跑
DNN 推理"的核心子集。

为什么单独存在：
  projects/opencv/examples_manifest.yaml 的 4 个 supported
  (gpt2/gemma3/qwen/vlm_inference) 都依赖用户自行导出的多 GB ONNX
  + tokenizer config.json（opencv 5.0.0 不自带，opencv_extra/testdata/
  dnn/llm/ 也不含 .onnx）。这是 manifest 自己写的"预期 bring-up 状
  态"。这个脚本走 Quick-start 用过的 mobilenetv2-12.onnx 路径（同
  字节入仓在 projects/opencv/fixtures/，13.9 MB），能在 CI 真实验
  证 NPU 端到端：DNN_BACKEND_CANN + DNN_TARGET_NPU + classic engine
  + GE compile + NPU forward 出 top-1=372 patas monkey。

不要打 patch：脚本纯 numpy + cv2 + argparse，零非 cv2 第三方依赖。
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
                        help='Path to BGR input image')
    parser.add_argument('--imagenet-topk', type=int, default=5,
                        help='Top-k classes to print')
    parser.add_argument('--expected-top1', type=int, default=372,
                        help='Expected top-1 class index (372=patas monkey for baboon.jpg)')
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f'test_npu_dnn: model not found: {args.model}', file=sys.stderr)
        return 2
    if not os.path.isfile(args.input):
        print(f'test_npu_dnn: input not found: {args.input}', file=sys.stderr)
        return 2

    import numpy as np
    import cv2

    # Quick-start §"为什么加 OPENCV_FORCE_DNN_ENGINE=1"：5.0.0 默认
    # ENGINE_AUTO 会选新引擎 onnx_importer2（不支持非 CPU 后端）。
    # 镜像里没设过这个变量（run_example.sh 也没 export），在这里强
    # 制一次以确保 classic 引擎走 DNN_BACKEND_CANN。
    os.environ.setdefault('OPENCV_FORCE_DNN_ENGINE', '1')

    net = cv2.dnn.readNetFromONNX(args.model, cv2.dnn.ENGINE_CLASSIC)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CANN)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_NPU)

    image = cv2.imread(args.input)
    if image is None:
        print(f'test_npu_dnn: cv2.imread returned None for {args.input}',
              file=sys.stderr)
        return 2

    # mobilenetv2-12 input contract: BGR 224x224, ImageNet mean/std.
    x = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0
    x = x[..., ::-1]                                # BGR -> RGB
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    blob = np.ascontiguousarray(x.transpose(2, 0, 1))[np.newaxis]  # 1xCxHxW

    net.setInput(blob)
    out = net.forward()
    if out is None or out.size == 0:
        print('test_npu_dnn: empty forward output', file=sys.stderr)
        return 3

    # mobilenetv2-12 输出是 logits（无 softmax），手工归一化取 top-k。
    logits = out.reshape(-1)
    e = np.exp(logits - logits.max())
    prob = e / e.sum()
    top_idx = np.argsort(-prob)[:args.imagenet_topk]

    print(f'test_npu_dnn: model bytes: {os.path.getsize(args.model)}')
    print(f'test_npu_dnn: output shape: {out.shape} dtype: {out.dtype}')
    print(f'test_npu_dnn: top class index: {int(top_idx[0])} '
          f'score: {float(prob[top_idx[0]]):.4f}')
    for i, idx in enumerate(top_idx):
        print(f'test_npu_dnn: top-{i + 1}: #{int(idx)} score={float(prob[idx]):.4f}')

    if int(top_idx[0]) != args.expected_top1:
        print(f'test_npu_dnn: FAIL top-1 mismatch (got {int(top_idx[0])} '
              f'expected {args.expected_top1})', file=sys.stderr)
        return 4

    # Quick-start §"驱动 25.5.2 × CANN 9.1.0 错配环境堆损坏"：在
    # DVPP atexit 撞坏堆块之前用 _exit(0) 绕过。CI runner 驱动配套
    # 正常时此调用无害，NPU 资源由驱动随进程回收。
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    sys.exit(main())
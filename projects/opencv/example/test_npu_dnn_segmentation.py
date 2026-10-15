#!/usr/bin/env python3
"""One-leg NPU DNN CANN inference smoke — 语义分割（非分类 layer 覆盖）。

覆盖 OpenCV 在 NPU 上的第一条通路里「分类之外」的 layer 子集：
FCN-ResNet50 是卷积 + 反卷积（deconv）+ 逐元素相加（eltwise skip
connection）的语义分割结构，用来验证 CANN 后端对 deconv / eltwise /
resize / concat 等非分类算子的覆盖——分类 example（mobilenetv2）只
覆盖到 conv + BN + pooling + FC + softmax，覆盖不到这些。

模型：fcn-resnet50-12.onnx（onnx/models validated，PASCAL VOC 21 类）。

实测结论（910B4 + CANN 9.1.0）：本模型在 CANN 后端 forward 失败——
GE 图编译成功（OM ~71MB），但 createOutputDataset 的 aclrtMalloc 返回
ret=100000。根因是**动态 H/W + 双输出**，非脚本问题：
  1) 输入/输出 H/W 是动态（input=(0,3,0,0)，output=(0,21,0,0)），GE 编译
     后输出 size 无法解析，CANN 后端 bindInput/outputWrappers 逐维 CV_CheckEQ
     只认静态 shape；
  2) 双输出（out + aux），输出 dataset 分配出错。

对比：mobilenetv2-12 与 super-resolution-10 也带 batch=0（动态）且都含
Shape/Gather 算子，但它们 H/W 静态 + 单输出，实测能上 NPU——所以 FCN 的
Shape/Gather/Cast 不是根因，动态 H/W + 双输出才是。本腿改用 super-resolution-10
（见 example/test_npu_dnn_superres.py），本脚本保留记录结论。

不 patch 上游：脚本纯 numpy + cv2 + argparse，零非 cv2 第三方依赖。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request

MODEL_URL = ('https://github.com/onnx/models/raw/'
             '491ce05590abb7551d7fae43c067c060eeb575a6/'
             'validated/vision/object_detection_segmentation/fcn/model/'
             'fcn-resnet50-12.onnx')


def ensure_model(path: str) -> bool:
    """Return True if the model file is present (downloaded if missing)."""
    if os.path.isfile(path):
        return True
    print(f'test_npu_dnn_segmentation: model missing, downloading from '
          f'onnx/models ...')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(MODEL_URL, path)
            break
        except OSError:
            if os.path.exists(path):
                os.unlink(path)
            if attempt == 4:
                raise
            time.sleep(10)
    return os.path.isfile(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True,
                        help='Path to fcn-resnet50-12.onnx '
                             '(downloaded if missing)')
    parser.add_argument('--input', required=True,
                        help='Path to BGR input image (e.g. baboon.jpg)')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f'test_npu_dnn_segmentation: input not found: {args.input}',
              file=sys.stderr)
        return 2
    try:
        if not ensure_model(args.model):
            print(f'test_npu_dnn_segmentation: model download failed: '
                  f'{args.model}', file=sys.stderr)
            return 2
    except OSError as exc:
        print(f'test_npu_dnn_segmentation: model download failed: {exc}',
              file=sys.stderr)
        return 2

    import numpy as np
    import cv2

    os.environ.setdefault('OPENCV_FORCE_DNN_ENGINE', '1')

    net = cv2.dnn.readNetFromONNX(args.model, cv2.dnn.ENGINE_CLASSIC)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CANN)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_NPU)

    image = cv2.imread(args.input)
    if image is None:
        print(f'test_npu_dnn_segmentation: cv2.imread returned None for '
              f'{args.input}', file=sys.stderr)
        return 2

    # FCN-ResNet50 输入契约：BGR 224x224，ImageNet mean/std 归一化。
    x = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0
    x = x[..., ::-1]                                # BGR -> RGB
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    blob = np.ascontiguousarray(x.transpose(2, 0, 1))[np.newaxis]  # 1xCxHxW

    net.setInput(blob)
    out = net.forward()
    if out is None or out.size == 0:
        print('test_npu_dnn_segmentation: empty forward output',
              file=sys.stderr)
        return 3

    # 输出 1x21x224x224，逐像素 argmax 得 21 类分割图。
    print(f'test_npu_dnn_segmentation: model bytes: '
          f'{os.path.getsize(args.model)}')
    print(f'test_npu_dnn_segmentation: output shape: {out.shape} '
          f'dtype: {out.dtype}')

    seg = np.argmax(out[0], axis=0)                 # 224x224 class map
    hist = np.bincount(seg.reshape(-1), minlength=out.shape[1])
    top = int(np.argmax(hist))
    frac = float(hist[top]) / seg.size
    print(f'test_npu_dnn_segmentation: dominant class {top} '
          f'coverage {frac:.3f}')

    # 非平凡性检查：分割图不能全是一个类（否则是退化输出）。
    if frac > 0.999:
        print(f'test_npu_dnn_segmentation: FAIL degenerate output '
              f'(single class {top} covers {frac:.3f})', file=sys.stderr)
        return 4

    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    sys.exit(main())

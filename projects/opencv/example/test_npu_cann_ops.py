#!/usr/bin/env python3
"""One-leg NPU cannops smoke — 经典 CV 算子的 AscendC/NPU 加速。

覆盖 OpenCV 在 NPU 上的第二条通路：contrib 的 cannops 模块（cv2.cann.*），
把经典 CV 算子丢到 Ascend NPU 上跑，并与 CPU 参考（cv2 同名函数）逐元素
对齐。实测在 910B4 + CANN 9.1.0 上通过的算子族：
  - 颜色转换 cvtColor（BGR2GRAY / BGR2RGB / GRAY2BGR）
  - 几何变换 flip / rotate / transpose
  - 通道操作 merge / split（往返 = 恒等）
  - 像素算术 add / subtract（array+array）
  - 位运算 bitwise_and / bitwise_not

刻意跳过的已知 broken 路径（910B4 + CANN 9.1.0 实测）：
  - resize / crop / cropResize / copyMakeBorder 走 DVPP VPC 通道，本环境
    hi_mpi_vpc_sys_create_chn 返回失败（"failed to create DVPP vpc channel"）；
  - threshold 走 AscendC kernel（910B 上 AI Core 越界，见 Quick-start）；
  - cvtColor 的 XYZ / YCrCb / YUV 系（非 32F 输入退化）。

不 patch 上游：脚本纯 numpy + cv2 + argparse，零非 cv2 第三方依赖。
"""
from __future__ import annotations

import argparse
import os
import sys


def _check(name: str, got, expected, atol: float = 1.0) -> bool:
    import numpy as np
    ok = bool(np.allclose(got, expected, atol=atol))
    status = 'OK' if ok else 'MISMATCH'
    print(f'test_npu_cann_ops: {name}: {status} '
          f'shape={getattr(got, "shape", None)}')
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True,
                        help='Path to BGR input image (e.g. baboon.jpg)')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f'test_npu_cann_ops: input not found: {args.input}',
              file=sys.stderr)
        return 2

    import numpy as np
    import cv2

    if not hasattr(cv2, 'cann'):
        print('test_npu_cann_ops: cv2 has no `cann` module — source build '
              'did not link cannops against the Python bindings',
              file=sys.stderr)
        return 3

    img = cv2.imread(args.input)
    if img is None:
        print(f'test_npu_cann_ops: cv2.imread returned None for {args.input}',
              file=sys.stderr)
        return 2

    # 大图会放大每个算子的单算子编译税，先退化到小图；算子正确性与尺度无关。
    img = cv2.resize(img, (256, 256)) if img.shape[0] > 512 else img

    cv2.cann.initAcl()
    cv2.cann.setDevice(0)

    failures = 0

    # --- 颜色转换 ---
    failures += not _check('cvtColor BGR2GRAY',
                           cv2.cann.cvtColor(img, cv2.COLOR_BGR2GRAY),
                           cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    failures += not _check('cvtColor BGR2RGB',
                           cv2.cann.cvtColor(img, cv2.COLOR_BGR2RGB),
                           cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    failures += not _check('cvtColor GRAY2BGR',
                           cv2.cann.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                           cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    # --- 几何变换 ---
    failures += not _check('flip (x-axis)',
                           cv2.cann.flip(img, 0), cv2.flip(img, 0))
    failures += not _check('rotate (90 CW)',
                           cv2.cann.rotate(img, 0), cv2.rotate(img, 0))
    failures += not _check('transpose',
                           cv2.cann.transpose(img), cv2.transpose(img))

    # --- 通道操作（merge 返回 AscendMat，需 .download()）---
    merged = cv2.cann.merge(cv2.cann.split(img))
    failures += not _check('split+merge = identity', merged.download(), img)

    # --- 像素算术（float32：uint8 饱和路径在 cannops 上与 CPU 不一致，
    #     float32 逐位一致；与 test_cannops 的 resize/arithmetic 同思路）---
    imgf = img.astype(np.float32)
    noisef = np.full(img.shape, 20, dtype=np.float32)
    failures += not _check('add (float32)',
                           cv2.cann.add(imgf, noisef), cv2.add(imgf, noisef))
    failures += not _check('subtract (float32)',
                           cv2.cann.subtract(imgf, noisef),
                           cv2.subtract(imgf, noisef))

    # --- 位运算 ---
    failures += not _check('bitwise_and',
                           cv2.cann.bitwise_and(img, img),
                           cv2.bitwise_and(img, img))
    failures += not _check('bitwise_not',
                           cv2.cann.bitwise_not(img), cv2.bitwise_not(img))

    cv2.cann.resetDevice()

    if failures:
        print(f'test_npu_cann_ops: FAIL {failures} operator(s) mismatch '
              f'CPU reference', file=sys.stderr)
        sys.stdout.flush()
        os._exit(4)

    print('test_npu_cann_ops: all operators match CPU reference')
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    sys.exit(main())
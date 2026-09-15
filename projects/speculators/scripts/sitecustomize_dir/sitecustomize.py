"""Copy CUDA_VISIBLE_DEVICES onto ASCEND_RT_VISIBLE_DEVICES at interpreter start.

Upstream examples mask cards with CUDA_VISIBLE_DEVICES=... on the
python/torchrun line. torch_npu reads ASCEND_RT_VISIBLE_DEVICES.
Job-level ASCEND_RT_* is the container slice; without this copy,
vLLM and the trainer would both see every card in that slice.
"""
import os

_cuda = os.environ.get('CUDA_VISIBLE_DEVICES')
if _cuda:
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = _cuda

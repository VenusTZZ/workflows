# open_clip

This directory contains the Ascend Quick Start guard for
[mlfoundations/open_clip](https://github.com/mlfoundations/open_clip); it does
not contain open_clip source code.

The guard follows the upstream README inference example and runs one real
pretrained `ViT-B-32` image/text similarity calculation on `npu:0`. It also
adapts upstream's `tests/test_training_simple.py` synthetic RN50 case into a
four-sample, single-card NPU training smoke. Together they cover real weight
loading and inference plus loss, backward, and optimizer execution.

The guard clones the latest upstream release selected by the shared Quick Start
engine, installs its training requirements and editable source, and uses the
upstream `docs/CLIP.png`. It deliberately does not include HCCL, FSDP, audio,
CoCa, or `torch.compile`. Model downloads use a persistent Hugging Face cache
mounted by the workflow.

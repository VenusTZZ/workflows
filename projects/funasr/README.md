# funasr

本目录是 [FunASR](https://github.com/modelscope/FunASR) 的看护配套数据，不是 FunASR 源码。Quick Start 流水线在 [.github/workflows/funasr-quick-start.yml](../../.github/workflows/funasr-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：推理加速；支持程度：新兴适配；阶段 A）。

## Quick Start 看护

文档在 `docs/Quick-start-Ascend.md`。流水线是 `.github/workflows/funasr-quick-start.yml`。文档方言见 [docs/markdown_doc_test_label.md](../../docs/markdown_doc_test_label.md)。无标签的 `shell` 块给用户复制，看护跳过。

### 看护范围

- **看护**：Python 版本检查；安装 `torch==2.9.0` + `torch_npu==2.9.0.post2` + `torchaudio==2.9.0` 并检查 `torch.npu.is_available() == True`；从源码安装 FunASR（克隆上游最新 release tag + `uv pip install -e .`）；在单卡 NPU 上跑 5 个冒烟：`paraformer-zh` 生产级 pipeline（`vad_model="fsmn-vad"` + `punc_model="ct-punc"` + `spk_model="cam++"`）、`iic/SenseVoiceSmall` 多语/情感/事件、`paraformer-zh-streaming` 流式识别、`iic/emotion2vec_plus_large` 情感识别、以及 `funasr` 命令行工具（`-f json` 抽取字段校验）。
- **不看护**（无标签块）：`npu-smi info`（设备表数值每次不同）；`pip install funasr` 二进制安装方式（用户自选路径）。
- **NPU 设备放置说明**：FunASR 的 `AutoModel(..., device="npu:0")` **原生**支持 NPU——内部 `is_npu_available()` 会 `import torch_npu` 并检查 `torch.npu.is_available()`，随后 `model.to("npu:0")` 放置权重；与 modelscope 的 `verify_device`（只认 cpu/cuda/gpu）不同，无需走 `.to('npu:0')` 兜底。看护目标：FunASR 的 ASR / VAD / 标点 / 说话人 / 情感 / 流式各链路在昇腾上都能 `AutoModel` 直接跑通。

### 触发

`funasr-quick-start.yml` 复用 [quick-start-template.yml](../../.github/workflows/quick-start-template.yml) 引擎。`schedule` 保持注释（接入阶段只留手动 `workflow_dispatch`），首次 dispatch 跑通后再打开。文档 `doc_url` 走 Contents API + `ref=${{ github.sha }}`（NPU runner 无法直连 raw.githubusercontent.com）。

模型权重缓存：FunASR 默认 `hub="ms"` 经 ModelScope `snapshot_download` 落盘到 `~/.cache/modelscope`（默认位置），workflow 的 `container_options` 把宿主机目录 bind-mount 到该路径，跨 run 复用。
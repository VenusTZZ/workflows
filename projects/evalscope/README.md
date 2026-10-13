# evalscope

本目录是 [EvalScope](https://github.com/modelscope/evalscope)（ModelScope 团队推出的轻量级大模型评测框架）的看护配套数据，不是上游源码。流水线位于 [`.github/workflows/evalscope-quick-start.yml`](../../.github/workflows/evalscope-quick-start.yml)。注册信息见根目录 [`projects.yaml`](../../projects.yaml)（分类：推理加速；支持程度：新兴适配；阶段 A；upstream：`modelscope/evalscope`）。

## 看护范围

- **Quick-start 文档测试**：[`docs/Quick-start-Ascend.md`](docs/Quick-start-Ascend.md) 遵循 [`docs/markdown_doc_test_label.md`](../../docs/markdown_doc_test_label.md) 标签契约（`#test` / `#test-result` 配对、id 唯一），覆盖单卡昇腾 NPU 完整流程：环境检查、`pip install "evalscope==1.11.0"` 安装、经 ModelScope 自动下载 Qwen/Qwen2.5-0.5B-Instruct 与两个 benchmark（`gsm8k` + `arc`），用 Native（transformers + accelerate）后端在 `device_map="auto"`（`ASCEND_RT_VISIBLE_DEVICES=0`）跑通两个数据集各 `--limit 5` 冒烟，并通过独立的 transformers 加载断言确认模型落到了 `npu:0`，最后校验报告里所有 `score` 落在 [0,1]。
- **版本矩阵**：CANN 9.1.0 / Python 3.12 / torch 2.9.0 / torch_npu 2.9.0.post2 / evalscope 1.11.0 / transformers（随 1.11.0 解析）/ modelscope，与 watch 镜像 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12` 对齐。
- **测试类**：`tests/test_quick_start_ascend.py` 基于 `src/workflows/markdown_doc_test_base.py` 端到端执行文档；`prepare_environment` 承载 CANN env source、CUDA 排除清单、`ASCEND_RT_VISIBLE_DEVICES=0` 卡号 pin、torch 栈 2.9.0 探针、safetensors 与 ModelScope 缓存清理。

## 触发方式

quick-start 目前仅 `workflow_dispatch` 手动触发；schedule 待手动 dispatch 跑绿后再启用。

## 已知问题

- 受限网络（CI 集群与部分开发机）无法直连 HuggingFace，因此本 quick-start 的模型与数据集全部走 ModelScope；`device_map="auto"` 在 `torch_npu` 安装就绪时会自动把模型放到 NPU。
- `evalscope` 的 Native 后端对 `torch` / `transformers` 的版本范围较宽，安装时通过 CUDA 排除清单 + 集群内 PyPI 缓存锁定 NPU 栈，避免被换成 CPU / CUDA 轮子。

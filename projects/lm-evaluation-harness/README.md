# lm-evaluation-harness

本目录是 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)（lm-eval，统一的大模型评测框架）的看护配套数据，不是上游源码。流水线位于 [`.github/workflows/lm-evaluation-harness-quick-start.yml`](../../.github/workflows/lm-evaluation-harness-quick-start.yml)。注册信息见根目录 [`projects.yaml`](../../projects.yaml)（分类：推理加速；支持程度：新兴适配；阶段 A；upstream：`EleutherAI/lm-evaluation-harness`）。

## 看护范围

- **Quick-start 文档测试**：[`docs/Quick-start-Ascend.md`](docs/Quick-start-Ascend.md) 遵循 [`docs/markdown_doc_test_label.md`](../../docs/markdown_doc_test_label.md) 标签契约（`#test` / `#test-result` 配对、id 唯一），覆盖单卡昇腾 NPU 完整流程：环境检查、`uv pip install "lm_eval[hf]"` 安装、经 ModelScope 自动下载 Qwen/Qwen2.5-0.5B-Instruct 与两个 benchmark 仓库（`modelscope/ai2_arc` 脚本型 + `allenai/winogrande` parquet 镜像），用 HuggingFace 后端在 `--device npu:0` 跑通官方 `arc_easy`（0-shot）与 `winogrande`（5-shot）各 `--limit 10` 冒烟，最后校验两个任务的 `acc` 落在 [0,1]。文档中两个任务 YAML 直接复制安装好的官方定义，只把 `dataset_path` 指向 ModelScope 本地目录。
- **版本矩阵**：CANN 9.1.0 / Python 3.12 / torch 2.9.0+cpu / torch_npu 2.9.0.post2 / lm-eval 0.4.13 / transformers `<5.0` / modelscope 1.37.0，与 watch 镜像 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12` 对齐。
- **测试类**：`tests/test_quick_start_ascend.py` 基于 `src/workflows/markdown_doc_test_base.py` 端到端执行文档；`prepare_environment` 承载 CANN env source、CUDA 排除清单、`ASCEND_RT_VISIBLE_DEVICES=0` 卡号 pin、uv 安装、torch 栈 2.9.0 探针、safetensors 与 ModelScope 缓存清理。

## 触发方式

quick-start 目前仅 `workflow_dispatch` 手动触发；schedule 待手动 dispatch 跑绿后再启用。

## 已知问题

- 受限网络（CI 集群与部分开发机）无法直连 HuggingFace / Xet 数据面，历史方案中的 `HF_ENDPOINT` 镜像与 `HF_HUB_DISABLE_XET` 均不可靠，因此本 quick-start 的模型与数据集全部走 ModelScope，任务定义用本地 YAML 复制改写。


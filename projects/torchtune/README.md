# torchtune

本目录是 [torchtune](https://github.com/meta-pytorch/torchtune) 的看护配套数据，不是 torchtune 源码。流水线在 `[.github/workflows/torchtune-quick-start.yml](../../.github/workflows/torchtune-quick-start.yml)`。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：训练加速；支持程度：基础支持；阶段 A）。

在昇腾 NPU 上把 Quick Start 文档里所有 `#test` / `#test-setup` 代码块跑通，对照 `#test-result` 做输出断言。脚本退出码非 0 即判红。

## 文档

`docs/Quick-start-Ascend.md` 走的是 `docs/markdown_doc_test_label.md` 的契约：每个 `shell` 代码块都带 `#test` / `#test-result` / `#test-setup` 标签加 `id=` / `store=` / `load='x>>y'` / `fuzzy='xxx'` 参数，runner（`tests/test_quick_start_ascend.py` 的 `MarkdownDocTestBase.run_template`）会按文档顺序抓取、解析、执行并比对预期。

文档覆盖：

- CANN / torch / torch_npu / modelscope 前置检查与安装；
- torchtune 二进制安装 + 源码安装两条路径；
- `tune --help` / `tune ls lora_finetune_single_device` CLI 自检；
- 从 ModelScope 下载 `Qwen/Qwen2.5-0.5B-Instruct`（~1 GB，单卡友好、无 gating）；
- `tune run lora_finetune_single_device --config qwen2_5/0.5B_lora_single_device` 跑 3 步 LoRA 微调并验证 checkpoint 落盘。

> 上游教程 [First Finetune Tutorial](https://meta-pytorch.org/torchtune/0.6/tutorials/first_finetune_tutorial.html) / [README](https://github.com/meta-pytorch/torchtune/blob/main/README.md) 默认走 Meta Llama 系列（gating + HF_TOKEN），本文档换用 `Qwen2.5-0.5B-Instruct` 以便 CI 在国内镜像 + ModelScope 拉取链路下开箱即用，CLI 命令与上游教程保持一致。

## 触发

`torchtune-quick-start.yml` 接受：

- `schedule`：每 6 小时轮询上游一次。`monitor` job（免费的 ubuntu-latest）对比三个信号与上次记录（状态存 actions/cache）：文档 hash、latest release tag、main HEAD SHA 任一变化才 checkout 上游、在 NPU runner 上跑测试，被测 ref 按 release > doc > commit 优先级取自变化的信号；全部无变化则 monitor 后直接结束。上次看护失败时，下个周期即使无变化也会自动重试（`record-outcome` job 回写成败）。
- `workflow_dispatch`：手动触发；绕过 monitor 门，直接拿 latest release tag 跑测试。

## Example 看护

`torchtune-examples.yml` 走共享的 examples 引擎（[`.github/workflows/examples-template.yml`](../../.github/workflows/examples-template.yml)），只在上游 release 变化时跑 `projects/torchtune/examples_manifest.yaml` 矩阵。

当前 6 条支持（都在 coder `hdc-stable-npu-4` 单卡 1.85 GiB NPU HBM、1 step 实测 exit 0）：

| Recipe | 配置 | 说明 |
|---|---|---|
| `lora_finetune_single_device.py` | `qwen2_5/0.5B_lora_single_device` | LoRA SFT，50 行 alpaca fixture |
| `full_finetune_single_device.py` | `qwen2_5/0.5B_full_single_device` | 全量 SFT，同 fixture |
| `lora_dpo_single_device.py` | `qwen2_5/0.5B_lora_single_device` + `DPOLoss` + preference fixture | LoRA DPO，8 行对话式偏好 |
| `generate.py` | `generation.yaml` 覆写为 Qwen2.5-0.5B | 单条 prompt 推理 |
| `quantize.py` | `quantization.yaml` 覆写为 Qwen2.5-0.5B | Int8DynActInt4 量化 |
| `dev/generate_v2.py` | `generation.yaml` 覆写 + `log_level=INFO` | 实验性推理 v2 |

每条 overlay 注释说明：为什么覆盖 `model._component_` / `tokenizer.path` / `checkpointer.*` / `dataset.source=json` / `checkpointer.model_type=QWEN2` / `log_level=INFO` —— 全部锚定到上游源码（upstream tags 静态分析 + 实际 run 验证）。

`unsupported` 注释 14 条，每条注明根因：

- 8 条 `*_distributed.py`：NPU 硬阻塞（`init_process_group("cuda:nccl,cpu:gloo")` 写死，torch_npu ship HCCL not NCCL，RuntimeError: Distributed package doesn't have NCCL built in）
- 1 条 `dev/grpo_full_finetune_distributed.py`：除 NCCL 外还需 verl/tensordict 栈
- 1 条 `knowledge_distillation_single_device.py`：default 8B teacher 超 32G HBM（候选，有 qwen2 1.5B→0.5B 配置理论能跑）
- 1 条 `ppo_full_finetune_single_device.py`：需独立 reward/value model
- 1 条 `eleuther_eval.py`：需 lm-eval + HF 数据集下载
- `recipes/configs/`：被 `scan.exclude` 排除（YAML 不能独立 launch）

# deepspeed

本目录是 [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) 的看护配套数据，不是 DeepSpeed 源码。example 流水线在 [.github/workflows/deepspeed-examples.yml](../../.github/workflows/deepspeed-examples.yml)，quick-start 流水线在 [.github/workflows/deepspeed-quick-start.yml](../../.github/workflows/deepspeed-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：训练加速；支持程度：基础支持；阶段 A）。

上游默认分支是 `master`。上游有 Ascend NPU 加速器支持（`accelerator/npu_accelerator.py`），由华为贡献。外部 Ascend CI（`Ascend/Ascend-CI` 的 `deepspeed.yaml`）已停摆（自 2026-06-11 起连续失败，基础设施故障）。本仓先走阶段 A：在本仓流水线把 example 跑通，再考虑往上游推。

## 仓库关系（分离模式）

`deepspeed-examples.yml` 是调用公共 [.github/workflows/examples-template.yml](../../.github/workflows/examples-template.yml) 的薄触发器，并使用 `examples_repo` 分离模式：

- 监控仓 / 源码安装：`deepspeedai/DeepSpeed`（主仓，发布 release，驱动 schedule；`run-example` 容器内 checkout 到 `target/`，DeepSpeed 从这里 `pip install -e` 安装）。
- Example 来源：`deepspeedai/DeepSpeedExamples`（不发布 release，永远跟随其默认分支 `master`；checkout 到 `examples/`，本目录 manifest 的 `path` 相对该仓根）。
- 这是 `examples_repo` 分离模式（见 [docs/examples-guard-engine.md](../../docs/examples-guard-engine.md)）的首个使用方。

## 清单

[examples_manifest.yaml](examples_manifest.yaml) 的 `scan.root` 为 DeepSpeedExamples 仓根，`include_extensions` 为 `.sh` / `.py`，`scan.exclude` 把被 import 的库、模型定义、测试等配套物剪枝掉。`files-only` 扫描模型的对账单位是入口文件。

当前 supported 共 10 条单卡（`linux-aarch64-a2-1`）小规模 example，统一用 CANN 9.1.0 镜像，模型走 ModelScope、数据集用仓内本地 fixture，全部用 `overlay_args` 压到 CI 规模：

| path（相对 examples 仓） | profile | 看护点 | 模型 / 数据 | 压规模 |
|---|---|---|---|---|
| `training/HelloDeepSpeed/run_ds.sh` | deepspeed | Roberta MLM（ZeRO-1 + CPU offload + BF16） | wikitext（setup patch 为 Salesforce/wikitext，ModelScope）+ roberta-base tokenizer | 2 层 10 步 |
| `applications/DeepSpeed-Chat/training/step1_supervised_finetuning/main.py` | ds_chat_sft | SFT（NPU-aware） | opt-125m（ModelScope）+ 8 行 local/jsonfile fixture | 1 epoch |
| `applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/main.py` | ds_chat_rw | Reward Model | 同上 | 1 epoch |
| `applications/DeepSpeed-Chat/training/step2_dpo_finetuning/main.py` | ds_chat_dpo | DPO（ref model 内存翻倍） | 同上 | 1 epoch |
| `applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/main.py` | ds_chat_rlhf | RLHF（官方 `--enable_test_mode`） | opt-125m ×2（ModelScope）+ fixture | test mode 5 步 |
| `inference/huggingface/text-generation/inference-test.py` | ds_infer | 推理（`--hf_baseline` 跳过 DS kernel） | opt-125m（ModelScope） | 8 token |
| `training/bf16_master_weight/train.py` | deepspeed | bf16 master-weight 对比 | 内置 SimpleTransformer + 合成数据 | 5 步 |
| `training/cifar/cifar10_deepspeed.py` | deepspeed | CIFAR10 分类（ZeRO-0 + BF16） | torchvision CIFAR-10 | 1 epoch |
| `training/pipeline_parallelism/train.py` | deepspeed | Pipeline parallelism（单卡 p=1） | torchvision CIFAR-10 | 10 步 |
| `training/offload_states/offload_states.py` | deepspeed | ZeRO offload_states | 随机合成数据 | 小规模 |

DeepSpeed-Chat 的 `--data_path local/jsonfile` 从 `applications/DeepSpeed-Chat/data/{train,eval}.json` 读取（JSON Lines，字段 `prompt`/`chosen`/`rejected`）；`scripts/setup_example.sh` 在对应 profile 下把 [fixtures/](fixtures/) 里的 8 行 fixture 拷到该目录。模型经 `modelscope.snapshot_download` 下载并 plant 到 HF hub cache，使 example 里硬编码的 `facebook/opt-125m` 离线可解析。

其余约 388 条列入 unsupported 并按族群加注释（多机多卡 mpi/NCCL、需 ImageNet/大模型、绑 CUDA 算子、NVMe 硬件、性能基准、compression 需 patch 等），见 manifest。清单与磁盘的差异只打印路径，不使 job 失败；例外：`supported` 条目的 path 已不在磁盘上时 manifest-check 立即判红。

## 触发

`deepspeed-examples.yml` 是薄触发器：

- `workflow_dispatch`：手动运行只接受可选 `target_ref`（`deepspeedai/DeepSpeed` 的分支/tag/SHA，留空由引擎解析为上游默认），不经过 monitor 门。`upstream_repo` / `examples_repo` 固定在 workflow 内，不再支持旧版 `target_repo` 覆盖。
- `schedule`：当前以注释保留、暂不启用。启用后由公共引擎对主仓 `deepspeedai/DeepSpeed` 做 release-only 监控（latest release tag 变化时触发，上次失败时下一周期以 `release-retry` 重试）；examples 仓无 release，始终跟随 `master`。

## 模型缓存边界

薄触发器不向公共引擎传宿主缓存卷，模型在各 matrix job 的容器内准备；同一 job 的 setup 与 run 步骤可复用该容器内文件，但不保证跨 job 或跨 workflow run 复用。本看护所需模型（opt-125m 等）与数据集（fixture / CIFAR-10）均可经 ModelScope 或直接下载得到，正常路径不使用 [cache-seed](../../cache-seed/README.md)；仅当某 example 硬编码了 ModelScope 也无镜像的资产时才按 cache-seed 流程兜底投递。

## Quick Start

`deepspeed-quick-start.yml` 看护本仓 [docs/Quick-start-Ascend.md](docs/Quick-start-Ascend.md)。该文档描述了在昇腾 NPU 上安装 DeepSpeed、验证加速器、单卡跑通 CIFAR10 示例并双卡体验分布式训练的完整流程。

- 监控信号：doc 哈希、上游 latest release、master HEAD SHA。按 doc > release > commit 优先级，任一变化触发测试。
- 测试内容：pip 安装 DeepSpeed → 安装配套 torchvision → `get_accelerator()._name == 'npu'` → CIFAR10 内联示例（ZeRO-1 + BF16，1 个 epoch）→ 双卡分布式训练（`--num_gpus 2`，HCCL 通信）。
- **当前为节约 NPU 资源，`schedule` 已注释，只保留手动 `workflow_dispatch`。**

## 已知边界

- 版本错位：DeepSpeed 安装自主仓被监控的 release 版本，examples 仓跟随 `master`，二者存在小幅错位的可能；若某 release 与 examples `master` 不兼容导致失败，按结果定位后可将 manifest 的 `target_ref` 固定或向上游反馈。
- compression 的 `bert/gpt2/cifar` 三个 `*_no_trainer` 入口硬编码 `torch.device("cuda")`，需 patch 后方可接入；gan 需去掉 `--cuda` 硬编码；data_efficiency / deepspeed_finetune_demo 依赖远程 HF 数据集，换本地 fixture 后可升级。均列为下一轮候选，当前在 manifest 中以 unsupported + 注释记录。

# deepspeed

本目录是 [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) 的看护配套数据，不是 DeepSpeed 源码。example 流水线在 `[.github/workflows/deepspeed-examples.yml](../../.github/workflows/deepspeed-examples.yml)`，quick-start 流水线在 `[.github/workflows/deepspeed-quick-start.yml](../../.github/workflows/deepspeed-quick-start.yml)`。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：训练加速；支持程度：基础支持；阶段 A）。

上游默认分支是 `master`。上游有 Ascend NPU 加速器支持（`accelerator/npu_accelerator.py`），由华为贡献。外部 Ascend CI（`Ascend/Ascend-CI` 的 `deepspeed.yaml`）已停摆（自 2026-06-11 起连续失败，基础设施故障）。本仓先走阶段 A：在本仓流水线把 example 跑通，再考虑往上游推。

## 仓库关系

- 监控目标：`deepspeedai/DeepSpeed`（主仓，源码安装）
- Example 来源：`deepspeedai/DeepSpeedExamples`（`training/` 目录）
- 流水线同时 checkout 两个仓库：主仓源码安装，Examples 仓跑 example

## 清单

- `examples_manifest.yaml` 扫描 `deepspeedai/DeepSpeedExamples` 的 `training/` 目录下的 `.sh` / `.py`。`training/HelloDeepSpeed/run_ds.sh` 已进 `supported`。`run_ds.sh` 封装了 `deepspeed --bind_cores_to_rank train_bert_ds.py --checkpoint_dir experiment_deepspeed $@`，已有 `$@` 透传，无需补丁。该 example 使用 Roberta 结构 Transformer 做 MLM 任务，展示 **ZeRO-1 + CPU Offload + BF16**。
- `profile` 是 `deepspeed`：从主仓源码安装 DeepSpeed 并验证 `ds_report` 能识别 NPU 加速器。
- runner 是 `linux-aarch64-a2-1`（1 张卡），`npu_devices` 是 `'0'`。
- 镜像使用 CANN 9.1.0 ubuntu22.04。
- `overlay_args`：`--num_layers 2 --num_heads 2 --h_dim 64 --num_iterations 10 --dtype bf16`（2 层 Transformer、10 步训练，压到 CI 规模）。
- 清单与磁盘的差异只打印路径，不使 job 失败；例外：`supported` 条目的 path 已不在磁盘上时 manifest-check 立即判红。

## 触发

`deepspeed-examples.yml` 接受：

- `schedule`：每 6 小时轮询上游主仓一次。`monitor` job（免费的 ubuntu-latest）对比三个信号与上次记录（状态存 actions/cache）：`examples/` 最新 commit、latest release tag、master HEAD SHA。任一变化才 checkout 双仓、在 NPU runner 上跑 example，被测 ref 按 examples > release > commit 优先级取自变化的信号；全部无变化则 monitor 后直接结束。上次看护失败时，下个周期即使无变化也会自动重试（`record-outcome` job 回写成败）。**当前为节约 NPU 资源，`schedule` 已注释，只保留手动 `workflow_dispatch`。**
- `workflow_dispatch`：手动指定 `target_repo` 与 `target_ref`（默认 `deepspeedai/DeepSpeed` 的 `master`），不经过 monitor 门。

## Quick Start

`deepspeed-quick-start.yml` 看护本仓 [docs/Quick-start-Ascend.md](docs/Quick-start-Ascend.md)。该文档描述了在昇腾 NPU 上安装 DeepSpeed、验证加速器、单卡跑通 CIFAR10 示例并双卡体验分布式训练的完整流程。

- 监控信号：doc 哈希、上游 latest release、master HEAD SHA。按 doc > release > commit 优先级，任一变化触发测试。
- 测试内容：pip 安装 DeepSpeed → 安装配套 torchvision → `get_accelerator()._name == 'npu'` → CIFAR10 内联示例（模块化展示 DeepSpeed 工作流，ZeRO-1 + BF16，1 个 epoch）→ 双卡分布式训练（同一脚本 `--num_gpus 2`，ZeRO-1 分片，HCCL 通信）。
- **当前为节约 NPU 资源，`schedule` 已注释，只保留手动 `workflow_dispatch`。**
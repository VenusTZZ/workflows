# Liger-Kernel examples 看护

本项目复用公共 `examples-template.yml`；薄触发器为
`.github/workflows/liger-kernel-examples.yml`，被测仓库是 `linkedin/Liger-Kernel`。
手动触发的 `target_ref` 可指定分支、tag 或 SHA；留空时由公共引擎解析上游最新
release。setup 从同一个 `$TARGET_ROOT` checkout 安装 Liger 源码，因此被测内核
版本与 examples 始终来自同一个 release。Bring-up 期间 schedule 保持注释关闭。

## 为什么可行

v0.8.3 已原生带 Ascend NPU 后端：内核实现位于
`src/liger_kernel/ops/backends/_ascend/`，按设备自动选中；上层有
`infer_device()`（NPU 上返回 `"npu"`）与 `infer_comm_backend()`（返回 `"hccl"`）
两个抽象。上游 README 的 Ascend 安装段 pin 了 `torch==2.9.0` /
`torch_npu==2.9.0` / `triton-ascend==3.2.2` / `CANN==9.1.0`，并挂了华为官方
Ascend-CI 的 `liger_kernel.yml` 徽章——本项目 setup 即复刻该配方。

镜像用 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`，
与本仓 tensordict、trl 等项目同款。CANN 基础镜像可能完全不带 PyTorch（tensordict
的 run #1 就是这么失败的），所以 setup 先探测可复用的 torch/torch_npu 对，缺失时
从集群 pip 缓存 + Ascend 源装 `torch==2.9.0` 与 `torch_npu==2.9.0.post2`，再断言
版本与 NPU 可用性。`triton-ascend` 只发布在
`https://triton-ascend.osinfra.cn/pypi/simple`：它的 wheel 名是 `triton_ascend-*`，
但装出来的 import 名仍是 `triton`（Ascend 版替换 PyPI 同名包），所以安装前先卸载
PyPI triton，幂等判断用 `pip show triton-ascend`。

## supported 清单

已有三条单卡任务在手动 workflow #5（commit `33c39b2`）全部通过，包含实际训练步骤和
三个成功的 `result.json`；被测版本为 v0.8.3。新增一条两卡 FSDP 任务，待下一轮手动
workflow 验证。

| example | 作用 | 压规模方式 |
|---|---|---|
| `examples/huggingface/training.py` | `AutoLigerKernelForCausalLM` 一行 monkey-patch 应用 Liger 内核，trl `SFTTrainer` 做 SFT | `--max_steps 3`、batch 2、seq 256，不开 FSDP |
| `examples/huggingface/run_qwen.sh` | 上述 SFT 的两卡 FSDP 配方，验证 torchrun、HCCL 和 full shard | 复刻上游 launcher；改用 ModelScope 0.5B、2 卡、2 步、每卡 batch 1、本地 8 行数据 |
| `examples/medusa/train.py` | 冻结 backbone、注入 medusa 多头，用 `fused_linear_cross_entropy` 训练多 token 预测头 | `--max_steps 1`、`--medusa_num_heads 2`、`--medusa_return True` |
| `examples/huggingface/training_multimodal.py` | Qwen2-VL 图文 SFT，monkey-patch 多模态 RoPE / RMSNorm / SwiGLU / FLCE | `--max_steps 1`、batch 1、seq 256，数据换 4 行本地 fixture |

四条都不改源码。文本训练与 Medusa 模型走 ModelScope 的 `Qwen/Qwen2.5-0.5B-Instruct`；多模态那条
走 ModelScope 的 `Qwen/Qwen2-VL-2B-Instruct`。设备由 liger 的 `infer_device()` 与
accelerate 自动落到 NPU。前两条带 upstream 自带的 `EfficiencyCallback`，其
`on_init_end` 强制要求 `--include_num_input_tokens_seen` 与 `--logging_steps 1`，
overlay 必须带这两个。

`run_qwen.sh` 原脚本固定 `torchrun` 四进程、Qwen2-7B、每卡 batch 48，而且不透传额外
参数。项目 runner 按该脚本启动 `training.py`，用 manifest 覆盖模型、数据和规模，并保持
`--fsdp "full_shard auto_wrap"` 与上游 `config/fsdp_config.json`。这是同一 Python
example 的另一种分布式启动配方，不算新增的独立训练程序。两卡任务要求
`linux-aarch64-a2-2`，任务启动前检查 NPU 数量，`torchrun --standalone` 分配独立端口。

fixture 全部由 setup 现生成，不往仓库塞二进制：

- hf 例的 `--dataset` 走 `datasets.load_dataset(path)`，需要一个**目录**（内含 `train.jsonl`），
  文件名不影响 split 名（都是 `train`）；
- 多模态例的 `--dataset` 同理需要目录，且必须自带 dataset card 声明配置名，否则
  `load_dataset(dir, "ai2d")` 抛 `BuilderConfig not found`。图片列用 `Sequence(Image())`，
  文本列声明成**单元素 list** 的 dict schema（写成 `Sequence({...})` 会在编码时抛
  `'list' object has no attribute 'get'`）。setup 在
  `$TARGET_ROOT/fixtures/cauldron_ai2d/` 下生成 4 行 112x112 合成图文。

多模态那条额外装 `torchvision==0.24.0`：`AutoProcessor` 走 torchvision 图像后端，
纯文本栈里没有它，缺了会抛 `AutoVideoProcessor requires the Torchvision library`。

## 依赖版本窗口（关键约束）

`transformers==4.57.1` + `trl==0.12.1`。这个窗口被 example 源码钉死：

- `from_pretrained(..., dtype=...)` 需要 transformers >= 4.56.0（4.55.x 只有
  `torch_dtype=`，传 `dtype=` 抛 `Object of type dtype is not JSON serializable`）。
- `Trainer(tokenizer=...)`（medusa 用）在 transformers 5.0 被彻底移除。
- `SFTTrainer(max_seq_length=...)`（hf 与多模态例用）从 trl 0.13.0 起移入 `SFTConfig`。
- `DataCollatorForCompletionOnlyLM`（hf 例用）在 trl 0.20.0 被删除。

窗口内唯一同时满足的组合即 4.57.1 + 0.12.1，已用逐字节相同的 `training.py` 与
`training_multimodal.py` 在 CPU 上跑通训练步验证；上卡实测由手动 workflow 完成。

## unsupported

2026-10-10 全量复核更新：最新 release 为 `v0.8.4`，新增两卡 FSDP ORPO 配方，
当前 supported 为 **5 条**。ORPO 用上游 FSDP 配置、ModelScope 原架构模型和原生
本地偏好数据加载，保留固定100步；仅关闭可选 loss 编译优化，不改示例源码。
此条仍待 Actions 验证。当前 unsupported 仅保留 Lightning 和两条 Megatron；
库、回调与重复启动包装移出清单，具体功能和接入限制记录在 manifest 注释中。
下方关于 ORPO 的旧“单卡不能接入”记录已被这次两卡配方取代。

`examples/alignment/run_orpo.py`、`examples/lightning/training.py`、
`examples/megatron/*` 及其余 `run_*.sh` 启动器入 unsupported，逐条在 manifest 里
写明原因。两个值得记下的硬阻塞：

- ORPO：`LigerORPOTrainer` 的损失调用无条件走 `_FSDPForwardRedirection()`，该 helper
  在 `src/liger_kernel/transformers/fsdp.py:40` 直接 assert 模型是 `FullyShardedDataParallel`；
  单卡未包装模型第一步就 AssertionError，而走 accelerate + FSDP 需要多卡 HCCL。
  叠加脚本无 argparse（`max_steps=100`/batch 32/1B 模型全硬编码）与 gated 模型。
- Lightning：`pl.Trainer(accelerator=infer_device())` 传 `"npu"`，而 Lightning 只注册
  cpu/cuda/mps/xla，且 `torch.optim.AdamW(fused=True)` 在 NPU 无 `_fused_adamw_` 实现；
  源码还把 MMLU 拆出固定 4096 行验证集，不能用很小的 CI fixture。
- Megatron 两条虽只需两进程，源码仍固定 `torch.cuda.set_device`、`.cuda()` 和 NCCL；
  增加 runner 卡数无法改变设备后端。

## 已知边界

medusa 例的 `max_steps` 必须保持 1：其 callback 的 MFU 分支 `_get_gpu_peak_tflops`
只识别 A100/H100/V100，Ascend 上返回 `None`，第 2 步的 TFLOPS 除法会崩。warmup
步数为 2，故 1 步安全；这是上游代码问题。

artifact 由公共引擎命名为 `liger-kernel-examples-<run_id>-<job_index>`。

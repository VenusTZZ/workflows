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

第一阶段 supported 共 10 条，按 example 的最小有效拓扑使用 1/2/4 卡 runner，统一用 CANN 9.1.0 镜像。模型走 ModelScope（`ms_download_models` 下载后经 `GITHUB_ENV` 传本地路径），数据集优先使用仓内 fixture，并用 `overlay_args` 压到 CI 规模：

| path（相对 examples 仓） | profile | 看护点 | 模型 / 数据 | 压规模 |
|---|---|---|---|---|
| `training/HelloDeepSpeed/run_ds.sh` | deepspeed / 1 卡 | Roberta MLM（ZeRO-1 + CPU offload + BF16） | wikitext 运行时重定向到 Salesforce/wikitext + roberta-base tokenizer | 2 层 10 步 |
| `applications/DeepSpeed-Chat/training/step1_supervised_finetuning`（exec: `main.py`） | ds_chat_sft | SFT（NPU-aware） | opt-125m（ModelScope）+ 8 行 local/jsonfile fixture | 1 epoch |
| `applications/DeepSpeed-Chat/training/step2_reward_model_finetuning`（exec: `main.py`） | ds_chat_rw | Reward Model | 同上 | 1 epoch |
| `applications/DeepSpeed-Chat/training/step2_dpo_finetuning`（exec: `main.py`） | ds_chat_dpo | DPO（ref model 内存翻倍） | 同上 | 1 epoch |
| `applications/DeepSpeed-Chat/training/step3_rlhf_finetuning`（exec: `main.py`） | ds_chat_rlhf | RLHF（官方 `--enable_test_mode`） | opt-125m ×2（ModelScope）+ fixture | test mode 5 步 |
| `inference/huggingface/text-generation/inference-test.py` | ds_infer | 推理（`--hf_baseline` 跳过 DS kernel） | opt-125m（ModelScope） | 8 token |
| `training/cifar`（exec: `run_ds.sh`） | ds_cifar | CIFAR10 分类（ZeRO-0 + BF16） | ModelScope 预置并校验的 CIFAR-10 | 1 epoch |
| `training/offload_states/offload_states.py` | deepspeed / 1 卡 | ZeRO offload_states | 随机合成数据 | 小规模 |
| `training/cifar/run_ds_moe.sh` | ds_cifar / 2 卡 | CIFAR10 MoE expert parallel（EP=2） | ModelScope 预置并校验的 CIFAR-10 | 1 epoch |
| `training/autotp_equivalence`（exec: `train.py`） | ds_autotp_equivalence / 4 卡 | AutoTP=1/3/4 loss 等价性 | Qwen3-0.6B（ModelScope）+ 随机 token | 每组 5 步 |

**启动方式的选择**：`cifar10_deepspeed.py` 的 main() 无条件读 launcher 注入的 `LOCAL_RANK` 并调 `init_distributed()`，因此普通 CIFAR 经支持 `$@` 的上游 `run_ds.sh` 启动。CIFAR MoE 的上游脚本不透传 `$@`，项目 runner 按原配方复刻两卡 launcher、EP=2 和 MoE 参数，再追加 CI overlay。DS-Chat 官方 training_scripts 硬编码 1.3B～66B 模型且不透传任意参数，项目 runner 等价执行 `deepspeed --num_gpus 1 main.py <overlay_args>`。AutoTP equivalence 同样复刻上游 `run_gpu.sh` 的 1/3/4 卡三次启动与 loss 比较，但显式传入 ModelScope 本地模型。offload_states 与 `--hf_baseline` inference 不依赖 launcher，直接运行 `.py`。

多卡条目启动前会检查 `ASCEND_RT_VISIBLE_DEVICES`：MoE 至少需要 2 卡，AutoTP equivalence 至少需要 4 卡；runner 未注入时分别使用 `0,1` 和 `0,1,2,3`。AutoTP 三次子运行使用独立 master port，避免进程组端口复用。

bf16_master_weight 与 pipeline_parallelism 曾进 supported，CI 实测其源码硬绑 CUDA（`torch.cuda.set_device` / `autocast(device_type="cuda")` / `--backend nccl`），装包无法解决，已移回 unsupported（需 patch，次轮候选）。

DeepSpeed-Chat 的 `--data_path local/jsonfile` 从 `applications/DeepSpeed-Chat/data/{train,eval}.json` 读取（JSON Lines，字段 `prompt`/`chosen`/`rejected`）；`scripts/setup_example.sh` 在对应 profile 下把 [fixtures/](fixtures/) 里的 8 行 fixture 拷到该目录。上游 `setup.py` 的 `find_packages(include=['dschat'])` 无法安装缺少根 `__init__.py` 的 namespace package，因此 setup 不依赖其空 editable wheel，而是把 `applications/DeepSpeed-Chat` 源码根目录写入 `PYTHONPATH` 并立即执行 `import dschat` 验证。上游 [issue #813](https://github.com/deepspeedai/DeepSpeedExamples/issues/813) 也记录了 DeepSpeed-Chat 在切换执行/缓存上下文后发生模块解析错误，但不是本次完全相同的报错。模型经 `ms_download_models` 从 ModelScope 下载到本地，路径写入 `GITHUB_ENV`（`OPT_125M_PATH` / `QWEN3_06B_PATH`），overlay_args 引用本地目录。setup 按上游 requirements/setup.py 显式安装依赖，但不会从 PyPI 覆盖镜像的 `torch + torch_npu` 或 `$TARGET_ROOT` 中的 DeepSpeed 源码；安装后会校验 `deepspeed.__file__` 位于目标源码树。

CIFAR 单卡和两卡 MoE 共用 `ds_cifar` profile。setup 从固定 revision 的 ModelScope 镜像下载 CIFAR-10 zip，先校验 SHA-256，再解压到上游脚本使用的 `training/cifar/data`，最后调用 torchvision 自带的官方逐文件 MD5 清单复核。这样保留 example 的 `download=True` 原始逻辑，但完整数据已存在时不会访问 CI 中超时的 Toronto 源；其余 `deepspeed` profile 不承担这次约 170 MB 的下载。

其余约 224 条列入 unsupported：同一逻辑 example 的 `.sh` 启动包装已并入对应 `.py` 条目，每一条都带一行内联中文注释，注明具体不支持原因（多机多卡 mpi/NCCL、需 ImageNet/大模型、绑 CUDA 算子、NVMe 硬件、性能基准、compression 需 patch、依赖远程 HF 数据集等），见 manifest。清单与磁盘的差异只打印路径，不使 job 失败；例外：`supported` 条目的 path 已不在磁盘上时 manifest-check 立即判红。

第二阶段以第一阶段 10/10 远程全绿为门槛；届时再评估加入 `linux-aarch64-a2-8` 的 HF AutoTP=8 和两卡 ZenFlow 单配置 smoke，未验收前不进入 supported，也不启用 schedule。

## 触发

`deepspeed-examples.yml` 是薄触发器：

- `workflow_dispatch`：手动运行只接受可选 `target_ref`（`deepspeedai/DeepSpeed` 的分支/tag/SHA，留空由引擎解析为上游默认），不经过 monitor 门。`upstream_repo` / `examples_repo` 固定在 workflow 内，不再支持旧版 `target_repo` 覆盖。
- `schedule`：当前以注释保留、暂不启用。启用后由公共引擎对主仓 `deepspeedai/DeepSpeed` 做 release-only 监控（latest release tag 变化时触发，上次失败时下一周期以 `release-retry` 重试）；examples 仓无 release，始终跟随 `master`。

## 模型缓存边界

薄触发器不向公共引擎传宿主缓存卷，模型和 CIFAR-10 在各 matrix job 的容器内准备；同一 job 的 setup 与 run 步骤可复用该容器内文件，但不保证跨 job 或跨 workflow run 复用。本看护所需模型（opt-125m 等）和 CIFAR-10 均有 ModelScope 来源，正常路径不使用 [cache-seed](../../cache-seed/README.md)；仅当 ModelScope 资产变得不可用时才按 cache-seed 流程兜底投递。

## Quick Start

`deepspeed-quick-start.yml` 看护本仓 [docs/Quick-start-Ascend.md](docs/Quick-start-Ascend.md)。该文档描述了在昇腾 NPU 上安装 DeepSpeed、验证加速器、单卡跑通 CIFAR10 示例并双卡体验分布式训练的完整流程。

- 监控信号：doc 哈希、上游 latest release、master HEAD SHA。按 doc > release > commit 优先级，任一变化触发测试。
- 测试内容：pip 安装 DeepSpeed → 安装配套 torchvision → `get_accelerator()._name == 'npu'` → CIFAR10 内联示例（ZeRO-1 + BF16，1 个 epoch）→ 双卡分布式训练（`--num_gpus 2`，HCCL 通信）。
- **当前为节约 NPU 资源，`schedule` 已注释，只保留手动 `workflow_dispatch`。**

## 已知边界

- 版本错位：DeepSpeed 安装自主仓被监控的 release 版本，examples 仓跟随 `master`，二者存在小幅错位的可能；若某 release 与 examples `master` 不兼容导致失败，按结果定位后可将 manifest 的 `target_ref` 固定或向上游反馈。
- compression 的 `bert/gpt2/cifar` 三个 `*_no_trainer` 入口硬编码 `torch.device("cuda")`，需 patch 后方可接入；gan 需去掉 `--cuda` 硬编码；data_efficiency / deepspeed_finetune_demo 依赖远程 HF 数据集，换本地 fixture 后可升级。均列为下一轮候选，当前在 manifest 中以 unsupported + 注释记录。

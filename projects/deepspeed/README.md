# deepspeed

本目录是 [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) 的看护配套数据，不是 DeepSpeed 源码。example 流水线在 [.github/workflows/deepspeed-examples.yml](../../.github/workflows/deepspeed-examples.yml)，quick-start 流水线在 [.github/workflows/deepspeed-quick-start.yml](../../.github/workflows/deepspeed-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：训练加速；支持程度：基础支持；阶段 A）。

上游默认分支是 `master`。上游有 Ascend NPU 加速器支持（`accelerator/npu_accelerator.py`），由华为贡献。外部 Ascend CI（`Ascend/Ascend-CI` 的 `deepspeed.yaml`）已停摆（自 2026-06-11 起连续失败，基础设施故障）。本仓先走阶段 A：在本仓流水线把 example 跑通，再考虑往上游推。

## 仓库关系（分离模式）

`deepspeed-examples.yml` 是调用公共 [.github/workflows/examples-template.yml](../../.github/workflows/examples-template.yml) 的薄触发器，并使用 `examples_repo` 分离模式：

- 监控仓 / 源码安装：`deepspeedai/DeepSpeed`（主仓，发布 release，驱动 schedule；`run-example` 容器内 checkout 到 `target/`，DeepSpeed 从这里 `pip install -e` 安装）。
- Example 来源：`deepspeedai/DeepSpeedExamples`（不发布 release，永远跟随其默认分支 `master`；checkout 到 `examples/`，本目录 manifest 的 `path` 相对该仓根）。
- 这是 `examples_repo` 分离模式（见 [docs/examples-guard-engine.md](../../docs/examples-guard-engine.md)）的首个使用方。

## 清单

[examples_manifest.yaml](examples_manifest.yaml) 的 `scan.root` 为 DeepSpeedExamples 仓根，`include_extensions` 为 `.sh` / `.py`。`supported` 与 `unsupported` 只登记 example 入口：被 import 的库、模型定义、安装脚本、单元测试、数据准备和结果处理工具不登记，也不为它们增加 exclude 字段。公共引擎只校验已声明的 supported 条目，不要求上游每个 `.py`/`.sh` 都出现在清单中。

当前 supported 共 38 条，按 example 的有效拓扑使用 1/2/4/8 卡 runner，统一用 CANN 9.1.0 镜像。Run #29 已验证前 19 条全部成功（原有 15 条 + 第一阶段新增 4 条）；本次扩展新增 19 条（pin_memory 基准 3 条、ZenFlow 微调、OPSD 主训练/学生/教师 smoke 3 条、finetune demo、SD 蒸馏、OPSD decode 基准、推理族 9 条），全部待 Actions 验收。模型走 ModelScope（`ms_download_models` 下载后经 `GITHUB_ENV` 传本地路径），数据集优先使用仓内 fixture，并用 `overlay_args` 压到 CI 规模：

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
| `training/tensor_parallel/hf_integration/train.py` | ds_hf_autotp / 8 卡 | HF Trainer AutoTP=8，含最终 TP 权重保存 | opt-125m（ModelScope）+ 16 行 Alpaca fixture | 总共 3 步（含 2 步预热），序列 128，batch 1 |
| `training/cifar/run_ds_prmoe.sh` | ds_cifar / 2 卡 | 残差 PR-MoE，EP=2、experts=2/4 | 与普通 CIFAR/MoE 相同 | 1 epoch |
| `training/data_efficiency/variable_batch_size_and_lr/variable_batch_size_and_lr_example.py` | ds_variable_batch / 1 卡 | 动态序列打包、batch 与 LR 缩放 | 内置小模型和 1000 条合成序列 | 上游原生 2 epoch，pipeline=0 |
| `training/DeepSpeed-ZenFlow/benchmark/zf_benchmark.py` | ds_zenflow / 2 卡 | ZenFlow CPU optimizer offload 单配置 smoke | 256 维、2 层模型和合成数据 | iteration=3 × update_interval=2，共 6 次循环 |
| `compression/reasoning_aware_compression/prune.py` | ds_rac_prune / 1 卡 | Wanda 校准剪枝，显式使用 NPU | opt-125m（ModelScope）+ 8 行 prompt JSONL | 4×32 token，首三分之一 block 的线性层 |
| `applications/DeepSpeed-Chat/training/step1_supervised_finetuning/prompt_eval.py` | ds_chat_prompt_eval / 1 卡 | 真正先 SFT、再对比基础/微调模型生成 | opt-125m + 本 job 训练的 SFT 检查点 | 8 行数据训练 1 epoch；内置 6 个 prompt 各生成 8 token |
| `applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/rw_eval.py` | ds_chat_reward_eval / 1 卡 | 初始化 reward head 的 NPU forward/评分 smoke | opt-125m + 上游内置偏好对 | 2 对样本，原生 padding 到 512；不验证评分质量 |
| `training/tensor_parallel/hf_integration/train_bench_length.py` | ds_hf_bench_length / 8 卡 | 固定长度 padding/label masking + TP=8 + 最终权重保存 | opt-125m + 16 行 Alpaca fixture | 3 步（含 2 步预热），序列 128，batch 1 |
| `training/DeepSpeed-SuperOffload/finetune_zero3.py` | ds_superoffload / 2 卡 | 官方 ZeRO-Offload 模式，ZeRO-3 参数/优化器 CPU 卸载 | opt-125m + fixture 生成的本地 Parquet 目录 | 3 步，序列 128；每卡 batch 1、全局 batch 2 |
| `benchmarks/pin_memory/model_tensor_offload/bench.py` | ds_pin_memory / 1 卡 | 模型张量 CPU 卸载：pinned/unpinned 两臂对比 | 内置合成小模型 | hidden 4、2 层、各臂 3 步 |
| `benchmarks/pin_memory/activation_offload/bench.py` | ds_pin_memory / 1 卡 | 激活卸载：checkpoint + pinned buffer 两臂对比 | 内置合成小模型 | hidden 64、2 层、各臂 3 步 |
| `benchmarks/pin_memory/h2d_d2h/bench.py` | ds_pin_memory / 1 卡 | 主机/NPU 双向拷贝带宽：pageable/torch/native 三种缓冲 | 无模型 | 1 MiB × 3 次迭代；跳过需 CUDA host registration 的 native-registered 臂 |
| `training/DeepSpeed-ZenFlow/finetuning/finetune_llama.py` | ds_zenflow_finetune / 1 卡 | ZenFlow CPU optimizer offload 微调 + rank0 保存 | opt-125m（ModelScope）+ 16 行本地 Alpaca（原生 `load_dataset("tatsu-lab/alpaca")` 本地目录解析） | 1 epoch = 16 步（上游固定 512 序列长度） |
| `training/opsd/main.py` | ds_opsd / 1 卡 | OPSD 在线蒸馏：rollout → 教师 CPU 缓存 logits → streamed KL | Qwen2.5-0.5B-Instruct 学生 + Qwen2.5-0.5B 教师（ModelScope，词表全等校验）+ 8 行 prompt fixture | 3 步、max_prompt 512/response 8 |
| `training/opsd/test_student_autotp_zero3.py` | ds_opsd / 2 卡 | 学生 AutoTP=2 + ZeRO-3 forward/backward/step smoke | Qwen2.5-0.5B-Instruct（env `STUDENT_MODEL` 上游原生接口） | 12 token、1 步；OFFLOAD=0 避开 torch_adam 冲突 |
| `training/opsd/test_teacher_autotp_zero3.py` | ds_opsd / 2 卡 | 教师 AutoTP=2 + ZeRO-3 参数 CPU offload + logit cache | Qwen2.5-0.5B（env `TEACHER_MODEL`） | 12 token 单次 forward_to_cache |
| `training/deepspeed_finetune_demo/finetune_llama.py` | ds_finetune_demo / 2 卡 | Moonlight 系微调 demo：DP + ZeRO-2、shifted-label 校验 | SmolLM2-135M（ModelScope；transformers 4.42.4 的 Llama rotary 与上游 `_reset_rotary_embeddings` 兼容，Qwen2/3 会触发确定性 TypeError）+ 16 行本地 Alpaca | 3 步、序列 128、总 batch 2 |
| `training/stable_diffusion/train_sd_distil_lora.py` | ds_sd_distil / 2 卡 | 教师 CFG 引导的全 UNet 蒸馏（文件名含 LoRA 但源码无 LoRA） | stable-diffusion-v1-5 完整 Diffusers 组件（ModelScope whitelist 下载，约 5.5 GB）+ 8 幅 64×64 本地图文（原生 `load_dataset("poloclub/diffusiondb","2m_first_10k")` 本地目录解析） | 64×64、FP32（教师/学生 dtype 一致性）、ZeRO-2、3 步 |
| `training/opsd/benchmarks/bench_decode_1p1r.py` | ds_opsd_decode / 1 卡 | raw decode+sampling 与 HybridEngineRollout.generate 双路径计时 | Qwen3-0.6B（ModelScope）+ 合成 token | 16 prompt token、4 new token、2 次迭代；无吞吐阈值 |
| `benchmarks/inference/bert-bench.py` | ds_legacy_inference_bench / 1 卡 | BERT DS inference engine + accelerator 计时 + fill-mask | bert-base-cased（ModelScope） | fp32、4 次 trials |
| `benchmarks/inference/gpt-bench.py` | ds_legacy_inference_bench / 1 卡 | OPT DS inference engine + generation 计时 | opt-125m（ModelScope） | fp32、8 token × 4 trials |
| `inference/huggingface/text-generation/ds-hf-compare.py` | ds_hf_ds_compare / 1 卡 | HF baseline 与 DS engine 生成一致性（断言 2 match / 0 mismatch） | opt-125m（ModelScope） | float32、2 个 prompt、12-16 token |
| `inference/huggingface/fill-mask/test-bert.py` | ds_fill_mask_bert / 1 卡 | BERT-large fill-mask + DS inference engine | bert-large-cased（ModelScope 同名本地目录，上游硬编码 ID） | 单次 forward；WORLD_SIZE=1 |
| `inference/huggingface/fill-mask/test-electra.py` | ds_fill_mask_electra / 2 卡 | ELECTRA 显式 injection_policy 的推理 TP=2 | google/electra-base-generator（ModelScope 同名本地目录） | 单次 forward |
| `inference/huggingface/fill-mask/test-roberta.py` | ds_fill_mask_roberta / 2 卡 | RoBERTa 显式 injection_policy 的推理 TP=2 | roberta-large（ModelScope 同名本地目录） | 单次 forward |
| `inference/huggingface/translation/test-t5-base.py` | ds_t5_translation / 2 卡 | T5 翻译 + 推理 TP=2（三 injection suffix 预检） | t5-base（ModelScope 本地资产视图，仅 config/generation_config 元数据限制 8 token） | 单句翻译 |
| `benchmarks/opsd/benchmark_hybrid_engine_rollout.py` | ds_hybrid_rollout / 1 卡 | HybridEngine rollout 容器 + 原生 NPU inference ops（NPUInference builder 预检） | opt-125m（ModelScope）+ 合成 token | bf16、1 batch、4 response token、2 次迭代 |
| `inference/huggingface/automatic-speech-recognition/test-wav2vec2.py` | ds_asr_ctc / 1 卡 | Wav2Vec2 CTC 前向 smoke（非 LibriSpeech 准确率） | wav2vec2-base-960h（ModelScope 同名本地目录）+ 2 条合成 1 秒 16 kHz 音频（原生 `load_dataset("librispeech_asr","clean",split="test")` 本地目录解析） | 2 条音频、WER 仅要求有限 |

**启动方式的选择**：`cifar10_deepspeed.py` 的 main() 无条件读 launcher 注入的 `LOCAL_RANK` 并调 `init_distributed()`，因此普通 CIFAR 经支持 `$@` 的上游 `run_ds.sh` 启动。CIFAR MoE 的上游脚本不透传 `$@`，项目 runner 按原配方复刻两卡 launcher、EP=2 和 MoE 参数，再追加 CI overlay。DS-Chat 官方 training_scripts 硬编码 1.3B～66B 模型且不透传任意参数，项目 runner 等价执行 `deepspeed --num_gpus 1 main.py <overlay_args>`。AutoTP equivalence 同样复刻上游 `run_gpu.sh` 的 1/3/4 卡三次启动与 loss 比较，但显式传入 ModelScope 本地模型。offload_states 与 `--hf_baseline` inference 不依赖 launcher，直接运行 `.py`。

多卡条目启动前会检查 `ASCEND_RT_VISIBLE_DEVICES`：MoE、PR-MoE、ZenFlow 基准、SuperOffload、OPSD 学生/教师 smoke、finetune demo、SD 蒸馏与三条 TP=2 填词/翻译推理入口至少需要 2 卡，AutoTP equivalence 至少需要 4 卡，两条 HF AutoTP 入口需要 8 卡；runner 未注入时按所需卡数选择从 0 开始的设备列表。这个变量只过滤子进程可见的物理 NPU 并把它们重新映射为进程内的逻辑设备，不负责创建 worker；实际 rank 数仍由 `deepspeed --num_gpus` 决定。各配方使用不同 master port，AutoTP 三次子运行也各自分配端口。pin_memory 三条基准与 OPSD decode 不经 deepspeed launcher：上游 driver 自带子进程编排或单进程语义，项目 runner 直接以选定设备执行原文件。

bf16_master_weight 与 pipeline_parallelism 曾进 supported，CI 实测其源码硬绑 CUDA（`torch.cuda.set_device` / `autocast(device_type="cuda")` / `--backend nccl`），装包无法解决，已移回 unsupported（需 patch，次轮候选）。

DeepSpeed-Chat 的 `--data_path local/jsonfile` 从 `applications/DeepSpeed-Chat/data/{train,eval}.json` 读取（JSON Lines，字段 `prompt`/`chosen`/`rejected`）；`scripts/setup_example.sh` 在对应 profile 下把 [fixtures/](fixtures/) 里的 8 行 fixture 拷到该目录。上游 `setup.py` 的 `find_packages(include=['dschat'])` 无法安装缺少根 `__init__.py` 的 namespace package，因此 setup 不依赖其空 editable wheel，而是把 `applications/DeepSpeed-Chat` 源码根目录写入 `PYTHONPATH` 并立即执行 `import dschat` 验证。上游 [issue #813](https://github.com/deepspeedai/DeepSpeedExamples/issues/813) 也记录了 DeepSpeed-Chat 在切换执行/缓存上下文后发生模块解析错误，但不是本次完全相同的报错。模型经 `ms_download_models` 从 ModelScope 下载到本地，路径写入 `GITHUB_ENV`（`OPT_125M_PATH` / `QWEN3_06B_PATH`），overlay_args 引用本地目录。setup 按上游 requirements/setup.py 显式安装依赖，但不会从 PyPI 覆盖镜像的 `torch + torch_npu` 或 `$TARGET_ROOT` 中的 DeepSpeed 源码；安装后会校验 `deepspeed.__file__` 位于目标源码树。

CIFAR 单卡、两卡 MoE 和 PR-MoE 共用 `ds_cifar` profile。setup 从固定 revision 的 ModelScope 镜像下载 CIFAR-10 zip，先校验 SHA-256，再解压到上游脚本使用的 `training/cifar/data`，最后调用 torchvision 自带的官方逐文件 MD5 清单复核。这样保留 example 的 `download=True` 原始逻辑，但完整数据已存在时不会访问 CI 中超时的 Toronto 源；其余 `deepspeed` profile 不承担这次约 170 MB 的下载。

Run #22 中 10 条已有 9 条通过；两卡 MoE 已完成两个 rank 的 HCCL 初始化并创建 EP=2 group，首次 forward 才在 DeepSpeed `sharded_moe._capacity()` 触发 `torch.compile`，随后因镜像没有 Triton 后端而报 `ModuleNotFoundError: triton`。这是 DeepSpeed 0.19.7 将 MoE helper 从 TorchScript 改为 `torch.compile` 后产生的可选编译路径（[issue #7835](https://github.com/deepspeedai/DeepSpeed/issues/7835)、[PR #7840](https://github.com/deepspeedai/DeepSpeed/pull/7840)）；后续 [PR #7875](https://github.com/deepspeedai/DeepSpeed/pull/7875) 的 fallback 无法捕获 `torch.compile` 在首次调用时才发生的懒编译失败。项目 runner 因此仅对 CIFAR MoE 命令设置 `TORCH_COMPILE_DISABLE=1`，让该 helper 走 eager；两卡 launcher、HCCL、MoE 和 EP=2 训练语义均保留，也不要求为一个未由 example 声明的可选优化安装版本敏感的 Triton-Ascend。

当前 83 条尚未接入的 example 列入 unsupported：同一逻辑 example 的启动包装尽量合并到主入口；每条上方保留一行中文注释，说明用途和当前未接入原因。本次扩展同时清理了上游 master 已删除的 5 条 stale 登记（compression 的 bert/gpt2/cifar 旧入口与 MoQ run_glue.py），并把 5 条纯编排/分析配套物移出清单。原有 124 条非 example 配套项（包括 `applications/DeepSpeed-Chat/chat.py` 启动包装和库/测试目录）已移出清单，不代表从上游删文件。真正执行训练、推理、评测或基准的入口仍保留，即使暂时需要已有检查点或外部服务。独立扫描可能把未登记的配套文件列为清单差异，但不影响公共引擎调度；supported 路径不存在时 manifest-check 仍会立即失败。

新增配方通过 CLI、launcher、依赖版本和 fixture 接入，不修改上游 Python/启动脚本，也不为新增条目安装 API monkey patch。HF、动态 batch 和 ZenFlow 的运行缓存/报告写入本次 CI 输出目录；HF config 根据上游模板在输出目录生成。RAC 只通过正常 `import torch_npu` 注册设备后运行原入口，使用校准 forward 的 Wanda 路径，避免仅在 CPU 上完成 magnitude 剪枝却误判 NPU 通过。OPT 的前馈层名为 fc1/fc2，不匹配上游 `scope=mlp` 的命名筛选，因此这里使用 `scope=all`。PR-MoE 与普通 MoE 均局部禁用可选的 `torch.compile`。

HF setup 用上游慢速 tokenizer 和各入口自身的 SupervisedDataset 单进程生成共享数据缓存，并检查 fixture 在序列长度 128 下仍保留训练 label，八个 rank 只读该缓存；RAC setup 检查本地 prompt 可以组成完整的 4×32 token 校准窗口。新增依赖安装锁定镜像 torch/torch_npu 和被测源码 DeepSpeed，并检查 accelerator 为可用 NPU；ZenFlow 与新增 CPU offload profile 在 setup 编译/加载 CPU Adam 扩展，提前暴露工具链问题。动态 batch 没有 step/epoch 裁剪参数，保留完整的原生小规模运行，不用超时终止冒充成功。原有 15 条已在 #28 全绿，但扩展阶段仍不启用 schedule。

Run #27 使用 DeepSpeed `v0.19.7`、transformers `4.57.6` 和 accelerate `1.15.0`。HF AutoTP 的上游 config 使用 `WarmupDecayLR`，`warmup_num_steps: auto` 由 HF 的 `--warmup_steps` 填充；原 CI overlay 设为 0，被 DeepSpeed 的正整数校验拒绝。现在设为 2，与调度器内部最小有效预热长度一致；总训练仍为 3 步，TP=8、模型、数据和上游 config 模板不变。后续 `ERR99999` 是此次 Python 异常后的伴随日志，不据此判定为 NPU 算子不兼容。

Run #29 的 19 条配方及其 publish-result 已全部成功。本次远程验收改为 38 条训练/推理 job 与对应 publish-result 全部成功；结果发布成功本身不等于训练通过。新增条目失败时依据真实日志定位，不通过改写上游源码兜底。

### 第一阶段新增配方的语义和验收

- **生成评测**：在同一 job 内用原 SFT `main.py`、8 行 fixture、1 epoch 训练并保存小检查点，然后执行原 `prompt_eval.py`。基础模型与微调模型是不同的目录，不将基础模型复制两份冒充微调。验收要求 SFT 成功、检查点存在，并完成内置 6 个 prompt 的两组 greedy generation。
- **奖励评分**：原 `rw_eval.py` 调用 `create_critic_model(..., rlhf_training=False)`，从基础 OPT 构造模型并新建评分头，不恢复训练后的 reward head。本条只看护两组偏好样本的 NPU forward 与有限分数输出，不能宣传为训练检查点恢复，也不要求随机头满足 good > bad。
- **固定长度 HF AutoTP**：沿用已通过的 OPT/TP=8/3 步/2 步预热，但调用 `train_bench_length.py` 自己的固定 padding 和 label masking。setup 预生成独立工作目录中的 `dataset_dict128.pkl`，不会复用 `train.py` 的 `dataset_dict.pkl`。其 sibling `utils.py` 顶层导入旧 `openai_object`，因此只在此 profile 安装 `openai==0.28.1`；训练不调用 OpenAI 服务，不需要 API key。验收包括三步训练和上游强制执行的最终 TP 权重保存。
- **SuperOffload 目录入口**：采用上游允许的 `zerooffload` 模式，生成两卡 ZeRO-3 配置，CPU 参数/优化器 offload、BF16、每卡 micro batch 1、全局 batch 2、GA=1、pin_memory=false。**不启用 `super_offload=true`，不声称验证 superchip 原生 SuperOffload 或 pinned-memory 优化。** 模型走 ModelScope，数据由已有 16 行 Alpaca fixture 生成本地 `train.parquet` 目录，原 `load_dataset(directory)` 直接读取，无 shim。`--attn_implementation eager` 避免默认 FlashAttention CUDA 依赖；`--bench_steps 3` 真正停止训练，`--warmup_steps 1` 是计时预热而非 LR 预热。上游 CPUAdam 将实际 LR 固定为 0.001，`--lr` 只影响标记，所以 CI 同样填写 0.001，不伪装成低 LR 已生效。不传 `--save_checkpoint`，避免当前上游 rank0-only 分支调用分布式保存的挂起风险；本条不覆盖检查点保存。验收要求 world size 2、ZeRO-3/CPU offload 生效，并完成三个有限 loss 的训练 step。

### 第二阶段扩展配方的语义和验收

- **pin_memory 三条基准**：上游 driver 自带两臂（或多模式）子进程编排与 `ARMRESULT/DRIVERRESULT/RESULT` JSON 行输出；项目 runner 直接以单卡执行 driver（不再套 deepspeed launcher），把 soft memlock 提升到既有 hard limit（native pinned 分配需要），并解析结果 JSON 断言 device=npu、步数与有限计时。不设置带宽/加速比阈值，pinned 臂慢于 unpinned 也算通过（看护的是功能与真实 NPU 路径）。h2d_d2h 跳过需要 CUDA host registration 的 native-registered 臂（上游自带 `--skip-native-register` 开关），保留 pageable/torch/native-unregistered 三种模式。
- **ZenFlow 微调**：单卡执行（上游末尾 rank0-only `save_checkpoint`，多卡会挂起）。数据用原生 `load_dataset("tatsu-lab/alpaca")` 的本地目录解析：CI 工作目录内放置 `tatsu-lab/alpaca/train.parquet`（16 行 fixture 生成），配合 `HF_DATASETS_OFFLINE=1`，任何回落 Hub 的行为都会显式失败。setup 在 offline 模式下用与运行完全相同的调用验证 16 行加载与原入口 `preprocess_alpaca`/`default_data_collator` 的 512-token 行为。断言 16 个有限 loss、"Training complete!"、真实 DS checkpoint 与 tokenizer 文件。
- **OPSD 主训练**：单卡、学生 ZeRO-0 + BF16、教师 ZeRO-3 参数 CPU offload、autotp_size=1、HybridEngineRollout 默认 `module.generate()` 路径（不启用 graph capture/shared prefill）。学生 Qwen2.5-0.5B-Instruct 与教师 Qwen2.5-0.5B 词表经 `get_vocab()/get_added_vocab()` 全等校验（蒸馏 KL 要求同词表；两者 EOS 策略不同属正常）。断言 3 步 `[opsd][step N] loss=... resp_tok=...` 均有限且 resp_tok>0。save_steps=500 避免 3 步内触发保存（上游 `% save_steps` 语义）。
- **OPSD 学生/教师 smoke**：两卡 AutoTP=2 + ZeRO-3；模型路径经上游原生 env 接口 `STUDENT_MODEL`/`TEACHER_MODEL` 注入。学生臂 OFFLOAD=0（上游 config 同时设 torch_adam=true 与 CPU offload 会冲突），教师臂 OFFLOAD=1（参数 offload，无优化器）。上游日志里的 `mem=` 字段调用 `torch.cuda.max_memory_allocated()`，CUDA 未初始化时按 torch 2.9 语义返回 0，不报错，也不作为 NPU 显存证据；断言只看 STUDENT_OK/TEACHER_OK 标记、loss 有限与 cache shape。
- **finetune demo**：两卡 DP + ZeRO-2 + BF16。模型必须用 Llama 架构的 SmolLM2-135M：上游 `_reset_rotary_embeddings` 会把 `max_seq_len_cached` 置 None，transformers 4.42.4 的 Llama rotary forward 不读该字段（Qwen2/3 的旧实现会 `int > None` TypeError，已实测复现）；profile 独立 pin `transformers==4.42.4`。setup 用真实上游 reset 函数 + tiny Llama 做 CPU forward/backward 预检。断言首 batch shifted-label/finite logits 与 3 个有限 loss、"Training complete!"。`--eval_steps 0` 避开评测路径的 `torch.cuda.empty_cache()`。
- **SD 蒸馏**：`accelerate launch --use_deepspeed` 两卡 ZeRO-2、**FP32**（`mixed_precision no`、DS fp16/bf16 均 false；上游教师 UNet `.to(accelerator.device)` 不转 dtype，bf16 输入会与 fp32 教师权重冲突）。模型走 ModelScope `AI-ModelScope/stable-diffusion-v1-5` 的完整 Diffusers 组件 whitelist（含 safety_checker，最终 pipeline 保存需要；不下载重复 .bin/root ckpt，共约 5.5 GB）。数据为 8 幅程序生成 64×64 RGB 图 + prompt 的本地 Parquet（datasets.Image feature），经原生 `load_dataset("poloclub/diffusiondb", "2m_first_10k")` 本地目录解析读取。断言 3 个有限 step loss 与训练后 pipeline 的 model_index/unet config/权重非空。
- **OPSD decode 基准**：单卡 runpy 直接执行原文件；`get_accelerator().current_device()` 返回的整数索引在 torch 2.9 + torch_npu 下经 PrivateUse1 解析为 NPU（runner 启动前用 `torch.empty().to(index)`/`torch.randint(device=index)` 实测断言）。raw decode+sampling 与 HybridEngine rollout 两条路径都必须完成且计时有限为正，不比较快慢。
- **推理族 9 条**：统一独立 profile pin `transformers==4.44.2`（4.51+ 在分布式已初始化时会把 pipeline device 重置回 CPU 权重设备；4.44.2 原生支持整数 rank 解释为 npu:rank）。硬编码 HF 模型 ID 的入口（bert-large-cased、electra、roberta、t5-base、wav2vec2）用 CI 工作目录内的 ModelScope 同名本地目录满足原生解析，运行 offline，缺失别名立即失败。所有入口经普通 `runpy` bootstrap 执行原文件并断言真实 NPU：pipeline/模型权重 device=npu、fill-mask score 有限、比较脚本 2 match/0 mismatch（原脚本 mismatch 仍 exit 0，不能只看退出码）、benchmark 4 次有限计时、HybridEngine 结果 JSON 的 device/响应长度/延迟字段、CTC 两条转录与有限 WER。T5 的本地资产视图只含 config/generation_config 元数据（max_new_tokens=8），权重与 tokenizer 仍符号链接原 snapshot；setup 预检三个 injection suffix 与 heads 可被 TP=2 整除。CTC 是合成音频前向 smoke，明确不是 LibriSpeech 识别准确率。

第二阶段 38/38 全绿前不启用 schedule。新增条目若 argparse/import 之后出现真实 NPU 算子或显存问题，按日志单独定位；不恢复已被证伪的旧 unsupported 理由，也不修改上游源码兜底。

## 触发

`deepspeed-examples.yml` 是薄触发器：

- `workflow_dispatch`：手动运行只接受可选 `target_ref`（`deepspeedai/DeepSpeed` 的分支/tag/SHA），不经过 monitor 门。留空时优先选择最新 release；release 查询失败或无 release 时，使用薄触发器显式传入的 `default_branch: master`。要测试最新源码可填 `master`，不能填该仓不存在的 `main`。`upstream_repo` / `examples_repo` 固定在 workflow 内，不再支持旧版 `target_repo` 覆盖。
- `schedule`：当前以注释保留、暂不启用。启用后由公共引擎对主仓 `deepspeedai/DeepSpeed` 做 release-only 监控（latest release tag 变化时触发，上次失败时下一周期以 `release-retry` 重试）；examples 仓无 release，始终跟随 `master`。

Run #26 的 latest-release 查询返回 HTTP 403，旧触发器未声明 `default_branch`，引擎回退到 `main`，导致 `manifest-check` 的主仓 checkout 失败；15 个 example 均未启动。这次修复只补齐项目默认分支，不改公共引擎或 example 配方，也不把查询失败当成 example/NPU 兼容问题。

## 模型缓存边界

薄触发器不向公共引擎传宿主缓存卷，模型和 CIFAR-10 在各 matrix job 的容器内准备；同一 job 的 setup 与 run 步骤可复用该容器内文件，但不保证跨 job 或跨 workflow run 复用。本看护所需模型（opt-125m 等）和 CIFAR-10 均有 ModelScope 来源，正常路径不使用 [cache-seed](../../cache-seed/README.md)；仅当 ModelScope 资产变得不可用时才按 cache-seed 流程兜底投递。

## Quick Start

`deepspeed-quick-start.yml` 看护本仓 [docs/Quick-start-Ascend.md](docs/Quick-start-Ascend.md)。该文档描述了在昇腾 NPU 上安装 DeepSpeed、验证加速器、单卡跑通 CIFAR10 示例并双卡体验分布式训练的完整流程。

- 监控信号：doc 哈希、上游 latest release、master HEAD SHA。按 doc > release > commit 优先级，任一变化触发测试。
- 测试内容：pip 安装 DeepSpeed → 安装配套 torchvision → `get_accelerator()._name == 'npu'` → CIFAR10 内联示例（ZeRO-1 + BF16，1 个 epoch）→ 双卡分布式训练（`--num_gpus 2`，HCCL 通信）。
- **当前为节约 NPU 资源，`schedule` 已注释，只保留手动 `workflow_dispatch`。**

## 已知边界

- 版本错位：DeepSpeed 安装自主仓被监控的 release 版本，examples 仓跟随 `master`，二者存在小幅错位的可能；若某 release 与 examples `master` 不兼容导致失败，按结果定位后可将 manifest 的 `target_ref` 固定或向上游反馈。
- 部分 compression 旧入口已从最新 examples master 删除，不再登记。GAN、random-LTD、ViT/ImageNet 和基础 AutoTP 等现存入口仍有必经的 CUDA 调用；多卡 runner 或替换数据不能解决设备绑定。finetune demo 已通过本地 fixture 和独立依赖配方接入，待本轮 NPU 验证。

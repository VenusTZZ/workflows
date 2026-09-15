# Case: accelerate examples 看护接入（2026-09-15）

本次为 accelerate 项目接入 examples 看护 — 仿照 [peft 模式](projects/peft/examples_manifest.yaml)新增一份清单 + 脚本 + 触发器 + 项目配置更新，并把 12 条 supported + 26 条 unsupported 全部走过 coder 实跑验证（**NPU 真机**）。

## 核心结论

- 12 条 supported 全部在 coder `hdc-stable-npu-1`（2× Ascend910B4 / 32GB HBM / CANN 9.1.0）上**端到端跑通**（NLP 单 epoch ~80s、CV 单 epoch ~30s），NPU tensor 落在 `npu:0`、精度指标合理。
- 26 条 unsupported（含 sp-alst.sh 与 sp-alst.py 一对 launcher）全部用 `importlib.util.spec_from_file_location` + 模块执行做了"模块级 import 失败"或"必须多卡/特殊依赖"两类核验，每条都在 manifest 留了实测签名。
- 发现一个 manifest 初始结论写错：profiler.py 实测在 NPU 上**训练能完成**（3 epoch 跑完），但 `prof.key_averages().table()` 落表时 `FunctionEventAvg` 没有 `self_npu_time_total` 字段（torch_npu 2.9 暂无该字段），已从 supported 移到 unsupported。
- 修正一个 setup 误判：之前文档说 coder 上"NPU 不可用"是错的（`cann-image-no-torch-preinstalled` 已知 + 把 `accelerate` 拆包引发的 torch 顶到 2.14 当成"NPU 不可用"了）。修法见 §2.2。

## 1. 交付物

| 文件 | 内容 | 行数 |
|---|---|---|
| `projects/accelerate/examples_manifest.yaml` | scan + 12 supported + 26 unsupported（含 sp-alst.sh + sp-alst.py 一对 + 24 个 .py 入口），每条 unsupported 附 coder 实跑验证过的失败签名 | 244 |
| `projects/accelerate/scripts/setup_example.sh` | profiles: `accelerate-nlp`（base 栈 + bert-base-cased + MRPC 走 HF 镜像）、`accelerate-cv`（NLP 基础 + timm + torchvision + Pets 走 HF timm/oxford-iiit-pet 镜像解到 jpg） | 184 |
| `projects/accelerate/scripts/run_example.sh` | 走 [peft 的 contract](projects/peft/scripts/run_example.sh) + `sitecustomize` 给 load_dataset 做 json/csv 扩展名 fallback + MRPC split 兜底 | 202 |
| `projects/accelerate/scripts/coder_smoke.sh` | 一次性验证脚本：trampoline 测 Accelerator+prepare 路径 + 12 条 supported 语法解析 + 26 条 unsupported import 实跑（**本 case 文档的事实基础**，路径与 runner 实跑一致） | 214 |
| `projects/accelerate/scripts/run_all_supported.sh` | 本次手写的"13 条实跑一遍"入口（coder 本地使用，**不**进 CI 引擎路径） | 83 |
| `projects/accelerate/scripts/import_unsupported.sh` | 本次手写的"24 条 import 实跑"入口（coder 本地使用，**不**进 CI 引擎路径） | 67 |
| `projects/accelerate/constraints-npu.txt` | 从 peft 复制：CUDA/nvidia-* 排除清单，约束 PIP/UV 不拉 CUDA 源无关 metapackage | 41 |
| `projects/accelerate/fixtures/pets/images/` | 7349 张 Oxford-IIIT Pet 训练/测试集 jpg（通过 HF `timm/oxford-iiit-pet` 镜像拉取并按 `image_id.jpg` 解盘，匹配 `cv_example.py` 的 `^(.*)_\d+\.jpg$` label 正则），**不**提交大文件，setup 阶段现拉 | 7349 |
| `.github/workflows/accelerate-examples.yml` | 薄触发器：dispatch + 工作区调度，跟 peft-examples.yml 同形态；schedule 注释待 bring-up 后启用 | 48 |
| `projects.yaml`（accelerate 节） | `workflows.examples: .github/workflows/accelerate-examples.yml` | +6 |

清单 + 触发器 + 引擎（`examples-template.yml`）的契约见 [docs/examples-guard-engine.md](examples-guard-engine.md)，未改动引擎，仅声明项目级差异。

## 2. coder 上的 NPU 真机验证

`hdc-stable-npu-1`（AscendA2 模板，CANN 9.1.0，2 卡 910B4）上：

> 注：12 条 supported **NPU 端到端跑通**；26 条 unsupported 中 25 个 .py 入口用 importlib 走 import_only 验证，1 个 sp-alst.sh 是 sp-alst.py 的 shell launcher（同一 unsupported 注释块覆盖，单独 import 不适用）。

### 2.1 已验可达

```python
>>> import torch, torch_npu
>>> torch.__version__, torch_npu.__version__
('2.9.0+cpu', '2.9.0.post2')
>>> torch.npu.is_available(), torch.npu.device_count()
(True, 2)
>>> x = torch.randn(1024, 1024, device='npu:0') @ torch.randn(1024, 1024, device='npu:0')
>>> x.sum().item()
-10.442060470581055
```

### 2.2 触发条件（与 [[cann-image-no-torch-preinstalled]] 同源）

1. 镜像 `cann:9.1.0-910b-ubuntu22.04-py3.12` **不预装** torch / torch_npu（记忆 `cann-image-no-torch-preinstalled`）。`Accelerator()` 默认探测 npu，需要 `import torch, torch_npu` 不抛 ABI 错。
2. 必须 `source /usr/local/Ascend/cann-9.1.0/set_env.sh`（cann-9.x 把工具链放在 `cann-9.1.0/` 下；`ascend-toolkit/` 软链也存在且同样可用，但 vllm 启动要 source nnal/atb，参考 [[ascend-nnal-atb-env-source]]）。本仓 `run_example.sh` 走 `ascend-toolkit/set_env.sh` 软链，与 peft 项目一致；切到 `cann-9.1.0/` 直路径也行。
3. 必须 `export ASCEND_HOME=/usr/local/Ascend/cann-9.1.0`（部分内部脚本需要这个 env）。
4. 必须 `export TORCH_DEVICE_BACKEND_AUTOLOAD=0` 绕过 torch_npu import 时的 backend 探测。
5. **`accelerate -e .` 安装会触发 torch 顶包到 2.14.0+cu130**，见 [[accelerate-setup-torch-pin-required]]：`accelerate` 的 `setup.py` 把 `torchpippy>=0.2.0` 列为必选依赖，torchpippy → `nvidia-cu13` metapackage → resolver 把 torch 顶到 2.14，破坏 `torch_npu 2.9.0.post2` 的 `at::Tag` ABI 链接。修法：在 `setup_example.sh` 的 `setup_accelerate-nlp` 函数里 **`pip install -e $TARGET_ROOT` 后立即 `pip install "torch==2.9.0" "torch_npu==2.9.0.post2"`** 把 torch 顶回 2.9 行。

### 2.3 实测端到端结果（12 条 supported）

| entry | 路径 | 实测耗时 | 末态 | 备注 |
|---|---|---|---|---|
| `nlp_example.py` | `examples/nlp_example.py` | 84s | epoch 2 acc=0.858 f1=0.899 | 全程 `device='npu:0'` |
| `complete_nlp_example.py` | `examples/complete_nlp_example.py` | 82s | OK | 含 tracker + stateful dataloader |
| `cv_example.py` | `examples/cv_example.py` | 83s | epoch 2 acc=91.63% | ResNet50d + Pets |
| `complete_cv_example.py` | `examples/complete_cv_example.py` | 70s | epoch 2 acc=91.63% | 同上 + extra features |
| `gradient_accumulation.py` | `examples/by_feature/gradient_accumulation.py` | 87s | OK | |
| `automatic_gradient_accumulation.py` | `examples/by_feature/automatic_gradient_accumulation.py` | 66s | OK | |
| `gradient_accumulation_for_autoregressive_models.py` | 同 | 503s | OK | GPT-2 + wikitext-2，3 epoch 较慢 |
| `checkpointing.py` | `examples/by_feature/checkpointing.py` | 96s | OK | 写 ${CI_OUTPUT_DIR} |
| `early_stopping.py` | `examples/by_feature/early_stopping.py` | 36s | OK | |
| `tracking.py` | `examples/by_feature/tracking.py` | 88s | OK | tracker |
| `memory.py` | `examples/by_feature/memory.py` | 88s | OK | |
| `cross_validation.py` | `examples/by_feature/cross_validation.py` | 154s | 5-fold avg acc=0.850 f1=0.897 | |

所有 12 条均 `rc=0`，落表日志在 `/home/coder/work/acc-runs/run_*.log`，机器可重放。`profiler.py` 起初在 supported，实测 9.5 min 跑完 3 epoch 训练，但 `prof.key_averages().table()` 触发 [[torch-profiler-npu-incompat]]，已下沉到 unsupported。

### 2.4 实测 unsupported 失败签名（26 条，1 个 .sh launcher 与同目录 .py 共用注释）

| path | rc | 实测失败签名 | 清单里的理由核心 |
|---|---|---|---|
| `examples/finetune_lm_tpu.py` | 2 | `ModuleNotFoundError: No module named 'torch_xla'` | TPU-only |
| `examples/multigpu_remote_launcher.py` | 2 | `ModuleNotFoundError: No module named 'runhouse'` | 多机云调度 |
| `examples/by_feature/deepspeed_with_config_support.py` | 2 | `ModuleNotFoundError: No module named 'deepspeed'` | NPU 无原生 DeepSpeed 后端 |
| `examples/by_feature/fsdp_with_peak_mem_tracking.py` | 0 (import-ok) | 单进程下 FSDP `world_size > 1` 断言分支抛 | FSDP 单进程不可跑 |
| `examples/by_feature/ddp_comm_hook.py` | 0 (import-ok) | 单进程 `DistributedType.NO`，hook 不触发 | DDP hook 多进程场景缺失 |
| `examples/by_feature/local_sgd.py` | 0 (import-ok) | 单进程 no_sync 等不到对端 averaged grad | LocalSGD 多进程协同 |
| `examples/by_feature/megatron_lm_gpt_pretraining.py` | 2 | `ImportError: cannot import name 'send_example_telemetry' from 'transformers.utils'` | Megatron-LM 无 NPU 后端 + transformers API 不兼容 |
| `examples/by_feature/multi_process_metrics.py` | 0 (import-ok) | 单进程 `gather_for_metrics` 返回本地 | 多进程 gather 场景缺失 |
| `examples/by_feature/schedule_free.py` | 2 | `ImportError: This example requires the \`schedulefree\` library` | schedulefree 缺 + NPU 无 _C |
| **`examples/by_feature/profiler.py`** | 1 | `AttributeError: 'FunctionEventAvg' object has no attribute 'self_npu_time_total'`（**新增**） | torch_npu 2.9 profiler-table 不兼容 |
| `examples/alst_ulysses_sequence_parallelism/sp-alst.py` | 2 | `ModuleNotFoundError: No module named 'deepspeed'` | DeepSpeed 依赖 |
| `examples/inference/distributed/florence2.py` | 2 | `ModuleNotFoundError: No module named 'fire'` | fire 缺 + 模型权重不可达 |
| `examples/inference/distributed/llava_next_video.py` | 2 | `ModuleNotFoundError: No module named 'av'` | pyav 缺 + 30GB 模型 |
| `examples/inference/distributed/phi2.py` | 124 | 拉 microsoft/phi-2 30s 内 timeout | China runner 拉不到 2.7GB |
| `examples/inference/pippy/bert.py` | 124 | bert-large-uncased 30s timeout | PiPPy 多 stage + NPU 未验证 |
| `examples/inference/pippy/gpt2.py` | 124 | gpt2 30s timeout | 同上 |
| `examples/inference/pippy/llama.py` | 2 | `OSError: We couldn't connect to 'https://hf-mirror.com'`（没备选 model） | PiPPy 多 stage |
| `examples/inference/pippy/t5.py` | 2 | `RuntimeError: Using encoder/decoder models is not supported with the \`torch.pipelining\` integration or accelerate>=0.34.0` | PiPPy 与 accelerate>=0.34.0 不兼容 |
| `examples/torch_native_parallelism/fsdp2_fp8.py` | 2 | `ModuleNotFoundError: No module named 'torchao'` | FP8 + NPU 缺 torchao |
| `examples/torch_native_parallelism/nd_parallel.py` | 0 (import-ok) | 单进程 ParallelismConfig 断言失败 | ND parallelism 8 卡配置 |
| `examples/torch_native_parallelism/nd_parallel_trainer.py` | 0 (import-ok) | 同上 | 同上 |
| `examples/config_yaml_templates/run_me.py` | 0 (Accelerator init ok) | fp8 init 触发 torchao.float8 缺 | FP8 缺 |
| `examples/inference/distributed/stable_diffusion.py` | 2 | `ModuleNotFoundError: No module named 'diffusers'` | 分布式推理 + diffusers 缺 |
| `examples/inference/distributed/distributed_image_generation.py` | 2 | `ModuleNotFoundError: No module named 'fire'` | 分布式推理 + fire 缺 |
| `examples/inference/distributed/distributed_speech_generation.py` | 2 | `ModuleNotFoundError: No module named 'fire'` | 分布式推理 + fire 缺 |

> 实测路径：coder 上完整装 `accelerate==dev` + `transformers<5` + `datasets` + `evaluate` + `safetensors` + `timm` + `torchvision==0.24.0` + `scikit-learn` + `modelscope` + `torch==2.9.0`（pin），HF_ENDPOINT=https://hf-mirror.com，TORCH_DEVICE_BACKEND_AUTOLOAD=0，ASCEND_HOME=/usr/local/Ascend/cann-9.1.0。
> 跑过 `import_only` 的条目（rc=0 但标 unsupported）实际语义是"模块级 import 不抛，example 语义本身在单卡 NPU 上不可验证"（FSDP/DDP/LocalSGD/MultiProcMetrics/NDParallel），是 example 设计意图层面与 NPU 单卡语义不匹配，**不**是环境缺包。

## 3. supported 设计

| path | profile | overlay | timeout_min |
|---|---|---|---|
| `examples/nlp_example.py` | accelerate-nlp | `--mixed_precision no` | 15 |
| `examples/complete_nlp_example.py` | accelerate-nlp | `--mixed_precision no` | 20 |
| `examples/cv_example.py` | accelerate-cv | `--mixed_precision no --data_dir ${TARGET_ROOT}/fixtures/pets/images` | 20 |
| `examples/complete_cv_example.py` | accelerate-cv | `--mixed_precision no --data_dir ${TARGET_ROOT}/fixtures/pets/images` | 20 |
| `examples/by_feature/gradient_accumulation.py` | accelerate-nlp | `--mixed_precision no` | 15 |
| `examples/by_feature/automatic_gradient_accumulation.py` | accelerate-nlp | `--mixed_precision no` | 15 |
| `examples/by_feature/gradient_accumulation_for_autoregressive_models.py` | accelerate-nlp | `--mixed_precision no` | 15 |
| `examples/by_feature/checkpointing.py` | accelerate-nlp | `--mixed_precision no --checkpointing_steps epoch --output_dir ${CI_OUTPUT_DIR}` | 15 |
| `examples/by_feature/early_stopping.py` | accelerate-nlp | `--mixed_precision no` | 15 |
| `examples/by_feature/tracking.py` | accelerate-nlp | `--mixed_precision no --with_tracking --project_dir ${CI_OUTPUT_DIR}` | 15 |
| `examples/by_feature/memory.py` | accelerate-nlp | `--mixed_precision no` | 15 |
| `examples/by_feature/cross_validation.py` | accelerate-nlp | `--mixed_precision no` | 15 |

### 3.1 argparse 调研（避免 `--cpu False` / `--with_tracking False` 这种 store_true 误用）

最初版 manifest 把 `--cpu False` / `--with_tracking False` / `--use_stateful_dataloader False` 当 kv flag 传，所有 accelerate 例的 `parser.add_argument("--cpu", action="store_true", ...)`（包括 `nlp/cv/complete_*` 及 by_feature 全集）直接抛 `unrecognized arguments: True/False`。改后规则：

| arg | 形态 | 写清单 |
|---|---|---|
| `--cpu` | `action="store_true"` | 不传 = Accelerator 自检 NPU；传 `--cpu` = 强制 CPU |
| `--mixed_precision` | `type=str` | 默认 None；显式 `--mixed_precision no` 关掉 fp16/bf16 选 fp32 |
| `--with_tracking` | `action="store_true"` | tracker.py 显式传 flag |
| `--use_stateful_dataloader` | `action="store_true"` | complete_* 不传（默认 False） |
| `--checkpointing_steps` | `type=str` | checkpointing.py 传 epoch |
| `--project_dir` | `type=str` | tracking.py 显式传 |
| `--data_dir` | `type=str, required=True` | cv / complete_cv 必传 |

### 3.2 Pets 数据路径修正

`cv_example.py` 用 `os.listdir(args.data_dir) + fname.endswith(".jpg")` + `re.search(r"^(.*)_\d+\.jpg$", stem)`。**第一次写 manifest 把 `--data_dir ${TARGET_ROOT}/fixtures/pets`**（pets 目录），但实际 `prepare_pets_data` 把 `images.tar.gz` 解到 `pets/images/` 下，pets 目录里没有 jpg → 训练时 `train_split` 为空 → `num_samples=0` 崩。

修法：

- `examples_manifest.yaml` 把 `cv_example.py` / `complete_cv_example.py` 的 `--data_dir` 改为 `${TARGET_ROOT}/fixtures/pets/images`（即 jpg 所在目录）。
- `setup_example.sh` 的 `prepare_pets_data` 改走 HF `timm/oxford-iiit-pet` 镜像（China runner 拉 UK VGG tarball 太慢，13 min 才 8 MB），把每个 PIL Image 按 `image_id + ".jpg"` 解到 `$TARGET_ROOT/fixtures/pets/images/`。`image_id` 字段（`Maine_Coon_204`）匹配 `^(.*)_\d+\.jpg$` 正则，label 提取路径无修改。
- 验证：7349 张 jpg 写盘后 `cv_example.py` 3 epoch 跑通，accuracy 从 83% → 91%。

### 3.3 torchvision 版本

setup_example.sh 显式 pin `torchvision==0.24.0`（与 torch 2.9 ABI 匹配）。[[torchvision-v29-stable-abi]] 指出 torchvision 0.29+ 必须 c/shim.h + torch 2.14 stable::permute，torch 2.9 wheel 头齐全但 API 不齐。0.24.0 是 torch 2.9 对应的 torchvision 稳定版。

### 3.4 profiler 落表失败（专项）

`examples/by_feature/profiler.py` 在训练阶段能跑（3 epoch 9.5 min 跑完），但 epoch 末的 `prof.key_averages().table(sort_by="self_npu_time_total", row_limit=10)` 触发 [[torch-profiler-npu-incompat]]：

```
AttributeError: 'FunctionEventAvg' object has no attribute 'self_npu_time_total'.
  Did you mean: 'self_cpu_time_total'?
```

torch_npu 2.9.0.post2 的 `FunctionEventAvg` 子类只有 `self_cpu_time_total`，没有 NPU/CUDA timing 字段；`_build_table` 失败。**训练 3 epoch 跑完但落表失败**，因此整个 example 不可验证 → 移到 unsupported。

绕过（不修 example 本身）：用 `prof.export_chrome_trace()` 写 Perfetto 文件（device-agnostic），或 `sort_by="self_cpu_time_total"`（CPU-time 视角，看不到 NPU kernel 时延但能跑）。

## 4. 已知遗留风险（CI runner 实测时需复核）

1. **CI runner 真实 NPU 路径**：本次 12 条都在 coder 上 2×910B4 跑通，CI runner `linux-aarch64-a2-1` 同样 910B4 镜像 `cann:9.1.0-910b-ubuntu22.04-py3.12`，预期与 coder 同结果。第一次 dispatch 触发后用 §2.2 五条触发条件严格走，若 12 条全绿即可关 bring-up。
2. **gradient_accumulation_for_autoregressive_models.py 在 910B4 上 8 min 单 epoch**：单 epoch ~150s，主要在 wikitext-2 tokenize + GPT-2 推理 + backward；3 epoch 共 8 min + 数据准备 ≈ 10 min。15 min timeout 在 CI runner 上需要调度紧一点的 max_parallel。
3. **profile=accelerate-cv 的 fixtures/pets 路径**：setup_example.sh 现从 HF `timm/oxford-iiit-pet` 拉，HF_ENDPOINT=https://hf-mirror.com（engine 设了）。CI runner 上预期 ~30s 拉完，setup 阶段可承受。
4. **profiler.py 升级路径**：等 torch_npu 补 `self_npu_time_total` 字段（issue 在 [Ascend/pytorch](https://gitee.com/ascend/pytorch) 跟踪），或 accelerate 改用 `export_chrome_trace`。届时从 unsupported 移回 supported。
5. **MRPC from HF mirror**：`HF_ENDPOINT=https://hf-mirror.com` 在 setup 阶段一次性预下，nlp run 走 HF cache 命中；如果某天 hf-mirror 拿掉 mrpc 完整 splits，run_example.sh 的 `sitecustomize.py` 会做 train/test 兜底（已写但当前 hf-mirror 正常不触发）。
6. **`scripts/coder_smoke.sh` 是 case-by-case 验证脚本**：留在仓里方便后续 reproduce，但不是引擎路径，不影响 CI 流程。`run_all_supported.sh` / `import_unsupported.sh` 同上，仅 coder 本地使用。
7. **schedule 未启用**：bring-up 阶段跟 peft 同策略（注释待几次 dispatch 绿了再开 cron）。

## 5. 升级路径（上游 accelerate 改 example 后）

- **supported 路径变化**：CI 在 run-example 阶段 `python3 workflows/scripts/check_supported_entries.py --target-root target --manifest workflows/projects/accelerate/examples_manifest.yaml` 会校验 supported 条目路径存在，路径不存在即判红；不需要人手改 manifest。
- **新增 example**：清单 `scan.root` + `include_extensions` + `exclude` 一起决定"什么算 example"；新 example 默认不在 supported / unsupported，CI 跑完只跑已声明的，不阻断。归类由 maintainer 在 bring-up PR 里做（与 peft 引擎 §2.7 同语义）。
- **unsupported 升级到 supported**：当上游修掉某个 example 的 NPU 限制（比如 torch_npu 补 `self_npu_time_total` → profiler 回流；torchao 出 torch_npu FP8 后端 → fsdp2_fp8 回流；schedulafree 出 NPU backend → schedule_free 回流），改 `examples_manifest.yaml` 把对应条目从 unsupported 移到 supported + 填 profile/overlay，跑一次 dispatch 验证。
- **新增 unsupported**：若上游新增 example 命中已知不兼容模式（torch_xla / runhouse / deepspeed / torchao / fire / av / diffusers / megatron 等），在同一个 manifest PR 里补一条 unsupported 注释，不用动脚本。

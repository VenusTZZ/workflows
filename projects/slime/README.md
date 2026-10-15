# slime examples guard

## 架构：上游 release 信号 × gitcode fork 被测

slime 上游 `THUDM/slime` 发布 release（当前 v0.3.2）但没有任何昇腾支持；
昇腾适配在 gitcode fork [Ascend/slime-ascend](https://gitcode.com/Ascend/slime-ascend)
（无 release，examples/ 只存在于 main/dev 分支，v0.3.2 发布分支已改为 submodule
结构）。公共引擎的 checkout 与 release 监控只认 GitHub，因此本看护的分工是：

- **触发信号与 manifest 校验树**：`upstream_repo: THUDM/slime`。schedule 由上游
  release tag 变化驱动（release-only），manifest-check 在上游 release 树上校验
  supported 路径存在性。
- **被测代码与执行树**：`setup_example.sh` 在 CI 时从 gitcode 浅克隆 fork main，
  记录 HEAD sha（`SLIME_FORK_HEAD_SHA`，写入运行日志与 GITHUB_ENV 供追溯），
  `pip install -e` 安装 fork 的 slime 包，launcher 在 fork 树内执行。
- **已知错位边界**：被测代码是 fork main HEAD，触发信号是上游 release tag —— 与
  deepspeed「主仓 release + examples 仓跟 master」同构。fork main 相对上游
  v0.3.2 的差异仅在编译期（README/CI/tests 结构调整），训练链路一致。

## 镜像与栈

- 镜像：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`
  （与 roll 同款，runner 已验证可拉起 NPU；fork py3.12 组件表官方支持 CANN 9.1.0 +
  torch_npu 2.10.0）。回退路径：`cann:9.0.0-910b-ubuntu22.04-py3.11`（fork
  Dockerfile.910b 原配）+ py311-cann9.0 wheel。
- 栈 pin（出处：fork `scripts/ascend_script/quick_install.sh` +
  `docker/npu_docker/v0.3.0/Dockerfile.910b...` + 官方组件表）：
  sglang v0.5.13 源码（pyproject_npu.toml）→ torch/torch_npu 2.10.0、torchvision 0.25.0 →
  sgl-kernel-npu `2026.08.21`（py312-cann9.1.0-910b-aarch64，GitHub release 直下）→
  mbridge@89eb1088 → Megatron-Bridge@dev_rl → Megatron-LM@1dcf0dafa →
  MegatronAdaptor@f707a3b6 → TransformerEngineNPU@47d60449 → triton-ascend 3.2.1 →
  Transformers（fork requirements 未 pin；#8 实际解析为 5.8.1）→ `pip install -e` fork →
  按 fork 的真实目录名应用 `docker/npu_patch/v0.3.0/` 补丁。
  #8 日志中 `Megatron-LM`、`TransformerEngineNPU`、`Megatron-Bridge` 三组补丁因
  仓库名与补丁目录名不同而被静默跳过；setup 现显式映射到 `megatron`、
  `transformer_engine_npu`、`megatron-bridge`，并在必需补丁缺失时立即失败。TE 补丁
  本身包含 RMSNorm 初始化修复，因此不再额外 cherry-pick 重复改动。
- 外部源码克隆失败时的回退链：sglang / Megatron-LM 走 gitcode `gh_mirrors` 镜像；
  MegatronAdaptor / TransformerEngineNPU / fork 本体原生就在 gitcode。

## 日志与步骤间配置

- #8 完整日志超过 14 MB，其中约 97% 是 Transformers 5.8.1 在 4 个转换进程中重复
  输出的 `Accessing ... image_processing_*_fast` 兼容别名 warning。setup 在转换时及
  后续运行步骤设 `TRANSFORMERS_VERBOSITY=error`，压掉重复 warning 并保留错误级日志。
- setup 同时把 fork、模型、转换 checkpoint、fixture 等路径写入 workspace 下的
  `deps/slime-example.env`；运行步骤先 source 此文件，再使用 GitHub 的 `GITHUB_ENV`
  传递值，避免容器 runner 未传递 file-command 环境变量时直接失败。

## 缓存事实

- 引擎的 `actions/cache` 只承担 monitor-state（release 信号状态），不承担模型缓存。
- 模型下载走 runner 既有持久 ModelScope 缓存（与 roll 同机制，`/data/ci-cache`），
  Qwen2.5-0.5B-Instruct 首次冷下载后跨 run 命中。
- pip 走集群缓存代理（`select_pip_index` 探测 `cache-service.nginx-pypi-cache`），
  失败回退清华源。
- 模型全走 ModelScope，数据用仓内 fixture。`sgl-kernel-npu` Release ZIP 则由
  `cache-seed/slime/curl_seeds.yaml` 一次性投递到共享缓存；三个 example setup
  校验命中后直接安装，不再分别访问 GitHub。#17 证明直连可能持续慢于五分钟；
  缓存缺失或损坏时仍允许 job 本地 Range 续传兜底，但不算缓存验收通过。

## supported 清单与压缩口径

### 已接入（fully-async #12、OPD #13 全绿；ReTool 待手动验收）

| example | runner | 配方出处 | 压缩口径 |
|---|---|---|---|
| `examples/fully_async/run-qwen2.5-0.5B-fully_async.sh` | a2-4 | fork NPU nightly `tests/tests_npu/nightly_CI/test_qwen2.5_0.5B_fully_async_short_npu.py`（昇腾已验证） | actor 1 + rollout 3、TP/PP/CP/EP 全 1、`--num-rollout 2`、response 1024（nightly 为 8192）、数据换仓内 16 行 fixture；显式关闭两类 dropout 并沿用 temperature 0.8 对齐训练/推理策略 |
| `examples/on_policy_distillation/run-qwen3-8B-opd.sh` | a2-8 | fork NPU ST `tests/tests_npu/st/test_qwen2.5_0.5B_opd_sglang_npu.py` | 同型号 Qwen2.5-0.5B student/teacher，Ray 中 actor 4 + rollout 3、独立 teacher 1 卡；2 个 rollout、fixture 16 行、response 1024。#13 已通过。 |
| `examples/retool/retool_qwen3_4b_rl.sh` | a2-8 | fork 自带 ReTool `.sh`（暂无 NPU 回归先例） | ModelScope Qwen3-4B-Instruct-2507、训练 4 卡 + 推理 4 卡隔离、推理 TP2、两次 rollout、8 行本地工具调用数学 fixture；待远程验证。 |

执行不直接跑上游 `.sh`（硬编码 `/root` 绝对路径、无 `"$@"` 透传），由
manifest `overlay_args` 承载 CI 训练配方；`run_example.sh` 映射引擎无法透传的
`MODEL_TYPE`/`TRAIN_SCRIPT`、管理 OPD teacher 与 ReTool 模块路径，然后调用 fork 自己的
`slime.utils.external_utils.command_utils.execute_train()`（ray start/submit、
NPU 资源注入都由 fork 框架代码完成）。模型 `--hf-checkpoint` 走 ModelScope
本地路径，`--ref-load` 用 fork 自带 `tools/convert_hf_to_torch_dist.py`
现转的 `_torch_dist` 目录（torchrun 4 procs）。

OPD 的上游 `.sh` 会启动 teacher、轮询健康接口，再启动 Ray 训练；但脚本写死
`/root/Qwen3-32B`、`nvidia-smi` 和 CUDA 卡号，不能直接用于昇腾 CI。项目 launcher
复刻相同步骤：在第 8 张 NPU 启动本地 SGLang teacher，确认
`/health_generate` 就绪后，以前 7 张卡执行 fork 的 `execute_train()`；结束时关闭
teacher。训练参数留在 manifest。环境仍是现有华为 CANN 镜像 + setup 按 fork
`quick_install.sh`/NPU Dockerfile 安装的 NPU 栈，不拉取新的国外容器镜像。

ReTool 的上游 `.sh` 包含 CUDA 检测、Ray 清理、四卡 colocate/TP2 配置与固定模型、
数据和 W&B。#16 在四卡共享布局下出现推理 health-check 超时和训练 actor 初始化断言；
项目 runner 改用八卡中训练/推理各四卡隔离，仍保留 TP2 和自定义生成/奖励函数。setup 补齐
`jinja2`、`psutil`，从 ModelScope 下载模型并用 fork 工具转为 `_torch_dist`。
`tool_sandbox.py` 会在容器内执行模型生成的 Python 代码，因此仅使用本地 fixture
与最小权限 workflow；这条尚无 fork NPU 端到端先例，需以远程训练日志验收。

`slime-examples #14` 的 ReTool 已完成环境安装与 SGLang 启动，但四卡 Ray 训练
actor 在创建时抛出无源码位置的 `AssertionError`。日志显示 fork 的
`execute_train()` 给 worker 写入 `0,1,2,3,4,5,6,7`，与 runner 实际四卡不一致；
项目 runner 现通过 `extra_env_vars` 覆盖为可见的四卡，并在再次失败时最多打印
四份 Ray worker 错误日志的短尾段。此修复需下一次手动 workflow 验证。

### 本次新增（待 NPU 手动验收）

| example | runner | 保留语义与 CI 规模 |
|---|---|---|
| `retool/retool_qwen3_4b_sft.sh` | a2-2 | 官方训练专用 SFT 模式、原生 assistant token mask；Qwen3-4B-Instruct-2507，8 行 messages fixture、TP2、两次 rollout |
| `on_policy_distillation/run-qwen3-8B-opd-megatron.sh` | a2-8 | 真实 Megatron 教师 checkpoint 自蒸馏；Qwen2.5-0.5B、训练 4 + rollout 4、TP2、两个短 rollout |
| `train_infer_mismatch_helper/run-qwen3-4b-mis.sh` | a2-8 | 保留原生 MIS/TIS 函数及 TP2/CP2；Qwen3-4B-Instruct-2507、训练/推理各 4 卡、两个短 rollout |

supported 共 11 条。SFT 的 `--debug-train-only` 来自官方 SFT 脚本，只关闭不需要的推理服务器，仍执行真实监督训练。其模型转换改用两个进程；setup 检查每行的监督 token mask。Megatron OPD 不启动 SGLang teacher，使用本地转换的 checkpoint 执行真实教师前向。MIS 使用 fork 原生 `mis.yaml`，不关闭 context parallel 或偏差校正。

另外新增多任务评测、multi-agent、Strands 和 Search-R1。多任务评测保留 `eval-config` 的原生多奖励分发，用数学与选择题 fixture 运行两个任务。multi-agent 使用有 thinking 输出的 ModelScope Qwen3-4B，保留源码固定的 5 solver、5 rewriter、1 selector。Strands 使用 Qwen3-4B-Instruct-2507，不重复套 chat template，固定 `strands-sglang==0.3.2` 并在 setup 检查真实 Python 工具调用依赖。两类 agent 必须通过原生 rollout/gradient dump 与日志的核心行为验证，否则失败。

Search-R1 使用上游检索器 CLI 的 `--retriever_name bm25`，不经过 dense Encoder 的 `model.cuda()`。CPU 工具检索配合 NPU 模型训练，原生 BM25 索引由八篇本地文档生成；使用 Java 11、Pyserini 0.25.0 和 CPU FAISS 包。runner 管理原生 `/retrieve` 服务并要求 rollout 真正访问搜索服务，仅服务 readiness 成功不能算 example 成功。

Geo3K 多轮视觉训练也已接入：ModelScope `Qwen/Qwen3-VL-2B-Instruct`、八张 128×128 本地图像、训练/推理各两卡 TP2。其原生 bridge 直接从 HF 目录加载权重，不使用文本模型转换器；保留原配置的三轮上限与 `calc_score` 环境。setup 用原生 Dataset/processor 检查 `pixel_values`、`image_grid_thw`，结果检查要求真实视觉张量、工具评分反馈和 mask 的 1→0→1 切换，证明图像与至少两轮交互均实际进入训练轨迹。

新增配方与接入限制记录在 manifest 注释中。本地没有 NPU，新增八条的硬件成功状态需要下一轮 Actions 确认。

### 剩余未接入入口

逐条理由见 `examples_manifest.yaml` 的 unsupported 注释。fork 专有视觉/GLM 入口受双仓路径校验约束；共同的 Geo3K 多轮入口已接入。

### 硬阻塞（不接入）

- `coding_agent_rl`：8 节点 64 rollout 卡 + E2B/远程 Docker sandbox + `SLIME_HEAD_HOST`
  反连 + `MOCK_SANDBOX` 引用的 `sandbox_mock.py` 不存在 + 模型/数据无公开源。
- `delta_weight_sync`：GLM-4.7-355B-A32B（约 700GB）16 节点 128 卡 + 共享盘，零覆盖入口。
- `tau-bench`：`generate_with_tau.py` 硬编码 gemini 用户模拟器且 `GEMINI_API_KEY`
  被源码强制覆写为 "NONE"，不改源码无法本地化。
- `geo3k_vlm`（单轮）：上游 release 树只有 35B-A3B + GitHub 私有 Megatron fork 的入口；
  通用版仅存在于 fork main，manifest-check 无法校验。

## 数据与模型

- 模型：全部走 ModelScope（Qwen2.5-0.5B-Instruct 已验证；ReTool 使用
  [Qwen3-4B-Instruct-2507](https://modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507)，需首次 CI 确认实际下载与转换）。
- 数据：`fixtures/ci_dapo_16.jsonl` 为 16 行 DAPO-Math-17k 同 schema 真实样本
  （`prompt` 为 `[{content, role}]` 消息列表、`label` 为字符串答案，deepscaler
  本地 CPU 校验）；ReTool 使用 `fixtures/ci_retool_math_8.jsonl`，提示模型先调用
  `code_interpreter` 再以 boxed 答案作答。不依赖 hf-mirror 数据集下载。

## 运行时环境契约

`run_example.sh` 按 fork ascend 启动器导出：`ASCEND_RT_VISIBLE_DEVICES`（按 profile
注入）、`RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1`、
`CUDA_DEVICE_MAX_CONNECTIONS=1`、HCCL 端口段、
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`（fork 原值；与 roll 的 vLLM
CaMemAllocator 场景不同，这里不 unset）、`WANDB_MODE=offline`。

`slime-examples #11` 首次进入训练后，fork 的 CI logprob 检查发现训练与 rollout 差异为
2.628（阈值 0.1）：Megatron 默认 attention/hidden dropout 均为 0.1，而 fork nightly
配方将两者设为 0。manifest 现对齐该配方并固定 rollout temperature 0.8。该次 run 的
fixture rollout reward 与 advantage 均为 0。#12 已完成 2 个 rollout 与训练 step、整体成功，
但 reward / advantage / loss / grad_norm 仍为 0，因此只能视为运行链路验证，
不能证明现有数学 fixture 产生了有效的优化信号。OPD 的纯蒸馏后处理设计上返回
标量 reward 0，学习信号应来自 teacher-logprob 的 KL 项；远程验收需检查该项与
teacher 请求成功，不能只看退出码。

## CI 接口

- 手动触发：`gh workflow run slime-examples`（`target_ref` 留空 = 上游最新 release，
  本次源码审查版本 v0.4.0）。
- artifact：`slime-examples-<run_id>-<job_index>`（公共引擎统一命名）。
- schedule：bring-up 期注释关闭；手动轮次全绿后由维护者决定启用（启用后由上游
  release tag 变化或上次 scheduled failure 触发）。
- `max_parallel: 1`（bring-up 阶段保持；避免 4 卡和 8 卡任务并发争用资源）。

## 本地验证

```bash
python -m pytest tests/test_check_supported_entries.py -q
python -m unittest discover -s projects/slime/tests -p 'test_*.py'
python -m pytest projects/roll/tests/test_roll_examples.py -q  # regression: shared-engine contract
bash -n projects/slime/scripts/setup_example.sh
bash -n projects/slime/scripts/run_example.sh
```

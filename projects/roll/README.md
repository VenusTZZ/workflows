# ROLL Examples NPU Guard

对上游 [alibaba/ROLL](https://github.com/alibaba/ROLL) 的 `examples/` 做 NPU 兼容性看护。
实现与 [TRL examples](../trl/README.md) 同构：本目录提供 manifest 与项目脚本，
[.github/workflows/roll-examples.yml](../../.github/workflows/roll-examples.yml) 是调用公共
[examples-template.yml](../../.github/workflows/examples-template.yml) 的薄触发器；公共引擎负责
release-only 监控、matrix 调度、结果校验与状态写回，本仓不修改公共引擎。

## 覆盖基线

- 对账快照：上游 latest release `v0.4.0`（commit `581046abce143e643221f8e292895c46a3b20112`），2026-10-10 全量重审。
- 对账单位是「完整运行配置」`.yaml`；根目录 `start_*_pipeline.py` 与各目录 `run_*.sh`
  是公共 launcher / 别名，不重复登记。
- `examples/` 快照共 146 个 YAML：`examples/config/` 的 9 个共享片段由一个
  unsupported 目录条目汇总，其余 137 个文件登记为 `20 supported` + `117 unsupported`；
  加上目录汇总项，manifest 共 `20 supported` + `118 unsupported`。
  相对旧 117-YAML 快照：新增 34 个上游文件全部登记；5 个 distill/onpolicy 旧路径
  在 v0.4.0 已迁移到 `examples/distill/` 下新路径，按新路径重新登记。

## Supported：二十条昇腾链路

| Example | CI 参数 | Runner | 覆盖 |
|---|---|---|---|
| `examples/qwen2.5-0.5B-agentic/agentic_rollout_sokoban.yaml` | manifest `overlay_args` | `linux-aarch64-a2-1` | 单环境多轮交互、vLLM-Ascend 生成、轨迹组装，无训练 |
| `examples/qwen2.5-0.5B-agentic/agentic_val_sokoban.yaml` | manifest `overlay_args` | `linux-aarch64-a2-2` | Sokoban 交互、vLLM rollout、GRPO advantage、FSDP2 backward + optimizer step |
| `examples/ascend_examples/qwen3_8b_rlvr_fsdp2.yaml` | manifest `overlay_args` | `linux-aarch64-a2-4` | RLVR 数据预处理、vLLM 生成、math_rule 奖励、reference log-prob、FSDP2 更新 |
| `examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake.yaml` | manifest `overlay_args` | `linux-aarch64-a2-2` | FrozenLake 环境 GRPO（gymnasium toy_text 纯 Python），待实跑 |
| `examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake-pg_var.yaml` | manifest `overlay_args` | `linux-aarch64-a2-2` | TOPR 策略梯度变体（agentic_actor_pg_worker），待实跑 |
| `examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake-pg_var_is_correct.yaml` | manifest `overlay_args` | `linux-aarch64-a2-2` | TOPR + 训推一致性校正（train_infer_correction），待实跑 |
| `examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake_gigpo.yaml` | manifest `overlay_args` | `linux-aarch64-a2-2` | FrozenLake GiGPO 步进优势（StepEnvManager），待实跑 |
| `examples/qwen2.5-0.5B-agentic/agentic_val_sokoban_gigpo.yaml` | manifest `overlay_args` | `linux-aarch64-a2-2` | Sokoban GiGPO（step_envs 片段），待实跑 |
| `examples/qwen2.5-0.5B-agentic/agentic_val_sokoban_lora.yaml` | manifest `overlay_args` | `linux-aarch64-a2-2` | LoRA 微调（all-linear/rank32，fsdp2 is_lora 分支），待实跑 |
| `examples/qwen2.5-0.5B-agentic/agentic_val_sokoban_dynamic_batching.yaml` | manifest `overlay_args` | `linux-aarch64-a2-2` | 训练/推理动态 batching（reference 覆盖回 hf_infer），待实跑 |
| `examples/qwen2.5-0.5B-agentic/agentic_val_sokoban_native.yaml` | manifest `overlay_args` | `linux-aarch64-a2-2` | 原生 Sokoban 环境步进 REINFORCE（AgentNativeStepEnvManager），关闭异步，待实跑 |
| `examples/qwen2.5-0.5B-agentic/agentic_val_sokoban_ppo.yaml` | manifest `overlay_args` | `linux-aarch64-a2-2` | PPO（gae + FSDP2 critic 与 train 共置卡 0），待实跑 |
| `examples/qwen2.5-0.5B-agentic/agentic_val_sokoban_sao.yaml` | manifest `overlay_args` | `linux-aarch64-a2-2` | SAO（skip_obs_gae + sao pg_variant + critic），待实跑 |
| `examples/qwen2.5-0.5B-agentic/agentic_val_sokoban_agent_runner.yaml` | manifest `overlay_args` | `linux-aarch64-a2-2` | AgentRunner 2.0（ProxyEnvManager + GEMRunner，gem-llm 已随 setup 安装），待实跑 |
| `examples/docs_examples/example_grpo.yaml` | manifest `overlay_args` | `linux-aarch64-a2-4` | 文档 GRPO，保留算法，数学奖励域，待实跑 |
| `examples/docs_examples/example_gspo.yaml` | manifest `overlay_args` | `linux-aarch64-a2-4` | GSPO 序列级重要性比率（importance_sampling=seq），待实跑 |
| `examples/docs_examples/example_ppo.yaml` | manifest `overlay_args` | `linux-aarch64-a2-4` | GAE + critic，与 actor_train 共置卡 0-1，warmup=0，待实跑 |
| `examples/docs_examples/example_topr.yaml` | manifest `overlay_args` | `linux-aarch64-a2-4` | 用公开配置补全权重与 GAE critic，原生 ActorPGWorker/topr，待实跑 |
| `examples/qwen2.5-7B-rlvr_megatron/rlvr_config_dynamic_batching.yaml` | manifest `overlay_args` | `linux-aarch64-a2-4` | actor/reference 保留 token 动态 batching，待实跑 |
| `examples/qwen2.5-7B-rlvr_megatron/rlvr_lora_fsdp2.yaml` | manifest `overlay_args` | `linux-aarch64-a2-4` | REINFORCE + rank32 LoRA，保留 train/infer 低秩权重，待实跑 |

阶段一已证明不依赖 `quay.io/ascend/roll` 的环境基线：国内 CANN 基础镜像启动 job，
再据 v0.3.0 的官方升腾环境文档安装固定版本组合 torch_npu / vLLM-Ascend / ROLL。
阶段二在此基线上恢复两卡 Agentic train 与四卡 RLVR。2026-10-10 扩展把 0.5B Agentic
家族里所有算法/环境/训练模式变体一次性接入（frozen_lake 系 4 条、sokoban 变体 7 条），
全部复用 `agentic_train_npu` profile 与两卡布局，每条 overlay 都在 v0.4.0 树上实测
Hydra compose 通过。异步变体（`*_async`、`*_as1`）与 rollout mock dump 在 CI 必须
关闭其特性开关，关闭后与对应主配置等价，不单独登记；`examples/config` 共享片段仍按
目录汇总。另将六条可原生配置覆盖的 RLVR 算法/训练模式接入四卡 profile；默认大模型、
Megatron 和多域远程数据不再单独作为这些条目的否定依据。数学 fixture 覆盖数学奖励域，
不覆盖上游 code_sandbox/LLM judge 域。PPO 的 critic_warmup 覆盖为 0，保证单步测试进入
actor 更新分支。所有新增条目待下一次手动 workflow 验收，本次未修改 workflow 的 schedule。

## 压缩策略

与 PEFT 一样，模型、数据、规模等 CI 参数集中放在 manifest 的 `overlay_args`，
数据保留在 `fixtures/`，不再维护本仓 `configs/`。
ROLL release 的三个 launcher 只接受 `--config_path` / `--config_name`，
不会向 Hydra 转发额外覆盖参数，因此 `scripts/run_example.sh` 内置配置适配逻辑：
用 Hydra Compose API 读取该 release 的原始 YAML（包括 defaults），应用 manifest 覆盖项，
再把临时 YAML 交给原始 launcher。上游 YAML、example、pipeline 和 worker 源码均不修改。

覆盖项使用 Hydra 原生语法：`++key=value` 添加或覆盖，`~key` 删除；
设备表达式保留为字符串，例如 `++actor_infer.device_mapping="list(range(0,1))"`。
切换 Megatron → FSDP2 时先删除再重建 `strategy_config`，避免混入上游后端专用键。
临时配置生成在上游 `examples/` 下的唯一目录中，运行结束后自动清理；
解析后的副本保存在 `$CI_OUTPUT_DIR/resolved_config.yaml`，可在 runner 日志对应路径诊断。
Sokoban 的交互模板直接继承上游 defaults，CI 只覆盖环境数量和动作/token 上限。
二十条任务共用 `Qwen/Qwen2.5-0.5B-Instruct`，ModelScope 使用当前 runner 可用缓存：

- 单卡 rollout：单环境、最多两次动作、序列 256。
- 两卡 Agentic train：单步 2 环境组、序列 256；FSDP2 train 占卡 0，vLLM 卡 1，hf reference 与 train 同卡。
- 四卡 RLVR：2 prompt x 2 采样、序列 192，8 行本地 math_rule fixture；FSDP2 train 0-1、vLLM 2、hf reference 3，reward 单副本。

这些是一次完整 pipeline step 冒烟测试，不是训练收敛复现，不验证原始大模型指标。

## 运行与结果

- matrix 并行上限为 4（原为 2），与 Accelerate 的默认配置一致；实际并发取决于空闲 runner。
- schedule：`45 */6 * * *`（release-only，仅在 release tag 变化或上一轮失败时重跑）；#20 已三条全绿，定时看护已恢复。
- 手动触发：`target_ref` 留空测最新 release，或显式指定 `main` / tag / SHA。
- 镜像：国内 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`；
  setup 固定安装 torch 2.10 + torch_npu 2.10.0.post4 + vLLM 0.23.0 +
  vLLM-Ascend 0.23.0rc1 + triton-ascend 3.2.1，再从被测 checkout 安装 ROLL。
- 多卡设备：setup 按 profile 注入 `ASCEND_RT_VISIBLE_DEVICES`（rollout=0，train=0,1，rlvr=0,1,2,3）；
  run 入口只在未注入时填单卡默认 0，并统一 unset 与 vLLM-Ascend 内存池冲突的 `PYTORCH_NPU_ALLOC_CONF`。
- 模型：仍通过 ModelScope `snapshot_download` 解析，复用 runner 现有
  `~/.cache/modelscope`。本阶段不新增 `cache-seed/roll`。

## 已知边界

- [roll-examples #105](https://github.com/cosdt-ci-test/workflows/actions/runs/38128320886)
  的 20 条任务均在 setup 的 vLLM 依赖解析阶段缺少 `packaging`，尚未进入 example。
  setup 现先显式安装 `packaging==25.0` 并检查解析 API，再读取 wheel 元数据；
  不再假设基础镜像或 `--no-deps` 安装会提供该依赖。完整 NPU 运行仍待下一轮验证。
- v0.4.0 的配置兼容：所有 supported 条目显式覆盖 `transfer_backend.backend_name=null`，
  使用本地 DataProto 传输。该 release 默认启用 TransferQueue（16 个存储单元），
  单卡 runner 的空闲 CPU 不足；即使 CPU 足够，`protocol.py` 也明确禁止 NPU 使用 RemoteBatch。
  RLVR 的 `tag_included` 必须匹配 fixture 的 `tag: math_rule`，不能使用 `source: ci_math`；
  v0.4.0 的 `update_dataset_domain` 不再对未匹配 tag 回退到 `math_rule`。
  以上修复对应 [roll-examples #90](https://github.com/cosdt-ci-test/workflows/actions/runs/37701575713)，
  配置修复已在 [#91](https://github.com/cosdt-ci-test/workflows/actions/runs/37713723839) 三条全绿；
  本次 manifest 覆盖参数迁移仍需下一次 Actions 验证。

- 首次实现只覆盖单节点 A2。A3 / Ascend 950、多机、SGLang、Megatron、外部沙箱、
  WebShop、SWE、视频/音频/VLM、私有 OSS/CPFS 数据集均不在 supported 范围。
- 公共引擎只校验并调度已声明的 supported 条目，不负责自动发现上游新增 example。

## 本地配置验证

使用公共 `scripts/check_supported_entries.py` 对上游 checkout 检查 manifest，
两个项目脚本可用 `bash -n` 检查语法。Hydra Compose 适配器已内置在
`run_example.sh` 中，运行时会解析上游 YAML/defaults 并保存 resolved_config.yaml。
整理时已对 v0.4.0 的 20 条配方验证配置组合、算法参数与 launcher 退出码。
各配方的功能与接入限制记录在 manifest 注释中，新增配方需完成真实 NPU 远程验收。

## Run #100 修复与文件结构

Run #100 的 20 条任务均因 PyArrow/NumPy 导入冲突失败，尚未验证新增配方的
完整 NPU 训练路径。上游 requirements_common.txt 要求 NumPy <2，并固定
datasets 3.1.0，但未约束 PyArrow，导致安装了要求 NumPy 2 的 PyArrow 26。
项目现在通过 constraints-npu.txt 对全部 setup 安装统一约束 NumPy 1.26.4、
PyArrow 20.0.0、pandas 2.2.3；保留上游 datasets 版本。setup 在启动 Ray 前
检查数据包导入与实际 Parquet 写入/读取，避免安装成功后才在所有 worker 中失败。

scripts/ 仅保留 setup_example.sh 和 run_example.sh。原 prepare_config.py 的
Hydra 参数展开、配置组合、路径检查与上游 launcher 调用全部合入 run_example.sh，
没有删除运行功能。tests/ 仅保留 test_quick_start_ascend.py 与必要的 __init__.py，
独立 quick-start workflow 不变。supported 仍为 20 条，公共引擎、上游源码及
其他项目均未修改；schedule 保持原配置。

## Run #101 安装修复

20 条任务都在 vLLM 安装阶段失败，尚未进入 example。通用 vLLM 0.23.0
wheel 的 CUDA 依赖图包含 humming-kernels[cu13]、nvidia-cutlass-dsl 等，
不能直接用于昇腾环境。setup 现在用 --no-deps 安装固定版本 vLLM 与
vLLM-Ascend，再从其 metadata 生成非 CUDA 依赖清单；torch/torch_npu 与
Triton 沿用项目的独立 NPU 配方，避免被通用 wheel 替换。

这是一套仅针对当前文本 CI 用例的依赖配方，不宣称满足通用 CUDA wheel
的所有 metadata：Transformers 保持 4.57.6，OpenCV 保持兼容 NumPy 1 的
4.11.0.86，Mistral common 不安装 image extra。视频/多模态和 CUDA 专用
kernel 仍不在支持范围；没有修改这些包或上游 example 的源码。
全程约束同时补齐 SciPy 1.16.3，避免 SciPy 1.18 与 NumPy 1.26.4 冲突。
setup 检查 SciPy sparse/optimize、Parquet 读写、Transformers 导入，以及
vLLM 实际选择的 NPU platform。20 条配方与并行上限 4 保持不变，
真实 NPU 训练仍需后续手动 Actions 验证。

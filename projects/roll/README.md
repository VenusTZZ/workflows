# ROLL Examples NPU Guard

对上游 [alibaba/ROLL](https://github.com/alibaba/ROLL) 的 `examples/` 做 NPU 兼容性看护。
`.github/workflows/roll-examples.yml` 是公共 `examples-template.yml` 的薄触发器；
本项目提供 manifest、安装/启动脚本和本地数据，不修改公共引擎或上游源码。

## 覆盖与验证状态

- 对账快照：上游 `v0.4.0`，commit `581046abce143e643221f8e292895c46a3b20112`。
- 对账单位是完整 YAML 运行配置；公共 `start_*_pipeline.py` 和 shell 启动别名不重复计数。
- 146 个 YAML 中，9 个 `examples/config/**` 共享配置片段由 scan.exclude 排除；
  空配置 `agentic_val_webshop.yaml` 不再登记为用例。
- manifest 现为 **28 supported / 108 unsupported**，原为 19 / 119。
  原有 19 条配方的解析结果保持不变，已在
  [Run #109](https://github.com/cosdt-ci-test/workflows/actions/runs/38150557380) 全部通过；
  本轮新增 9 条已完成本地配置/数据验证，**尚未通过 NPU CI 验收**。
- 不按比例凑覆盖率：关闭关键特性后与已支持配方等价的配置，不重复宣称新增能力。

## 本轮新增用例

| 配置 | 覆盖 | CI 规模与设备 |
|---|---|---|
| `qwen3-30BA3B-sft_fsdp2/sft_config.yaml` | 原生 SFT 流程，替换小型稠密模型，不宣称覆盖原 MoE 架构 | 0.5B，8 行本地监督数据，1 epoch，batch 2，单卡 FSDP2 |
| `dpo_examples/qwen2.5-3B_dpo_megatron.yaml` | 原生 DPO 偏好训练，以 FSDP2 替换 Megatron 后端 | 0.5B，8 对 chosen/rejected，1 epoch，actor/reference 双卡隔离 |
| `distill/off_policy/llm/distill_megatron.yaml` | 离线 KL 蒸馏，保留教师 logits 与蒸馏损失 | 0.5B 学生 / 1.5B 教师，8 行数据，双卡，Ray 传输 top-k logits |
| `docs_examples/example_raft_pp.yaml` | 保留上游 RAFT++ 算法配置 | 0.5B，本地 math_rule 数据，四卡，1 step |
| `docs_examples/example_reinforce_pp.yaml` | 保留上游 REINFORCE++ 算法配置 | 0.5B，本地 math_rule 数据，四卡，1 step |
| `distill/on_policy/llm/onpolicy_distill_config.yaml` | 原生在线蒸馏，学生采样后计算教师分布 | 0.5B 学生 / 1.5B 教师，四卡 runner，1 step |
| `distill/on_policy/llm/opsd_distill_config.yaml` | 原生 OPSD，自蒸馏教师使用参考解答上下文 | 同源 0.5B 模型，本地 reference_solution，四卡，1 step |
| `qwen3_agentic_gem/gem_game_guess_the_number.yaml` | 原生 StepConcat GEM 猜数字环境及训练，不走 GEMRunner | 数字 1–3，最多 2 轮，双卡，1 step |
| `qwen3_agentic_gem/gem_rg_letter_counting.yaml` | 原生 reasoning-gym 字母计数环境及训练 | 8 个环境样本，最多 1 轮，双卡，1 step |

路径均相对于上游 `examples/`。原有 19 条覆盖 rollout、Sokoban/FrozenLake、
GRPO/GiGPO/TOPR、PPO/SAO、GSPO、LoRA 与动态 batching，详细配方见 manifest。

## 配置与资产压缩方法

模型、设备、训练预算和数据路径集中在 manifest 的 `overlay_args`。
`run_example.sh` 使用 Hydra Compose API 读取上游配置及 defaults，应用覆盖参数，
生成临时 YAML，再交给原始 launcher；不修改上游配置、pipeline、worker 或模型代码。
解析后的配置保存到 `$CI_OUTPUT_DIR/resolved_config.yaml`。

- 模型来自 ModelScope：学生为 `Qwen/Qwen2.5-0.5B-Instruct`；
  独立蒸馏教师为 `Qwen/Qwen2.5-1.5B-Instruct`。下载后以本地路径传给原生流程。
- 原有 RLVR 使用 `ci_math_8.jsonl`，不改变已通过的数据。
- 新增 SFT/DPO/离线蒸馏使用 `ci_supervised_8.jsonl`，包含 instruction、output、
  chosen/rejected 和 question_zh/answer_zh，按原生编码器读取。
- 在线蒸馏/OPSD 使用 `ci_opsd_8.jsonl`；OPSD 增加明确的参考解答，不伪造教师 logits。
- SFT/DPO/离线蒸馏同时限制数据量和 epoch：它们从 dataloader 推导迭代预算，
  不能只依赖顶层 max_steps。8 行数据、batch 2、1 epoch 将完整流程限制在数个批次。
- 切换 Megatron → FSDP2 时删除并重建 strategy_config，避免混入后端专用键。
- 离线蒸馏使用原生 Ray logits 后端，不依赖 CUDA IPC；学生/教师 tokenizer 的 token ID 必须相同。
- 新 profile 在 setup 校验本地 tokenizer/fixture；GEM profile 实际检查原生 reset/step。
- 这些是训练链路冒烟测试，不复现原始大模型指标，也不验证收敛效果。

overlay_args 已逐条去重和删除冗余默认值覆盖，共精简 197 项。
每次删除均比较 v0.4.0 上完整解析后的配置和值类型；结果必须完全相同。
模型/数据、设备、后端和必要的兼容修复仍显式声明，不另建公共配置或修改引擎。

## 暂未接入的真实限制

默认大模型、远程数据或 Megatron 配方本身不作为一概否定的理由；
可以保持算法语义的条目已替换模型、数据和后端。仍保留 unsupported 的主要原因是：

- v0.4.0 的 WorkerConfig 会对非 Megatron 后端强制关闭 sequence packing；
  改成 FSDP2 后不能声称测试 packing。
- NpuPlatform.apply_ulysses_patch 返回 None，FSDP2 的 context parallel 会退化为 1；
  仅保留 SP=2 参数不代表测试了真实序列并行。
- GEMRunner 缺少 ProxyEnvManager 要求的 set_episode_context；新增两个 GEM 用例走
  原生 StepConcat 环境管理器，不掩盖该接口问题。
- ROCK/SWE/WebShop 等需要外部沙箱或服务；视频、音频、VLM、扩散等需要对应的
  真实模型、数据和推理依赖，不能用文本奖励域或文本模型替代。
- 异步/mock 配方若必须关闭自身核心功能才能跑通，不作为额外覆盖计数。
- 上游 MindSpeed/Megatron NPU 支持提案尚未合入当前基线，不引入其源码补丁。

每条具体限制见 manifest 条目前的中文注释。后续若原生能力改善，可重新评估，
不把 unsupported 当成永久不能运行。

## 运行与环境

- workflow 调用公共引擎，matrix 并行上限为 4；本轮不改并发和 schedule。
- schedule 为 `45 */6 * * *`，release-only。
- 手动触发 `target_ref` 留空测试最新 release，或指定 tag/SHA/main。
- 镜像为国内 CANN 9.1.0 / Python 3.12；沿用已经通过的固定 torch 2.10、
  torch_npu 2.10.0.post4、vLLM 0.23.0、vLLM-Ascend 0.23.0rc1、
  triton-ascend 3.2.1 组合，从被测 checkout 安装 ROLL。
- NumPy 1.26.4、PyArrow 20.0.0、pandas 2.2.3、SciPy 1.16.3 仍由原有 constraints 保护。
- setup 按 profile 分配 1/2/4 卡；run 清除与 vLLM 内存池冲突的
  PYTORCH_NPU_ALLOC_CONF。不改变原有三个 profile 的依赖版本或设备布局。
- 使用 ModelScope 当前可用缓存，但薄触发器不保证跨 job/跨 run 宿主模型缓存；
  本轮不新增 cache-seed bundle。
- artifact 名为 `roll-examples-<run_id>-<job_index>`。

## 验证与迁移边界

本地验证包括 YAML/schema/path 检查、28 条 Hydra compose/resolve、原有 19 条
配方不变断言、原生数据编码及 token 预算、ModelScope tokenizer 对齐、
原生 GEM reset/step、公共 supported 检查器与 shell/内嵌 Python 语法检查。

本地机器不提供 NPU，以上验证不等同于训练通过。本轮仅修改 workflows 的
fix/roll 分支；由用户提交并运行 NPU CI，28 条验收成功后再迁移到 ascend-docs。
公共引擎、quick-start 测试、其他项目和 ascend-docs 本轮均不修改。

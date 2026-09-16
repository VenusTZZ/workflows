# ROLL Examples NPU Guard

对上游 [alibaba/ROLL](https://github.com/alibaba/ROLL) 的 `examples/` 做 NPU 兼容性看护。
实现与 [TRL examples](../trl/README.md) 同构：本目录提供 manifest 与项目脚本，
[.github/workflows/roll-examples.yml](../../.github/workflows/roll-examples.yml) 是调用公共
[examples-template.yml](../../.github/workflows/examples-template.yml) 的薄触发器；公共引擎负责
release-only 监控、matrix 调度、结果校验与状态写回，本仓不修改公共引擎。

## 覆盖基线

- 对账快照：上游 `main` 于 2026-09-16 检出（commit `192b1a01ea61c113b2deb543f7b115783038dff8`）。
- 对账单位是「完整运行配置」`.yaml`；根目录 `start_*_pipeline.py` 与各目录 `run_*.sh`
  是公共 launcher / 别名，不重复登记。
- `examples/` 共 117 个 YAML：排除 `examples/config/` 9 个共享片段和
  `agentic_val_webshop.yaml`（仅有 Hydra defaults 的空壳）后，剩余 107 个逐一分类：
  `3 supported` + `104 unsupported`，无遗漏、无重复。

## Supported：官方三类 Ascend 路径

| Example | CI 配置 | Runner | 覆盖 |
|---|---|---|---|
| `examples/qwen2.5-0.5B-agentic/agentic_val_sokoban.yaml` | `configs/ci_agentic_train.yaml` | `linux-aarch64-a2-2` | Sokoban 环境交互、vLLM rollout、GRPO reward/advantage、FSDP2 backward + optimizer step |
| `examples/qwen2.5-0.5B-agentic/agentic_rollout_sokoban.yaml` | `configs/ci_agentic_rollout.yaml` | `linux-aarch64-a2-1` | 单环境多轮交互、vLLM-Ascend 生成、轨迹组装，无训练 |
| `examples/ascend_examples/qwen3_8b_rlvr_fsdp2.yaml` | `configs/ci_rlvr.yaml` | `linux-aarch64-a2-4` | RLVR 数据预处理、vLLM 生成、math_rule 奖励、reference log-prob、FSDP2 训练更新 |

选择依据是上游官方 Ascend 文档列出的三类能力（Agentic / Agentic-Rollout / RLVR），
并使用官方配套栈：FSDP2 训练、vLLM-Ascend 推理、HCCL 通信、训练型 Atlas A2。
Megatron 在昇腾当前不支持，因此使用 Megatron train/infer 的配置全部归入 unsupported。

## 压缩策略

不修改上游 example、pipeline 或 worker 源码；只替换输入规模。三条任务都使用
`Qwen/Qwen2.5-0.5B-Instruct`（ModelScope 预下载），把模型规模、设备数、序列长度、
环境/采样数量和训练步数压到单节点 CI 规模，但保留被看护 example 对应的
pipeline、launcher、训练/推理后端与核心路径：

- Agentic train：2 NPU（FSDP2 训练 / vLLM 推断分置），单步 2 环境组，序列 256。
- Agentic-Rollout：1 NPU vLLM，单环境多轮交互，序列 256。
- RLVR：4 NPU（FSDP2 训练 2 + vLLM 1 + 参考模型 1），2 prompt x 2 采样，
  序列 192，8 行本地 `math_rule` fixture。

这些是一次完整的 pipeline step 冒烟测试，不是训练收敛复现，不验证原始大模型指标。

## 运行与结果

- schedule：`45 */6 * * *`（release-only，首次冷启动测最新 release；仅在 release tag
  变化或上一轮失败时重跑）。
- 手动触发：`target_ref` 留空测最新 release，或显式指定 `main` / tag / SHA。
- 镜像：官方 `quay.io/ascend/roll:main-a2` 的当前 arm64 digest，在 manifest 中固定；
  容器内预装 torch 2.10 + torch_npu 2.10.0.post4 + vLLM 0.23.0 + vLLM-Ascend 0.23.0rc1，
  项目脚本只以 `--no-deps --no-build-isolation -e` 覆盖被测 ref 的 ROLL 源码。
- 模型缓存在单个 matrix job 容器内，跨 job / 跨 run 不承诺复用。

## 已知边界

- 首次实现只覆盖单节点 A2。A3 / Ascend 950、多机、SGLang、Megatron、外部沙箱、
  WebShop、SWE、视频/音频/VLM、私有 OSS/CPFS 数据集均不在 supported 范围。
- 13 条非首选 FSDP2 配置标记为“当前未纳入 / 待 NPU 实测”，不声称原理上不支持。
- 公共引擎只校验并调度已声明的 supported 条目，不负责自动发现上游新增 example。

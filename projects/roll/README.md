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
  `1 supported` + `106 unsupported`，无遗漏、无重复。

## Supported：阶段一单卡 rollout 基线

| Example | CI 配置 | Runner | 覆盖 |
|---|---|---|---|
| `examples/qwen2.5-0.5B-agentic/agentic_rollout_sokoban.yaml` | `configs/ci_agentic_rollout.yaml` | `linux-aarch64-a2-1` | 单环境多轮交互、vLLM-Ascend 生成、轨迹组装，无训练 |

阶段一先建立不依赖 `quay.io/ascend/roll` 的环境基线：用国内 CANN
基础镜像启动 job，再按 quick-start 已实测的固定版本组合安装
torch_npu / vLLM-Ascend / ROLL。两卡 Agentic train 和四卡 RLVR 暂放
unsupported，待单卡 rollout 手动验收全绿后逐项恢复。

## 压缩策略

不修改上游 example、pipeline 或 worker 源码；只通过 CI Hydra 配置压缩输入规模。
阶段一使用 `Qwen/Qwen2.5-0.5B-Instruct`，ModelScope 继续使用 runner
现有持久缓存；单卡 vLLM-Ascend、单环境、最多两次动作，序列长度 256。

这是一次完整的 rollout pipeline 冒烟测试，不是训练收敛复现，不验证原始大模型指标。

## 运行与结果

- schedule：阶段一保持注释关闭；单卡 rollout 手动验收全绿后再恢复
  release-only 定时看护。
- 手动触发：`target_ref` 留空测最新 release，或显式指定 `main` / tag / SHA。
- 镜像：国内 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`；
  setup 固定安装 torch 2.10 + torch_npu 2.10.0.post4 + vLLM 0.23.0 +
  vLLM-Ascend 0.23.0rc1 + triton-ascend 3.2.1，再从被测 checkout 安装 ROLL。
- 模型：仍通过 ModelScope `snapshot_download` 解析，复用 runner 现有
  `~/.cache/modelscope`。阶段一不新增 `cache-seed/roll`。

## 已知边界

- 首次实现只覆盖单节点 A2。A3 / Ascend 950、多机、SGLang、Megatron、外部沙箱、
  WebShop、SWE、视频/音频/VLM、私有 OSS/CPFS 数据集均不在 supported 范围。
- 两卡 Agentic train 和四卡 RLVR 是明确的后续恢复项，不声称原理上不支持。
- 公共引擎只校验并调度已声明的 supported 条目，不负责自动发现上游新增 example。

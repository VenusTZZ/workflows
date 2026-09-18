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

## Supported：三条升腾链路（阶段二）

| Example | CI 配置 | Runner | 覆盖 |
|---|---|---|---|
| `examples/qwen2.5-0.5B-agentic/agentic_rollout_sokoban.yaml` | `configs/ci_agentic_rollout.yaml` | `linux-aarch64-a2-1` | 单环境多轮交互、vLLM-Ascend 生成、轨迹组装，无训练 |
| `examples/qwen2.5-0.5B-agentic/agentic_val_sokoban.yaml` | `configs/ci_agentic_train.yaml` | `linux-aarch64-a2-2` | Sokoban 交互、vLLM rollout、GRPO advantage、FSDP2 backward + optimizer step |
| `examples/ascend_examples/qwen3_8b_rlvr_fsdp2.yaml` | `configs/ci_rlvr.yaml` | `linux-aarch64-a2-4` | RLVR 数据预处理、vLLM 生成、math_rule 奖励、reference log-prob、FSDP2 更新 |

阶段一已证明不依赖 `quay.io/ascend/roll` 的环境基线：国内 CANN 基础镜像启动 job，
再据 v0.3.0 的官方升腾环境文档安装固定版本组合 torch_npu / vLLM-Ascend / ROLL。
阶段二在此基线上恢复两卡 Agentic train 与四卡 RLVR，三条并行验收。

## 压缩策略

不修改上游 example、pipeline 或 worker 源码；只通过 CI Hydra 配置压缩输入规模。
三条任务共用 `Qwen/Qwen2.5-0.5B-Instruct`，ModelScope 继续使用 runner 现有持久缓存：

- 单卡 rollout：单环境、最多两次动作、序列 256。
- 两卡 Agentic train：单步 2 环境组、序列 256；FSDP2 train 占卡 0，vLLM 卡 1，hf reference 与 train 同卡。
- 四卡 RLVR：2 prompt x 2 采样、序列 192，8 行本地 math_rule fixture；FSDP2 train 0-1、vLLM 2、hf reference 3，reward 单副本。

这些是一次完整 pipeline step 冒烟测试，不是训练收敛复现，不验证原始大模型指标。

## 运行与结果

- schedule：阶段二仍保持注释关闭；三条全绿后再由你决定是否恢复 release-only 定时看护。
- 手动触发：`target_ref` 留空测最新 release，或显式指定 `main` / tag / SHA。
- 镜像：国内 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`；
  setup 固定安装 torch 2.10 + torch_npu 2.10.0.post4 + vLLM 0.23.0 +
  vLLM-Ascend 0.23.0rc1 + triton-ascend 3.2.1，再从被测 checkout 安装 ROLL。
- 多卡设备：setup 按 profile 注入 `ASCEND_RT_VISIBLE_DEVICES`（rollout=0，train=0,1，rlvr=0,1,2,3）；
  run 入口只在未注入时填单卡默认 0，并统一 unset 与 vLLM-Ascend 内存池冲突的 `PYTORCH_NPU_ALLOC_CONF`。
- 模型：仍通过 ModelScope `snapshot_download` 解析，复用 runner 现有
  `~/.cache/modelscope`。本阶段不新增 `cache-seed/roll`。

## 已知边界

- 首次实现只覆盖单节点 A2。A3 / Ascend 950、多机、SGLang、Megatron、外部沙箱、
  WebShop、SWE、视频/音频/VLM、私有 OSS/CPFS 数据集均不在 supported 范围。
- 公共引擎只校验并调度已声明的 supported 条目，不负责自动发现上游新增 example。

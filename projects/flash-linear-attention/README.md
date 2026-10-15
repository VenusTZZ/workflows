# 在昇腾 NPU 上使用 Flash Linear Attention

[Flash Linear Attention（FLA）](https://github.com/fla-org/flash-linear-attention) 提供线性注意力层和语言模型实现。本目录提供昇腾环境的入门指南，以及从本地文本训练语言模型的使用示例。

| 你想完成的任务 | 阅读入口 |
| --- | --- |
| 安装 FLA，检查 NPU 环境，运行一次前向和反向计算 | [Quick Start](docs/Quick-start-Ascend.md) |
| 使用自己的文本训练一个小型语言模型，保存后重新加载并续写文本 | [文本训练与生成示例](example/README.md) |

当前环境基线为单张 Ascend 910B、Linux aarch64、Python 3.11、CANN 9.0.0，以及 FLA v0.5.2 的 `[npu]` 依赖配套。具体版本、安装步骤和运行命令见各指南。CANN、Torch-NPU 和 Triton-Ascend 需要按版本配套使用，不能只升级其中一个组件。

Quick Start 已有 NPU 运行通过记录；文本训练示例也已在 2026-09-28 [完成 NPU 端到端验证](https://github.com/cosdt-ci-test/workflows/actions/runs/36403948431)，覆盖默认配置下的训练、模型保存、重载和带缓存生成。该结果限于文档列出的配套与示例配置，不代表所有模型和参数组合都已验证。

工作流接入、版本监控、结果校验和开发者测试说明见[看护维护文档](docs/examples-maintenance.md)。

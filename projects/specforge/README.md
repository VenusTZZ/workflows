# specforge

本目录是 [SpecForge](https://github.com/sgl-project/SpecForge) 的看护配套数据，不是 SpecForge 源码。看护流水线在 `.github/workflows/specforge-quick-start.yml`（端到端 smoke）和 `.github/workflows/specforge-examples.yml`（examples 看护）。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：训练加速；支持程度：基础支持；阶段 A）。

## 端到端 smoke（Quick-start）

在昇腾 NPU 上把 Quick Start 文档里所有 `#test` / `#test-setup` 代码块跑通，对照 `#test-result` 做输出断言。脚本退出码非 0 即判红。

文档覆盖（[Quick-start-Ascend.md](docs/Quick-start-Ascend.md)）：

- 镜像预装 torch 2.10.0 + torch_npu 2.10.0 + sglang 0.5.18 + CANN 9.0.0 + Python 3.11.15（`check-torch` 步校验；specforge 源码 `pip install --no-deps .` 跳过上游 `pyproject.toml` 的 sglang==0.5.14 pin）；再装 modelscope 1.37.0 + **mooncake-transfer-engine 0.3.13.post1 从阿里云源**（PyPI 上 Mooncake 的 CUDA 变体不能装；华为云 ascend 源未挂 npu flavor）；
- specforge 源码安装（`git clone <ref>` + `pip install --no-deps .`），PyPI 二进制 wheel 作为可选路径；
- `specforge --help` / `specforge train --help` CLI 自检；
- **端到端 smoke**：在 4 卡 NPU 上起 `mooncake_master` + SGLang capture server（卡 0）+ `specforge train` 1 步训练（卡 1）。Smoke 拆成 5 个独立 `#test` 块（`smoke-download-model` / `smoke-apply-patches` / `smoke-start-mooncake` / `smoke-start-sglang` / `smoke-train`），每段独立失败定位；mooncake + sglang 是 `nohup` 后台进程跨段复用，store `model_path` 把下载路径传到 sglang / trainer 段。

## Examples 看护

注册 specforge 上游 `examples/` 下的训练入口到看护清单，ci 端到端跑通则视为 supported；其余按"为什么跑不了"分类列出。规格与 [docs/examples-guard-engine.md](../../docs/examples-guard-engine.md) 一致（release-only monitor + manifest-check + run-example matrix + validate-results + save-monitor-state）。

清单 [`examples_manifest.yaml`](examples_manifest.yaml)（2026-09-16 静态分析 + `--plan` 反证）：

- **supported**（1 条）：
  - `examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-npu.yaml` — 与 Quick-start 端到端 smoke 同 recipe，CI 已绿；overlay 把训练规模压到 1 步、序列长度 512、锚点 32、`nproc_per_node=1`（让 4 卡 runner 的 capture 卡 0 / trainer 卡 1 / buffer 卡 2-3 排得下）。
- **unsupported**（83 条，按阻塞原因分 9 类，每条带理由注释）：
  - **A. 模型超单卡 32GB HBM**（dense > 8B / MoE > 35B 等 33 条；recipe 默认 `bfloat16` 加载 + 无 FP8 NPU 路径 + 无 FSDP offload 默认值）；
  - **B. trainer nproc_per_node > 2 → 4 卡塞不下**（17 条，recipe 默认 8 trainer + capture + buffer）；
  - **C. managed-local full-stack recipe**（10 条，yaml 锁死 7-14 trainer + 1-6 capture server，overlay 改不了 capture_servers[]）；
  - **D. 两节点脚本**（5 条；shell 硬性校验 NUM_NODES=2 + HEAD_IP + 共享盘）；
  - **E. AMD target**（2 条，依赖 ROCm + FA4 后端，NPU 路径不可用）；
  - **F. offline recipe**（5 条，`data.hidden_states_path` 指向预计算 `.ckpt` 文件，必须先跑 online 把特征写出来；当前 online 没在 NPU 上跑通，offline 不可达）；
  - **G. data_regeneration 脚本**（2 条，是 specforge train 的"上游"步骤，不是 training 入口本身）；
  - **H. utility 脚本**（1 条，`sync_distributed_checkpoints.py` 是 2-node checkpoint relay helper，单独跑无业务行为）；
  - **I. 单卡理论可跑（≤1B/4B，1 nproc）但未在 coder NPU 上端到端验证**（11 条；`specforge train -c <recipe> --plan` 在 coder npu-1（torch 2.9 + torch_npu 2.9）能产出 producer/consumer process plan，验证了 CLI + YAML schema 解析，但 specforge 强约束 torch==2.13.0 而 CI 镜像只能装 torch 2.10.0，0.5B/1B 在 NPU 上的真实 step 行为未实测）。

依赖线 pin 见 [scripts/setup_example.sh](scripts/setup_example.sh) 注释；runner / 镜像 / 超时逐条来自清单（与 peft 同模式）。

## 触发

`specforge-quick-start.yml` 接受：

- `schedule`：每 6 小时轮询上游一次（详见 [Quick-start-Ascend.md](docs/Quick-start-Ascend.md)）。
- `workflow_dispatch`：手动触发。

`specforge-examples.yml` 接受 `workflow_dispatch`（bring-up 阶段 schedule 注释，与 peft / trl 同策略；几轮手动绿后启用 cron）。

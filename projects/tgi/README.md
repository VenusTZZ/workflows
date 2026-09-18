# tgi

本目录看护 Text Generation Inference（TGI）在昇腾 NPU 上的 Quick Start：
从源码构建（Rust launcher/router + Python server）、部署推理服务，对
Qwen3-0.6B 做端到端 `/generate` 验证——**单卡基线 + 双卡 HCCL 张量并行**
（`--num-shard 2`）都跑，双卡回复必须与单卡基线逐字一致。

## 看护对象与前提

- **上游仓库**：`VenusTZZ/text-generation-inference`（TGI 官方仓库的 fork）。
  官方 release 尚未包含 Ascend NPU 支持，NPU 适配以 **release** 形式发布在
  该 fork 上（从 `feat/ascend-npu` 分支打 tag）。
- **看护信号**：quick-start 引擎轮询 fork 的最新 release + 本目录
  `docs/Quick-start-Ascend.md` 的 hash。**因此每次把官方上游的新改动合入
  `feat/ascend-npu` 并验证通过后，必须在 fork 上打一个新 release**，
  否则流水线没有可测的新 ref。
- **看护范围**：文档第 1（环境检查）、3（依赖与工具链）、4（构建安装）、
  5（启动服务并验证推理：单卡基线 + 双卡张量并行）节，以及第 2 节的模型
  缓存命中。**不看护**：`npu-smi info` 的机器相关数值输出（版本号、功耗、
  温度等）。

## 绿灯含义

**绿灯 = 用被测 tag 的源码构建出 TGI，在两张 NPU 上真实跑通 `/generate`：
单卡回复非空，双卡（`--num-shard 2`，HCCL 张量并行）回复与单卡基线逐字
一致。** 不是"编译通过"，也不是"进程起来了"——`smoke-tp2` 块轮询 `/info`
就绪后发真实请求，并断言双卡输出与单卡贪心解码基线完全相同。
（若 CI 栈上出现极罕见的浮点序差异导致贪心输出不同，把一致性断言降级为
"非空"并记录——目前本地双栈均逐字一致。）

## 目录内容

| 文件 | 用途 |
|---|---|
| `docs/Quick-start-Ascend.md` | 被看护的快速入门文档（`#test`/`#test-result` 标签契约见 `docs/markdown_doc_test_label.md`） |
| `tests/test_quick_start_ascend.py` | 文档端到端测试（`NPU_READY=true` 时执行文档标签块） |
| `tests/test_project_contract.py` | 静态契约测试（注册表条目、标签配对、关键内容），无卡可跑 |
| `../cache-seed/tgi/ms_seeds.yaml` | Qwen3-0.6B 的 ModelScope 缓存 plant 清单 |

## 版本基线

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| 镜像 | `cann:9.1.0-910b-ubuntu22.04-py3.12`（SWR ascendhub） | 与 peft/tensorflow 同基线 |
| CANN / Python | 9.1.0 / 3.12 | 镜像自带 |
| torch / torch_npu | 2.9.0 / 2.9.0.post2 | 华为 ascend 源 |
| transformers / kernels | 4.57.6 / 0.5.0 | `kernels` 必须 0.5.0（构建后端 lockfile） |
| 模型 | Qwen/Qwen3-0.6B（约 1.2 GB） | cache-seed plant，CI 零下载 |

runner 为 `linux-aarch64-a2-2`（两张 910B4）：单卡基线与双卡 HCCL 张量
并行同机验证。双卡的硬前提是**被测 tag 包含 launcher 的 LOCAL_RANK 注入**
（transformers 原生 TP 读取 `LOCAL_RANK`；官方上游没有这段代码，因此
release 必须从 fork 的 `feat/ascend-npu` 分支打）。四卡（`--num-shard 4`）
本地已验证（见 TGI fork 的 `docs/npu/hccl-multicard-design.md`），暂不在
quick-start 看护范围。

## 触发契约

- 薄触发器 `.github/workflows/tgi-quick-start.yml` → 共享引擎
  `quick-start-template.yml`；cron 在 bring-up 调绿前保持注释，手动
  `workflow_dispatch` 不写 monitor 状态。
- 模型/数据零下载：`ms_seeds.yaml` 由 `cache-seed` workflow dispatch
  投递后，经 `container_options` 的 modelscope 缓存卷命中本地。

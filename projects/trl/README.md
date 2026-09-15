# trl

本目录是 [TRL](https://github.com/huggingface/trl)（huggingface/trl，主流大模型后训练库：SFT / DPO / GRPO / PPO 等）的看护配套数据，不是 TRL 源码。流水线位于 [`trl-quick-start.yml`](../../.github/workflows/trl-quick-start.yml) 和 [`trl-examples.yml`](../../.github/workflows/trl-examples.yml)。注册信息见根目录 [`projects.yaml`](../../projects.yaml)（分类：训练加速；支持程度：新兴适配；阶段 A；upstream：`huggingface/trl`）。

上游目前没有昇腾 NPU CI，本仓按 [`docs/guarding-examples.md`](../../docs/guarding-examples.md) 的看护标准接入阶段 A 看护：用 quick-start 流水线看护「昇腾上可安装、可跑通最小后训练流程」的基线，用 examples 流水线看护上游示例代码的可用性，确保 TRL 与最新版本在昇腾 NPU 上保持兼容。

## 看护范围

- **Quick-start 文档测试**：[`docs/Quick-start-Ascend.md`](docs/Quick-start-Ascend.md) 遵循 [`docs/markdown_doc_test_label.md`](../../docs/markdown_doc_test_label.md) 标签契约（`#test` / `#test-setup` / `#test-result` 配对、id 唯一），覆盖单卡昇腾 NPU 完整流程：环境与 NPU 检查、安装 TRL（PyPI 二进制）、经 ModelScope 自动下载 Qwen2.5-0.5B-Instruct（网络环境无法直连 HuggingFace 时经 ModelScope 获取模型）、用 ModelScope 数据集 `HuggingFaceH4/ultrafeedback_binarized` 跑通最小 SFT LoRA 与偏好优化 DPO LoRA 双方法、验证两份 LoRA 适配器产物。`tests/test_quick_start_ascend.py` 基于 `src/workflows/markdown_doc_test_base.py` 端到端执行文档。文档含版本矩阵，与 CI 镜像 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12` 对齐。
- **Examples 清单看护**：[`examples_manifest.yaml`](examples_manifest.yaml) 由 `scripts/bootstrap_manifest.py` 扫描上游 `examples/` 生成（TRL 新布局为目录式 example，`scan.unit: mixed`）。当前 supported 为 2 条单卡（`linux-aarch64-a2-1`、`npu_devices: '0'`、profile `peft_lora`）小规模 example：`examples/dpo_reduce_hallucinations`（DPO LoRA）与 `examples/tpo_ultrafeedback`（TPO LoRA），均用 `overlay_args` 压到 CI 规模（8 条 fixture、`max_steps 2`、输出到 `${CI_OUTPUT_DIR}`），其余全部列入 unsupported 并注明原因。`scripts/setup_example.sh` / `run_example.sh` 遵循「项目运行脚本契约」，只修改 CI 工作区内的目标仓副本，绝不向上游写操作。
- **触发方式**：quick-start 已开启 schedule 轮询（`cron: '0 */3 * * *'`）并保留 `workflow_dispatch` 手动触发；examples 维持 `workflow_dispatch`（schedule 注释保留在 YAML 中）。开启 schedule 后，monitor 对比上游信号（examples 树 commit / latest release tag / main HEAD，或文档 hash），有变化才占用 NPU；上次失败时下个周期自动重试。

## 能力覆盖矩阵

- **已验证**：二进制与源码安装（quick-start）、单卡最小 SFT LoRA（quick-start）、DPO LoRA 多模态 VLM（examples）、TPO LoRA（examples）。
- **未验证**：GRPO、PPO 与 reward modeling、全参微调、多卡分布式、蒸馏与 KTO/ORPO/CPO。上游 examples 当前没有非 vLLM 的最小 GRPO 入口，补覆盖需自行编写并先推上游。
- **共享缓存**：examples 挂共享根 `/data/ci-cache/modelscope`，DPO / TPO 两个 matrix job 复用同一份权重，避免重复下载；quick-start 挂 `trl` 子目录，与 examples 的模型缓存互不借用。
- **残留不自动清理**：examples 不调用缓存清理（挂共享根与自动 purge 互斥，属本仓约定），残留损坏分片由人工定向清理。
- **单卡串行**：runner `linux-aarch64-a2-1` 单卡，两个 supported example 串行排队，冷启动下载只发生一次，之后命中缓存。

## 看护周期计划

### 日常检查（每个轮询周期）

- 确认 monitor 状态与失败重试（`*-retry` 信号）是否收敛：偶发失败应在下一周期重试后转绿，不应连续多个周期失败。
- 失败时按 artifact 中的 `result.json`（quick-start：`trl-quick-start-<run_id>`；examples：`trl-examples-<run_id>-<job_index>` 与 `trl-manifest-check`）与 Actions 日志定位：先分清是环境问题（镜像 / 依赖安装 / 网络）、上游代码变更，还是本仓清单 / 文档过期。
- manifest-check 中 supported 条目的 path 在磁盘消失会立即判红，属最高优先级处理项。

### 定期维护（每周）

- 复核 manifest 差集：`manifest_check_result.json` 中的 `new_paths` 视为「提示有待办」——评估能否在 CI 跑通，能则补充 supported 条目（path / profile / runner / npu_devices / image / timeout_minutes / overlay_args，必要时扩展 `setup_example.sh` 分支），否则保留在 unsupported 并加注释说明原因；`stale_paths` 及时从清单移除或修正。
- 核对文档与上游版本的一致性：版本矩阵（CANN / torch / torch_npu / transformers / trl / 模型）是否与最新 release 匹配；注意上游 examples 已改为目录式组织，昇腾社区旧文档中的 `examples/scripts/dpo.py` 等路径已过时，不可照抄。
- 清理 unsupported 中的 stale 条目（上游已删除的路径）。

### 版本更新全面测试（上游 release 时）

- 上游发布新 release 时（monitor 的 release 信号），对 quick-start 文档与**全部 supported examples** 做全量回归。
- 根据回归结果更新文档版本矩阵、examples 清单，必要时更新配套镜像（当前为 CANN 9.1.0）。
- 若新版引入不兼容变更，先在本仓记录已知问题（见下），再评估是锁版本、改清单还是向上游提 issue。

## 问题响应与 issue 跟踪机制

- **跟踪对象**：
  - 上游 [huggingface/trl](https://github.com/huggingface/trl) 的 NPU / Ascend 相关 issue 与 PR（关键词：NPU、Ascend、torch_npu、CANN）；
  - 本仓两条流水线（`trl-quick-start` / `trl-examples`）的失败记录。
- **响应路径**：
  1. 流水线判红 → 从 `result.json` 与 Actions 日志定位失败层（环境 / 上游代码 / 清单 / 文档）；
  2. 属上游问题 → 检索上游 issue，已有则跟踪并引用，没有则向上游提 issue（附 NPU 环境信息与最小复现）；
  3. 属本仓问题 → 修文档 / 清单 / 脚本，手动 `workflow_dispatch` 验证转绿；
  4. 短期无法修复 → 记入下方已知问题清单；若某 supported example 持续不可跑通，先移回 unsupported 并注释原因，避免长期占红。
- **修复验证**：任何修复都通过手动触发对应流水线确认全绿后，才视为闭环。

## 已知问题清单

| 问题 | 状态 | 说明 |
| --- | --- | --- |
| 上游 issue [#5495](https://github.com/huggingface/trl/issues/5495)：CANN 8.5.0 + TRL 0.25.1 GRPO backward 报错 | 持续观察 | 当前看护镜像为 CANN 9.1.0（`cann:9.1.0-910b-ubuntu22.04-py3.12`），未复现该问题；supported 清单暂未纳入 GRPO 类 example，后续纳入 GRPO 或切换 CANN 版本时需持续观察此问题是否复现。 |
| 昇腾社区旧文档路径过时 | 已规避 | 上游 examples 已改为目录式布局，`examples/scripts/sft.py` / `examples/scripts/dpo.py` 等旧路径不存在，文档与清单一律按新布局维护。 |

## quick-start 引擎兼容性核查结论

对照 TRL 需求逐项核查了共享引擎 [`.github/workflows/quick-start-template.yml`](../../.github/workflows/quick-start-template.yml) 的输入契约（`project` / `test_runner` / `image` / `container_options` / `timeout_minutes` / `upstream_repo` / `doc_url` / `doc_path` / `test_command`）与引擎固定行为（monitor 信号优先级、cluster pip/uv 镜像 env、cache I/O 在 ubuntu-latest 上的分工）：

- TRL 的差异点（单卡 `linux-aarch64-a2-1`、CANN 9.1.0 镜像、davinci0 设备挂载 + ModelScope 缓存挂载、`huggingface/trl` 上游、本仓文档路径、`python -m unittest tests.test_quick_start_ascend -v 2>&1` 测试入口）**全部可通过现有 inputs 表达**；
- cache key（`quick-start-monitor-state-trl-`）、artifact 命名（`trl-quick-start-<run_id>`）与 result.json 校验由引擎按 `inputs.project` 自动派生，无需项目侧干预。

**结论：未修改 `quick-start-template.yml`，共享引擎零改动，不影响现有调用方（peft 等）。**

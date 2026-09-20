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
  transformers 5.3.0 → `pip install -e` fork → `git am docker/npu_patch/v0.3.0/*` 六组补丁。
- 外部源码克隆失败时的回退链：sglang / Megatron-LM 走 gitcode `gh_mirrors` 镜像；
  MegatronAdaptor / TransformerEngineNPU / fork 本体原生就在 gitcode。

## 缓存事实

- 引擎的 `actions/cache` 只承担 monitor-state（release 信号状态），不承担模型缓存。
- 模型下载走 runner 既有持久 ModelScope 缓存（与 roll 同机制，`/data/ci-cache`），
  Qwen2.5-0.5B-Instruct 首次冷下载后跨 run 命中。
- pip 走集群缓存代理（`select_pip_index` 探测 `cache-service.nginx-pypi-cache`），
  失败回退清华源。
- **不使用 cache-seed**：模型全走 ModelScope、数据用仓内 fixture、sgl-kernel wheel
  是 GitHub release 资产（CI 本身就在 GitHub 上，带 `--retry 3` 直下即可）。

## supported 清单与压缩口径

### 阶段一（当前）

| example | runner | 配方出处 | 压缩口径 |
|---|---|---|---|
| `examples/fully_async/run-qwen2.5-0.5B-fully_async.sh` | a2-4 | fork NPU nightly `tests/tests_npu/nightly_CI/test_qwen2.5_0.5B_fully_async_short_npu.py`（昇腾已验证） | actor 1 + rollout 3、TP/PP/CP/EP 全 1、`--num-rollout 2`、response 1024（nightly 为 8192）、数据换仓内 16 行 fixture |

执行不直接跑上游 `.sh`（硬编码 `/root` 绝对路径、无 `"$@"` 透传），由
`scripts/ci_train_driver.py` 复刻同一 `train_async.py` 调用并注入 CI 参数；
模型 `--hf-checkpoint` 走 ModelScope 本地路径，`--ref-load` 用 fork 自带
`tools/convert_hf_to_torch_dist.py` 现转的 `_torch_dist` 目录（torchrun 4 procs）。

### 阶段二（阶段一远程绿后）

- `examples/on_policy_distillation/run-qwen3-8B-opd.sh`：sglang teacher 模式，
  复刻 fork `tests/tests_npu/st/test_qwen2.5_0.5B_opd_sglang_npu.py`（0.5B student +
  同模型 teacher，7 train + 1 teacher，a2-8）。原脚本 teacher Qwen3-32B 在 64GB 卡放不下。
- `examples/retool/retool_qwen3_4b_rl.sh`：4 卡 colocate（TP2、engine 2），模型换
  `Qwen/Qwen3-4B-Instruct-2507`（ModelScope 不可达时降 Qwen3-0.6B）；sandbox 为
  纯本地 subprocess。

### 阶段三候选

`strands_sglang`、`multi_agent`、`search-r1`（需 mock 检索服务器）、
`geo3k_vlm_multi_turn`、`train_infer_mismatch_helper`、`eval_multi_task`；
逐条评估理由见 `examples_manifest.yaml` unsupported 注释。

### 硬阻塞（不接入）

- `coding_agent_rl`：8 节点 64 rollout 卡 + E2B/远程 Docker sandbox + `SLIME_HEAD_HOST`
  反连 + `MOCK_SANDBOX` 引用的 `sandbox_mock.py` 不存在 + 模型/数据无公开源。
- `delta_weight_sync`：GLM-4.7-355B-A32B（约 700GB）16 节点 128 卡 + 共享盘，零覆盖入口。
- `tau-bench`：`generate_with_tau.py` 硬编码 gemini 用户模拟器且 `GEMINI_API_KEY`
  被源码强制覆写为 "NONE"，不改源码无法本地化。
- `geo3k_vlm`（单轮）：上游 release 树只有 35B-A3B + GitHub 私有 Megatron fork 的入口；
  通用版仅存在于 fork main，manifest-check 无法校验。

## 数据与模型

- 模型：全部走 ModelScope（Qwen2.5-0.5B-Instruct 已验证；阶段二候选
  Qwen3-4B-Instruct-2507 接入前先验可达性）。
- 数据：`fixtures/ci_dapo_16.jsonl` 为 16 行 DAPO-Math-17k 同 schema 真实样本
  （`prompt` 为 `[{content, role}]` 消息列表、`label` 为字符串答案，deepscaler
  本地 CPU 校验），不依赖 hf-mirror 数据集下载。

## 运行时环境契约

`run_example.sh` 按 fork ascend 启动器导出：`ASCEND_RT_VISIBLE_DEVICES`（按 profile
注入）、`RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1`、
`CUDA_DEVICE_MAX_CONNECTIONS=1`、HCCL 端口段、
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`（fork 原值；与 roll 的 vLLM
CaMemAllocator 场景不同，这里不 unset）、`WANDB_MODE=offline`。

## CI 接口

- 手动触发：`gh workflow run slime-examples`（`target_ref` 留空 = 上游最新 release，
  当前 v0.3.2）。
- artifact：`slime-examples-<run_id>-<job_index>`（公共引擎统一命名）。
- schedule：bring-up 期注释关闭；手动轮次全绿后由维护者决定启用（启用后由上游
  release tag 变化或上次 scheduled failure 触发）。
- `max_parallel: 1`（bring-up），阶段二起可提 2。

## 本地验证

```bash
python -m pytest projects/slime/tests tests/test_check_supported_entries.py -q
python -m pytest projects/roll/tests/test_roll_examples.py -q  # regression: shared-engine contract
bash -n projects/slime/scripts/setup_example.sh
bash -n projects/slime/scripts/run_example.sh
python -m py_compile projects/slime/scripts/ci_train_driver.py
```

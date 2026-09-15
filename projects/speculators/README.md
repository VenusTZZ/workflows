# speculators

本目录是 [speculators](https://github.com/vllm-project/speculators) 的看护配套数据，不是上游源码。example 流水线在 [.github/workflows/speculators-examples.yml](../../.github/workflows/speculators-examples.yml)。Quick Start 流水线在 [.github/workflows/speculators-quick-start.yml](../../.github/workflows/speculators-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：推理加速；支持程度：基础支持；阶段 A）。

上游默认分支是 `main`。上游 GitHub Actions / Buildkite 覆盖 NVIDIA GPU，没有 Ascend / CANN / torch_npu CI，因此按阶段 A 在本仓落地。库本身已经识别 NPU，见 `is_torch_npu_available` 与 `--draft-attn-impl sdpa`。训练 example 默认仍走 CUDA 可见设备和 flex attention。看护用 `sitecustomize` 把 `CUDA_VISIBLE_DEVICES` 同步到 `ASCEND_RT_VISIBLE_DEVICES`；训练额外改工作副本里的 `scripts/launch_vllm.py`，评测走 PATH 上的 `vllm` shim；sdpa 仍由清单 `overlay_args` 传入。

example 线的 `schedule` 保持注释。Quick Start 的 cron 是既成决定，本线不跟着打开。

## 绿灯含义

两个 `profile` 的绿灯不是同一句话。未知 profile 会在安装任何包之前非 0 退出。

- **`evaluate`**：上游 `examples/evaluate/example_qwen3_8b_dflash_humaneval.sh` 能拉起 `vllm serve` 并跑完 GuideLLM HumanEval 的 CI 规模请求。进程 exit 0 但日志里没有 NPU 设备锚点，会被判红。这不是 draft 训练绿。
- **`train`**：上游 online 训练脚本走完 prepare-data、vLLM hidden-state 抽取、`speculators.train` 的 CI 规模步数。exit 0 但日志里没有 NPU 设备锚点，会被判红。`--draft-attn-impl sdpa` 是文档写明的昇腾路径，不是改算法。MTP 那条不传该旗标。

清单仍要求每条 `supported` 写 `npu_devices`，所以 evaluate 的单卡 job 和 train 的双卡 / 四卡 job 都跑在 NPU runner 上。

## 看护范围

example 线会跑清单 `supported` 里的 8 条：1 条 DFlash 评测，7 条 Qwen 训练。整份清单允许诚实红，不把跑红的条目改成 `unsupported`。

在 910B4 32GB HBM 上按清单字段跑过之后：

- 绿：`dspark_qwen3_0_6b_sharegpt_online.sh`、`dflash_qwen3_8b_ultrachat_online_5k.sh`、`dflash_qwen3_8b_ultrachat_online_5k_bestpractices.sh`、`eagle3_qwen3_8b_ultrachat_online_5k.sh`、`mtp_qwen3_5_9b_gsm8k_online.sh`
- 诚实红：`example_qwen3_8b_dflash_humaneval.sh` 在 vllm-ascend 0.23 的 DFlash drafter `dummy_run` 里触发 RoPE `positions.shape[0] == num_tokens` 断言；`dflash2_qwen3_8b_ultrachat_online_5k.sh` 和 `peagle_qwen3_8b_ultrachat_online_5k.sh` 在 CI 规模序列长度下训练卡 OOM

四卡训练脚本默认 2 卡给 vLLM、2 卡给 trainer。CI 的 `linux-aarch64-a2-8` 按这个切分跑。非连续物理卡号的调试机上 2 卡 HCCL 可能超时，那种机器要把 `NUM_TRAIN_GPUS` 收到 1，不要把这个收卡写进 CI 默认。

不跑的条目和原因写在 [examples_manifest.yaml](examples_manifest.yaml)：Eagle3 convert 缺魔搭上的草稿且现有补丁吃不了单行 convert；Llama-3.1 offline 有魔搭复刻但还没 remap / 真机验证；Llama-4 Maverick 对不上 910B4 HBM；Kimi-K3 是三机 NVL72 / Mooncake / NVLink，权重约 1.56 TB。

`setup_example.sh` 只做 profile 门和 CANN `source`，安装在 `setup_example.py`。`--no-deps` 装 speculators 和 guidellm，再钉 `numpy==1.26.4`，因为 triton-ascend 3.2.2 不能跟 numpy 2 共存。同一次安装还要 editable 装仓内 `hs_connectors`，否则 `speculators.train` 会在 import 阶段失败。工作副本会把 hub id 先 `snapshot_download` 成本地目录，并改写 speculator `config.json` 里的 verifier 路径，避开 vLLM 0.23 走 ModelScope 列文件时的 `KeyError: Type`。

Quick Start 是另一条线，文档在 `docs/Quick-start-Ascend.md`。

## 触发

example 线是脚本型 B 族：`examples` 树、`/releases/latest` tag、默认分支 HEAD 三信号，失败重试写成 `<signal>-retry`。手动入口是 `target_repo` / `target_ref`，没有 `force`。NPU job 不上传 artifact；`result.json` 由 `ubuntu-latest` 上的 `publish-result` 按 job 名 `run-example (${{ matrix.example.path }})` 回看 conclusion 后上传。cache 前缀是 `speculators-examples-monitor-state-`，与 Quick Start 引擎的 `monitor-state-speculators-` 不互为前缀。

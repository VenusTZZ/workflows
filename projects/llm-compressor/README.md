# llm-compressor

本目录是 [llm-compressor](https://github.com/vllm-project/llm-compressor) 的看护配套数据，不是上游源码。example 流水线在 [.github/workflows/llm-compressor-examples.yml](../../.github/workflows/llm-compressor-examples.yml)。Quick Start 流水线在 [.github/workflows/llm-compressor-quick-start.yml](../../.github/workflows/llm-compressor-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：推理加速；支持程度：基础支持；阶段 A）。

上游默认分支是 `main`，最新正式 release 是 `0.13.0`。上游 GitHub Actions / Buildkite 覆盖 CPU、NVIDIA GPU 和 Intel XPU，没有 Ascend / CANN / torch_npu CI，因此按阶段 A 在本仓落地。

两条线的 `schedule` 都保持注释。

## 绿灯含义

### example 线

三个 `profile` 都在昇腾上跑，未知 profile 会在安装任何包之前非 0 退出。判断标准和 CUDA 机器对齐：脚本在 CUDA 上会用到 GPU 的，这里装 `torch_npu`，让 `get_main_device` 落到 npu。不卸 `torch_npu` 来换绿灯，也不改 example 文件。写死 `cuda` 的脚本留在 `unsupported`。

example 的 setup 安装 CANN 9.1.0 官方推荐组合 `torch==2.12.0+cpu` 与 `torch_npu==2.12.0`。PyPI 上的 Linux `torch==2.12.0` 轮子依赖 `cuda-toolkit`，CPU 轮子从 `download.pytorch.org` 按文件 URL 安装，后续 pip 用约束钉住 `torch==2.12.0+cpu`。音频 example 仍在 `unsupported`：PyPI 的 aarch64 `torchcodec` 轮子链接 `libtorch_cuda.so`，升到 torch 2.12 也加载不了。

- **`npu_inference`**：`examples/compressed_inference/fp8_compressed_inference.py`。加载公开 FP8 TinyLlama 后，`compressed_model` 和 `inputs` 都要在 `npu:0` 上完成 `generate`。进程 exit 0 但张量在 CPU 上判红。这条当前是诚实红：权重能到 `npu:0`，`generate` 在 FP8 解压时触发 `aclnnInplaceCopy`，错误码 561103。
- **`npu_oneshot`**：单卡 `oneshot` 量化。成功条件是脚本自己的 `model` 上至少有参数在 `npu`。FP8 解压失败、GPTQ 的 `aclnnLinalgCholesky` 超过 8192 维，都是诚实红。绿灯只表示这次量化用了 npu，不是某种打包格式能在昇腾上推理。MXFP、NVFP4 也按这条理解。
- **`npu_ddp`**：脚本无条件调用 `init_dist`，必须由 `torchrun` 拉起。清单挂 `linux-aarch64-a2-2`，`npu_devices` 为 `0,1`。看护按这个卡数启动，不改 example 文件。昇腾上分布式后端是 gloo。gloo 若拒绝 npu 设备，是诚实红。

`run_example.sh` 调 `run_guard.py`，用 `runpy.run_path` 执行原始脚本。`oneshot` 若收到字符串数据集且没带 `splits`，会补上 `train[:num_calibration_samples]`，避免把整份语料 tokenize 进 32GiB cgroup。这只包在看护进程里，不改 example 文件。门禁模型从 ModelScope 种进 Hugging Face 缓存里脚本写的那个 id，然后设 `HF_HUB_OFFLINE` 和 `TRANSFORMERS_OFFLINE`。校准数据集不强制离线，因为有的脚本在运行时才算出 slice。`HF_HUB_DISABLE_XET=1` 避开 Xet 数据面。`TORCHDYNAMO_DISABLE=1` 避免 transformers 5 默认 `torch.compile` 把超时吃完。

### Quick Start 线

文档绿灯表示：按 `docs/Quick-start-Ascend.md` 装上 `torch==2.10.0` / `torch_npu==2.10.0.post4` 和当前 release 的 `llmcompressor`，对 `nm-testing/tinysmokeqwen3` 做单层 W4A16 GPTQ，保存后重载，前向张量在 `npu:0`。探测块的 `torch.npu.is_available()` 不能单独当成量化成功。

## 看护范围

打了标签、会被看护的步骤：

- example：压缩推理；单卡 `oneshot`，含 FP8、GPTQ、AWQ、AutoRound、SpinQuant、多模态里放得进 32GiB 的模型；双卡 `init_dist` 脚本
- Quick Start：检查 Python、安装 torch / torch_npu、安装 llm-compressor、`oneshot()` GPTQ、保存、重载、NPU 前向

无标签、**不看护**的步骤：

- `source set_env.sh` 和 `export PATH`（Quick Start 测试在 `prepare_environment` 里做等价的 CANN 注入）
- `npu-smi info`（设备表每次不同，正文只要求退出码 0）
- 清单 `unsupported` 里的其余上游 example，原因写在 [examples_manifest.yaml](examples_manifest.yaml)：`.agents` 技能模板、脚本写死 cuda、音频解码依赖的 torchcodec aarch64 轮子链接 CUDA、权重放不进 32GiB

## 触发

example 线是脚本型 B 族：`examples` 树、`/releases/latest` tag、默认分支 HEAD 三信号，失败重试写成 `<signal>-retry`。手动入口是 `target_repo` / `target_ref`，没有 `force`。NPU job 不上传 artifact；`result.json` 由 `ubuntu-latest` 上的 `publish-result` 按 job 名 `run-example (${{ matrix.example.path }})` 回看 conclusion 后上传。

Quick Start 走共享引擎 `quick-start-template.yml`：互斥优先级 `release` > `doc` > `retry`。`doc_url` 走 GitHub Contents API（`ref=${{ github.sha }}`）。cache 前缀是引擎拥有的 `monitor-state-llm-compressor-`，与 example 线的 `llm-compressor-examples-monitor-state-` 以及 `llm-d` 的 `monitor-state-llm-d-` 都不互为前缀。

# openrlhf

本目录是 [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 的看护配套数据，不是上游源码。Quick Start 流水线在 [.github/workflows/openrlhf-quick-start.yml](../../.github/workflows/openrlhf-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：训练加速；支持程度：新兴适配；阶段 A）。

当前只落地 Quick Start 线，没有 example 清单和 `*-examples.yml`。

上游主干没有昇腾专用代码，也没有昇腾 CI。`openrlhf.cli.train_sft` 通过 DeepSpeed 调用 `torch.cuda.*`。文档因此在用户看得见的步骤里写入 `sitecustomize.py`（调用 `torch_npu.contrib.transfer_to_npu`），把 `torch.cuda` 映射到 NPU。`setup.py` 把 wheel 标签写成 `manylinux1_x86_64`，aarch64 上不能 `pip install` 该包装；文档把克隆下来的源码树加入 `PYTHONPATH`。`openrlhf.models.actor` 在 import 时会加载 `flash_attn`（与 `--ds.attn_implementation` 无关）；昇腾上没有 CUDA 版，文档用一条可见命令写入只满足 import 的占位包，真正调用 packing / ring attention API 会立刻失败。

`upstream_repo` 指向 `OpenRLHF/OpenRLHF`。monitor 跟上游 Release 走。文档用隐藏 `#test-setup` 把 `$UPSTREAM_REF` 填进 `git clone --branch <ref>`。共享引擎仍会把 `result.json` 的 `target_ref` 写成解析到的最新 Release tag。

## Quick Start 看护范围

文档在 `docs/Quick-start-Ascend.md`。方言见 [docs/markdown_doc_test_label.md](../../docs/markdown_doc_test_label.md)。

打了 `#test`、会被看护的步骤：

- 检查 Python 版本
- 安装 `torch==2.11.0` / `torch_npu==2.11.0`，断言 `npu_available True`
- 写入 `sitecustomize.py`，使后续 Python 进程（含 DeepSpeed worker）调用 `transfer_to_npu`
- 在工作目录写入 `flash_attn` 占位包（只满足 import）
- 按 `$UPSTREAM_REF` 克隆上游并安装 SFT 路径依赖（不含 `flash-attn` / `bitsandbytes` / `pynvml` / `ray`）
- 从 ModelScope 取 `Qwen/Qwen2.5-0.5B-Instruct`，链到 `/root/openrlhf-qs/model`
- `deepspeed --module openrlhf.cli.train_sft` 跑 4 步 LoRA。工作负载块的预期输出含设备锚点（`npu::npu_format_cast`、`device='npu'`）以及 `Train step of epoch 0: 100%` / `exits successfully.`

隐藏 `#test-setup`（页面不渲染，看护会跑）：

- 把 `$UPSTREAM_REF` 填进克隆命令
- 校验 / 预置 / 写回 `/root/.cache/cosdt-ci-test/openrlhf/` 里的源码树（跨 run 复用；可见块仍写官方 GitHub URL）
- 把可见的 `tiny_sft.jsonl` 写到工作目录（用户按正文用编辑器保存同一份）

无标签、**不看护**的步骤：

- `source set_env.sh` 和 `export PATH`（测试进程在 `prepare_environment` 里做等价的 CANN 注入，并且覆盖写回 `os.environ`）
- `npu-smi info`（设备表每次不同，正文只要求退出码 0）
- 可见的 `tiny_sft.jsonl`（无标签围栏，不当命令执行）
- 上游 README 的 Docker / `pip install openrlhf[vllm]`、PPO / Ray / vLLM、Flash Attention 2、QLoRA、`bitsandbytes`、多卡

`schedule` 保持注释。`doc_url` 走 GitHub Contents API（`ref=${{ github.sha }}`），不走 `raw.githubusercontent.com`。

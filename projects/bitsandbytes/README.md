# bitsandbytes

本目录是 [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) 的看护配套数据，不是 bitsandbytes 源码。example 流水线在 [.github/workflows/bitsandbytes-examples.yml](../../.github/workflows/bitsandbytes-examples.yml)。Quick Start 流水线在 [.github/workflows/bitsandbytes-quick-start.yml](../../.github/workflows/bitsandbytes-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)。分类：推理加速；支持程度：基础支持；阶段 A。

上游默认分支是 `main`。上游官方不支持昇腾，见 [issue #1847](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1847)，也没有健康的昇腾 CI。本仓先走阶段 A：在本仓流水线把能跑的路径跑通。

## 绿灯含义

example 线现在有两个 profile，绿灯含义不同，不要互相代替。

- `cpu`：上游 `examples/cpu/cpu_training.py` 在主机 CPU 上还能用 bitsandbytes 的 AdamW 跑完一小段微调，loss 按脚本自己的判据下降。脚本写死在 CPU 上训练，不把张量放到 `npu:0`。**这条绿灯不是昇腾推理绿。**
- `npu`：上游 `examples/compile_inference.py` 按原文加载 8-bit `google/gemma-2-2b-it`，`torch.compile` 之后生成 32 个 token。权重必须经 `torch_npu` 的 `NPUCachingAllocator` 分配。NPU 上的 4-bit 前向仍看 Quick Start 文档 `docs/Quick-start-Ascend.md`。

## 清单

- `examples_manifest.yaml` 的 `scan.root` 是上游 `examples/`。`supported` 只收这条目录里的脚本，不跑上游 `tests/`。
- `examples/cpu/cpu_training.py`：`profile: cpu`，挂 `linux-aarch64-a2-1`、`npu_devices: '0'`、镜像 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。overlay 把规模压到 `--steps 5 --max_length 64 --batch_size 1`。脚本会下载 `JackFram/llama-68m` 和 `yahma/alpaca-cleaned`；NPU runner 走 `HF_ENDPOINT=https://hf-mirror.com`，不改 example 正文。
- `examples/compile_inference.py`：`profile: npu`，同一 runner 与镜像。脚本没有 CLI，没有 overlay。`google/gemma-2-2b-it` 在 Hugging Face 上是 gated 仓，runner 无 token 时 Hub 返回 403。setup 从 ModelScope 的公开镜像 `LLM-Research/gemma-2-2b-it` 拉权重，种进 Hugging Face 缓存布局 `models--google--gemma-2-2b-it/`，再设 `HF_HUB_OFFLINE=1`，这样 `from_pretrained("google/gemma-2-2b-it")` 仍用原文 id。`torch.compile` 在 CANN 9.1.0 上需要 `torch==2.10.0`、`torch-npu==2.10.0.post4`、`triton-ascend==3.2.2`；社区版 `triton` 不能装。`triton-ascend` 必须 `--no-deps --force-reinstall`，并单独补 `pybind11`。Triton / Inductor 缓存在工作副本里，不写 `/root/.triton`。
- 其余上游 example 放进 `unsupported`，只表示本看护体系当前不跑它们：
  - `examples/int8_inference_huggingface.py`：gated 的 `meta-llama/Llama-2-7b-hf`，并且写死 `torch.cuda.mem_get_info()` / `torch.cuda.device_count()`。
  - `examples/xpu/paged_xpu_training.py` 与 `examples/xpu/benchmark_paged_memory.py`：断言 `torch.xpu.is_available()` 或 CUDA，Intel GPU 专用。

未知 `profile` 在任何 `pip install` 之前非 0 退出，并打印已支持列表。当前是 `cpu` 与 `npu`。

## 防假绿

1. **mock 原生库。** `import bitsandbytes` 成功不能当绿灯。`cextension` 加载失败会换成 `ErrorHandlerMockBNBNativeLibrary`。`setup_example.sh` 装完后断言 `type(ce.lib).__name__` 不是这个 mock 类，并打印 `BNB_BACKEND`。CPU 与 NPU example 上原生库后端都是 `CPU`，这是上游 default 后端，不是装错了。
2. **loss 哨兵。** `cpu_training.py` 在 loss 没有下降时仍 exit 0，只打 `WARNING`。`run_example.sh` 在进程结束后要求日志里出现 `OK: Loss decreased as expected`，否则非 0。
3. **CPU profile 不扫 NPU 设备锚点。** 这条 example 本来就不该上卡。不要用 `npu:0` 去卡它。
4. **NPU compile 设备锚点。** `compile_inference.py` 默认 verbosity 不打印 `npu:0`。`run_example.sh` 要求日志出现 `NPUCachingAllocator`，这是 `torch_npu` 分配器才会打的警告；还要求出现脚本写死的 prompt `Write me a poem about Machine Learning`，证明 generate 跑完。只 exit 0 或只生成了诗、但走了 CPU，都不算绿。

## 触发

### example 线

`bitsandbytes-examples.yml` 有两种入口。`monitor` job 跑在 `ubuntu-latest`，不占 NPU。

- `schedule`：cron 写在文件里但是注释掉的。接入阶段保持注释，不要打开。
- `workflow_dispatch`：手动触发。默认 `force=false`，和定时走同一套监控、同一份 cache。只有 `force=true` 才跳过监控门、必跑，并且不读不写 monitor cache。`target_repo` / `target_ref` 只在 `force=true` 时有意义。

两个监控信号都跑，是「或」，互不跳过、没有优先级、失败不重试：

1. 清单 `supported` 各 `path` 在上游 `main` 上的文件内容哈希。Contents API 的 blob SHA；目录会递归到文件。404 记成 `MISSING`，哈希会变。本信号亮了，测的是这一轮解析到的 `main` commit SHA。
2. `/releases/latest` 的 release **id**，数字，不是 tag 字符串。本信号亮了，测的是该 release tag 当前指到的 commit SHA。bitsandbytes 有 GitHub Releases，不要跳过这个信号。

谁亮了就测谁的树。都没亮则 `targets` 为空，后面的 job 跳过，不写 `result.json`，不产包。`force=true` 时 `targets` 只有一项，`reason=manual`。

NPU job 不上传 artifact。`result.json` 由托管 runner 上的 `validate-results` 按 job 名回看 conclusion 后上传。job 显示名必须保持 `run-example (${{ matrix.example.path }} @ ${{ matrix.example.target_ref }})`，与 `EXPECTED_JOB_NAME` 全等匹配。

### Quick Start 线

文档在 `docs/Quick-start-Ascend.md`。流水线是 `.github/workflows/bitsandbytes-quick-start.yml`，只是共享模板 `quick-start-template.yml` 的薄触发器。文档方言见 [docs/markdown_doc_test_label.md](../../docs/markdown_doc_test_label.md)：围栏 info 行用 `#test` / `#test-setup` / `#test-result`。无标签的 `shell` 块给用户复制，看护跳过。

Quick Start **不**抄本项目 example 线的「两信号或、无重试」。它走共享模板自己的监控：互斥优先级 `release` > `doc` > `retry`。字面 `retry`，不是 `-retry`。cache 前缀是模板拥有的 `quick-start-monitor-state-bitsandbytes-`，不要再发明第二套前缀。

这是相对本项目 example 线的**有意偏离**：本仓 Quick Start 触发器统一走共享模板，触发语义以 `quick-start-template.yml` 为准。

`schedule` 保持注释。`force` 行为由模板解释。文档 URL 走 GitHub Contents API，`Accept: application/vnd.github.raw`，不走 `raw.githubusercontent.com`。

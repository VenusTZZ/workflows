# bitsandbytes

本目录是 [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) 的看护配套数据，不是 bitsandbytes 源码。example 流水线在 [.github/workflows/bitsandbytes-examples.yml](../../.github/workflows/bitsandbytes-examples.yml)。Quick Start 流水线在 [.github/workflows/bitsandbytes-quick-start.yml](../../.github/workflows/bitsandbytes-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：推理加速；支持程度：基础支持；阶段 A）。

上游默认分支是 `main`。上游官方不支持昇腾（[issue #1847](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1847)），也没有健康的昇腾 CI。本仓先走阶段 A：在本仓流水线把能跑的路径跑通。

## 绿灯含义

`bnb4bit` profile 绿，表示 bitsandbytes 的 **default（设备无关）后端** 4-bit 量化数学在昇腾上还能跑。**不是**官方支持昇腾，**不是**昇腾专用 kernel 绿了。权重仍然走 CPU 原生库打包，计算落到 `torch.npu` 设备。

**红灯可能是「上游从未支持」而不是回归。** 分诊时先看是缺算子（`torch_npu` 侧）还是缺 kernel（bitsandbytes 侧没有 default 实现）。不要把红灯默认理解成最近一次提交把昇腾弄坏了。

## 清单

- `examples_manifest.yaml` 的 `scan.root` 是上游 `examples/`，用来发现新增 example。`supported` 指向上游测试文件 `tests/test_ops.py` 与 `tests/test_linear4bit.py`，不是 `examples/`。共享的 `scripts/check_examples_manifest.py` 允许 `supported.path` 落在扫描根之外，不需要项目私有 checker。
- 上游 `examples/` 五条全部放进 `unsupported`，只表示本看护体系当前不跑它们：
  - `examples/compile_inference.py` 会下载 `google/gemma-2-2b-it`（约 5 GB），`device_map="auto"`，还依赖 `torch.compile`（NPU 上 Inductor 不可用）。
  - `examples/int8_inference_huggingface.py` 会下载需要授权的 `meta-llama/Llama-2-7b-hf`，并且写死 `torch.cuda.mem_get_info()` / `torch.cuda.device_count()`。
  - `examples/cpu/cpu_training.py` 是 CPU 训练，还要下载 `JackFram/llama-68m` + `yahma/alpaca-cleaned`，不是昇腾 example。
  - `examples/xpu/paged_xpu_training.py` 与 `examples/xpu/benchmark_paged_memory.py` 断言 `torch.xpu.is_available()` 或 CUDA，Intel GPU 专用。
- 明确不看护 8-bit 训练、8-bit 优化器、分页优化器。这些路径在上游没有 default 实现，昇腾上必挂，属于「从未支持」而不是回归。
- 明确不看护 `tests/test_functional.py` 的 `test_4bit_quant`。误差阈值按 CUDA kernel 标定，default 实现的数值差容易造成假红。
- 每条 `supported` 都挂 `linux-aarch64-a2-1`、`npu_devices: '0'`、镜像 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。
- `overlay_args`：`test_ops` 用 `-k Test4bitBlockwiseQuantOps`；`test_linear4bit` 用 `-k "not serialization and not compile and not fsdp"`。FSDP 用例会 `torchrun` 子进程，子进程不 `import torch_npu`，不是 default 4-bit 数学路径。过滤器以 coder 上 `--collect-only` 和一次实跑为准。
- 不需要 `fixtures/`。本项目不下模型。

## 防假绿

三个守卫都是运行时断言，不把布尔值穿进 workflow YAML。

1. **mock 原生库。** `import bitsandbytes` 成功不能当绿灯。`cextension` 加载失败会换成 `ErrorHandlerMockBNBNativeLibrary`。`setup_example.sh` 装完后断言 `type(ce.lib).__name__` 不是这个 mock 类，并打印 `BNB_BACKEND`（昇腾上预期为 `CPU`，这不是 NPU 证据）。
2. **pytest 日志设备锚点。** `run_example.sh` 设置 `BNB_TEST_DEVICE=npu`，把输出 tee 到日志，要求出现 `[npu` 或 `-npu]`，并拒绝 `[cpu` / `-cpu]` / `[cuda` / `-cuda]`。上游 node id 把 device 放在最后（`[fp16-bf16-npu]`），只搜 `[npu` 会把真绿判成假绿。环境变量没生效时，上游 helper 会退回 `["cpu"]`，进程仍可能 exit 0。
3. **`passed` 标记。** 日志里必须有 `passed`。过滤器写错、一条都没选中时，pytest 退出码是 5；这条再兜一层。

未知 `profile` 在任何 `pip install` 之前非 0 退出，并打印已支持列表。当前只有 `bnb4bit`。

`run_example.sh` 用 pytest 插件 `scripts/bnb_npu_bootstrap.py`（`-p bnb_npu_bootstrap`）在测试进程里 `import torch_npu`。上游 `tests/` 和 `conftest.py` 都不 import 它。插件避免改被测树。

## 触发

### example 线

`bitsandbytes-examples.yml` 有两种入口。`monitor` job 跑在 `ubuntu-latest`，不占 NPU。

- `schedule`：cron 写在文件里但是注释掉的。接入阶段保持注释，不要打开。
- `workflow_dispatch`：手动触发。默认 `force=false`，和定时走同一套监控、同一份 cache。只有 `force=true` 才跳过监控门、必跑，并且不读不写 monitor cache。`target_repo` / `target_ref` 只在 `force=true` 时有意义。

两个监控信号都跑，是「或」，互不跳过、没有优先级、失败不重试：

1. 清单 `supported` 各 `path` 在上游 `main` 上的文件内容哈希（Contents API 的 blob SHA；目录会递归到文件。404 记成 `MISSING`，哈希会变）。本信号亮了，测的是这一轮解析到的 `main` commit SHA。
2. `/releases/latest` 的 release **id**（数字，不是 tag 字符串）。本信号亮了，测的是该 release tag 当前指到的 commit SHA。bitsandbytes 有 GitHub Releases（例如 `0.50.1`），不要跳过这个信号。

谁亮了就测谁的树。都没亮则 `targets` 为空，后面的 job 跳过，不写 `result.json`，不产包。`force=true` 时 `targets` 只有一项，`reason=manual`。

**已知局限：** 监控的是测试文件，不是 `bitsandbytes/backends/default/ops.py`。被测代码变了但测试文件没变时，要等到下一个 release 才会重跑。这和本仓其他项目一致，不要为此再加第三个信号。

NPU job 不上传 artifact。`result.json` 由托管 runner 上的 `validate-results` 按 job 名回看 conclusion 后上传。job 显示名必须保持 `run-example (${{ matrix.example.path }} @ ${{ matrix.example.target_ref }})`，与 `EXPECTED_JOB_NAME` 全等匹配。

### Quick Start 线

文档在 `docs/Quick-start-Ascend.md`。流水线是 `.github/workflows/bitsandbytes-quick-start.yml`，只是共享模板 `quick-start-template.yml` 的薄触发器。文档方言见 [docs/markdown_doc_test_label.md](../../docs/markdown_doc_test_label.md)：围栏 info 行用 `#test` / `#test-setup` / `#test-result`。无标签的 `shell` 块给用户复制，看护跳过。

Quick Start **不**抄本项目 example 线的「两信号或、无重试」。它走共享模板自己的监控：互斥优先级 `release` > `doc` > `retry`（字面 `retry`，不是 `-retry`）。cache 前缀是模板拥有的 `quick-start-monitor-state-bitsandbytes-`，不要再发明第二套前缀。

这是相对本项目 example 线的**有意偏离**：本仓 Quick Start 触发器统一走共享模板，触发语义以 `quick-start-template.yml` 为准。

`schedule` 保持注释。`force` 行为由模板解释。文档 URL 走 GitHub Contents API（`Accept: application/vnd.github.raw`），不走 `raw.githubusercontent.com`。

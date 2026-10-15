# Ollama Ascend 快速入门接入与验证状态

本目录保存 Ollama 的原生 CANN 接入补丁和快速入门看护入口。调查日期为 **2026-10-13**；已完成 prototype 的源码与补丁兼容性验证，**CANN 原生构建和 NPU 推理均尚未验证**。

[用户文档](docs/Quick-start-Ascend.md)、[测试夹具](tests/test_quick_start_ascend.py)和手动 [workflow](../../.github/workflows/ollama-quick-start.yml) 均已落地，首次原生构建和 NPU 推理验收尚未完成。

## 上游调查快照（2026-10-13）

- 调查时 [Ascend PR #5872](https://github.com/ollama/ollama/pull/5872) 尚未合入，head 为 `1be1fa6b63397ffa5ae7e9b0578d1fcd517e497e`。
- 调查时最新 release 为 `v0.40.3`，提交为 `eab97e9f92b9a25c2d52d2cc6c1b1c99bd9fae21`；调查的 main 为 `a5276525531a75dda69b69226aa5e369cd3933e7`。这些版本是调查快照，不是后续运行的固定版本。
- 旧 PR 约修改 1.1 万行，对这两个版本执行 `git apply --check` 均失败，涉及构建入口、设备发现和已经移除的旧 GGML 目录。
- 调查版本使用独立 `llama-server`。`v0.40.3` 固定 llama.cpp `b11351`，该源码已包含 CANN 后端；应从同一版本源码编译，避免复用旧 PR 的二进制。旧 PR 的 GGML backend ABI 为 1，目标版本为 3。

## 原型范围

[ollama-cann.patch](patches/ollama-cann.patch) 使用目标版本自带的 CANN 实现，增加五处接入，不搬运旧 PR 的 ACL 探测代码或算子源码：

| 上游文件 | 改动 |
|---|---|
| `cmake/local.cmake` | 接受 `OLLAMA_LLAMA_BACKENDS=cann`，构建 `ggml-cann`，传递 CANN 安装目录和 SoC 型号 |
| `discover/llama_server.go` | 将 `CANN0` 等设备名归类为 `CANN` |
| `discover/native_probe_platform.go` | 将 `libggml-cann.so` 加入原生设备探测列表 |
| `discover/runner.go` | 将父进程可见设备列表映射回子进程筛选 ID |
| `ml/device.go` | 通过 `ASCEND_RT_VISIBLE_DEVICES` 传递实际选中的 NPU |

补丁另外包含四项 Go 回归测试，覆盖设备分类、原生库探测、可见设备编号映射和子进程设备选择。已有的通用安装布局与动态库搜索机制可以复用，后端安装在 `build/lib/ollama/cann/`。

## 已完成的 prototype 验证记录

- 原型补丁在上述 release 和 main 的干净检出中均通过 `git apply --check`。
- 在 release 检出中实际应用补丁成功。
- Linux 顶层 CMake configure 成功；生成的 CANN 子构建命令包含 `GGML_CANN=ON`、`OLLAMA_GPU_BACKEND=cann`，并正确转发 toolkit 目录和 `SOC_TYPE`。
- 使用临时 Go 1.26.9 工具链运行 `go test ./discover ./ml -run TestCANN -count=1 -v`，四项补丁回归测试通过。
- 尚未完成原生模块编译、服务加载或 NPU 推理。

`git apply --check` 只证明补丁适用性，顶层 CMake configure 只证明构建入口接通，Go 回归测试只覆盖接入逻辑。以上结果均不能证明 CANN 原生构建或 NPU 推理成功。

## 快速入门与看护入口

安装、构建、模型导入和推理操作集中在 [Quick-start-Ascend.md](docs/Quick-start-Ascend.md)，README 不重复维护实验命令。

用户文档保持前置条件、加载 CANN、源码安装、用法示例、外部链接的结构。Ollama 走原生 CANN 后端，不需要为它安装 torch / torch_npu。

首次示例选用 Qwen2.5-0.5B-Instruct 的 Q4_0 GGUF，从 ModelScope 下载，通过 Modelfile 和 `ollama create` 导入，再分别完成 CLI 文本生成与本地 `/api/chat` 非流式请求。该格式在 llama.cpp 的 CANN 支持表内，避免默认模型标签落到不支持的量化格式。

用户文档通过相对链接 `../patches/ollama-cann.patch` 引用补丁，要求用户手动下载到当前目录。夹具从 checkout 中的 assets（`projects/ollama/patches/ollama-cann.patch`）复制补丁到文档执行的当前目录，完成这一可见人工步骤，再执行文档中的补丁检查与应用；补丁不适用时应报告版本漂移。

文档以前台 `ollama serve` 展示服务启动。夹具负责启动该前台服务、等待就绪，以及成功或失败后的进程组清理；源码、构建和本地调试报告使用临时工作目录并由夹具清理。原生构建后先通过 `llama-server --list-devices --offline` 验证 CANN 模块加载及设备枚举，再验收非空回复、正数生成 token、`/api/ps` 中正数设备显存占用，以及最近一次模型加载日志中的正数 CANN buffer 和完整 offload。CPU 回退不得视为通过，也不得用旧的完整 offload 记录掩盖后续加载失败。

手动 workflow 复用与 dgl 相同的 `quick-start-template.yml`：`project: ollama`、`upstream_repo: ollama/ollama`、runner `linux-aarch64-a2-2`、CANN `9.1.0-910b-ubuntu22.04-py3.12` 镜像，测试 job 总预算为 240 分钟。文档通过 GitHub Contents API 按本次 `github.sha` 读取，测试入口为 `python -m unittest tests.test_quick_start_ascend -v 2>&1`。首次验证仅启用 `workflow_dispatch`，没有 schedule。

共享引擎解析官方 release，并通过 `UPSTREAM_REF` 固定本次源码检出。夹具须检出该 ref 后检查并应用本地补丁，结果 artifact 的 `target_ref` 保持实际构建的官方版本；源码 commit SHA 和补丁 SHA256 记录在夹具报告及 job 日志中。当前模板没有 `fixed_ref` 输入，不能将旧 PR head 当作官方 release 的替代版本。

GitHub 要求 `workflow_dispatch` 的 workflow 文件先存在于默认分支，之后才能从 UI 选择功能分支运行。新增文件仅推送到功能分支时还不能直接 dispatch。

容器绑定 `/data/ci-cache/ollama:/ollama-cache`，持久缓存分别使用 `/ollama-cache/toolchains`、`/ollama-cache/downloads`、`/ollama-cache/models`、`/ollama-cache/go-build`、`/ollama-cache/go-mod` 和 `/ollama-cache/ccache`。缓存用于复用工具链、下载、模型与编译缓存；源码和构建工作目录仍由临时目录承载。

夹具的本地调试报告约定如下，文件首先写入临时工作目录的 `reports/`，清理前复制到 `/ollama-cache/reports/<本次临时目录名>/`，保留在宿主挂载目录中：

| 报告路径 | 验收证据 |
|---|---|
| `reports/ollama-compatibility.json` | 官方 ref、源码 commit SHA、补丁 SHA256、补丁兼容性与应用结果；兼容性检查不代表原生构建或 NPU 验证通过 |
| `reports/server.log` | 服务完整日志，检查 CANN 设备、模型 buffer 和 offload 层数 |
| `reports/chat-response.json` | `/api/chat` 非流式响应，检查非空回复及正数生成 token |
| `reports/loaded-models.json` | `/api/ps` 响应，检查模型加载及正数设备显存占用 |

详细构建与推理证据输出到 job 日志：夹具打印包含源码 SHA、补丁 SHA256、响应与设备检查结果的 JSON 汇总，失败时打印服务日志尾部（log tail）。原始服务日志及 JSON 另保留在挂载缓存的 `reports/` 子目录，便于检查首次加载过程；共享引擎仅上传 `result.json`，这些额外文件不会自动进入 artifact。当前仍需完成首次上卡验收，原生 CANN 构建、服务加载和 NPU 推理的实际结果尚无证据。

## 参考资料

- [Ollama 开发构建文档](https://github.com/ollama/ollama/blob/v0.40.3/docs/development.md)
- [llama.cpp CANN 后端](https://github.com/ggml-org/llama.cpp/blob/b11351/docs/backend/CANN.md)
- [Ollama 导入 GGUF 模型](https://docs.ollama.com/import)
- [Ollama 本地快速入门](https://docs.ollama.com/quickstart)

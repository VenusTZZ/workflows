# TensorDict examples 看护

本项目复用公共 `examples-template.yml`；薄触发器为 `.github/workflows/tensordict-examples.yml`，被测仓库是 `pytorch/tensordict`。手动触发的 `target_ref` 可指定分支、tag 或 SHA；留空时由公共引擎选择最新 release。Bring-up 期间 schedule 保持注释关闭。

2026-10-10 重审最新 release v0.14.3 和 main 全部 14 个 Python 教程：supported 5→8，unsupported 9→6。原有 5 条调度与执行契约不变；新增切片/布尔掩码、异步流/nested densify 和 memmap 存储三个入口，尚待 NPU 实跑。各教程的接入限制记录在 manifest 注释中。

项目 setup 先探测 `torch`/`torch_npu`：若镜像里有兼容版本就复用，否则按 quick-start 的已验证配方安装 `torch==2.9.0` 与 `torch_npu==2.9.0.post2`；再从受测 checkout 安装 TensorDict 源码，且源码安装使用 `--no-deps` 防止覆盖 NPU 栈。启动器先设置 PyTorch 默认设备为 `npu:0`，再用 `runpy` 执行未经修改的上游教程；最后检查原教程生成的 TensorDict 叶子或函数式/导出结果确实位于 NPU。这样既运行了原教程的完整代码与断言，也不会把 CPU 回落误报为 NPU 成功。五个 supported 教程都只使用小型合成张量，不下载模型或数据，无需 ModelScope/cache-seed。

新增切片保留原 CPU BoolTensor，检查真实 6 行 NPU 结果；流教程完整运行约 11 秒，检查初始/异步 nested 叶子在 NPU、10 个桶。memmap 本身必须位于 CPU：原教程完成后以公开 API 填充其 memmap_like，并验证 NPU→CPU 映射→NPU 内容一致，不声称映射存储在 NPU。没有源码补丁、设备 monkeypatch 或 compile 禁用。剩余入口分别具有显式 CPU/CUDA 设备、无 backend 参数的 Inductor 编译、固定大规模写盘、ZipStore CPU/NPU 比较等实际约束。

手动验收时检查 manifest-check 展开 8 个唯一 job；所有日志出现 `NPU tutorial passed`、没有 CPU 回落。混合掩码、strided nested 与映射拷贝均为真实待上卡路径，不兼容时明确失败，不能跳过源代码。8 个 publish-result 及整体 success 后才考虑 schedule。artifact 为 `tensordict-examples-<run_id>-<job_index>`。

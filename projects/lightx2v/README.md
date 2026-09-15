# LightX2V

本目录是 [LightX2V](https://github.com/ModelTC/LightX2V)（轻量级图像/视频生成推理框架）的看护配套数据，不是 LightX2V 源码。流水线在 [.github/workflows/lightx2v-quick-start.yml](../../.github/workflows/lightx2v-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：推理加速；支持程度：新兴适配；阶段 A）。

在单卡昇腾 NPU 上跑通 LightX2V：文档自装 torch 栈（torch 2.9.0 + torchvision 0.24.* + torch_npu 2.9.0.post2 + triton 3.5.*）→ 源码安装（lightx2v 未发布到 PyPI；`git clone` 后 `uv pip install --no-deps` 装代码，再按 wan2.1 t2v 路径模块级实际导入的清单补齐依赖）→ 验证 NPU 分发（`PLATFORM=ascend_npu` → `platform: ascend_npu npu` + `npu available: True`）→ 用官方 `LightX2VPipeline` Python API 配仓库自带的 NPU 配置 `configs/platforms/ascend_npu/wan_t2v.json` 跑 Wan2.1-T2V-1.3B 文生视频，内嵌 `snapshot_download` 自动把 ~17.6 GB 权重拉到 ModelScope 默认缓存，产物做 MP4 结构校验。多卡并行 / 量化 / 服务化部署为文档结尾一句指引链接，不在看护范围。

依赖清单为什么只装这几个、为什么 `--no-deps`：上游 `pyproject.toml` 声明了只在 x86_64 提供预编译包的依赖（如 `decord`），整包解析在 aarch64 runner 上装不动。看护清单是走一遍 `lightx2v` + `lightx2v.common.ops` + `lightx2v.models.runners.wan.wan_runner` 的模块级 import 闭包得到的，闭包上其余三方导入全部是函数级且被 `try` / `except ImportError` 兜住的可选后端（CUDA / 其他加速器的 attention 与量化）；`cv2` / `decord` / `torchaudio` 根本不在闭包上，所以测试类也不再需要 stub。

## 触发

`lightx2v-quick-start.yml` 接受：

- `schedule`：每 6 小时轮询上游一次，上游新 release / 文档有变化才在 NPU 上跑 `tests.test_quick_start_ascend`（当前暂时注释，跑通后恢复）。
- `workflow_dispatch`：手动 trigger。

文档跟随上游最新 release tag，clone 块经 store/load 把引擎注入的 `UPSTREAM_REF` 打进 `git clone --branch <ref>`，监视什么就装什么，无需 fixed_ref。

cwd 是 `workflows/projects/lightx2v`（测试类 chdir 到 `/root/lightx2v-test` 钉文档执行目录，ModelScope 默认缓存 `~/.cache/modelscope` 由宿主卷 `/data/ci-cache/modelscope/lightx2v` 持久化）。环境契约：`MONITORED_DOC_URL` / `UPSTREAM_REF` / `NPU_READY` 由 engine 注入；CANN env source、CUDA 排除清单、卡号 pin、torch 栈探针、ModelScope 缓存校验、文档执行目录 chdir 等纯 CI 侧准备都在测试类的 `prepare_environment` 钩子里，文档保持纯用户视角。

详细 trigger 模式与 cache I/O 流程参见父引擎 [quick-start-template.yml](../../.github/workflows/quick-start-template.yml) 与项目文档说明 [docs/guarding-examples.md](../../docs/guarding-examples.md)。


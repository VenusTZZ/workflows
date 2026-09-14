# xDiT

本目录是 [xDiT](https://github.com/xdit-project/xDiT)（PyPI 包名 [xfuser](https://pypi.org/project/xfuser/)）的看护配套数据，不是 xDiT 源码。流水线在 [.github/workflows/xdit-quick-start.yml](../../.github/workflows/xdit-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：推理加速；支持程度：新兴适配；阶段 A）。

在昇腾 NPU 上跑通 xfuser：文档自装 torch 栈（torch 2.9.0 + torch_npu 2.9.0.post2 + triton 3.5.*）→ PyPI 安装 xfuser（`uv pip install xfuser`）→ 验证 NPU 分发（`npu hccl True`）→ 最小 SD3 脚本（内嵌 `snapshot_download` 自动拉模型，约 28 GB，走 ModelScope 默认缓存）经 `torchrun --nproc_per_node=1` 生成一张 256×256 单步图片并做 PNG 结构校验 → 同脚本加 `--ulysses_degree 2` 展示 2 卡序列并行。多卡高级用法（PipeFusion / CFG / Ring）为文档内指引链接，不在看护范围。

## 触发

`xdit-quick-start.yml` 接受：

- `schedule`：每 6 小时轮询上游（当前暂时注释，2 卡改造跑通后恢复）。
- `workflow_dispatch`：手动 trigger。

cwd 是 `workflows/projects/xdit`（测试类 chdir 到 `/root/xdit-test`，ModelScope 默认缓存 `~/.cache/modelscope` 由宿主卷 `/data/ci-cache/modelscope/xdit` 持久化）。

# xDiT

本目录是 [xDiT](https://github.com/xdit-project/xDiT)（PyPI 包名 [xfuser](https://pypi.org/project/xfuser/)）的看护配套数据，不是 xDiT 源码。流水线在 [.github/workflows/xdit-quick-start.yml](../../.github/workflows/xdit-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：推理加速；支持程度：新兴适配；阶段 A）。

在昇腾 NPU 上跑通 xfuser：文档先安装 torch 2.9.0 + torch_npu 2.9.0.post6，再通过 pip 安装 xfuser 并打印其版本；运行示例前安装 triton 3.5.0 与 ModelScope。测试夹具把文档中的 SD3 Python 代码块写入 `sd3_npu.py`，脚本通过 ModelScope 自动下载约 28 GB 的模型 → `torchrun --nproc_per_node=1` 生成单卡图片 → 同脚本加 `--ulysses_degree 2` 生成双卡图片。脚本分别打印推理耗时和带有并行参数的图片路径；图片的 PNG 结构校验（存在 + 大小下限 + 魔数）由测试夹具完成。多卡高级用法（PipeFusion / CFG / Ring）为文档内指引链接，不在看护范围。

## 触发

`xdit-quick-start.yml` 接受：

- `schedule`：每 6 小时轮询上游（当前暂时注释，2 卡改造跑通后恢复）。
- `workflow_dispatch`：手动 trigger。

cwd 是 `workflows/projects/xdit`（测试类 chdir 到 `/root/xdit-test`，ModelScope 默认缓存 `~/.cache/modelscope` 由宿主卷 `/data/ci-cache/modelscope/xdit` 持久化）。

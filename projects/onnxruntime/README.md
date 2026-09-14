# onnxruntime

本目录是 [onnxruntime](https://github.com/microsoft/onnxruntime) 的看护配套数据，不是 onnxruntime 源码。example 流水线在 [.github/workflows/onnxruntime-examples.yml](../../.github/workflows/onnxruntime-examples.yml)。Quick Start 流水线是另一条线，文件在 `.github/workflows/onnxruntime-quick-start.yml`。注册信息见根目录 [projects.yaml](../../projects.yaml)。分类：推理加速；支持程度：基础支持；阶段 A。

软件本体仍是 `microsoft/onnxruntime`。example 线 checkout 的是用户会抄的样例仓 [microsoft/onnxruntime-inference-examples](https://github.com/microsoft/onnxruntime-inference-examples)，默认分支 `main`。主仓 `samples/` 几乎空了，样例已经迁到这个仓。

上游有 CANN Execution Provider，没有健康的昇腾 CI。本仓先走阶段 A。能在昇腾机器上跑起来的 example 都进 `supported`。当前两条都是主机 CPU 脚本，不要把它们读成昇腾推理绿。CANN 推理看护在 Quick Start 线。example 线禁止改工作副本源码。

## 清单

`examples_manifest.yaml` 的扫描根是样例仓根目录，单位是 `.py` 文件。重新生成会整文件覆盖 `--output`。生成器不会合并已填的 `profile`、`overlay_args`。

```bash
python3 scripts/bootstrap_manifest.py \
  --target-root /path/to/onnxruntime-inference-examples \
  --output projects/onnxruntime/examples_manifest.yaml \
  --scan-root . \
  --include-extension .py
```

`supported` 有两条。都挂 `linux-aarch64-a2-1`、`npu_devices: '0'`、镜像 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。不再从源码 `--use_cann` 编译。每次 run 从华为云 PyPI 装当时的 CPU 包 `onnxruntime` / `onnx`，不钉次版本；新轮子把 example 搞挂，就要红。仍写 `numpy<2`：这些轮子按 NumPy 1.x 编，pip 拉到 NumPy 2 时 `import onnxruntime` 会在 example 开始前失败，那是看护噪音。两条都不要装 `onnxruntime-cann`：`quantize_static` 会自己建会话，CANN 轮子在场会把校准抢到 NPU 上。

- `cpu-python` 额外从华为云 PyPI 先装 torch 的 Python 依赖，含 `onnxscript`，因为 `torch==2.9.0` 的 `torch.onnx.export` 会 import 它。再从阿里云 CPU find-links 用 `--no-index` 装钉死的 `torch==2.9.0`。不要用上游 `python/api/requirements.txt` 里的 `download.pytorch.org/whl/cu128`。
- `quant-cpu` 只加 `pillow`，不装 `torch`。

`unsupported` 只表示本看护体系当前不跑，不是社区支不支持。不要把 CUDA / OpenVINO / Azure / TensorRT 样例从 `unsupported` 改回去。那些在本机没有对应后端。未知 `profile` 在装任何包之前非 0 退出，并打印 `cpu-python` `quant-cpu`。

## 绿灯含义

- `cpu-python`，路径 `python/api/getting_started.py`。按上游原文跑：用 `torch` 导出加法图，会话只注册 `CPUExecutionProvider`，有 CUDA 才插 CUDA。昇腾机器上没有 CUDA，就是 CPU。日志必须出现加法结果，形态是 NumPy 打印的 `[5. 7. 9.]` 这类。**绿灯 = 上游入门脚本在 CPU 上算出了加法，不是昇腾推理绿。**
- `quant-cpu`，路径 `quantization/image_classification/cpu/run.py`。这是 CPU 静态量化工具，按上游原文跑。日志必须有 `Calibrated and quantized model saved.`，并且写出 `mobilenetv2-7.quant.onnx`。日志里出现 `CANNExecutionProvider` 必须红。**这条不是昇腾推理绿。**

## 不改工作副本

`setup_example.sh` 只装依赖、检查文件在不在。禁止改 `TARGET_ROOT` 里的源码，禁止 `git add` / `commit` / `push` 回被测仓。规模压缩只走清单 `overlay_args`，那是 `run.py` 本来就认的 CLI。

## 不看护的相邻仓库

不看护 [microsoft/onnxruntime-training-examples](https://github.com/microsoft/onnxruntime-training-examples)。仓库已归档。CANN 没有训练 EP。

不看护主仓 `microsoft/onnxruntime` 里的 `onnxruntime/test/providers/cann` gtest，也不再编 `samples/cxx`。gtest 不是用户 example；`samples/cxx` 默认不注册 CANN EP。

## 触发

`onnxruntime-examples.yml` 有两种入口。`monitor` 跑在 `ubuntu-latest`，不占 NPU。

- `schedule`。cron 写在文件里但是注释掉的。接入阶段保持注释，不要打开。
- `workflow_dispatch`。手动触发。默认 `force=false`，和定时走同一套监控。只有 `force=true` 才跳过监控门。`target_repo` 和 `target_ref` 只在 `force=true` 时有意义。默认 `target_repo` 是 `microsoft/onnxruntime-inference-examples`。

两个监控信号是「或」，没有优先级，失败不重试。

1. 清单 `supported` 各 `path` 在样例仓 `main` 上的 Contents API 哈希。404 记成 `MISSING`。
2. `/releases/latest` 的 release 数字 id。该仓目前 0 个 release，404 时这个信号保持不变，不让 monitor 红。

`Decide targets` 成功之后才写 `.monitor-state`。任一步失败都不保存候选状态。

## Quick Start

文档在 `docs/Quick-start-Ascend.md`。Quick Start 是从 GitHub 当前正式 Release 源码编译 `onnxruntime-cann`，和 example 线不是同一套制品：example 追 CPU 包 `onnxruntime`，文档追上游源码 tag。索引用 `https://repo.huaweicloud.com/repository/pypi/simple`。昇腾专用索引没有 `onnxruntime-cann`。

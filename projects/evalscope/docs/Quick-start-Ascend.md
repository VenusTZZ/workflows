# 快速开始：在昇腾 NPU 上用 EvalScope 跑一次 GSM8K + ARC 评测

> **阅读本文前**，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。

[EvalScope](https://github.com/modelscope/evalscope) 是 ModelScope 团队推出的轻量级大模型评测框架，统一封装了 LLM / VLM / RAG / AIGC 等多类评测后端，并提供可视化看板。本文用其**原生（Native）后端**（基于 transformers + accelerate），在单卡昇腾 NPU 上安装 EvalScope，对 [Qwen/Qwen2.5-0.5B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct) 跑通 `gsm8k`（数学，0-shot）与 `arc`（AI2 推理，含 ARC-Easy / ARC-Challenge）两个官方数据集各 `--limit 5` 的冒烟评测，最后校验两个数据集的 `Accuracy` 都落在 [0,1] 区间。模型与数据集均经 ModelScope 自动下载，避免直连 HuggingFace。

---

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为**单卡**。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装并可 `source set_env.sh` |
| Python | 3.12 |
| PyTorch | `torch==2.9.0` 与 `torch_npu==2.9.0.post2`，见下文安装 |
| EvalScope | 从 PyPI 安装（撰写时 `1.11.0`），见下文 |
| 模型 / 数据 | ModelScope 上的 Qwen2.5-0.5B-Instruct 与评测数据集 |

**配套机器**：Atlas 900 A2（Ascend 910B4，64 GB × 1）。**配套镜像**：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`（与本文 `torch` / `torch_npu` 栈对齐）。

---

## 1. 加载 CANN 环境

新开终端后 CANN 变量不会自动生效。`npu-smi` 常见位于 `/usr/local/sbin`，需要把该目录加入 `PATH`。

```shell #test-setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
test -n "$ASCEND_HOME_PATH"
command -v npu-smi
```

`ASCEND_HOME_PATH` 应非空，`npu-smi` 应能找到。

---

## 2. 检查环境是否就绪

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

检查 torch / torch_npu 是否装好且 NPU 设备可用：

```shell #test id="check-torch"
python -c "import torch, torch_npu; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="check-torch" fuzzy='xxx'
torch=xxx
torch_npu=xxx
is_available: True
count: 1
```

```{admonition}
:class: note
如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。
```

---

## 3. 安装 EvalScope

EvalScope 的 Native 后端基于 transformers，因此评测机需已具备 §2 的 `torch` / `torch_npu` 栈。直接用 `pip` 安装 EvalScope 本体（安装时会顺带拉取 transformers / modelscope 等导入期依赖），装完打印版本并再次确认 NPU 栈未被污染：

```shell #test id="install-evalscope"
python -m pip install "evalscope==1.11.0" "numpy>=1.23.4,<2"
python -c "import evalscope, transformers, modelscope, torch, torch_npu; print('evalscope', evalscope.__version__); print('transformers', transformers.__version__); print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu_available', torch.npu.is_available())"
```

输出结果如下：

```shell #test-result id="install-evalscope" fuzzy='xxx'
evalscope 1.11.0
transformers xxx
torch 2.9.0...
torch_npu 2.9.0.post2
npu_available True
```

`npu_available` 必须是 `True`。`False` 时不要继续，先查 CANN、驱动和可见设备。若 `torch` 不再以 `2.9.0` 开头，说明后面的包把 NPU 栈换掉了，回到 §2 重装 `torch` / `torch_npu` 后再装 EvalScope。

---

## 4. 准备模型与数据

权重从 ModelScope 拉取，避免直连 HuggingFace。第一次约 1 GB，之后走本地缓存（见 §7 流水线挂载的 `/root/.cache/modelscope`）。

```shell #test-setup store="model_path"
python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-0.5B-Instruct'))"
```

确认权重目录完整：

```shell #test id="check-model" load="model_path>>model_path"
test -f "<model_path>/config.json" && echo has_config True
```

输出结果如下：

```shell #test-result id="check-model"
has_config True
```

`gsm8k` 与 `arc` 数据集在评测时由 EvalScope 通过 ModelScope 自动下载到本地缓存（`--limit 5` 只取少量样本，下载很快）。

---

## 5. 在 NPU 上评测

### 5.1 先确认模型能够加载到 NPU

下面这段用 transformers 直接把模型以 `device_map="auto"` 加载，并打印第一个参数的设备，断言其为 `npu` 开头。EvalScope 的 Native 后端走的是同一套 transformers + accelerate 加载机制，因此这一步等价于验证 EvalScope 在 NPU 上的加载路径。

```shell #test id="npu-device-check" load="model_path>>model_path"
python -c "
from transformers import AutoModelForCausalLM
import torch
m = AutoModelForCausalLM.from_pretrained('<model_path>', torch_dtype='auto', device_map='auto')
dev = str(next(m.parameters()).device)
print('evalscope_npu_device', dev)
assert dev.startswith('npu'), dev
print('EVALSCOPE_NPU_DEVICE_OK')
"
```

输出结果如下：

```shell #test-result id="npu-device-check" fuzzy='xxx'
...
evalscope_npu_device npu:0
EVALSCOPE_NPU_DEVICE_OK
```

`evalscope_npu_device` 必须是 `npu:0`。若打印 `cpu`/`cuda`，说明 `torch_npu` 没装上或可见设备为空，回到 §2–§3 排查。

### 5.2 运行 EvalScope 评测

`evalscope eval` 命令在指定数据集上用默认配置评测模型。`--model` 指向本地权重，`--model-args` 传入 `device_map="auto"` 让 accelerate 自动把模型放到 NPU，`--work-dir` 指定固定输出目录便于结果校验，`--limit 5` 是跑通口径（不代表真实榜单值）。评测机已通过环境变量 `ASCEND_RT_VISIBLE_DEVICES=0` 限定单卡。

```shell #test id="run-eval" load="model_path>>model_path"
evalscope eval \
  --model <model_path> \
  --datasets gsm8k arc \
  --limit 5 \
  --model-args '{"device_map": "auto"}' \
  --work-dir outputs/evalscope_npu_qs \
  2>&1 | tee outputs/evalscope_npu_qs_run.log
echo "EVALSCOPE_NPU_DONE"
```

输出结果如下（评测日志较长，此处仅校验末尾标记与模型名）：

```shell #test-result id="run-eval" fuzzy='xxx'
...
Qwen2.5-0.5B-Instruct ...
...
EVALSCOPE_NPU_DONE
```

---

## 6. 检查评测结果

EvalScope 会把评测报告写入 `--work-dir` 下的 `reports/` 目录（JSON）。下面读取报告，断言所有出现的 `score` 字段都落在 [0,1] 区间并打印，作为冒烟通过的数值依据：

```shell #test id="check-acc"
python << 'PY'
import glob
import json

found = []
for path in glob.glob('outputs/evalscope_npu_qs/reports/*.json'):
    try:
        data = json.load(open(path, encoding='utf-8'))
    except Exception:
        continue

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == 'score' and isinstance(v, (int, float)):
                    found.append(float(v))
                else:
                    walk(v)
        elif isinstance(o, list):
            for it in o:
                walk(it)

    walk(data)

assert found, 'no score found in evalscope report'
for s in found:
    assert 0.0 <= s <= 1.0, f'score out of range: {s}'
print('evalscope scores in [0,1]:', [round(s, 4) for s in found])
PY
```

输出结果如下：

```shell #test-result id="check-acc" fuzzy='xxx'
...
evalscope scores in [0,1]: ...
```

更多数据集、批量评测与可视化看板用法见 [EvalScope 官方文档](https://evalscope.readthedocs.io/zh-cn/latest/)。

---

## 7. 故障排查

| 现象 | 可能原因 | 建议 |
| --- | --- | --- |
| `torch.npu.is_available()` 为 `False` | 未 `source set_env.sh`，或设备未挂进容器 | 重做 §1–§2 |
| `pip` 把 `torch` 换成 CPU / CUDA 轮子 | 安装 EvalScope 时未锁定 NPU 栈 | 卸掉后按 §2–§3 重装 `torch` / `torch_npu` 再装 EvalScope |
| `evalscope_npu_device` 为 `cpu` | `torch_npu` 没装上，或可见设备为空 | 检查 `npu_available` 和 `ASCEND_RT_VISIBLE_DEVICES` |
| 评测极慢或显存/OOM | 模型被加载到 CPU，或 `limit` 过大 | 确认 §5.1 通过；CI 已用 `--limit 5` 压规模 |
| ModelScope 下载超时或落到 HTML | 出口到 modelscope.cn 失败 | 检查网络后重跑；不要改成 HuggingFace 直连 |
| `pip install evalscope` 解析到错误 `torch` 版本 | 依赖范围与本地 NPU 栈不兼容 | 固定 `torch==2.9.0` `torch_npu==2.9.0.post2`，必要时用集群内 PyPI 缓存 |

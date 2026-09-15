# Quick Start（Ascend NPU）

在双卡昇腾 NPU 上跑通 [Accelerate](https://github.com/huggingface/accelerate) 的三大核心能力：

- **训练代码适配**（`acc-launch` / `acc-launch-bf16` / `acc-train-single` / `acc-prepare`）：多卡 DDP 训练（fp32 + bf16）+ 单卡训练 + 单卡推理三档场景都过 `Accelerator.prepare`，训练三档额外走 `Accelerator.backward`；
- **分布式评估**（`acc-gather-multi`）：DDP 双进程下 `gather_for_metrics` 真的跨卡集合通信（hccl 后端），而不是退回单进程 identity；
- **大模型推理**（`acc-empty-weights` / `acc-dispatch`）：`init_empty_weights` 把大模型骨架以 meta 占位符方式创建（0 显存），`load_checkpoint_and_dispatch` 真实走一遍权重跨卡分片（不依赖 hub，自建 toy checkpoint）；

DDP 训练与跨卡集合通信都需要至少 2 张 NPU；单卡 runner 上 `accelerate launch --num_processes 2` 会直接报「visible devices 不够」。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的 `torch` + `torch_npu`，且 `torch` 能正常 `import` 并 `torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）

Accelerate 通过 `torch_npu` 间接支持昇腾 NPU（无需在 Accelerate 端引入额外后端），只要 `torch` / `torch_npu` 已就绪，Accelerate 的 `Accelerator` 类就能正确探测到设备并把张量放到 NPU 上。

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，64 GB × 2）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post2 |
| transformers | `<5.0` |
| accelerate | 本仓库当前 main 分支源码（`uv pip install -e .`，对应 release tag 见下） |

### 检查前置是否满足

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：
```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

检查 torch / torch_npu 是否装好且 NPU 设备可用，下面用 **python** 跑：

```python #test id="check-torch"
import torch, torch_npu
print('torch=', torch.__version__)
print('torch_npu=', torch_npu.__version__)
print('is_available:', torch.npu.is_available())
print('count:', torch.npu.device_count())
```

输出结果如下：

```shell #test-result id="check-torch" fuzzy='xxx'
torch= xxx
torch_npu= xxx
is_available: True
count: xxx
```

确认能看到 NPU 设备：

```shell
npu-smi info
```

输出如下类似结果：

```
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.2                   Version: 25.5.2                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
| 1     910B4               | OK            | 89.9        39                0    / 0             |
| 1                         | 0000:42:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 0                                                            |
| No running processes found in NPU 1                                                            |
+===========================+===============+====================================================+
```

> 如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动；如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

## 安装 accelerate

### 使用 uv 进行安装

通过 PyPI 镜像直接装最新 release 的二进制 wheel：

```shell #test id="acc-install-binary"
uv pip install --index-url https://mirrors.aliyun.com/pypi/simple accelerate
python -c "import accelerate; print('accelerate', accelerate.__version__)"
```

输出结果类似如下：

```shell #test-result id="acc-install-binary" fuzzy='xxx'
accelerate xxx
```
- xxx 表示最新的版本号

<!--
```shell #test-setup
uv pip uninstall accelerate -y
```
-->

### 从源码安装

<!-- 工作流注入的 UPSTREAM_REF（最新 release tag）通过这个隐藏的 #test-setup 捕获并注入到下方 install 命令中-->
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到最新 release tag，安装并且验证：

```shell #test id="acc-install-source" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/huggingface/accelerate.git
cd accelerate
uv pip install -e .
python -c "import accelerate; print('accelerate', accelerate.__version__)"
```
\<ref> 为安装的最新的 release tag

输出结果类似如下：

```shell #test-result id="acc-install-source" fuzzy='xxx'
accelerate xxx
```
- xxx 表示最新的版本号

> 如果机器在国内、无稳定外网，请参考 [huawei pypi 镜像](https://repo.huaweicloud.com/ascend/repos/pypi) 配置 `pip install -i ...`。

### CLI 自检：`accelerate env`

`accelerate env` 打印出当前环境对 PyTorch / 分布式后端 / 设备信息的探测结果。在昇腾 NPU 上跑应当能看到 `torch_npu` 已被识别：

```shell #test id="acc-env"
ASCEND_RT_VISIBLE_DEVICES=0,1 accelerate env
```

输出结果类似如下：

```shell #test-result id="acc-env" fuzzy='xxx' fuzzy='...'
...
Copy-and-paste the text below in your GitHub issue

- `Accelerate` version: xxx
- Platform: xxx
- `accelerate` bash location: xxx
- Python version: xxx
- Numpy version: xxx
- PyTorch version: xxx
- PyTorch accelerator: NPU
- System RAM: xxx
- CANN version: xxx
...
```

其中 `PyTorch accelerator: NPU` 是 accelerate 探测到 `torch_npu` 后给出的标识；`CANN version` 一行只有在 NPU 环境才会出现。如果 `PyTorch accelerator` 不是 `NPU`，多半是 `torch_npu` 没被 import 到——回到「基础软件」一节检查。

## 训练代码适配

下面把上游 [Quicktour](https://github.com/huggingface/accelerate/blob/main/docs/source/quicktour.md) 里「训练代码适配」一节的训练循环压到最小，目标是验证 `Accelerator.prepare` / `Accelerator.backward` 在 NPU 上跑通。模型只用一个小线性层，但走的是 Accelerate 的全套适配路径。

### 写最小训练脚本

保存为 `train_npu.py`：

```shell #test-setup store="script_path"
cat > train_npu.py <<'PY'
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator


def build():
    # Toy regression: y = 2 * x + 1, learned by a single linear layer.
    x = torch.linspace(-1.0, 1.0, 64).unsqueeze(1)
    y = 2 * x + 1 + 0.05 * torch.randn_like(x)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    model = nn.Linear(1, 1)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    return loader, model, optim


def main():
    accelerator = Accelerator()
    loader, model, optim = build()
    model, optim, loader = accelerator.prepare(model, optim, loader)
    loss_fn = nn.MSELoss()
    for step, (xb, yb) in enumerate(loader):
        preds = model(xb)
        loss = loss_fn(preds, yb)
        accelerator.backward(loss)
        optim.step()
        optim.zero_grad()
        # 3 steps are enough to drive loss below 0.5 on this toy task.
        if step == 2:
            break
    # accelerator.print only emits on the main process, so the #test-result
    # below sees exactly one line regardless of --num_processes.
    accelerator.print(
        f"device={accelerator.device.type} "
        f"final_loss={loss.item():.4f}"
    )


if __name__ == "__main__":
    main()
PY
echo "${PWD}/train_npu.py"
```

### 用 `accelerate launch` 启动

```shell #test id="acc-launch" load="script_path>>path"
ASCEND_RT_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision no <path>
```

其中 `<path>` 是上面「写最小训练脚本」一节生成的脚本绝对路径（即 `${PWD}/train_npu.py`）

输出结果类似：

```shell #test-result id="acc-launch" fuzzy='xxx'
device=npu final_loss=xxx
```

### 单卡训练

 `Accelerator()` + 完整训练循环（forward + `accelerator.backward` + `optim.step`），验证单卡 NPU 上完整训练链路。下面用 **python** 跑：

```python #test id="acc-train-single"
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator

x = torch.linspace(-1.0, 1.0, 64).unsqueeze(1)
y = 2 * x + 1 + 0.05 * torch.randn_like(x)
ds = TensorDataset(x, y)
loader = DataLoader(ds, batch_size=8, shuffle=True)
model = nn.Linear(1, 1)
optim = torch.optim.SGD(model.parameters(), lr=0.1)

accelerator = Accelerator()
model, optim, loader = accelerator.prepare(model, optim, loader)
loss_fn = nn.MSELoss()
for step, (xb, yb) in enumerate(loader):
    preds = model(xb)
    loss = loss_fn(preds, yb)
    accelerator.backward(loss)
    optim.step()
    optim.zero_grad()
    if step == 2:
        break
print(f'device={accelerator.device.type} final_loss={loss.item():.4f}')
```

输出结果如下：

```shell #test-result id="acc-train-single" fuzzy='xxx'
device=npu final_loss=xxx
```

### 独立 `Accelerator.prepare`

`Accelerator()` + `accelerator.prepare(model)` 跑一次 forward，验证 Accelerate 在 NPU 上：

1. **设备探测**：`Accelerator()` 自动识别 `torch_npu`、拿到 `device='npu:0'`；
2. **NPU 放置**：`accelerator.prepare(model)` 在非 distributed 上下文只做 `model.to(self.device)`，权重真在 NPU 上分配；
3. **真 kernel 跑**：`prepared(x)` 在 `npu:0` 上跑一次 1×1 matmul + bias add，**不是只 alloc**——NPU kernel 真跑了一次。

下面用 **python** 跑：

```python #test id="acc-prepare"
import torch
from torch import nn
from accelerate import Accelerator

accelerator = Accelerator()
model = nn.Linear(1, 1)
prepared = accelerator.prepare(model)
# prepare moves the model to the device, NOT your inputs —
# data tensors must be placed explicitly.
x = torch.tensor([[0.5]], device=accelerator.device)
y = prepared(x)
print(f'device={y.device.type}')
print(f'shape={list(y.shape)}')
```

输出结果如下：

```shell #test-result id="acc-prepare"
device=npu
shape=[1, 1]
```

`accelerator.prepare(model)` 这一行是 Accelerate 的核心 abstraction——同一份脚本不写一行 `.cuda()` / `.npu()`，能自动适配 CPU / CUDA / NPU / XPU / MPS。

### 启动 bf16 混合精度路径

把「写最小训练脚本」一节生成的脚本 `train_npu.py` 再跑一遍，但把 `--mixed_precision` 从 `no` 切到 `bf16`，验证 Accelerate 的 autocast 包装在 NPU 运行。：

```shell #test id="acc-launch-bf16" load="script_path>>path"
ASCEND_RT_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision bf16 <path>
```
>其中 `<path>` 是上面「写最小训练脚本」一节生成的脚本绝对路径（即 `${PWD}/train_npu.py`）

输出结果如下：

```shell #test-result id="acc-launch-bf16" fuzzy='xxx'
device=npu final_loss=xxx
```

## 分布式评估

分布式评估 = 多张 NPU 协同算一个验证集上的指标。`Accelerator()` 起 DDP 后，每个 rank 只看到数据集的 `1/world_size` 子集——必须把各 rank 算出的预测 / 标签汇总到主进程，再算最终 metric；否则算出来的只是「rank 0 看到的 1/N 数据」上的 metric，不是全量数据上的。

```python
validation_dataloader = accelerator.prepare(validation_dataloader)  # DDP 自动按 rank 切数据
for inputs, targets in validation_dataloader:                       # 每张卡独立看 1/world_size 子集
    predictions = model(inputs)                                       # 每张卡独立 forward
    all_predictions, all_targets = accelerator.gather_for_metrics(   # 跨卡 all_gather 汇总到主进程
        (predictions, targets)
    )
    metric.add_batch(all_predictions, all_targets)                    # 主进程算 metric（accuracy / F1 / ...）
```

下面抽 `gather_for_metrics` 这一行用最小可执行断言验它在 NPU 上真的跨卡做 `all_gather`（hccl 后端）：

```shell #test-setup store="gather_script_path"
cat > gather_npu.py <<'PY'
import torch
from accelerate import Accelerator

accelerator = Accelerator()
x = torch.tensor([1, 2, 3], device=accelerator.device)
(gathered,) = accelerator.gather_for_metrics((x,))
# accelerator.print only emits on the main process — without this guard
# the #test-result below would see two copies of each line.
accelerator.print(f'world={accelerator.num_processes}')
accelerator.print(f'device={gathered.device.type}')
accelerator.print(f'gathered={gathered.tolist()}')
PY
echo "${PWD}/gather_npu.py"
```

```shell #test id="acc-gather-multi" load="gather_script_path>>path"
ASCEND_RT_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 <path>
```

>其中 `<path>` 是`echo "${PWD}/gather_npu.py"` 的输出

输出结果如下：

```shell #test-result id="acc-gather-multi" fuzzy='...'
world=2
device=npu
gathered=[1, 2, 3, 1, 2, 3]
```

## 大模型推理

Accelerate 的大模型推理分两半：先用 [`init_empty_weights`](https://huggingface.co/docs/accelerate/main/en/usage_guides/big_modeling#initializing-an-empty-model) 在 `meta` device 上建空骨架（参数都是 meta 占位符，0 显存），再用 [`load_checkpoint_and_dispatch`](https://huggingface.co/docs/accelerate/main/en/usage_guides/big_modeling#sharding-checkpoints) 把分片权重塞进多张设备。本节两半都覆盖——meta 骨架用 1 层 toy config 验证 0 显存，dispatch 用 20 层 toy checkpoint + 显式 `device_map` 验证权重真分到两张卡上。真实场景（Mixtral-8x7B ~90 GB）跳到上游 [Big model inference](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference) 教程。

### init_empty_weights

下面用 **python** 跑：

```python #test id="acc-empty-weights"
from transformers import LlamaConfig, LlamaForCausalLM
from accelerate import init_empty_weights

# Inline config: 1 层 / 1 head / hidden=64 —— 没有 hub 网络依赖。
config = LlamaConfig(
    vocab_size=32,
    hidden_size=64,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=1,
    max_position_embeddings=64,
)
with init_empty_weights():
    model = LlamaForCausalLM(config)
n_params = sum(p.numel() for p in model.state_dict().values())
first_dev = next(model.parameters()).device
print(f'empty_model_params={n_params}')
print(f'device={first_dev}')
```

输出结果如下（参数数固定、device 固定为 meta —— 用 `fuzzy='xxx'` 让版本无关的字段差异不会挂）：

```shell #test-result id="acc-empty-weights" fuzzy='xxx'
empty_model_params=xxx
device=meta
```

`device=meta` 是 `init_empty_weights` 的合同行为：参数不在真实设备上，而是 PyTorch 的 meta 占位符，所以这一步**不占 NPU 显存**，CI 上跑稳。

### load_checkpoint_and_dispatch

在 `init_empty_weights` 建好的空骨架上跑 `load_checkpoint_and_dispatch`，验证 Accelerate 在 NPU 上真的把权重分到不同卡——前 10 层放 `npu:0`、后 10 层放 `npu:1`。CI 上不依赖 HF Hub / ModelScope 下载，自建 toy checkpoint。

先用一个 **python** setup 块造出 toy checkpoint，路径写到 `dispatch_ckpt` 供下面 `acc-dispatch` 用 `load='dispatch_ckpt>>ckpt'` 引用：

```python #test-setup store="dispatch_ckpt"
import os, torch, gc
from transformers import LlamaConfig, LlamaForCausalLM
from accelerate import init_empty_weights

# 20 层 / hidden=512 ≈ 80M 参数 bf16，~160 MB
config = LlamaConfig(
    vocab_size=32000, hidden_size=512, intermediate_size=1024,
    num_hidden_layers=20, num_attention_heads=8, max_position_embeddings=64,
)
with init_empty_weights():
    model = LlamaForCausalLM(config)
# save_pretrained 需要真实 tensor，先 to_empty 到 CPU + 填随机权重
model.to_empty(device='cpu')
with torch.no_grad():
    for p in model.parameters():
        p.data = torch.randn(p.shape, device='cpu', dtype=p.dtype) * 0.02
os.makedirs('/tmp/fake-llama-dispatch', exist_ok=True)
model.save_pretrained('/tmp/fake-llama-dispatch', safe_serialization=True)
del model
gc.collect()
print('/tmp/fake-llama-dispatch')
```

下面用 **python** 跑，`checkpoint='<ckpt>'` 里的 `<ckpt>` 由上面 `dispatch_ckpt` setup 块通过 `load='dispatch_ckpt>>ckpt'` 注入（替换后是合法 python 字符串字面）：

```python #test id="acc-dispatch" load="dispatch_ckpt>>ckpt"
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

config = LlamaConfig(
    vocab_size=32000, hidden_size=512, intermediate_size=1024,
    num_hidden_layers=20, num_attention_heads=8, max_position_embeddings=64,
)
# 显式 device_map：前 10 层 + embed 在 npu:0，后 10 层 + norm + lm_head 在 npu:1
device_map = {'model.embed_tokens': 'npu:0'}
device_map.update({f'model.layers.{i}': 'npu:0' for i in range(10)})
device_map.update({f'model.layers.{i}': 'npu:1' for i in range(10, 20)})
device_map.update({'model.norm': 'npu:1', 'lm_head': 'npu:1'})

with init_empty_weights():
    skeleton = LlamaForCausalLM(config)
loaded = load_checkpoint_and_dispatch(
    skeleton,
    checkpoint='<ckpt>',
    device_map=device_map,
    no_split_module_classes=['LlamaDecoderLayer'],
)

# 验证：前 / 后层落在不同卡
dev_first = loaded.model.layers[0].self_attn.q_proj.weight.device
dev_last = loaded.model.layers[-1].self_attn.q_proj.weight.device

# 跑一次 forward：dispatch hook 自动跨卡转 tensor
x = torch.tensor([[1, 2, 3]], device=loaded.device)
out = loaded(input_ids=x).logits

print(f'first_layer_device={dev_first}')
print(f'last_layer_device={dev_last}')
print(f'out_device={out.device.type}')
print(f'out_shape={list(out.shape)}')
```

输出结果如下：

```shell #test-result id="acc-dispatch"
first_layer_device=npu:0
last_layer_device=npu:1
out_device=npu
out_shape=[1, 3, 32000]
```
# Quick Start (Ascend NPU)

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载（`/dev/davinci*` 等）。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，32 GB × 2 — CANN / 驱动预留约 2.5 GB，每张卡 PyTorch 实测可用 ~30 GB）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.12.0 |
| torch_npu | 2.12.0 |
| triton-ascend | 3.5.0+dev20260701（Ascend nightly 源） |
| modelscope | 最新release |
| torchtitan | 最新 release |
| 训练配置 | 单卡 Step 12：`torchtitan/models/llama3/config_registry.py::llama3_debugmodel`（debugmodel：dim=256 / 6 层 / 16 head / vocab 2048，~6 M 参数）；多卡 Step 13：`torchtitan/models/llama3/config_registry.py::llama3_8b`（Llama 3 8B：dim=4096 / 32 层 / 32 head / 8 kv head，FSDP shard=2 + cpu_offload + 全量 bf16 装得下） |


### 检查前置是否满足

确认能看到 NPU 设备：

```shell
npu-smi info
```

输出类似：

```
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.2                   Version: 25.5.2                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 5     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 5                                                            |
+===========================+===============+====================================================+
```

> 如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动。

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：
```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

装 `torch` + `torch_npu` ：

```shell #test-setup
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.12.0
uv pip install --extra-index-url https://mirrors.aliyun.com/pypi/simple torch_npu==2.12.0
```

检查 torch / torch_npu 是否装好且 NPU 设备可用：

```shell #test id="check-torch"
python -c "import torch, torch_npu; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="check-torch" fuzzy='xxx'
torch= xxx
torch_npu= xxx
is_available: True
count: 2
```

> 如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

### 安装 modelscope

```shell #test-setup
uv pip install modelscope
```

打印安装版本：
```shell #test id="modelscope-install"
python -c "import modelscope; print('modelscope', modelscope.__version__)"
```

输出结果如下：
```shell #test-result id="modelscope-install" fuzzy='xxx'
modelscope xxx
```
### 安装 triton-ascend

NPU 上 `torch.compile`/inductor 走到 `torch_npu._inductor` 时需要 Ascend 的 Triton fork **triton-ascend**（为 `triton` 模块提供 Ascend 后端）；社区版 `triton` 只有 CUDA 后端，装了会在训练第一步报 `RuntimeError: 0 active drivers ([]). There should only be one.`。triton-ascend 的版本号对齐它 fork 的 triton 基线：torch 2.12 配套 triton 3.5，因此用 Ascend nightly 源 3.5.0 线的末位构建。注意华为云源的 triton-ascend（3.2.1 起的稳定版与全部 nightly）都声明 `triton==3.5.0` 依赖、文件设计为覆盖社区版目录：稳定 3.2.x 的 fork 基线（3.2）与所钉社区版（3.5.0）错配，混装后 `import triton` 报 `cannot import name 'Language'`；nightly 3.5.0 线基线匹配且 wheel 为完整 fork 可独立成立，但 `--no-deps` 仍然必要——不让社区版进环境，`triton/` 目录只归属 triton-ascend 一个包。`--no-deps` 跳过的依赖里，被 Ascend 后端运行期 import 的只有 **pybind11**（后端 utils/driver 在首次编译 kernel 时用它把 `npu_utils.cpp` 现场编成扩展，需要容器里有 g++/clang++ 和已 source 的 CANN 环境），单独补装；其余（numpy/pytest/pandas 等）是上游打包夹带的测试依赖，运行期不 import：

```shell #test-setup
uv pip install --no-deps --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi/nightly triton-ascend==3.5.0+dev20260701
uv pip install pybind11
```

打印安装版本：
```shell #test id="triton-install"
python -c "from importlib.metadata import version; print('triton-ascend', version('triton-ascend'))"
```

输出结果如下：
```shell #test-result id="triton-install" fuzzy='xxx'
triton-ascend xxx
```

## 安装 torchtitan

### 通过 uv 安装

torchtitan 上 PyPI 有 `py3-none-any` 的纯 Python wheel（450 KB），不依赖 native build——`uv pip install torchtitan` 直接装最新 stable release，不用管版本号：

```shell #test id="torchtitan-install-binary"
uv pip install --extra-index-url https://mirrors.aliyun.com/pypi/simple/ torchtitan
python -c "import torchtitan; print('torchtitan', torchtitan.__version__)"
```

输出结果类似如下：

```shell #test-result id="torchtitan-install-binary" fuzzy='xxx'
torchtitan xxx
```
- xxx 表示最新的版本号

<!--
```shell #test-setup
uv pip uninstall torchtitan -y
```
-->

### 从源码安装

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="torchtitan-install-source" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/pytorch/torchtitan.git
cd torchtitan
uv pip install -e .
python -c "import torchtitan; print('torchtitan', torchtitan.__version__)"
```
\<ref> 为工作流注入的最新 release tag

输出结果类似如下：

```shell #test-result id="torchtitan-install-source" fuzzy='xxx'
torchtitan xxx
```
- xxx 表示最新的版本号

## 跑训练

### 下载 tokenizer

```shell #test-setup id="modelscope-download-tokenizer" store="ms_tokenizer_path"
python -c "from modelscope import snapshot_download; print(snapshot_download('LLM-Research/Llama-3.2-1B', allow_patterns=['*.json', '*.model', 'tokenizer*']))" | tail -n 1
```

> 该下载演示 modelscope `snapshot_download` 的标准用法（多卡训练改为 debugmodel 后，训练自身使用 torchtitan 自带的 tests/assets/tokenizer，不依赖此路径）。

验证 tokenizer 关键文件都落盘：

```shell #test id="modelscope-verify-tokenizer" load="ms_tokenizer_path>>ms_tokenizer_path"
ls <ms_tokenizer_path>
```

```shell #test-result id="modelscope-verify-tokenizer" fuzzy='...' 
config.json
configuration.json
generation_config.json
original
special_tokens_map.json
tokenizer.json
tokenizer_config.json
```

文件名是 Llama 3 tokenizer 必备文件，确认 snapshot_download 命中正确。

### 兼容性补丁

torchtitan v0.3.0 + torch 2.12 + torch_npu 2.12.0 + triton-ascend 3.5.0 是一个双方生态都未验证过的组合，剥到最底层是两个硬限制 + 一个算子缺口，本文档最终只保留两处 `sed`：

**限制一：flex attention 在这套 NPU 栈上编不出来（硬墙，无法绕过，只能换 backend）。** v0.3.0 的语言模型路径强制 flex/varlen（`sdpa` 被 `config_utils.py::get_attention_config` 显式禁用），flex 必经 inductor 编译 Triton kernel。逐层剥开（顺序即迭代顺序）：`separate_full_blocks` 参数 torch ≥2.13 才有 → mask 图的双归约 kernel torch_npu codegen 不支持 → torch_npu×triton-ascend 六处 API 断层（`DeferredLine`、`triton_key` 路径、launch hooks、设备白名单无 `npu`、lowering 全量白名单误杀 flex 模板与 `aten.index`、`define_kernel` 签名漂移）——这些全部可用 sed 修复，且修法均与 torch_npu master 一致。但最后一层是 **CANN 9.1.0 的 bishengir-compile 编译器本身**编不了 inductor 生成的 flex 模板 kernel：`'hivm.hir.store' op only support store ub to gm currently!` / `'scf.for' op Failed to collect vector loop tiling info`（BiShengIR 流水线报错，在编译器二进制里，无法 patch）。因此本文档把 llama3 的 attention backend 切到 **SDPA**（`config_utils.py` 解除 sdpa 禁用 + `llama3/__init__.py` 默认 backend 改 `sdpa`）——trainer 本就支持 maskless SDPA 路径（靠 `is_causal`），torch_npu 的 SDPA 走 aclnn flash attention，是 NPU 生态的标准 attention 路径（vllm-ascend 同款）。代价：SMOKE 不再验证 flex kernel 本身，文档 masking 语义为纯 causal（对 2 步训练验证无影响）。

**限制二：NPU 算子缺口——`aclnnIndex` 不支持 complex64。** llama3 注册表默认 `ComplexRoPE`（complex64 缓存），forward 里 `rope_cache[positions]` 索引落到 `aclnnIndex` 直接报 `AclNN_Parameter_Error: not implemented for DT_COMPLEX64`。换成数学等价的实数实现 `CosSinRoPE`（cos/sin 缓存 + rotate-half）；它不支持 llama scaling，需一并把 `scaling="llama"` 改为 `"none"`——llama scaling 只影响 >8k 长上下文的频率插值，对本文档 256 seq 的 smoke 数值无影响。

**限制三：v0.3.0 的 `ChunkedLossWrapper` 在 NPU 上 backward 崩。** 该 wrapper 在 forward 内部逐 chunk 调 `chunk_loss.backward()`（backward-inside-forward）+ FSDP unshard/reshard 交错，在 NPU 上触发 `RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet`（meta 张量泄漏）。换成标准的 `CrossEntropyLoss`（上游同 registry 提供 `llama3_debugmodel_ce_loss` 同款配置，数学等价、只是无峰值内存优化）。

**限制四：多卡下默认的 `spmd_types` 后端在 torch 2.12 上不可用。** v0.3.0 默认 SPMD 后端 `spmd_types` 用惰性注解标记参数分布，需要 torch ≥2.13 的 FSDP `dp_mesh_dims` 把注解翻译成 DTensor；torch 2.12 的 FSDP 只认真 DTensor，多卡（dp_shard>1）时报 `ValueError: When dp_mesh_dims is provided, all parameters must be DTensors... Got plain tensor`。切到 `full_dtensor` 后端（`--parallelism.spmd-backend full_dtensor`，参数经 `distribute_tensor` 成为真 DTensor）即可，无需改代码；单卡不受影响（size-1 mesh 时 torchtitan 自己跳过该路径）。

**限制五：`set_pg_timeouts` 用了 torch 2.13+ 的 API。** `distributed/utils.py` 调 `torch.distributed.set_timeout(timeout, group)`——该模块级 API torch 2.13 才有，step 1 之后（`train()` 里调整 PG 超时）必炸 `AttributeError`。torch 2.12 等价物是实例方法 `ProcessGroup.set_timeout(timeout)`。

**限制六：8B 规模模型在本栈上不可用（多卡改用 debugmodel）。** 8B 走 `--training.enable-cpu-offload` 路径时权重在 CPU 上经 DTensor dispatch 逐参数 `init_weights`，实测单个 `trunc_normal_` 超过 1.5 小时不完成（CPU 持续 133% 在 `normal_fill`，DTensor in-place op 反复重派发，接近活锁）——即使加大 OMP 线程也不缓解。多卡章节因此用 `debugmodel` 双卡（FSDP shard=2 + HCCL 双卡 + DTensor 参数分布 + bf16 全部覆盖），8B 规模验证留待上游修复后恢复。

安装侧另有两点配合（见「安装 triton-ascend」一节）：`--no-deps` 防止 wheel 声明的社区版 `triton==3.5.0` 依赖混入覆盖 fork 文件；单独补装被跳过依赖里唯一被运行期 import 的 `pybind11`。

> 退役条件：限制二随 CANN 的 `aclnnIndex` 支持 complex64（或 torch_npu 补转换实现）后可移除；限制一的 SDPA 切换随 bisheng/triton-ascend 支持编译 inductor flex 模板 kernel 后整体回退（届时 flex 路径还需带上六处 torch_npu 断层的 sed，修法已在上游 master 验证过——等待回合 2.12 补丁版或 2.13）；限制三、四、五随 torch_npu 发布配套 torch ≥ 2.13 的版本自然消失。

### 单卡训练

用 `torchrun --nproc_per_node=1` 在 1 张 NPU 上跑 `debugmodel` 真跑 2 步，验证配置解析、初始化、加载 tokenizer、build dataloader、forward + backward 整条链路能跑通。`llama3_debugmodel` 是 torchtitan 自带的最小 smoke 配置（dim=256 / 6 层 / 16 head / vocab 2048，~6 M 参数量），单卡 30 GB 完全够装。走真实 HCCL backend（`--comm.mode default`）让 c10d 把 `npu` 路由到 `hccl`，1-rank 下所有集合通信都是 self-barrier，不会真的有跨卡流量；不要用 `--comm.mode fake_backend` —— 它只注册 `fake` PG，v0.2.2 在 step 1 之后调 `set_pg_timeouts` → `torch.distributed.barrier(device_ids=[npu:0])` 时会因 `default_device_backend_map["npu"]="hccl"` 但当前 PG 是 `fake` 抛 `RuntimeError: No backend type associated with device type npu`。多卡 FSDP shard 训练见下一节「多卡训练」（8B 规模在本栈暂不可用，见「兼容性补丁」限制六）：

```shell #test id="torchtitan-train-debug" load="upstream_ref>>ref"
cd torchtitan && git checkout -f <ref>
sed -i 's/^    ComplexRoPE,$/    ComplexRoPE,\n    CosSinRoPE,/; s/ComplexRoPE\.Config(/CosSinRoPE.Config(/; s/scaling="llama",/scaling="none",/' torchtitan/models/llama3/__init__.py
sed -i 's/attn_backend: str = "flex",/attn_backend: str = "sdpa",/' torchtitan/models/llama3/__init__.py
sed -i 's/    VarlenAttention,$/    VarlenAttention,\n    ScaledDotProductAttention,/' torchtitan/models/common/config_utils.py
sed -i 's/    elif backend == "sdpa":/    elif backend == "sdpa":\n        return ScaledDotProductAttention.Config()\n    elif backend == "sdpa_banned":/' torchtitan/models/common/config_utils.py
sed -i '/^        loss=ChunkedLossWrapper.Config($/,/^        ),$/c\        loss=CrossEntropyLoss.Config(\n            global_vocab_size=decoder_vocab_size(model_spec),\n        ),' torchtitan/models/llama3/config_registry.py
sed -i 's|        torch.distributed.set_timeout(timeout, group)|        (group if group is not None else torch.distributed.distributed_c10d._get_default_group()).set_timeout(timeout)|' torchtitan/distributed/utils.py
ASCEND_RT_VISIBLE_DEVICES=0 \
torchrun --nproc_per_node=1 \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    -m torchtitan.train \
    --module llama3 \
    --config llama3_debugmodel \
    --comm.mode default \
    --training.steps 2 \
    --training.local-batch-size 1 \
    --training.seq-len 256 \
    --metrics.log-freq 1 \
    --metrics.disable-color-printing \
    --dump-folder /tmp/torchtitan-quickstart
```

输出结果类似如下：

```shell #test-result id="torchtitan-train-debug" fuzzy='xxx' fuzzy='...'
[titan] xxx - root - INFO - torchtitan version: xxx
[titan] xxx - root - INFO - Building llama3 debugmodel
...
[titan] xxx - root - INFO - Sleeping 2 seconds for other ranks to complete
[titan] xxx - root - INFO - Training completed
[titan] xxx - root - INFO - Process group destroyed
```

### 多卡训练

用 torchrun 起 2 个 rank 跑 `debugmodel` 真分布式训练，`--training.steps 2` 真跑 2 步。`data_parallel_shard_degree = -1` 在双卡下解析成 2，FSDP 把 params / grads / Adam state 按 shard 分摊到两张卡，验证 HCCL 双卡集合通信 + FSDP shard>1 + DTensor 参数分布整条链路。`--parallelism.spmd-backend full_dtensor` 是多卡必须项（默认的 `spmd_types` 后端要 torch ≥2.13 的 FSDP 注解翻译，见「兼容性补丁」限制四）；叠 `--training.dtype bfloat16` 验证混合精度：

```shell #test id="torchtitan-train-2card" load="upstream_ref>>ref"
cd torchtitan && git checkout -f <ref>
sed -i 's/^    ComplexRoPE,$/    ComplexRoPE,\n    CosSinRoPE,/; s/ComplexRoPE\.Config(/CosSinRoPE.Config(/; s/scaling="llama",/scaling="none",/' torchtitan/models/llama3/__init__.py
sed -i 's/attn_backend: str = "flex",/attn_backend: str = "sdpa",/' torchtitan/models/llama3/__init__.py
sed -i 's/    VarlenAttention,$/    VarlenAttention,\n    ScaledDotProductAttention,/' torchtitan/models/common/config_utils.py
sed -i 's/    elif backend == "sdpa":/    elif backend == "sdpa":\n        return ScaledDotProductAttention.Config()\n    elif backend == "sdpa_banned":/' torchtitan/models/common/config_utils.py
sed -i '/^        loss=ChunkedLossWrapper.Config($/,/^        ),$/c\        loss=CrossEntropyLoss.Config(\n            global_vocab_size=decoder_vocab_size(model_spec),\n        ),' torchtitan/models/llama3/config_registry.py
sed -i 's|        torch.distributed.set_timeout(timeout, group)|        (group if group is not None else torch.distributed.distributed_c10d._get_default_group()).set_timeout(timeout)|' torchtitan/distributed/utils.py
ASCEND_RT_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    --local-ranks-filter 0 \
    --tee 3 \
    -m torchtitan.train \
    --module llama3 \
    --config llama3_debugmodel \
    --comm.mode default \
    --parallelism.spmd-backend full_dtensor \
    --training.dtype bfloat16 \
    --training.steps 2 \
    --training.local-batch-size 1 \
    --training.seq-len 256 \
    --metrics.log-freq 1 \
    --metrics.disable-color-printing \
    --dump-folder /tmp/torchtitan-quickstart-2card
```

输出结果类似如下：

```shell #test-result id="torchtitan-train-2card" fuzzy='xxx' fuzzy='...'
[default0]:[titan] xxx - root - INFO - torchtitan version: xxx
[default0]:[titan] xxx - root - INFO - Building device mesh with parallelism: xxx
[default0]:[titan] xxx - root - INFO - Building llama3 debugmodel
...
[default0]:[titan] xxx - root - INFO - Sleeping 2 seconds for other ranks to complete
[default0]:[titan] xxx - root - INFO - Training completed
[default0]:[titan] xxx - root - INFO - Process group destroyed
```

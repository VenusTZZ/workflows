# torchtitan v0.3.0 × torch 2.12 × 昇腾 NPU 问题分析报告

| | |
| --- | --- |
| 时间 | 2026-09-04 ~ 2026-09-07（首版）/ 2026-09-17（no-patch 复盘） |
| 看护对象 | [Quick-start-Ascend.md](../projects/torchtitan/docs/Quick-start-Ascend.md)（C 类自维护文档） |
| 上游信号 | torchtitan **v0.3.0**（2026-09-03 发布，首个 stable release，此前 v0.2.2 均为 prerelease） |
| 触发原因 | 上游发版自动触发重测，CI 失败 |
| 首次结论 | **CI 全绿**（run 34087315070，单卡 + 双卡训练各 2 步完整跑通） |
| 2026-09-17 复盘 | **撤掉全部 7 条 sed patch + 4 个 launcher 里的 `TORCH_NPU_DEVICE_CAPABILITY=9.0` env var**。按 no-patch policy，torchtitan 的「支持 torch_npu / 设备无关」主张应在 upstream 实现里兑现，不在 setup 层绕。CI 现在预期失败，**3 条** supported entry 留为「能支持但因上游 bug 失败」的占位（原 4 条 #1 default llama3_debugmodel 1card 与 #2 ce_loss 变体共享 launcher，仅 --config 区别，no-patch 下 default 的 ChunkedLossWrapper 路径是已知 NPU meta tensor leak，没法既留着 default 又让 ce_loss 单独信号，所以合并掉），等 torch_npu / torchtitan 各自修好对应问题后自动转绿。详见 §六。 |
| 环境 | Ascend 910B4 × 2 / CANN 9.1.0 / Python 3.12.13 / torch 2.12.0+cpu / torch_npu 2.12.0 / triton-ascend 3.5.0+dev20260701 |

## 一、问题概述

torchtitan v0.3.0 发布次日，看护流水线按 release 信号自动重测失败。此前的
v0.2.2 时代文档是绿的，v0.3.0 是一个大版本：Python 配置重构、模型层重构
（models/common 目录）、flex attention 成为语言模型唯一强制的 attention 路径。

从第一层报错剥到最后一层，共 **14 层独立问题**，最终收敛为**一个根本矛盾**：

> **torchtitan v0.3.0 的基线是 torch 2.14 nightly（release notes 明示 validated
> with PyTorch 2.14.0），而昇腾全栈（torch_npu / triton-ascend）最高只配套到
> torch 2.12。两个生态之间隔着两个大版本，这个组合双方都没有验证过。**

torch_npu 配套矩阵最高到 torch 2.12.0（↔ CANN 9.1.0），triton-ascend 稳定源
只有 3.2.x（torch 2.10 时代基线）——生态内被普遍验证过的组合是 **torch 2.10 +
triton-ascend 3.2.2**（本仓 flagscale / aibrix / ms-swift / llm-d 等项目的钉版
即是）。本文档为了跟 torchtitan 最新 release，用到了 torch_npu 配套上限
（torch 2.12）+ triton-ascend 3.5.0 nightly，各组件都处于各自生态的边缘。

## 二、问题分层与修复过程

14 层问题按修复手段分四类。全部修复已归档在文档「兼容性补丁」一节，每条带
根因、报错原文、退役条件。

### 2.1 编译器硬墙（1 层，无法 patch，只能绕）

| 层 | 报错 | 定性 |
| --- | --- | --- |
| flex kernel 编译 | `'hivm.hir.store' op only support store ub to gm currently!` / `'scf.for' op Failed to collect vector loop tiling info` | CANN 9.1.0 的 bishengir-compile（毕昇编译器）编不了 inductor 生成的 flex attention 模板 kernel。报错在**编译器二进制内部**，不是 Python 层，无法 sed |

**绕法**：v0.3.0 语言模型路径强制 flex/varlen（`sdpa` 被
`config_utils.py::get_attention_config` 显式禁用），且 varlen 的
`torch.nn.attention.varlen.varlen_attn` 是 CUDA FA 专属 API，NPU 无实现——
唯一可行解是把 llama3 的 attention backend 切回 **SDPA**（trainer 本就支持
maskless SDPA 路径，torch_npu 的 SDPA 走 aclnn flash attention，是 NPU 生态
的标准 attention 实现，NPU 推理/训练框架的通用选择）。代价：smoke 不再验证 flex kernel
本身，masking 语义从 block-causal 退化为纯 causal（对 2 步训练验证无影响）。

**注意**：到达这堵墙之前，flex 路径上还叠着 7 层可修的 Python 层断层（见
2.2/2.3），逐层修穿之后才撞到墙。这 7 层的修法（与 torch_npu master 一致）
已留档，待上游编译器支持后可整体恢复 flex 路径。

### 2.2 torch_npu 2.12.0 × triton-ascend 3.5.0 组合断层（8 层，sed 可修）

这类问题的共性：torch_npu 2.12.0 的 inductor 集成代码按 **torch 2.12-dev**
内部 API 写，而 triton-ascend 3.5.0 fork 又改了 **triton 3.5** 的 API——
三方（torch / torch_npu / triton-ascend）在这个组合上互相都对不上。上游
master 均已修复但未回合 2.12.0，修复手段全部是 sed 镜像上游 master 修法：

| # | 报错 | 根因 | 修法（= master 修法） |
| --- | --- | --- | --- |
| 1 | `TypeError: create_block_mask() got an unexpected keyword argument 'separate_full_blocks'` | torchtitan v0.3.0 传的参数 torch ≥2.13 才有 | torch 2.12 内部固定该值为 True，删参数语义不变（**注**：切 SDPA 后此路径不再执行，sed 保留为空操作） |
| 2 | `RuntimeError: 0 active drivers ([]). There should only be one.` | 装了社区版 triton（只有 CUDA 后端） | 换 triton-ascend fork |
| 3 | `ImportError: cannot import name 'Language' from 'triton.backends.compiler'` | 稳定源 triton-ascend wheel 声明依赖社区版 `triton==3.5.0`，两个包互写 `triton/` 目录形成残缺混合树 | nightly 源 3.5.0 线（fork 基线与所钉社区版一致）+ `--no-deps` 安装 |
| 4 | `ModuleNotFoundError: No module named 'pybind11'` | `--no-deps` 跳过的依赖里唯一被 ascend 后端运行期 import 的 | 单独补装 |
| 5 | `AttributeError: 'DeferredLine' object has no attribute 'find'` | torch 2.12 inductor 的 codegen 行是 `DeferredLine` 对象，torch_npu 按老 API 当字符串用 | 四个缓冲区循环统一先解包（master 的 `_iter_codegen_lines()` 修法） |
| 6 | `ImportError: cannot import name 'triton_key' from 'triton.compiler.compiler'` | triton-ascend fork 把函数挪到了 `triton.runtime.cache`；模块在而名字缺失抛 ImportError，torch_npu 的 `except ModuleNotFoundError` 接不住 | 改 import 路径 |
| 7 | `AttributeError: type object 'CompiledKernel' has no attribute 'launch_enter_hook'` | triton 3.5 把 launch hooks 挪进了 `triton.knobs.runtime` | 三处引用统一指向新位置 |
| 8 | `NameError: 'r2 is not defined'`（kernel 源码级残缺） | mask 图的 cumsum/scatter 被融合成**双归约轴 kernel**，torch_npu codegen 源码自述不支持双归约，生成引用未定义变量的残缺代码 | 去掉 `create_block_mask` 的 `torch.compile`，改跑 eager（一次性张量计算，语义不变） |

另外三处同类断层在切换 SDPA 后仍被踩到并修复：

| # | 报错 | 根因 | 修法 |
| --- | --- | --- | --- |
| 9 | `LoweringException: 'Subgraph' object has no attribute 'dtype'` | torch_npu 2.12.0 启动时把 torch 全部 inductor lowering 过一遍自己的白名单（约 70 个基础算子），**不在白名单一律替换成 eager fallback**——flex 模板 lowering 被误杀 | flex 两个 HOP 加白 |
| 10 | `SubgraphLoweringException: Buffers cannot be created while lowering a pointwise subgraph` | mask_mod 子图里 `offsets[document_id[q_idx]]` 用的 `aten.index` 也被白名单误杀（stock torch 本有正经 lowering） | `aten.index` 加白 |
| 11 | `TypeError: NPUTritonScheduling.define_kernel() missing 1 required positional argument: 'traced_graph_hash'` | torch_npu 按 torch 2.12-dev 的 4 参签名写，torch 2.12.0 正式版模板 codegen 只传 3 参 | 该参数加默认值 `None`（函数体内有空值 guard） |

### 2.3 NPU 算子 / 后端覆盖缺口（3 层，绕开而非修复）

| # | 报错 | 根因 | 绕法 |
| --- | --- | --- | --- |
| 12 | `AclNN_Parameter_Error: Tensor self not implemented for DT_COMPLEX64` | llama3 注册表默认 `ComplexRoPE`（complex64 缓存），`rope_cache[positions]` 索引落到 CANN `aclnnIndex`，该算子不支持 complex64 | 换数学等价的实数实现 `CosSinRoPE`；其不支持 llama scaling，一并改 `scaling="none"`（只影响 >8k 长上下文的频率插值，对 smoke 无影响） |
| 13 | `ValueError: When dp_mesh_dims is provided, all parameters must be DTensors... Got plain tensor` | v0.3.0 默认 `spmd_types` 后端用惰性注解标记参数分布，需要 torch ≥2.13 的 FSDP `dp_mesh_dims` 把注解翻译成 DTensor；torch 2.12 的 FSDP 只认真 DTensor | 多卡命令加 `--parallelism.spmd-backend full_dtensor`（纯 CLI，`distribute_tensor` 产真 DTensor；单卡 size-1 mesh 时 torchtitan 自己跳过该路径不受影响） |
| 14 | `AttributeError: module 'torch.distributed' has no attribute 'set_timeout'` | v0.3.0 的 `set_pg_timeouts` 用了 torch 2.13+ 的模块级 API，step 1 之后调整 PG 超时必炸 | sed 改为 torch 2.12 的实例方法 `ProcessGroup.set_timeout(timeout)` |

### 2.4 规模性限制（1 层，降级处理）

| # | 现象 | 定性 |
| --- | --- | --- |
| 8B 模型 init 活锁 | 8B + `--training.enable-cpu-offload` 路径权重在 CPU 上经 DTensor dispatch 逐参数 `init_weights`，py-spy 抓到单个 `trunc_normal_` 持续 >1.5 小时不完成（CPU 133% 在 `normal_fill`，DTensor in-place op 反复重派发） | debugmodel（6M 参数）在同路径秒级完成，纯规模放大问题 |

**处理**：多卡 smoke 从 8B 降级为 debugmodel 双卡——FSDP shard=2 / HCCL 双卡
集合通信 / DTensor 参数分布 / bf16 混合精度全部仍被覆盖，只损失 8B 规模本身。

### 2.5 附带发现（文档/测试工程问题）

| 现象 | 修复 |
| --- | --- |
| ChunkedLossWrapper（v0.3.0 默认 loss，forward 内部逐 chunk backward + FSDP unshard 交错）在 NPU 上触发 meta 张量泄漏：`The tensor has a non-zero number of elements, but its data is not allocated yet` | 换标准 `CrossEntropyLoss`（上游 registry 本就提供 `llama3_debugmodel_ce_loss` 同款配置） |
| flex kernel 的 `max_autotune` + `coordinate_descent_tuning` 在 NPU 上逐 config 编译实测，极易吃满命令超时 | torchtitan 源码注释自荐 "keep max_autotune disabled for faster compilation"，关掉（切 SDPA 后整条路径不再执行） |
| 测试框架命令超时 1300s 不够 NPU 首次编译 | `DEFAULT_COMMAND_TIMEOUT` 提到 3600s |
| v0.3.0 日志行变化（`Starting job: ...` → `Building llama3 debugmodel`）导致 `#test-result` 锚点失配 | 锚点更新为真实日志行 |
| 单卡步骤的 sed 修改残留在工作树，双卡步骤 `git checkout` 在 stdout 打 `M file` 行污染锚点 | checkout 改 `git checkout -f` |

## 三、方法论：三层递进的定位手段

1. **CI 黑盒迭代**（前 9 层）：每轮 30~40 分钟，靠报错堆栈逐层定位。缺点是慢，且两轮主 job 日志被 GitHub 侧丢失（自建 runner 长任务日志归档失败，连续复现 2 次，只能重跑碰运气）。
2. **HTTP Range 请求读 wheel**（多个关键判定）：不下整个 150MB+ 的 wheel，用 Range 请求读 zip 中央目录定位单文件再拉取，直接验证发行版元数据（triton-ascend 三个 wheel 的 `Requires-Dist`、文件清单），把"混装覆盖"这类问题的根因钉死在证据上而不是推测。
3. **coder workspace 本地复现**（后 5 层）：在 coder 910B workspace 复刻CI 环境（同 Python / torch / torch_npu / triton-ascend / CANN），把单轮迭代从 30+ 分钟压到分钟级；配合 py-spy dump --locals --native 精确抓到8B init 活锁的现场（具体到某个 FSDPLinear 的 weight 参数卡在`normal_fill` 内核）。workspace 验证全绿后才推 CI 确认——最后三轮 CI修的只是测试锚点，训练本体一次通过。

## 四、过程中的流程纠偏

| 问题 | 纠偏 |
| --- | --- |
| 补丁越打越多（一度 11 处 sed） | 重新梳理收敛：按根因归档为六条限制，随生态版本演进有明确退役条件；其中纯 CLI 切换（spmd-backend）和安装约定（`--no-deps` + pybind11）不算 sed |
| 8B 路径反复消耗 CI 时间 | workspace 定性活锁后果断降级 debugmodel 双卡，保住多卡验证面 |

## 五、对看护机制的启示

1. **C 类看护的价值被验证**：这次问题不是我们文档写错，而是上游大版本跨了两个 torch 大版本——正是看护机制要捕获的那类变化。分层定位信息14 层、每层报错原文）直接产出了可上报上游的 issue 材料。
2. **日志可靠性是薄弱点**：自建 runner 长任务主 job 日志归档失败复现 2 次， 只能靠重跑。
3. **本地复现环境是刚需**：coder workspace 复刻环境把迭代速度提升一个数量级，且 py-spy --locals/--native 是定位 NPU 上"活着但不动"类问题的关键工具。建议给每个 NPU 看护项目沉淀一份一键复刻脚本。

## 六、2026-09-17 no-patch 复盘

第一次跑通（2026-09-04~07）后，setup_example.sh 累计打上了 7 条 sed patch + 每个 launcher 顶上一行 `export TORCH_NPU_DEVICE_CAPABILITY=9.0`，CI 才转绿。复盘时重新审视：

- torchtitan v0.3.0 release notes 自称 device-agnostic，torch_npu 在支持矩阵里
- setup 层打的 7 条 sed 中，**4 条（patch 1-2-3、patch 5）改的是 torchtitan 上游源码**，本质上是在帮 torchtitan 适配一个尚未 ship 的 torch / 一个未支持 complex64 的 aclnn 算子；按 no-patch policy 这种适配应推给上游 PR 而不是闭门 patch
- patch 4（ChunkedLossWrapper → CE loss）、patch 6/7（flex → sdpa）是**绕开 NPU 栈的限制**，本质也是上游决策
- `TORCH_NPU_DEVICE_CAPABILITY=9.0` 是绕 torch_npu 自己的 c10d shim bug（None[0] 崩溃），跟 torchtitan 完全无关；用户的判断「torchtitan 已经支持 torch_npu」暗含「那 torch_npu 应该自己处理这个 bug 而不是让上游训练框架 export env var」

据此在 commit `publish-hdc` 上撤掉：

- `apply_compat_patches` 整个函数（7 条 sed + 调用点）
- 4 个 launcher 里的 `export TORCH_NPU_DEVICE_CAPABILITY=9.0` 行
- setup_example.sh 顶部「7 条 patch 列表」长注释改为「no-patch policy」说明
- supported entry 数量从 4 砍到 3：默认 `llama3_debugmodel` 1card 跟
  `#2` ce_loss 变体共用同一 launcher（仅 `--config` flag 不同），default
  走 ChunkedLossWrapper 在 NPU 是 case doc §2.5 已知的 meta tensor leak，
  no-patch 下两条同 launcher 重复跑零增量信号 → default 这条挪到
  unsupported「配置精简」分类
- 同步删 setup_example.sh 里 `run_llama3_debugmodel_1card.sh` 的生成
  （不再被 manifest 引用，避免 setup 留孤儿 launcher）
- Quick-start-Ascend.md 的「兼容性补丁」一节仍保留作为历史档案（不撤，那份是 patch 知识的载体）

**预期后果**：3 条 supported entry（1card_ce、2card、sft_1card）现在 CI 全失败。原 4 条里的 default `llama3_debugmodel` 1card 已合并到 unsupported「配置精简」分类（与 #2 共享 launcher + no-patch 下 ChunkedLossWrapper 是已知 NPU bug，留两条同 launcher 没信号）。剩下 3 条的 failure 落点会是：

| Entry | 预计首个 failure 点 | 上游归属 |
|---|---|---|
| 1card_ce | `OffsetBasedRNGTracker` → `c10d broadcast` → `None[0]` TypeError | torch_npu shim（撤 env var 后必现）|
| 2card | 同上 + `--parallelism.spmd-backend full_dtensor` 路径上的 `dp_mesh_dims` 限制 | torch_npu shim + torch 2.13-only FSDP API |
| sft_1card | 同 1card_ce + `create_block_mask(separate_full_blocks=...)` TypeError | torch 2.13-only kwarg（patch 6 撤了）|

这正是 policy 想要的 signal：CI 替我们盯住「上游的 device-agnostic 主张 vs 实际可用度」之间的差距。等任意上游修了对应 issue，CI 自动转绿，不需要再 review / 撤 patch。

**How to apply**：新加 NPU 看护项目时，setup 链默认不写 sed patch、不设 env var workaround。设备兼容性只在「上游已 ship 但我们的 pin 没跟上」时通过 pin 升级解决；上游尚未 ship 的限制作为 supported entry 的预期失败留下，由对应的上游 issue 跟踪。

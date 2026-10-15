# FLA examples 看护维护说明

用户运行入口是[文本训练与生成示例](../example/README.md)。本文说明环境依据、验证边界和 CI 维护方式。

## 昇腾配套和支持依据

当前基线为 FLA v0.5.2、CANN 9.0.0、Python 3.11、Torch 2.7.1、Torch-NPU 2.7.1.post4、torchvision 0.22.1、Triton-Ascend 3.2.1。依据包括：

- [FLA v0.5.2 的 pyproject.toml](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/pyproject.toml)：`[npu]` 固定 Python 侧的配套依赖。该版本 `INSTALL.md` 中的 Torch-NPU 示例没有写出 `post4`，安装以实际依赖声明为准。
- [FLA v0.5.2 的 Ascend A2 CI](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/.github/workflows/ascend-a2-ci.yml)：CANN 9.0.0 / Python 3.11，执行 modules、ops/utils、GDN、KDA、AttnRes 测试。
- [Triton-Ascend 官方镜像配套](https://github.com/triton-lang/triton-ascend/blob/main/docker/OVERVIEW.md#release-321)：3.2.1 / CANN 9.0.0 / Torch-NPU 2.7.1.post4。
- [Triton-Ascend 版本兼容性矩阵](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/release_note.md)：3.2.1 对应 CANN 9.0.0，3.2.2 对应 CANN 9.1.0，不能把两个版本系列的组件混合后视为同一验证环境。

版本配套和上游算子覆盖是实现依据，不等于完整示例通过认证。当前状态如下：

| 路径 | 证据与验证边界 |
| --- | --- |
| Gated Delta Rule、卷积、归一化、激活 | v0.5.2 存在 Ascend backend 实现，且相关模块/算子进入上游 A2 CI；不代表全部 shape 和 dtype 均已验证 |
| 本仓 Quick Start 的 GatedDeltaNet 前后向 | [运行 33314431867](https://github.com/cosdt-ci-test/workflows/actions/runs/33314431867) 已通过；只证明该文档覆盖的路径 |
| 两层完整语言模型、BF16 AdamW 更新 | [运行 36403948431](https://github.com/cosdt-ci-test/workflows/actions/runs/36403948431) 完成 20 步 NPU 训练及完整结果校验 |
| checkpoint 保存和重新加载 | 上述运行在 Transformers 4.57.6 上通过，`checkpoint_roundtrip_ok=true` |
| `generate(use_cache=True)` | 上述运行生成 32 个新 token，设备为 NPU；验证了缓存生成调用路径，未独立比较有缓存与无缓存的数值等价性 |

不能仅凭 `torch.npu.is_available()` 或 `IS_NPU=True` 把上述路径都标记为通过。首轮实机运行应保留具体 FLA commit、包版本、训练日志、重载结果和生成结果。

2026-09-28 的成功运行使用工作流代码 `6acf52c`、FLA v0.5.2（`9c8e42e762fce087c27b673af4922795d9edb85e`）。同一训练片段的 loss 从 3.84375 降至 0.494140625，`weights_changed=true`，保存/重载检查与生成检查全部通过。实际 Transformers 为 4.57.6；Torch-NPU 为 2.7.1.post4，安装的 Triton-Ascend 包为 3.2.1。报告中的 `torch.__version__=2.7.1+cpu` 和 `triton.__version__=3.2.0` 是模块版本字符串；实际运行设备及 FLA 后端均为 `npu`。

示例中的配置含义：

- `fuse_swiglu=False`：在 v0.5.2 的 [`GatedMLP`](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/modules/mlp.py) 中，仍执行 `fla.modules.activations.swiglu`，再调用输出投影；不是纯 PyTorch SwiGLU。FLA 的 [Ascend modules backend](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/modules/backends/triton_ascend/__init__.py) 注册了对应的前后向实现。
- `fuse_cross_entropy=False` 和 `fuse_linear_cross_entropy=False`：语言模型使用 `torch.nn.CrossEntropyLoss`，其 NPU 执行由 Torch-NPU/CANN 提供，已在上述默认示例运行中验证。
- Transformers 提供模型接口和生成调度；PyTorch/Torch-NPU 负责普通张量运算，FLA/Triton-Ascend 负责相应的自定义算子。改用纯 PyTorch 组织模型也不能省略底层配套和具体路径验证。

## 工作流和版本策略

`.github/workflows/flash-linear-attention-examples.yml` 调用公共 `examples-template.yml`。manifest 的 `source: project` 指向本仓 `example/train_text.py`，被测 FLA 则从目标上游 checkout 安装。项目代码、语料和参数更新都可以触发新的看护运行。

运行环境使用单卡 `linux-aarch64-a2-1`、SWR `cann:9.0.0-910b-ubuntu22.04-py3.11`，超时 120 分钟。`setup_example.sh` 安装目标 checkout 的 `.[npu]`，保留上游声明的 NPU 配套；同时使用项目的 `constraints-npu.txt` 限定 Transformers API 兼容范围。新 release 更换 CANN 配套或 Transformers API 要求时，需要同步审查镜像、约束、示例和用户文档。

FLA v0.5.2 的 `GatedDeltaNetForCausalLM._tied_weights_keys` 是列表，而 Transformers 5.17.0 的 [`_get_tied_weight_keys`](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/modeling_utils.py) 调用其 `.keys()`，使 `save_pretrained` 报错。Transformers [4.57.6 的实现](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/modeling_utils.py) 接受列表。约束采用 `transformers<5`，与目标 FLA 的要求共同解析，不改写上游类属性、不跳过保存，也不锁定一个补丁版本。上游迁移元数据并验证保存、重载和生成后，再调整这一约束；如果未来上游明确要求 5.x，依赖冲突应触发人工复核，不能静默忽略。

安装脚本在所有 pip 安装之前显式设置与 Quick Start 相同的集群缓存主索引及 `PIP_TRUSTED_HOST`，避免退回直连 PyPI。公共模板提供的华为云额外索引和安装 FLA 时的 Triton-Ascend 索引保持不变；日志会打印本次主索引。这是 NPU runner 内的网络配置，用户本地安装不使用该集群地址。

工作流合入默认分支后，每六小时的第 45 分钟检查最新 release 和本项目文件变化；失败后重试，成功且无变化时跳过。状态缓存为 `examples-monitor-state-flash-linear-attention_*`，与 Quick Start 分离。它不持续跟随上游 main 的每次提交。

在 GitHub Actions 中选择 **flash-linear-attention-examples → Run workflow**。`target_ref` 留空会选择最新 release，查询失败时公共模板回退到 main；也可以填写 tag、分支或 SHA。显式复现基线：

```bash
gh workflow run flash-linear-attention-examples.yml \
  --repo cosdt-ci-test/workflows \
  -f target_ref=v0.5.2
```

用户 README 中的 v0.5.2 安装命令用于复现配套，不会把自动看护固定在该版本。Quick Start 的独立 schedule 不在此变更中启用。

## 执行与结果

CI 调用用户可独立运行的 `train_text.py`，参数为 20 步、batch size 2、序列长度 128、生成 32 个 token。`validate_example.py` 检查：

- 实际运行使用 NPU，FLA 导入路径属于目标 checkout；
- loss 和梯度范数有限，完成参数更新，固定训练片段上的 loss 下降；
- checkpoint 和词表存在，重载前后相同输入的 logits 在 `rtol=1e-2, atol=1e-2` 内一致；
- 生成非空续写，文本文件与报告一致。

这些是运行验收条件，不是泛化能力、文本质量或性能指标。进程失败或缺少产物会使 job 失败。

NPU runner 的 `output/text-generation/` 保存模型、词表、生成文本和 `metrics.json`。公共模板仅上传通过 schema 校验的任务级 `result.json`，尚不上传模型或文本；需要长期保留实机日志和详细产物时，应另行设计其导出方式。

## 本地开发检查

在 workflows 仓库根目录，使用已安装 PyYAML、`mistune>=3,<4` 的 Python 环境：

```bash
python -m unittest discover -s projects/flash-linear-attention/tests -v
python scripts/check_supported_entries.py \
  --target-root /path/to/flash-linear-attention \
  --manifest projects/flash-linear-attention/examples_manifest.yaml
actionlint .github/workflows/flash-linear-attention-examples.yml
shellcheck projects/flash-linear-attention/scripts/*.sh
```

这些本地测试覆盖数据处理、CLI 转发、配置和结果校验，不执行 NPU 模型。Quick Start 的端到端测试在未设置 `NPU_READY=true` 时跳过。上面的成功记录来自单独的真实 NPU 工作流；后续改变配套或示例行为后仍需重新实测。

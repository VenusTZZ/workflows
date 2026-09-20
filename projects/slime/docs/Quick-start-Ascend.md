# slime-ascend smoke runbook

> 本文件是 bring-up 阶段的占位 runbook：完整的手动验证记录（含装栈耗时、
> 首轮失败点与修复）在首次人工验收时补齐。当前看护的执行配方全部
>固化在 `projects/slime/scripts/` 中，见下。

## 手动验证步骤（当前 CI 即按此执行）

1. Runner：Atlas A2 910B（`linux-aarch64-a2-4`），CANN 9.1.0 镜像
   `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。
2. 环境：`bash projects/slime/scripts/setup_example.sh slime_fully_async`
   （克隆 gitcode fork main → 装 sglang/torch/sgl-kernel-npu/Megatron 栈 →
   打 npu_patch v0.3.0 补丁 → ModelScope 拉 Qwen2.5-0.5B-Instruct →
   torchrun 4 进程转 `_torch_dist` ref 权重）。
3. 执行：`bash projects/slime/scripts/run_example.sh examples/fully_async/run-qwen2.5-0.5B-fully_async.sh`
   （actor 1 卡 + rollout 3 卡，`--num-rollout 2`，16 行 fixture）。
4. 验收：`train_async.py` 完成 2 个 rollout + optimizer step、exit 0。

## 配方出处

- 训练参数：fork `tests/tests_npu/nightly_CI/test_qwen2.5_0.5B_fully_async_short_npu.py`（昇腾已验证）。
- 装栈顺序与 pin：fork `scripts/ascend_script/quick_install.sh` +
  `docker/npu_docker/v0.3.0/Dockerfile.910b.ubuntu22.04.cann90.latest`。
- 运行时环境契约：fork ascend 启动器（`RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES`、
  HCCL 端口段、`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`）。


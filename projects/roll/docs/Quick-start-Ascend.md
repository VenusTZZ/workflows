# Quick Start: ROLL on Ascend NPU

在昇腾 NPU 上安装 ROLL，并用 FrozenLake agentic 强化学习跑通一次完整的训练闭环。

## 前置条件

- **硬件**：Atlas 900 A2 PODc / Ascend 910B 训练系列，单卡。
- **软件**：已装好 CANN，Python 版本不低于 3.10。参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)。

**本文档示例版本：**

| 组件 | 版本 | 来源 |
|---|---|---|
| Python | 3.12 | - |
| CANN | 9.1.0 | 昇腾官方 |
| torch | 2.10.0 | PyPI（CPU 版本） |
| torch_npu | 2.10.0.post4 | PyPI |
| triton-ascend | 3.2.1 | 华为 Ascend PyPI |
| vLLM / vLLM-Ascend | 0.23.0 / 0.23.0rc1 | GitHub 源码 |
| ROLL | main | GitHub 源码 |

torch 与 torch_npu 版本严格配套，按 CANN 兼容矩阵选择，参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)。ROLL 与 vLLM-Ascend 没有昇腾 PyPI 包，需源码安装。

### 检查环境

**检查 Python 版本。**

```shell #test id="check-py"
python --version
```

```shell #test-result id="check-py" fuzzy='xxx'
Python 3.xxx
```

## 安装 torch NPU 栈

**安装 torch 与 torch_npu。** 装完校验 NPU 运行时，is_available 须为 True。

```shell #test id="install-torch"
pip install torch==2.10.0 torchvision==0.25.0 >/dev/null 2>&1
pip install --no-deps torch-npu==2.10.0.post4 >/dev/null 2>&1
python -c "import torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('is_available', torch.npu.is_available()); print('count', torch.npu.device_count())"
```

```shell #test-result id="install-torch" fuzzy='xxx'
torch xxx
torch_npu xxx
is_available True
count 1
```

## 安装 vLLM-Ascend

**从源码安装 vLLM 与 vLLM-Ascend。** vLLM 策略为 ROLL 提供高吞吐 rollout，昇腾运行时由 vLLM-Ascend 插件提供，两包版本必须配套。装完用 triton-ascend 顶替 CUDA 版 triton，再 import 验证。

```shell #test id="install-vllm"
git clone -b v0.23.0 https://github.com/vllm-project/vllm.git vllm-src &&
git clone -b v0.23.0rc1 https://github.com/vllm-project/vllm-ascend.git vllm-ascend-src &&
cd vllm-src && VLLM_TARGET_DEVICE=empty pip install -e . >/dev/null 2>&1 && cd .. &&
cd vllm-ascend-src && git submodule update --init --recursive >/dev/null 2>&1 && pip install -r requirements.txt --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi >/dev/null 2>&1 && SOC_VERSION=ascend910b1 pip install -e . --no-build-isolation --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi >/dev/null 2>&1 && cd .. &&
pip uninstall -y triton >/dev/null 2>&1 &&
pip install --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi triton-ascend==3.2.1 >/dev/null 2>&1 &&
python -c "import vllm, vllm_ascend; print('vllm', vllm.__version__)"
```

```shell #test-result id="install-vllm" fuzzy='xxx'
vllm xxx
```

## 安装 ROLL

**克隆仓库并安装依赖。** PyPI 上的 roll 包名与本项目无关，从 GitHub 源码安装。在仓库内安装官方 agentic 依赖清单，按昇腾镜像配套 transformers 版本，补齐 agentic 环境注册所需的两个包，最后可编辑模式安装 ROLL 并校验环境管理模块可导入。

```shell #test id="install-roll"
git clone https://github.com/alibaba/ROLL.git &&
pip install -r ROLL/requirements_common.txt >/dev/null 2>&1 &&
pip install "transformers==4.57.6" "tensorboard==2.20.0" >/dev/null 2>&1 &&
pip install reasoning-gym==0.1.23 >/dev/null 2>&1 &&
pip install --no-deps gem-llm==0.0.4 >/dev/null 2>&1 &&
pip install -e ./ROLL >/dev/null 2>&1 &&
python -c "import roll.pipeline.agentic.env_manager.traj_env_manager; print('roll ok')"
```

```shell #test-result id="install-roll"
roll ok
```

## 运行示例：FrozenLake agentic 强化学习

FrozenLake 是 ROLL 官方快速入门的示例：Qwen2.5-0.5B-Instruct 作为策略模型，在 4×4 冰面网格中逐轮输出移动方向，绕开冰洞到达终点，环境按结果返回奖励。ROLL 用 Ray 把角色编排为独立 worker 集群：actor_train 用 FSDP2 策略更新权重，actor_infer 与 reference 用 HF 推理策略分别负责生成动作和参考概率，三个角色共享同一张 NPU，GRPO 用组内采样基线替代 critic。模型权重首次运行自动下载到默认缓存 ~/.cache/modelscope。

**写入示例配置。** 单卡昇腾版配置：训练后端 fsdp2_train，推理与参考模型后端 hf_infer，设备映射只留卡 0，批量收缩，只跑 2 步。

```python #test-setup id="write-config"
from pathlib import Path

config = """
hydra:
  run:
    dir: .
  output_subdir: null

exp_name: "roll-quick-start-npu"
seed: 42
logging_dir: ./output/logs
output_dir: ./output
system_envs:
  USE_MODELSCOPE: '1'

track_with: tensorboard
tracker_kwargs:
  log_dir: ./output/tensorboard

num_gpus_per_node: 1

max_steps: 2
save_steps: 1000
logging_steps: 1
eval_steps: 1000
resume_from_checkpoint: false

rollout_batch_size: 8
val_batch_size: 2
sequence_length: 2048
max_tokens_per_step: 128

ppo_epochs: 1
adv_estimator: "grpo"
init_kl_coef: 0.0
whiten_advantages: true
entropy_loss_coef: 0

pretrain: Qwen/Qwen2.5-0.5B-Instruct
reward_pretrain: Qwen/Qwen2.5-0.5B-Instruct

actor_train:
  model_args:
    attn_implementation: fa2
    disable_gradient_checkpointing: false
    dtype: bf16
    model_type: ~
  training_args:
    learning_rate: 1.0e-6
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 2
  data_args:
    template: qwen2_5
  strategy_args:
    strategy_name: fsdp2_train
    strategy_config:
      fsdp_size: 1
      param_dtype: bf16
      reduce_dtype: bf16
      reshard_after_forward: true
      offload_policy: false
  device_mapping: list(range(0,1))
  infer_batch_size: 1

actor_infer:
  model_args:
    disable_gradient_checkpointing: true
    dtype: bf16
  generating_args:
    max_new_tokens: 128
    temperature: 0.99
    num_return_sequences: 1
  data_args:
    template: qwen2_5
  strategy_args:
    strategy_name: hf_infer
    strategy_config: ~
  device_mapping: list(range(0,1))
  infer_batch_size: 1

reference:
  model_args:
    attn_implementation: fa2
    disable_gradient_checkpointing: true
    dtype: bf16
    model_type: ~
  data_args:
    template: qwen2_5
  strategy_args:
    strategy_name: hf_infer
    strategy_config: ~
  device_mapping: list(range(0,1))
  infer_batch_size: 1

train_env_manager:
  max_env_num_per_worker: 8
  num_env_groups: 4
  group_size: 2
  tags: [FrozenLake]
  num_groups_partition: [4]

val_env_manager:
  max_env_num_per_worker: 2
  num_env_groups: 2
  group_size: 1
  tags: [FrozenLake]
  num_groups_partition: [2]

custom_envs:
  FrozenLake:
    env_type: frozen_lake
    max_steps: 10
    max_tokens_per_step: 128
    env_manager_cls: roll.pipeline.agentic.env_manager.traj_env_manager.TrajEnvManager
    agent_runner_cls: null
    use_thread_lock: true
    agent_system_template: You're a helpful assistant. You are a good game player. You are aiming to get high reward in the game.
    agent_template: |
      Turn {turn_idx}:
      Observation:
      {observation}
      Strictly follow this format:
      1. output format is '<answer> [your answer] </answer>' with no extra text.
      2. You have {actions_left} actions left.
      3. Max response length: {max_response_length} words (tokens).
      Decide the next action:
    env_config:
      action_pattern: <answer>(.*?)</answer>
      max_steps: 10
      format_penalty: -0.01
      is_slippery: false
"""

path = Path("ROLL/examples/agentic_frozen_lake_npu/quick_start_npu.yaml")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(config.lstrip("\n"), encoding="utf-8")
print("config written", path)
```

**启动训练。** 从仓库根目录运行 agentic pipeline 入口脚本，ROLL 自动拉起 Ray 集群，训练日志实时输出到终端。校验训练正常收尾。

```shell #test id="run-agentic"
cd ROLL && python examples/start_agentic_pipeline.py --config_path agentic_frozen_lake_npu --config_name quick_start_npu
```

```shell #test-result id="run-agentic"
...
pipeline complete!
...
```

**校验训练产物。** 训练指标写入 output/tensorboard，校验事件文件存在。

```shell #test id="verify-output"
ls ROLL/output/tensorboard/events*
```

```shell #test-result id="verify-output" fuzzy='xxx'
ROLL/output/tensorboard/events.out.tfevents.xxx
```

## 更多用法

多卡并行、vLLM 高吞吐 rollout、Megatron 后端与 SFT/DPO 等其他 pipeline 见官方文档：https://alibaba.github.io/ROLL/docs/QuickStart/single_node_quick_start

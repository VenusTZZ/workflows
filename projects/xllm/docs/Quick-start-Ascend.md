# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上快速验证 xllm 在线服务推理。

## 前置条件

### 硬件

Atlas 900 A2 训练系列产品或者 Ascend 910B 系列产品，并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的 `torch` + `torch_npu`，且 `torch` 能正常 `import` 并 `torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）

### 本文档示例使用的版本

**配套镜像**：

swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/xllm-ai/xllm-ai:xllm-0.10.0-release-hb-rc2-arm

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11 |
| CANN | 9.1.0 |
| torch | 2.9.0 |
| torch_npu | 2.9.0.post2 |
| xllm | 官方 release 镜像预装 (v0.10.0) |
| 模型 | [Qwen2-7B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2-7B-Instruct) |

> 说明：CI 使用 xllm 官方 release 镜像 `swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/xllm-ai/xllm-ai:xllm-0.10.0-release-hb-rc2-arm`，**已预装 xllm v0.10.0 及其依赖**，无需从源码编译，启动即用。

### 前置安装

确认能看到 NPU 设备：

```shell #test id="check-npu"
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
| 0     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
```

> 如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动。

输出结果如下（表格内容随环境变化，仅校验命令执行成功）：

```shell #test-result id="check-npu"
...
```

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：
```shell #test-result id="check-py" fuzzy='xxx'
Python 3.11.xxx
```

检查 torch / torch_npu 是否装好且 NPU 设备可用：

```shell #test id="check-torch"
python -c "import torch, torch_npu; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="check-torch"
torch= 2.9.0+cpu
torch_npu= 2.9.0.post2
is_available: True
count: 1
```

> 如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

## 验证 xllm 安装

xllm 已预装于官方 release 镜像，无需编译，验证可导入（镜像内 xllm 包未暴露 `__version__`，此处打印模块路径确认安装位置）：

```shell #test id="check-xllm"
python -c "import xllm; print('xllm import ok:', xllm.__file__)"
```

输出结果如下：

```shell #test-result id="check-xllm" fuzzy='xxx'
xllm import ok: xxx
```

## 在线服务用例

参考 [xllm 在线服务文档](https://docs.xllm-ai.com/zh/getting_started/online_service/)，选取 **LLM 客户端调用 → HTTP 调用（chat 模式）** 用例进行验证。先按 [xllm 启动文档](https://docs.xllm-ai.com/zh/getting_started/launch_xllm/) 的 NPU 方式在单卡上启动服务，再通过 OpenAI 兼容接口发起一次对话请求（按文档说明调整参数：`stream: false`、`max_tokens: 10` 以便快速验证）。

一个命令完成"启动服务 → 等待就绪 → chat 请求 → 停止服务"：

```shell #test id="serve-chat"
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && source /usr/local/Ascend/nnal/atb/set_env.sh

rm -f /tmp/xllm-serve.log
ASCEND_RT_VISIBLE_DEVICES=0 xllm \
  --model /root/.cache/modelscope/Qwen2-7B-Instruct \
  --port 9977 \
  --master_node_addr=127.0.0.1:9748 \
  --nnodes=1 \
  --node_rank=0 \
  --block_size=128 \
  --max_memory_utilization=0.86 \
  --communication_backend="hccl" \
  --enable_prefix_cache=false \
  --enable_chunked_prefill=true \
  --enable_schedule_overlap=true \
  > /tmp/xllm-serve.log 2>&1 &
XLLM_PID=$!
trap 'kill $XLLM_PID 2>/dev/null || true' EXIT

response=""
for i in $(seq 1 30); do
  kill -0 "$XLLM_PID" 2>/dev/null || break
  response=$(curl -s --max-time 30 http://localhost:9977/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen2-7B-Instruct",
      "max_tokens": 10,
      "temperature": 0,
      "stream": false,
      "messages": [
        {
          "role": "system",
          "content": "You are a helpful assistant."
        },
        {
          "role": "user",
          "content": "hello xllm"
        }
      ]
    }') || response=""
  if [ -n "$response" ] && echo "$response" | grep -q '"choices"'; then
    break
  fi
  sleep 10
done

if ! echo "$response" | grep -q '"choices"'; then
  echo "xllm service did not become ready in time; last 50 lines of /tmp/xllm-serve.log:"
  tail -50 /tmp/xllm-serve.log
  exit 1
fi

echo "$response" | python -c "import json, sys; d = json.load(sys.stdin); print('chat content:', d['choices'][0]['message']['content'])"
```

输出结果如下：

```shell #test-result id="serve-chat" fuzzy='xxx'
chat content: xxx
```

> 注意：模型路径 `/root/.cache/modelscope/Qwen2-7B-Instruct` 是 CI 环境通过 ModelScope 预先下载的目录（挂载自 CI 缓存 `/data/ci-cache/modelscope/xllm`）。本地运行时请用 `modelscope` 自行下载该模型到对应目录。
>
> 服务日志输出到 `/tmp/xllm-serve.log`，排查启动失败时可查看该文件。更多客户端调用方式（completions 模式、Beam Search、`/v1/sample`、Python 调用、VLM）见 [在线服务文档](https://docs.xllm-ai.com/zh/getting_started/online_service/)。

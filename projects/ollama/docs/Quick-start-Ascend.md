# Ollama (Ascend NPU)

Ollama 提供本地大语言模型的导入、管理和推理服务。本文从源码构建 CANN 后端，在单张昇腾 NPU 上运行 Qwen2.5-0.5B-Instruct，并通过命令行和 HTTP API 完成对话。

昇腾接入目前需要应用本文提供的实验补丁，尚未包含在 Ollama 官方发布包中。

## 前置条件

### 硬件

Atlas A2 系列的 Ascend 910B4 设备，并完成设备驱动安装及设备挂载。其他 910B 型号需将构建命令中的 `SOC_TYPE` 改为实际芯片型号。

### 基础软件

在运行示例前，需要准备：

- Linux、Python 3.12，以及 Git、curl 和 C/C++ 编译器。
- Go 1.26.0 或更新版本，参考 [Go 安装文档](https://go.dev/doc/install)。
- 可用的 CANN，参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)。

本文使用 CANN 9.1.0；模型采用 CANN 支持的 Q4_0 GGUF 格式。

## 加载 CANN 环境

在当前终端加载 CANN 的库路径和运行时配置：

```shell #test-setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 安装 Ollama

**安装构建工具。** CMake 用于编译 Ollama 及其推理后端，Go 用于构建服务程序。

```shell #test-setup
pip install cmake==3.31.10
```

```shell #test id="build-tools"
go version
cmake --version
```

```shell #test-result id="build-tools" fuzzy='xxx'
go version goxxx
cmake version 3.31.10xxx
```

**获取源码和补丁。** 将 [CANN 接入补丁](../patches/ollama-cann.patch)下载并保存为当前目录的 `ollama-cann.patch`，再获取最新发布版本的源码。

```shell #test-setup
git clone --depth 1 --branch "$(curl -fsSL https://api.github.com/repos/ollama/ollama/releases/latest | python -c 'import json, sys; print(json.load(sys.stdin)["tag_name"])')" https://github.com/ollama/ollama.git ollama
```

**应用补丁。** 补丁启用 CANN 后端构建，并让服务能够发现和选择 NPU。

```shell #test-setup
git -C ollama apply --check ../ollama-cann.patch && git -C ollama apply ../ollama-cann.patch
```

**构建 CANN 后端。** 构建产物保存在 `ollama/build/`，服务程序生成在 `ollama/ollama`。

```shell #test-setup
cmake -S ollama -B ollama/build -DCMAKE_BUILD_TYPE=Release -DOLLAMA_LLAMA_BACKENDS=cann -DOLLAMA_BUILD_PARALLEL=8 -DCANN_INSTALL_DIR="$ASCEND_HOME_PATH" -DSOC_TYPE=Ascend910B4 -DOLLAMA_VERSION="$(git -C ollama describe --tags --exact-match)"
```

```shell #test-setup
cmake --build ollama/build --parallel 8
```

## 用法示例

### 启动推理服务

**启动服务。** 在当前终端前台运行服务，使用第 0 张 NPU；模型数据保存在当前目录的 `ollama-models/`。

```shell #test-setup
OLLAMA_LLM_LIBRARY=cann ASCEND_RT_VISIBLE_DEVICES=0 OLLAMA_MODELS=./ollama-models ./ollama/ollama serve
```

另开一个终端，在同一目录执行后续命令。

**查看服务版本。** 通过本地接口确认服务已经启动。

```shell #test id="server-version"
curl -fsS http://127.0.0.1:11434/api/version
```

```shell #test-result id="server-version" fuzzy='xxx'
{"version":"xxx"}
```

### 导入 GGUF 模型

**下载模型。** GGUF 将权重、模型配置和 tokenizer 信息保存在一个文件中，Ollama 可直接导入。这里使用约 429 MB 的 Q4_0 四位量化模型，减少首次下载和内存开销；已有文件时复用下载。

```python #test-setup
from pathlib import Path
from urllib.request import urlopen
import shutil

model = Path('qwen2.5-0.5b-instruct-q4_0.gguf')
if not model.exists():
    url = 'https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/master/' + model.name
    partial = model.with_suffix('.gguf.part')
    with urlopen(url, timeout=120) as response, partial.open('wb') as output:
        shutil.copyfileobj(response, output)
    partial.replace(model)
```

**创建模型。** Modelfile 指定本地权重和上下文长度，`num_gpu` 请求将全部模型层交给加速设备。

```python #test-setup
from pathlib import Path

Path('Modelfile').write_text(
    'FROM ./qwen2.5-0.5b-instruct-q4_0.gguf\n'
    'PARAMETER num_ctx 1024\n'
    'PARAMETER num_gpu 999\n'
    'PARAMETER num_predict 64\n', encoding='utf-8',
)
```

```shell #test-setup
./ollama/ollama create qwen2.5-ascend -f Modelfile
```

### 命令行对话

命令行直接向模型发送消息，适合快速体验本地对话。

```shell #test id="cli-chat"
./ollama/ollama run qwen2.5-ascend "用一句话介绍你自己。" --verbose
```

输出包含模型回复和生成速度等指标：

```shell #test-result id="cli-chat" fuzzy='xxx'
xxx
```

### HTTP API 对话

HTTP API 将本地模型接入应用程序，`stream: false` 让一次请求返回完整回复。

```shell #test-setup
curl -fsS http://127.0.0.1:11434/api/chat -H 'Content-Type: application/json' -d '{"model":"qwen2.5-ascend","messages":[{"role":"user","content":"用一句话介绍你自己。"}],"stream":false,"options":{"num_predict":64,"temperature":0}}' -o chat-response.json
```

```python #test id="api-chat"
import json
from pathlib import Path

response = json.loads(Path('chat-response.json').read_text(encoding='utf-8'))
print('model:', response['model'])
print('reply:', response['message']['content'])
print('tokens:', response['eval_count'])
```

```shell #test-result id="api-chat" fuzzy='xxx'
model: qwen2.5-ascendxxx
reply: xxx
tokens: xxx
```

## 外部链接

- [Ollama 官方仓库](https://github.com/ollama/ollama)
- [Ollama 官方文档](https://docs.ollama.com/)
- [llama.cpp CANN 后端](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/CANN.md)

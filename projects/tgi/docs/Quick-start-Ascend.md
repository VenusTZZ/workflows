# Quick Start (Ascend NPU)

在昇腾 NPU 上从源码构建并部署 Text Generation Inference（TGI）推理服务，
对 Qwen3-0.6B 完成端到端文本生成：单卡基线 + 双卡 HCCL 张量并行（`--num-shard 2`）。

本文档覆盖上游官方 release 中**尚未包含**的 Ascend NPU 支持：构建、安装、
启动与推理验证使用
[cosdt-ci-test/text-generation-inference](https://github.com/cosdt-ci-test/text-generation-inference)
（TGI 官方仓库的 fork，Ascend 适配已合入该 fork 的 `main`）。

## 前置条件

### 硬件

Atlas 900 A2 PODc（Ascend 910B × 2），并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 3.12 环境
- 可用的 CANN 9.1.0（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- Rust 工具链（本文档「安装 Rust 工具链」小节会通过 rustup 安装，无需提前准备）

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，32 GB × 2）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.9.0 |
| torch_npu | 2.9.0.post2 |
| transformers | 4.57.6 |
| kernels | 0.5.0 |
| 模型 | [Qwen/Qwen3-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B) |

## 1. 检查环境

### 确认 NPU 设备

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
| 0     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
| 1     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+---------------------------+---------------+----------------------------------------------------+
| No running processes found in NPU 0                                                            |
+---------------------------+---------------+----------------------------------------------------+
```

> 如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动。
> 本文档的双卡验证需要**至少两张卡**可见。

### 检查 Python 版本

```shell #test id="check-python"
python --version
```

输出结果如下：

```shell #test-result id="check-python" fuzzy='xxx'
Python 3.12.xxx
```

### 检查 NPU 设备运行时可用

```shell #test id="check-npu-runtime"
python -c "import torch, torch_npu; print(f'torch={torch.__version__}'); print(f'torch_npu={torch_npu.__version__}'); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="check-npu-runtime" fuzzy='xxx'
torch=xxx
torch_npu=xxx
is_available: True
count: 2
```

> 如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

## 2. 安装依赖与工具链

### 系统依赖

TGI 是 Rust + Python 双栈项目：Rust 侧编译需要 C/C++ 工具链与 protobuf 编译器，
PyO3 嵌入 Python 需要开发头文件：

```shell #test-setup
apt-get update -qq
apt-get install -y -qq build-essential protobuf-compiler pkg-config libssl-dev curl git python3-dev
```

### Python 依赖

torch / torch_npu 从华为昇腾源安装（与 CANN 9.1.0 匹配的 2.9 系列），其余
Python 依赖用 uv 安装：

```shell #test-setup
python -m pip install -q uv
uv pip install \
  --index-url https://repo.huaweicloud.com/ascend/repos/pypi/simple \
  "torch==2.9.0" "torch_npu==2.9.0.post2"
uv pip install \
  "transformers==4.57.6" "accelerate==1.15.0" "modelscope==1.37.0" \
  "kernels==0.5.0" "grpcio-tools>=1.69.0" "mypy-protobuf>=3.6.0"
```

> `kernels` 必须固定 0.5.0：它是 TGI Python server 的构建后端，0.5.0 自带
> `kernels.lockfile`，构建时不会去下载 CUDA 专属内核；更新的版本缺该文件，
> 在无 CUDA 的 aarch64 昇腾机器上会构建失败。

### 安装 Rust 工具链

```shell #test-setup
export RUSTUP_DIST_SERVER=https://rsproxy.cn
export RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
curl -fsSL https://rsproxy.cn/rustup-init.sh -o /tmp/rustup-init.sh
sh /tmp/rustup-init.sh -y --default-toolchain 1.85.1 --profile minimal
mkdir -p "$HOME/.cargo"
printf '[source.crates-io]\nreplace-with = "rsproxy-sparse"\n[source.rsproxy-sparse]\nregistry = "sparse+https://rsproxy.cn/index/"\n' \
  > "$HOME/.cargo/config.toml"
export PATH="$HOME/.cargo/bin:$PATH"
```

> 使用 rsproxy.cn（字节跳动 Rust 社区镜像）拉取工具链与 crates.io 依赖，规避
> 国内网络访问 static.rust-lang.org / crates.io 的不稳定。TGI 仓库根目录的
> `rust-toolchain.toml` 固定 1.85.1，因此这里直接安装该版本。

## 3. 下载基础模型

默认使用 **ModelScope** 下载 Qwen3-0.6B（约 1.2 GB，落到默认缓存
`~/.cache/modelscope`）：

```shell #test-setup store="model_path"
python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen3-0.6B'))" | tail -n 1
```

## 4. 构建并安装 TGI

### 获取源码

克隆 fork 并 checkout 到本次看护的 ref（工作流注入 `UPSTREAM_REF`，`<ref>`
为该 ref；fork 上未打 release 时引擎按 fallback 链解析为 `main` HEAD）：

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test-setup load="upstream_ref>>ref"
git clone --depth 1 --branch "<ref>" \
  https://github.com/cosdt-ci-test/text-generation-inference.git tgi \
  || { git clone --depth 1 \
         https://github.com/cosdt-ci-test/text-generation-inference.git tgi \
       && git -C tgi fetch --depth 1 origin "<ref>" \
       && git -C tgi checkout -q FETCH_HEAD; }
```

> `<ref>` 为要安装的 release tag 或 commit SHA（也可替换为任意分支名）。
> 手工执行时无需 `UPSTREAM_REF`，直接把 `<ref>` 换成分支名（如 `main`）
> 或 commit SHA。

### 编译 Rust 二进制

编译 launcher 与 router（`--profile release-opt` 为上游提供的发布优化 profile）。
PyO3 嵌入 Python 需要 `PYO3_PYTHON` 指向环境里的 Python，protobuf 代码生成
需要 `PROTOC`：

```shell #test-setup
cd tgi
export PYO3_PYTHON="$(command -v python)"
export PROTOC="$(command -v protoc)"
cargo build --profile release-opt \
  -p text-generation-launcher -p text-generation-router-v3
```

### 安装 Python server

```shell #test-setup
cd tgi
uv pip install --no-build-isolation -e server
make -C server gen-server-raw
```

### 检查构建产物

```shell #test id="check-build"
$PWD/tgi/target/release-opt/text-generation-launcher --version
python -c "import text_generation_server; print('server import ok')"
```

输出结果如下：

```shell #test-result id="check-build" fuzzy='xxx'
text-generation-launcher xxx
server import ok
```

## 5. 启动服务并验证推理

### 单卡基线

启动 TGI（单卡、bfloat16、贪心解码），轮询就绪后调用 `/generate` 做一次真实
推理，输出作为双卡验证的基线（此块为标准输出捕获块，回复存入
`reply_single`，不参与输出比对）：

```shell #test-setup store="reply_single" load="model_path>>model_path"
set -e
cd tgi
export PATH="$PWD/target/release-opt:$PATH"
export ATTENTION=flashdecoding-npu PREFIX_CACHING=0 CUDA_GRAPHS=0
export ASCEND_VISIBLE_DEVICES=0
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256

text-generation-launcher \
  --model-id "<model_path>" \
  --num-shard 1 --port 8080 \
  --max-total-tokens 128 --max-input-tokens 100 \
  > tgi-launcher-single.log 2>&1 &
LID=$!
cleanup() {
  kill "$LID" 2>/dev/null || true
  for _ in $(seq 1 15); do kill -0 "$LID" 2>/dev/null || break; sleep 2; done
  pkill -TERM -f "text-generation[-]server" 2>/dev/null || true
  sleep 3
}
trap cleanup EXIT

for i in $(seq 1 60); do
  curl -4fs http://127.0.0.1:8080/info >/dev/null 2>&1 && break
  kill -0 "$LID" 2>/dev/null || { echo "launcher exited early"; tail -50 tgi-launcher-single.log; exit 1; }
  sleep 5
done
curl -4fs http://127.0.0.1:8080/info >/dev/null || { echo "server not ready"; tail -50 tgi-launcher-single.log; exit 1; }

curl -4fs http://127.0.0.1:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{"inputs":"What is 1+1? Answer:","parameters":{"max_new_tokens":16,"do_sample":false}}' \
  | python -c 'import json, sys; print(json.load(sys.stdin)["generated_text"].strip().replace("\n", " "))'
```

### 双卡张量并行验证

用两张卡（`--num-shard 2`，HCCL 张量并行）再跑一次相同请求，验证回复非空、
且与单卡基线**逐字一致**（贪心解码下张量并行不改变输出）：

```shell #test id="smoke-tp2" load="model_path>>model_path" load="reply_single>>reply_single"
set -e
cd tgi
export PATH="$PWD/target/release-opt:$PATH"
export ATTENTION=flashdecoding-npu PREFIX_CACHING=0 CUDA_GRAPHS=0
export ASCEND_VISIBLE_DEVICES=0,1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256

# wait until the single-card service above has fully released port 8080
for i in $(seq 1 30); do
  curl -4fs http://127.0.0.1:8080/info >/dev/null 2>&1 || break
  sleep 2
done

text-generation-launcher \
  --model-id "<model_path>" \
  --num-shard 2 --port 8080 \
  --max-total-tokens 128 --max-input-tokens 100 \
  > tgi-launcher-tp2.log 2>&1 &
LID=$!
cleanup() {
  kill "$LID" 2>/dev/null || true
  sleep 3
  pkill -TERM -f "text-generation[-]server" 2>/dev/null || true
}
trap cleanup EXIT

for i in $(seq 1 90); do
  curl -4fs http://127.0.0.1:8080/info >/dev/null 2>&1 && break
  kill -0 "$LID" 2>/dev/null || { echo "launcher exited early"; tail -50 tgi-launcher-tp2.log; exit 1; }
  sleep 5
done
curl -4fs http://127.0.0.1:8080/info >/dev/null || { echo "server not ready"; tail -50 tgi-launcher-tp2.log; exit 1; }

REPLY=$(curl -4fs http://127.0.0.1:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{"inputs":"What is 1+1? Answer:","parameters":{"max_new_tokens":16,"do_sample":false}}' \
  | python -c 'import json, sys; print(json.load(sys.stdin)["generated_text"].strip().replace("\n", " "))')
[ -n "$REPLY" ] || { echo "empty reply"; exit 1; }
[ "$REPLY" = "<reply_single>" ] || { echo "TP2 reply differs from single-card baseline"; echo "single: <reply_single>"; echo "tp2:    $REPLY"; exit 1; }
echo "TGI-TP2-OK: $REPLY"
```

输出结果如下（`...` 为模型回复内容，与单卡基线一致）：

```shell #test-result id="smoke-tp2"
TGI-TP2-OK: ...
```

## 6. 停止服务

```shell
kill "$(pgrep -f text-generation-launcher)"
```

> 若 `npu-smi info` 里仍有残留的 `text-generation-server` 进程，用
> `pkill -f text-generation-server` 清理后再重新启动（残留进程会占用 NPU 与
> 端口，导致新实例报 `EJ0003 Failed to bind the IP port`）。

## 小贴士

- **更多卡**：`--num-shard N` 配合 `ASCEND_VISIBLE_DEVICES=0,1,...,N-1` 即可
  做 N 卡 HCCL 张量并行（双卡为本文档看护范围；实现细节见 TGI NPU 适配设计
  文档 `docs/npu/hccl-multicard-design.md`）。
- **输出一致性对比**：`do_sample=false` 贪心解码下，相同模型与参数的输出是
  确定的——双卡验证正是用它断言张量并行不改变输出。
- **常用查询**：`curl -4 http://127.0.0.1:8080/info` 查看服务信息，
  `curl -4 http://127.0.0.1:8080/v1/chat/completions` 走 OpenAI 兼容接口。

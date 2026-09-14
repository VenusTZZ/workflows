# 快速开始：在昇腾 NPU 上用 ONNX Runtime 做第一次推理

本文在单卡昇腾 NPU 上从 [ONNX Runtime](https://github.com/microsoft/onnxruntime) 当前正式 Release 源码编译 `onnxruntime-cann`，当场生成一个最小加法模型，并用 CANN Execution Provider 做一次推理。包名是 `onnxruntime-cann`，导入名仍是 `onnxruntime`。不要再装一份 CPU 包 `onnxruntime`，两个包会抢同一个导入名。

> **阅读本文前**，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 装好 CANN 与驱动。

---

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为**单卡**。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit 与驱动已安装，并能 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| Python | 3.12 |
| 编译 | gcc-12、g++-12、cmake 3.28 到 3.31、ninja、git |
| 包管理 | `python -m pip` |

常见容器里 `npu-smi` 位于 `/usr/local/sbin` 或 `/usr/local/bin`。先把这两个目录放进 `PATH`，再加载 CANN：

```shell
export PATH=/usr/local/sbin:/usr/local/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 确认 NPU 在线

```shell
npu-smi info
```

命令退出码应为 0，并打印设备表。功耗、温度、HBM 占用每次都不同，不必和任何截图逐字一致。

若提示找不到 `npu-smi`，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载，例如 `/dev/davinci0`。

---

## 本文档验证过的版本

**配套机器**

- **机器类型**：Atlas 900 A2（Ascend 910B，单卡）
- **操作系统**：Ubuntu 22.04

**配套镜像**

`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`

**软件版本**

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| onnxruntime-cann | 当前 GitHub 正式 Release 源码编译 |
| numpy | 1.26.x，必须 `<2` |
| 编译工具 | gcc-12、cmake 3.28–3.31、ninja |

官方 [CANN Execution Provider](https://onnxruntime.ai/docs/execution-providers/community-maintained/CANN-ExecutionProvider.html) 兼容表目前只列了 ONNX Runtime 1.20.0 / 1.21.0 / 1.22.1 对应 CANN 8.2.0。本文按上表在 CANN 9.1.0 上从当前正式 Release 源码编译，没有跟着那张旧表降版本。

---

## 安装编译工具

Ubuntu 22.04 默认 gcc 11 编不了 aarch64 上的 ONNX Runtime，需要 gcc-12。cmake 需要 3.28 及以上；cmake 4 会让 FetchContent 失败，不要装。`python -m pip` 装的 cmake 在当前解释器的 scripts 目录，要把它放到 `PATH` 前面，否则会继续用系统自带的 3.22。编 Python wheel 还要 `packaging`、`wheel`、`setuptools`，否则 `setup.py bdist_wheel` 会失败。第三方包走华为云通用 PyPI。

```shell #test id="toolchain"
export DEBIAN_FRONTEND=noninteractive
if ! command -v gcc-12 >/dev/null || ! command -v g++-12 >/dev/null || ! command -v ninja >/dev/null; then
  apt-get update
  apt-get install -y gcc-12 g++-12 ninja-build git
fi
python -m pip install --index-url https://repo.huaweicloud.com/repository/pypi/simple \
    'cmake>=3.28,<4' 'numpy<2' packaging wheel setuptools
export PATH="$(python -c 'import sysconfig; print(sysconfig.get_path("scripts"))'):$PATH"
hash -r
gcc-12 --version | head -n 1
cmake --version | head -n 1
```

输出结果如下：

```shell #test-result id="toolchain"
...gcc-12...
cmake version 3.3...
```

---

## 获取源码

工作目录为 `/root/onnxruntime-qs`。把 `<UPSTREAM_REF>` 换成 [Releases](https://github.com/microsoft/onnxruntime/releases) 里当前正式 tag。源码来自官方 GitHub。首次编译可能要几十分钟。

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

<!--
```shell #test-setup load="upstream_ref>>UPSTREAM_REF"
wd=/root/onnxruntime-qs
ci=/root/.cache/cosdt-ci-test/onnxruntime
ref='<UPSTREAM_REF>'
mkdir -p "$wd/dist"
for w in "$ci/wheels/$ref"/onnxruntime_cann-*.whl; do
  [ -f "$w" ] || continue
  if python -m zipfile -l "$w" >/dev/null 2>&1; then
    cp -a "$w" "$wd/dist/"
  else
    rm -f "$w"
  fi
done
if [ -d "$ci/src/$ref/.git" ] && [ ! -d "$wd/onnxruntime/.git" ]; then
  rm -rf "$wd/onnxruntime"
  git clone --depth 1 "$ci/src/$ref" "$wd/onnxruntime"
fi
```
-->

```shell #test id="clone" load="upstream_ref>>UPSTREAM_REF"
mkdir -p /root/onnxruntime-qs
if [ ! -d /root/onnxruntime-qs/onnxruntime/.git ]; then
  GIT_TERMINAL_PROMPT=0 GIT_HTTP_VERSION=HTTP/1.1 git clone --depth 1 --branch "<UPSTREAM_REF>" \
    https://github.com/microsoft/onnxruntime.git /root/onnxruntime-qs/onnxruntime
fi
ls /root/onnxruntime-qs/onnxruntime/build.sh
```

输出结果如下：

```shell #test-result id="clone"
/root/onnxruntime-qs/onnxruntime/build.sh
```

---

## 编译 onnxruntime-cann

开启 CANN Execution Provider，并打出 Python wheel。`--build_wheel` 会带上 pybind。首次验证不编单元测试，缩短编译时间。并行编译超过 32 路时内存容易打满，本文封顶 32。工作目录里如果已有 `dist/onnxruntime_cann-*.whl`，这一步会跳过编译。

<!--
```shell #test-setup
wd=/root/onnxruntime-qs
ci=/root/.cache/cosdt-ci-test/onnxruntime
if [ -d "$ci/cmake-mirror" ] && [ -d "$wd/onnxruntime" ]; then
  ln -sfn "$ci/cmake-mirror" "$wd/onnxruntime/.cmake-mirror"
fi
```
-->

```shell #test id="compile"
cd /root/onnxruntime-qs
if ! compgen -G "dist/onnxruntime_cann-*.whl" >/dev/null; then
  cd onnxruntime
  export CC=gcc-12 CXX=g++-12
  njobs=$(nproc)
  if [ "$njobs" -gt 32 ]; then
    njobs=32
  fi
  MIRROR=()
  if [ -d .cmake-mirror ]; then
    MIRROR+=(--cmake_deps_mirror_dir "$PWD/.cmake-mirror")
  fi
  ./build.sh --config Release --build_shared_lib --use_cann --build_wheel \
    --parallel "$njobs" --skip_tests --skip_submodule_sync \
    --compile_no_warning_as_error --allow_running_as_root \
    --cmake_generator Ninja \
    --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
    "${MIRROR[@]}"
  mkdir -p /root/onnxruntime-qs/dist
  cp -a build/Linux/Release/dist/onnxruntime_cann-*.whl /root/onnxruntime-qs/dist/
fi
ls /root/onnxruntime-qs/dist/onnxruntime_cann-*.whl
```

输出结果如下：

```shell #test-result id="compile"
...onnxruntime_cann-...whl
```

<!--
```shell #test-setup load="upstream_ref>>UPSTREAM_REF"
wd=/root/onnxruntime-qs
ci=/root/.cache/cosdt-ci-test/onnxruntime
ref='<UPSTREAM_REF>'
mkdir -p "$ci/wheels/$ref"
for w in "$wd/dist"/onnxruntime_cann-*.whl; do
  [ -f "$w" ] || continue
  base=$(basename "$w")
  dest="$ci/wheels/$ref/$base"
  if [ -f "$dest" ]; then
    continue
  fi
  cp -a "$w" "${dest}.part"
  mv "${dest}.part" "$dest"
done
if [ -d "$wd/onnxruntime/.git" ] && [ ! -d "$ci/src/$ref/.git" ]; then
  mkdir -p "$ci/src"
  rm -rf "$ci/src/${ref}.part"
  git clone --depth 1 "$wd/onnxruntime" "$ci/src/${ref}.part"
  mv "$ci/src/${ref}.part" "$ci/src/$ref"
fi
```
-->

---

## 安装 wheel 与算子编译依赖

把刚编出的 `onnxruntime-cann` 装进当前 Python。`onnx` 用来在下一步当场生成模型，不从网上下载权重。这个 wheel 按 NumPy 1.x 编译，不钉 `numpy<2` 时 pip 会拉到 NumPy 2，`import onnxruntime` 会直接失败。

CANN 编译算子还要用 `decorator`、`scipy`、`attrs`、`psutil`、`sympy`。不装的话，会话能建起来，第一次 `sess.run()` 会报 `aclgrphBuildInitialize` 或 `aclopCompileAndExecute`。`scipy` 钉在 1.15 以下，避免把 NumPy 升到 2。

```shell #test id="install"
python -m pip install --index-url https://repo.huaweicloud.com/repository/pypi/simple \
    /root/onnxruntime-qs/dist/onnxruntime_cann-*.whl \
    onnx \
    'numpy<2' \
    decorator \
    'scipy>=1.11,<1.15' \
    attrs \
    psutil \
    sympy
python -c "from importlib.metadata import version; print('onnxruntime-cann', version('onnxruntime-cann'))"
```

输出结果如下：

```shell #test-result id="install"
...onnxruntime-cann ...
```

---

## 确认昇腾后端

```shell #test id="providers"
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

输出结果如下。列表里必须有 `CANNExecutionProvider`，前后还可能有 `CPUExecutionProvider` 等：

```shell #test-result id="providers"
...CANNExecutionProvider...
```

若没有 `CANNExecutionProvider`，先看文末「常见问题」，不要继续推理。列表里有 CPU 并不等于这次推理会走 CPU。真正决定后端的是下一节创建 `InferenceSession` 时传入的 `providers`。

---

## 造一个最小 ONNX 模型

用已安装的 `onnx` 在工作目录写一个两向量相加的图，保存为 `add_model.onnx`。不下载任何文件。

```shell #test id="make-model"
cd /root/onnxruntime-qs
python <<'PY'
import onnx
from onnx import TensorProto, helper

x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2])
graph = helper.make_graph(
    [helper.make_node("Add", ["X", "Y"], ["Z"])],
    "add",
    [x, y],
    [z],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
onnx.save(model, "add_model.onnx")
print("wrote add_model.onnx")
PY
```

输出结果如下：

```shell #test-result id="make-model"
wrote add_model.onnx
```

---

## 用昇腾跑第一次推理

下面这段关掉了 CPU 回退。ONNX Runtime 默认 `enable_fallback=1`。CANN 会话创建失败时，它会静默改在 CPU 上重建会话，加法结果仍然正确，进程退出码也是 0，看起来像昇腾已经跑通。

官方 CANN 示例常把 `CPUExecutionProvider` 接在后面。第一次验证不要抄那种写法。这里只注册 `CANNExecutionProvider`，并同时关掉会话级和 Python 级回退。装错包、CANN 没加载、版本对不上时，进程必须失败。

输入是 `[1.0, 2.0]` 和 `[3.0, 4.0]`，昇腾上的加法结果应是 `[4.0, 6.0]`。

```shell #test id="infer"
cd /root/onnxruntime-qs
python <<'PY'
import numpy as np
import onnxruntime as ort

so = ort.SessionOptions()
so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
sess = ort.InferenceSession(
    "add_model.onnx",
    sess_options=so,
    providers=["CANNExecutionProvider"],
    enable_fallback=False,
)
providers = sess.get_providers()
print(providers)
assert providers[0] == "CANNExecutionProvider", providers
x = np.array([1.0, 2.0], dtype=np.float32)
y = np.array([3.0, 4.0], dtype=np.float32)
out = sess.run(None, {"X": x, "Y": y})[0]
print("result", [float(v) for v in out])
PY
```

输出结果如下。第一项必须是 `CANNExecutionProvider`。列表里是否还出现 CPU，以本机实际打印为准。

```shell #test-result id="infer"
...CANNExecutionProvider...
result [4.0, 6.0]
```

---

## 常见问题

| 现象 | 可能原因 | 建议 |
| --- | --- | --- |
| `get_available_providers()` 没有 `CANNExecutionProvider` | 没 `source set_env.sh`，或装的是 CPU 包 `onnxruntime`，或两个包叠在一起 | 重新 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`。若叠装过 CPU 包，先 `python -m pip uninstall -y onnxruntime onnxruntime-cann`，再只装本文编出的本地 wheel |
| `import onnxruntime` 报找不到 CANN 动态库 | 当前 shell 没有 CANN 环境变量 | 先 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| 创建 `InferenceSession` 失败，日志提到 CANN / ACL 版本 | 本文验证的是 CANN 9.1.0 + 当前正式 Release 源码。官方兼容表只写到 ORT 1.20–1.22.1 ↔ CANN 8.2.0 | 对齐本文版本表，确认 `./build.sh` 带了 `--use_cann` |
| 会话能建，`get_providers()` 第一项也是 `CANNExecutionProvider`，但 `sess.run()` 报 `aclgrphBuildInitialize` 或 `aclopCompileAndExecute("Add")` / `ACL_ERROR_FAILURE` | 当前 Python 环境缺 CANN 算子编译依赖，`import tbe` 失败 | 回到「安装 wheel 与算子编译依赖」，确认 `decorator`、`scipy`、`attrs`、`psutil`、`sympy` 都装上了，再 `source` 一次后重跑推理 |
| 推理结果正确，但 `get_providers()` 里同时出现 CPU | 创建会话时把 CPU 写进了 `providers`，或没有关掉回退。有的版本建好 CANN 会话后仍会在列表里留下 CPU | 不要把 `CPUExecutionProvider` 写进创建会话时的 `providers`。第一项必须是 `CANNExecutionProvider`，并且推理没有落到 CPU 回退日志 |
| `npu-smi: command not found` | `npu-smi` 在 `/usr/local/sbin` 或 `/usr/local/bin`，不在默认 `PATH` | `export PATH=/usr/local/sbin:/usr/local/bin:$PATH` 后再执行 `npu-smi info` |
| `setup.py bdist_wheel` 报 `No module named 'packaging'` | 当前 Python 环境缺打 wheel 的包 | 回到「安装编译工具」，确认 `packaging`、`wheel`、`setuptools` 都装上了 |
| `gcc` 报 `-march=armv8.2-a+bf16` 或 cmake 版本过低 | 用了 Ubuntu 默认的 gcc 11 或 cmake 3.22 | 回到「安装编译工具」，确认 `gcc-12` 与 cmake 3.28–3.31 |
| cmake 卡在下载 protobuf 等依赖 | FetchContent 直连 GitHub 过慢或失败 | 为 `./build.sh` 准备 `--cmake_deps_mirror_dir`，指向按 URL 路径展开的本地镜像目录 |

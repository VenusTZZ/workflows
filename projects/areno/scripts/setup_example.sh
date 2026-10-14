#!/usr/bin/env bash
# 为某个 supported AReno example 准备 CI 环境。
# $1 = manifest profile；未知 profile 在任何安装前即失败。
# AReno 从目标 checkout（TARGET_ROOT，即引擎解析到的最新 release）源码安装；
# v0.9.0 已自带 NPU 后端。安装时现场编译 areno.accel._areno_accel_npu
# （Ascend C + CMake，无 Python 回退）。
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
FALLBACK_PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"

source_vendor_env() {
  # CANN/ATB 的 set_env.sh 假设登录 shell，且直接引用未加 ${VAR:-} 的变量；
  # 这里临时关掉严格模式再 source。
  local vendor_file="$1"
  if [[ ! -f "$vendor_file" ]]; then
    echo "vendor env script not found, skipping: $vendor_file"
    return 0
  fi
  set +eu
  # shellcheck disable=SC1090
  source "$vendor_file"
  set -eu
}

select_pip_index() {
  # runner 在国内：优先集群 pip 缓存，退回清华镜像；Ascend 源经
  # PIP_EXTRA_INDEX_URL（workflow 已设）供 torch_npu wheel。
  if python -c "
import urllib.error
import urllib.request
try:
    urllib.request.urlopen('${CLUSTER_PIP_INDEX}', timeout=3)
except urllib.error.HTTPError:
    pass
" 2>/dev/null; then
    export PIP_INDEX_URL="$CLUSTER_PIP_INDEX"
    export PIP_TRUSTED_HOST="$CLUSTER_PIP_HOST"
  else
    export PIP_INDEX_URL="$FALLBACK_PIP_INDEX"
    unset PIP_TRUSTED_HOST
  fi
  echo "pip index: $PIP_INDEX_URL"
}

ensure_build_tools() {
  # 编译 Ascend C 扩展需要 cmake 与 binutils（nm）。
  command -v cmake >/dev/null 2>&1 || {
    apt-get update -qq >/dev/null
    apt-get install -y -qq cmake binutils >/dev/null
  }
  command -v cmake && command -v nm
}

ensure_torch_stack() {
  # 镜像不带 torch；torch 从 Ascend/华为源取（与 torch_npu 配套，华为 CDN 在
  # 国内快），避开上游 +cpu find-links 的单连接慢下载。
  if python -c "import torch, torch_npu; print('found torch', torch.__version__, 'torch_npu', torch_npu.__version__)"; then
    echo "reusing image torch stack"
    return
  fi
  echo "installing torch==2.10.0 + torch_npu==2.10.0.post2 from the Ascend repo"
  if ! python -m pip install --extra-index-url "$ASCEND_PIP_INDEX" \
      torch==2.10.0 torch_npu==2.10.0.post2; then
    echo "Ascend index lacked the pair; falling back to the upstream +cpu find-links"
    python -m pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.10.0
    python -m pip install --extra-index-url "$ASCEND_PIP_INDEX" torch_npu==2.10.0.post2
  fi
  python -c "import torch, torch_npu; print('torch', torch.__version__, 'torch_npu', torch_npu.__version__)"
}

install_areno() {
  source_vendor_env /usr/local/Ascend/ascend-toolkit/set_env.sh
  # 只编译 AReno 与其 Ascend C 扩展，跳过 CI 用不到的两个重包：flash-attn-npu
  # （只服务 --attn-backend flash，leg 固定 native）与 flash-linear-attention
  # （只服务 KDA/线性注意力模型，leg 用 dense 底座，且 areno 在 NPU 上把
  # lightning attn 路由到自研 accel.npu.seg_la）。故用 --no-deps 装 AReno，再按
  # requirements/npu.txt 从 wheel 安装其余依赖。
  python -m pip install -e "$TARGET_ROOT" --no-build-isolation --no-deps
  local deps
  deps=$(grep -vE '^[[:space:]]*(#|flash-attn-npu|flash-linear-attention)' \
    "$TARGET_ROOT/requirements/npu.txt")
  # shellcheck disable=SC2086
  python -m pip install $deps
  python - <<'PY'
import importlib.util as u
if u.find_spec('areno.accel._areno_accel_npu') is None:
    raise SystemExit(
        'areno.accel._areno_accel_npu MISSING - installer did not take the '
        'NPU branch; check that torch_npu is installed before AReno')
print('areno NPU extension ok')
PY
}

install_pi_toolchain() {
  # pi（pi.dev）是外部 Node.js CLI，示例通过自带环境变量
  # （ARENO_PI_EXECUTABLE / ARENO_PI_PYTHON / ARENO_PI_RESULT_DIR）驱动。全部从
  # 国内可达的镜像安装：Node 走 npmmirror 二进制、pi 走 npmmirror 的 npm 包
  # @earendil-works/pi-coding-agent、fd/ripgrep 走 apt、SWE-smith 数据用示例
  # generate_dataset.py 从 ModelScope 生成。路径经 GITHUB_ENV 传出。
  local node_ver="${ARENO_PI_NODE_VERSION:-22.19.0}"
  if ! command -v node >/dev/null 2>&1; then
    echo "installing Node ${node_ver} (aarch64) from npmmirror"
    curl -fsSL "https://cdn.npmmirror.com/binaries/node/v${node_ver}/node-v${node_ver}-linux-arm64.tar.gz" \
      | tar -xz -C /usr/local --strip-components=1
  fi
  node --version
  npm --version

  echo "installing pi from the npmmirror npm registry"
  npm install -g --ignore-scripts --registry=https://registry.npmmirror.com \
    @earendil-works/pi-coding-agent
  local pi_bin
  pi_bin="$(command -v pi)"
  "$pi_bin" --version

  # pi 的 bash 工具需要 fd / ripgrep。
  command -v rg >/dev/null 2>&1 || {
    apt-get update -qq >/dev/null
    apt-get install -y -qq fd-find ripgrep >/dev/null
  }
  command -v fd >/dev/null 2>&1 || ln -sf /usr/bin/fdfind /usr/local/bin/fd
  command -v fd && command -v rg

  # 示例的任务 Python（pi 的 bash 工具 + 打分器）只需要 pytest，主环境已有。
  local task_py
  task_py="$(command -v python)"
  # generate_dataset.py 的数据准备依赖（modelscope + pyarrow）。
  python -m pip install -q -r "$TARGET_ROOT/examples/agentic/pi/requirements.txt"

  local data="$GITHUB_WORKSPACE/pi-swesmith.jsonl"
  local results="$GITHUB_WORKSPACE/pi-results"
  mkdir -p "$results"
  echo "generating SWE-smith tasks from ModelScope -> $data"
  python "$TARGET_ROOT/examples/agentic/pi/generate_dataset.py" \
    --python "$task_py" --output "$data"

  {
    echo "ARENO_PI_EXECUTABLE=$pi_bin"
    echo "ARENO_PI_PYTHON=$task_py"
    echo "ARENO_PI_RESULT_DIR=$results"
    echo "ARENO_PI_DATA=$data"
  } >> "$GITHUB_ENV"
}

ensure_media_deps() {
  # requirements/default.txt 的多模态媒体栈在 requirements/npu.txt 里被有意省略。
  # VL 的 Fast 图像处理器（Qwen2VLImageProcessorFast，见 checkpoint 的
  # preprocessor_config.json）需要 torchvision：缺它 AutoProcessor.from_pretrained
  # 抛错，被 load_processor 吞成 None，报 "multimodal rows require a checkpoint
  # processor"。镜像里带的 torchvision 是残缺 stub（import 能过但没有
  # transforms.v2，pip 以为已满足不装真包），故强制重装与当前 torch 配对的真包
  # （--no-deps 只动它自己）；dataset_generator 用 Pillow。
  local torch_minor tv_ver
  torch_minor="$(python -c 'import torch; print(torch.__version__.split("+")[0].split(".")[1])')"
  tv_ver="0.$((torch_minor + 15)).0"  # torch 2.N <-> torchvision 0.(N+15)（2.6->0.21 ... 2.10->0.25）
  echo "installing torchvision ${tv_ver} to match the local torch"
  python -m pip install -q --force-reinstall --no-deps "torchvision==${tv_ver}"
  python -m pip install -q Pillow
  # 功能校验：不看 torchvision.__version__（残缺 stub 没有该属性）。
  python - <<'PY'
import torch
import torchvision
import torchvision.transforms.v2  # noqa: F401 - transformers Fast 图像处理器依赖

print("media deps ok: torch", torch.__version__, "torchvision", getattr(torchvision, "__version__", "n/a"), torchvision.__file__)
PY
}

install_triton_fla_stack() {
  # triton-ascend + flash-linear-attention：Ling 的 KDA（训练 chunk kernel 是
  # areno 自研 triton kernel accel/kernels/kda_fla/chunk_intra.py；推理增量直接调
  # fla.ops.kda.fused_recurrent_kda，见 accel/npu/kda.py:29）与 Qwen3.5 的
  # GatedDeltaNet（训练/推理都走 fla 的 chunk/fused_recurrent gated_delta_rule，
  # qwen3_5/model.py:420 无条件 _require_fla_gdn）都依赖它。npu.txt 的注释点名
  # CANN 配套栈含 triton-ascend，镜像未带——从 Ascend 源补装（--no-deps 防止
  # pip 顺带换 torch）。fla 同样 --no-deps：其依赖里的 triton 会从 PyPI 拉
  # CUDA 版把 triton-ascend 顶掉。
  python -m pip install -q --no-deps --extra-index-url "$ASCEND_PIP_INDEX" "triton-ascend==3.2.2"
  # triton-ascend 3.2.2 的 METADATA 依赖是 vendor 测试环境 pin，不能照装：
  # - triton==3.5.0：wheel 已自带完整 triton 包体，装 PyPI 的会用 CUDA 版覆盖它；
  # - numpy==1.26.4：降级会破坏 torch 2.10/torch_npu/pandas/transformers 栈；
  # - scipy/attrs/decorator/pyyaml/pytest-xdist：全量扫描 wheel 的 import，均不在
  #   运行链上（纯测试 pin；pytest 环境已有）。运行链上真正缺的只有 pybind11
  # （backends/ascend/utils.py）与 psutil（backends/ascend/runtime/autotuner.py，
  # 在 import triton 链上），两者纯 Python、无版本冲突；einops 是 fla 的运行依赖。
  python -m pip install -q pybind11 psutil einops
  python -m pip install -q --no-deps \
    "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git@e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2"
  # 与 areno 相同的导入链做功能校验（覆盖 Ling 的 KDA 与 Qwen3.5 的 GDN 两组）。
  python - <<'PY'
import triton
from fla.ops.gated_delta_rule import (  # noqa: F401 - Qwen3.5 GDN（dynamo_wrappers.py:17-19）
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
from fla.ops.kda import fused_recurrent_kda  # noqa: F401 - Ling KDA 推理（accel/npu/kda.py:29）

print("triton-ascend + fla ok: triton", triton.__version__)
PY
}

# 已知 profile。当前各 profile 共用同一套准备（下面的 AReno NPU 源码构建）；
# 需要额外依赖的 profile 在这里按 PROFILE 分支。pi 已不支持（QLoRA 为 CUDA-only），
# 保留 pi_npu 与 install_pi_toolchain 以备将来试非 QLoRA 变体。
SUPPORTED_PROFILES="bash_game_npu codebreaker_npu coding_npu duelgrid_npu shopping_npu terminal_hacking_npu tictactoe_npu tictactoe_no_tool_npu math_npu alpaca_npu tictactoe_image_npu pi_npu"
case "$PROFILE" in
  bash_game_npu | codebreaker_npu | coding_npu | duelgrid_npu | shopping_npu | terminal_hacking_npu | tictactoe_npu | tictactoe_no_tool_npu | math_npu | alpaca_npu | tictactoe_image_npu | pi_npu) ;;
  *)
    echo "unknown profile: ${PROFILE} (supported: ${SUPPORTED_PROFILES})" >&2
    exit 1
    ;;
esac

select_pip_index
python -m pip install -U pip 'setuptools>=69' wheel
ensure_build_tools
ensure_torch_stack
install_areno
# 供 coding 用例的 run_command 跑测试（其任务在 rollout 时执行 `python -m pytest`）。
python -m pip install -q pytest
# agentic 用例的 run_agent.py 经 OpenAI 兼容客户端打本地 rollout 代理；AReno 的
# NPU 依赖列了 openai，但镜像可能已满足 openai 而未带其 httpx 依赖，故两个都装。
python -m pip install -q openai httpx

# 逐 profile 的额外工具链（AReno 之外的外部依赖）。
case "$PROFILE" in
  bash_game_npu | terminal_hacking_npu) install_triton_fla_stack ;;
  tictactoe_image_npu) ensure_media_deps; install_triton_fla_stack ;;
  pi_npu) install_pi_toolchain ;;
esac

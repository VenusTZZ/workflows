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
  # 只编译 AReno 与其 NPU 扩展：装了 torch_npu 后根 setup.py 只走 NPU 分支
  # （areno/accel/csrc/npu/setup.py），CUDA 扩展（csrc/*.cu）不会编。另跳过两个
  # 重包：flash-attn-npu（只服务 --attn-backend flash，leg 固定 native）与
  # flash-linear-attention（Ling/Qwen3.5 的 leg 由 install_triton_fla_stack 按
  # profile 另装）。故用 --no-deps 装 AReno，再按 requirements/npu.txt 装 wheel。
  # Ascend C kernel 编译默认单线程——csrc/npu/setup.py 只在设置 MAX_JOBS 时才给
  # cmake 传 --parallel（torch 的 BuildExtension 也读它）。用满核并行（封顶 32：
  # ccec 单任务内存可观，防极端多核节点打爆内存）。
  local jobs
  jobs="$(nproc)"
  jobs=$(( jobs > 32 ? 32 : jobs ))
  export MAX_JOBS="${MAX_JOBS:-$jobs}"
  echo "building areno with MAX_JOBS=$MAX_JOBS"
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
  # --limit 1：冒烟只取 1 个 sqlparse 修复任务（默认 4 任务，README 配方是
  # 4 x 4 = 16 个 attempt、每个 32 次调用 / 1800s 预算，CI 压不动）。
  python "$TARGET_ROOT/examples/agentic/pi/generate_dataset.py" \
    --python "$task_py" --output "$data" --limit 1

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
  # "$@" 是调用方追加的媒体包（ave 的 av/librosa/soundfile）——不能丢，
  # 丢了会装不上音频链，transformers 的 load_audio 到 rollout 才报缺 librosa。
  python -m pip install -q Pillow "$@"
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

install_ave_media_deps() {
  # ave_event_recognition：音视频路径需 av（解容器）/ librosa（transformers 的
  # load_audio 用它读音频）/ soundfile（areno 直接 import）以及抽音轨用的 ffmpeg；
  # 图像走 ensure_media_deps。
  ensure_media_deps "av>=12" "librosa>=0.11" "soundfile>=0.13"
  command -v ffmpeg >/dev/null 2>&1 || {
    apt-get update -qq >/dev/null
    apt-get install -y -qq ffmpeg >/dev/null
  }
  command -v ffmpeg
  # 功能校验：librosa 的 import 依赖 numba/numpy 配套——坏了的话 transformers
  # 的 load_audio 只会笼统报 "requires the librosa library ... not found"，
  # 在 setup 阶段直接 import 才能提前看到真实报错。
  python - <<'PY'
import av
import librosa
import soundfile

import torchvision
import torchvision.transforms.v2  # noqa: F401

print("ave media stack ok: librosa", librosa.__version__, "| soundfile", soundfile.__version__, "| av", av.__version__)
PY
}

stage_ave_fixture() {
  # 上游示例的数据是整包 AVE（数十 GB 视频），CI 用一段短合成音视频 + 单行
  # manifest 代替，让 loader/agent/reward/train 全链路照常跑。片段必须 >= 32 帧：
  # 视频 processor 固定采样 num_frames=32，少于此会报 "num_frames exceeds
  # total_num_frames"；故取 5s@8fps = 40 帧（时长不影响采样后的视频 token 数）。
  # judge 由 run_example.sh 的存根服务提供（引擎不校验奖励数值、只认退出码）。
  local dir="$GITHUB_WORKSPACE/ave-fixture"
  local manifest="$dir/manifest.jsonl"
  if [[ ! -f "$manifest" ]]; then
    mkdir -p "$dir"
    ffmpeg -nostdin -loglevel error -y \
      -f lavfi -i "testsrc=size=224x224:rate=8:duration=5" \
      -f lavfi -i "sine=frequency=440:sample_rate=16000:duration=5" \
      -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "$dir/clip.mp4"
    ffmpeg -nostdin -loglevel error -y -i "$dir/clip.mp4" -vn -ac 1 -ar 16000 "$dir/clip.wav"
    python - "$dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
row = {
    "id": "ci-ave-00000",
    "dataset_root": str(root),
    "split": "train",
    "video_id": "ci-ave",
    "video_path": "clip.mp4",
    "audio_path": "clip.wav",
    "event_classes": ["Speech"],
    "qualities": ["good"],
    "start_seconds": 0.0,
    "end_seconds": 5.0,
}
(root / "manifest.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote ave fixture manifest {root / 'manifest.jsonl'}")
PY
  fi
  echo "AVE_DATASET=$manifest" >> "$GITHUB_ENV"
}

# 已知 profile。当前各 profile 共用同一套准备（下面的 AReno NPU 源码构建）；
# 需要额外依赖的 profile 在这里按 PROFILE 分支。pi 走非 QLoRA 变体：QLoRA 在
# areno 源码里 CUDA-only，但与 pi 机制正交，去掉即全参训练 Ling（4 卡 TP +
# adam-4bit + triton-ascend/fla）。
SUPPORTED_PROFILES=(
  bash_game_npu codebreaker_npu coding_npu duelgrid_npu shopping_npu
  terminal_hacking_npu tictactoe_npu tictactoe_no_tool_npu math_npu
  alpaca_npu tictactoe_image_npu ave_event_recognition_npu pi_npu
)
case " ${SUPPORTED_PROFILES[*]} " in
  *" ${PROFILE} "*) ;;
  *) echo "unknown profile: ${PROFILE} (supported: ${SUPPORTED_PROFILES[*]})" >&2; exit 1 ;;
esac

select_pip_index
python -m pip install -U pip 'setuptools>=69' wheel
ensure_build_tools
ensure_torch_stack
install_areno
# areno 看护统一钉 transformers==5.15.*：requirements/npu.txt 只写 >=5.15,<6，
# pip 会取区间内最新（曾装到 5.19.0），而 Gemma4 视觉塔在 5.19 的
# create_bidirectional_mask 抛 IndexError（mask 维数不符）。5.15 是 areno 声明
# 的 gemma4 多模态基线（gemma4/model.py:707,827 "requires transformers>=5.15"）；
# 统一钉住也让各 leg 环境确定、不随上游漂移（tokenizers 由 pip 配套降级）。
python -m pip install -q "transformers==5.15.*"
python -c "import transformers; print('transformers pinned:', transformers.__version__)"
# 供 coding 用例的 run_command 跑测试（其任务在 rollout 时执行 `python -m pytest`）。
python -m pip install -q pytest
# agentic 用例的 run_agent.py 经 OpenAI 兼容客户端打本地 rollout 代理；AReno 的
# NPU 依赖列了 openai，但镜像可能已满足 openai 而未带其 httpx 依赖，故两个都装。
python -m pip install -q openai httpx

# 逐 profile 的额外工具链（AReno 之外的外部依赖）。
case "$PROFILE" in
  bash_game_npu | terminal_hacking_npu) install_triton_fla_stack ;;
  tictactoe_image_npu) ensure_media_deps; install_triton_fla_stack ;;
  ave_event_recognition_npu) install_ave_media_deps; stage_ave_fixture ;;
  pi_npu) install_pi_toolchain; install_triton_fla_stack ;;
esac

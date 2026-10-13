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

# 已知 profile。当前各 profile 共用同一套准备（下面的 AReno NPU 源码构建）；
# 需要额外依赖的 profile 在这里按 PROFILE 分支。pi 已不支持（QLoRA 为 CUDA-only），
# 保留 pi_npu 与 install_pi_toolchain 以备将来试非 QLoRA 变体。
SUPPORTED_PROFILES="bash_game_npu codebreaker_npu coding_npu duelgrid_npu shopping_npu terminal_hacking_npu tictactoe_npu tictactoe_no_tool_npu math_npu alpaca_npu tictactoe_image_npu ave_event_recognition_npu pi_npu"
case "$PROFILE" in
  bash_game_npu | codebreaker_npu | coding_npu | duelgrid_npu | shopping_npu | terminal_hacking_npu | tictactoe_npu | tictactoe_no_tool_npu | math_npu | alpaca_npu | tictactoe_image_npu | ave_event_recognition_npu | pi_npu) ;;
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

ensure_media_deps() {
  # requirements/default.txt 的多模态媒体栈在 requirements/npu.txt 里被有意省略
  # （npu.txt 只收公共依赖）。但 VL 的 Fast 图像处理器
  # （Qwen2VLImageProcessorFast，见 checkpoint 的 preprocessor_config.json）需要
  # torchvision：缺它 AutoProcessor.from_pretrained 直接抛错，被
  # load_processor 吞成 None，随后报 "multimodal rows require a checkpoint
  # processor"。音视频还要 av / librosa / soundfile。dataset_generator 用 Pillow。
  # 装时用 constraints 把 torch 钉在现装的 CANN 版，避免 pip 顺带把 torch 换成
  # PyPI 的 CPU 版而破坏 torch_npu。
  local constraints
  constraints="$(mktemp)"
  echo "torch==$(python -c 'import torch; print(torch.__version__)')" > "$constraints"
  python -m pip install -q -c "$constraints" Pillow torchvision "$@"
  rm -f "$constraints"
  python -c "import torchvision, PIL; print('media deps ok: torchvision', torchvision.__version__)"
}

install_vl_image_deps() {
  # tictactoe_image：图像路径只需 Pillow + torchvision。
  ensure_media_deps
}

install_ave_media_deps() {
  # ave_event_recognition：音视频路径需 av（解容器）/ librosa（重采样）/
  # soundfile（areno 直接 import）以及抽音轨用的 ffmpeg。
  ensure_media_deps "av>=12" "librosa>=0.11" "soundfile>=0.13"
  command -v ffmpeg >/dev/null 2>&1 || {
    apt-get update -qq >/dev/null
    apt-get install -y -qq ffmpeg >/dev/null
  }
  command -v ffmpeg
}

stage_ave_fixture() {
  # 上游示例的数据是整包 AVE（数十 GB 视频），CI 用一段 10s 合成音视频 +
  # 单行 manifest 代替，让 loader/agent/reward/train 全链路照常跑。judge 由
  # run_example.sh 的存根服务提供（引擎不校验奖励数值、只认退出码）。
  local dir="$GITHUB_WORKSPACE/ave-fixture"
  local manifest="$dir/manifest.jsonl"
  if [[ ! -f "$manifest" ]]; then
    mkdir -p "$dir"
    ffmpeg -nostdin -loglevel error -y \
      -f lavfi -i "testsrc=size=224x224:rate=8:duration=10" \
      -f lavfi -i "sine=frequency=440:sample_rate=16000:duration=10" \
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
    "end_seconds": 10.0,
}
(root / "manifest.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote ave fixture manifest {root / 'manifest.jsonl'}")
PY
  fi
  echo "AVE_DATASET=$manifest" >> "$GITHUB_ENV"
}

# 逐 profile 的额外工具链（AReno 之外的外部依赖）。
case "$PROFILE" in
  tictactoe_image_npu) install_vl_image_deps ;;
  ave_event_recognition_npu) install_ave_media_deps; stage_ave_fixture ;;
  pi_npu) install_pi_toolchain ;;
esac

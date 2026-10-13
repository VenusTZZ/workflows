#!/usr/bin/env bash
# 运行一个 supported AReno 用例（从 CI 工作区）。
# $1 = manifest 条目路径（相对 examples 根），仅用于存在性检查。逐用例参数都在
# manifest 的 overlay_args（OVERLAY_ARGS，JSON 数组）里：dataset / loader /
# reward / agent 入口 / algo 与 CI 缩放参数。共享的单卡 NPU 冒烟档（native
# attention、ModelScope 的 Qwen3-0.6B、每个 profile 一个 checkpoint）就在这里，
# 因此新增用例只改 manifest。禁止 git add/commit/push。
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
EXAMPLES_ROOT="${EXAMPLES_ROOT:-$TARGET_ROOT}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"
PROFILE="${PROFILE:?PROFILE is required}"

EXAMPLE_PATH="$EXAMPLES_ROOT/$EXAMPLE_REL"
[[ -e "$EXAMPLE_PATH" ]] || { echo "example not found: $EXAMPLE_PATH" >&2; exit 1; }

mkdir -p "$CI_OUTPUT_DIR"

source_vendor_env() {
  # CANN 的 set_env.sh 假设登录 shell 且引用未加 ${VAR:-} 的变量；临时关严格
  # 模式再 source。
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

source_vendor_env /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

python -c "import torch, torch_npu; print('NPU available:', torch.npu.is_available(), 'devices:', torch.npu.device_count())"

expand_overlay() {
  # workflow 把 manifest.overlay_args 序列化成 JSON。逐项按原引号展开，并让
  # ${FIXTURE_DIR} / ${EXAMPLES_ROOT} 等 CI 路径只在本 job 环境里解析。
  python - <<'PY'
import json
import os
import shlex

raw = os.environ.get('OVERLAY_ARGS', '').strip()
if not raw or raw in ('null', '""'):
    raise SystemExit(0)
try:
    items = json.loads(raw)
except json.JSONDecodeError as exc:
    raise SystemExit(f'OVERLAY_ARGS is not valid JSON: {exc}') from exc
if items in (None, ''):
    raise SystemExit(0)
if not isinstance(items, list):
    raise SystemExit(
        f'OVERLAY_ARGS must be a JSON array, got {type(items).__name__}')
tokens = []
for item in items:
    if not isinstance(item, str) or not item.strip():
        raise SystemExit('OVERLAY_ARGS items must be non-empty strings')
    tokens.extend(shlex.split(os.path.expandvars(item), posix=True))
print(' '.join(shlex.quote(token) for token in tokens))
PY
}

eval "EXTRA_ARGS=( $(expand_overlay) )"

echo "running $EXAMPLE_REL (profile $PROFILE) with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

# 共享的单卡 NPU 冒烟档。--attn-backend native 避开与 NPU graph capture 不兼容的
# flash-attn-npu kvcache 路径。逐用例参数在 overlay_args 里；argparse 取最后一次
# 出现，故 overlay 会覆盖这里。
save="$CI_OUTPUT_DIR/${PROFILE}-ckpt"
cd "$TARGET_ROOT"
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --model-hub modelscope \
  --backend npu \
  --attn-backend native \
  --tp-size 1 \
  --world-size 1 \
  --save-path "$save" \
  --save-interval 1 \
  "${EXTRA_ARGS[@]}"
ls -d "$save"/step_* >/dev/null

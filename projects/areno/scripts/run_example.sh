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

start_ave_judge() {
  # ave_event_recognition 的 reward 需要一个 OpenAI 兼容 judge 服务
  # （JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_API_KEY，缺失即报错）。引擎不校验
  # 示例输出数值、只认退出码，且该 judge 是上游示例的外部依赖；故 CI 起一个
  # 标准库存根 judge（始终返回合法 JSON），把 JUDGE_* 指向它，训练结束由 trap
  # 收尾。同卡 a2-1 不再起第二个模型，避免与训练抢 NPU/端口。
  local port="${AVE_JUDGE_PORT:-18081}"
  nohup python - "$port" >/tmp/ave-judge.log 2>&1 <<'PY' &
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self._send({
            "id": "ave-judge-stub",
            "object": "chat.completion",
            "model": "ave-judge-stub",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps({"score": 7.5})},
            }],
        })

    def log_message(self, *args):
        pass


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
PY
  local pid=$!
  export JUDGE_BASE_URL="http://127.0.0.1:${port}/v1"
  export JUDGE_MODEL="ave-judge-stub"
  export JUDGE_API_KEY="unused"
  for _ in $(seq 1 30); do
    if curl -fsS -X POST "http://127.0.0.1:${port}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"judge","messages":[{"role":"user","content":"ping"}]}' >/dev/null 2>&1; then
      echo "ave judge stub ready on :$port (pid=$pid)"
      AVE_JUDGE_PID=$pid
      return 0
    fi
    sleep 1
  done
  echo "FAILED - ave judge stub did not become ready on :$port" >&2
  tail -20 /tmp/ave-judge.log >&2 || true
  return 1
}

cleanup_ave_judge() {
  [[ -n "${AVE_JUDGE_PID:-}" ]] && kill "$AVE_JUDGE_PID" 2>/dev/null || true
}

# 需要外部 OpenAI 兼容服务的 profile 在训练前把 sidecar 起好（同 specforge /
# slime / aibrix 的做法）。
case "$PROFILE" in
  ave_event_recognition_npu)
    start_ave_judge
    trap cleanup_ave_judge EXIT
    ;;
esac

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

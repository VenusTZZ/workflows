#!/usr/bin/env bash
# Run one supported AReno example from the CI working copy.
# $1 is the manifest entry path (relative to the examples root). The
# profile selects the recipe. Overlay CLI args come from OVERLAY_ARGS
# (JSON array, possibly []). Never git add/commit/push.
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
  # CANN env scripts assume a login shell and reference optional variables
  # without ${VAR:-} guards; relax strict mode while sourcing vendor code.
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
  # The workflow serializes manifest.overlay_args as JSON. Expand each item
  # with shell quoting intact, then allow CI paths such as ${CI_OUTPUT_DIR}
  # to resolve only in this job's environment.
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

echo "running $EXAMPLE_REL with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

run_bash_game_npu() {
  # Repo fixture convention: the deterministic position set is a checked-in
  # fixture (projects/areno/fixtures/ci_bash_game_8.jsonl, generated once by
  # the upstream dataset_generator.py at seed 42, same schema); --dataset-path
  # is supplied through overlay_args as ${FIXTURE_DIR}/... so the upstream
  # generator is not re-run in the leg. run_agent.py calls the local rollout
  # proxy (submit_move tool) and reward.py scores against the hidden oracle.
  # Qwen3-0.6B from ModelScope keeps a single 1-card leg within budget.
  local dir="$EXAMPLES_ROOT/examples/agentic/bash_game"
  areno train \
    --ckpt Qwen/Qwen3-0.6B \
    --model-hub modelscope \
    --dataset-loader-fn "$dir/dataset_loader.py" \
    --reward-fn-path "$dir/reward.py" \
    --agent-fn "$dir/run_agent.py" \
    --backend npu \
    --attn-backend native \
    --tp-size 1 \
    --world-size 1 \
    --save-path "$CI_OUTPUT_DIR/bash_game-ckpt" \
    --save-interval 1 \
    "${EXTRA_ARGS[@]}"
  ls -d "$CI_OUTPUT_DIR"/bash_game-ckpt/step_* >/dev/null
}

case "$PROFILE" in
  bash_game_npu)
    cd "$TARGET_ROOT"
    run_bash_game_npu
    ;;
  *)
    echo "unknown profile: $PROFILE" >&2
    exit 2
    ;;
esac

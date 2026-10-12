#!/usr/bin/env bash
# Run one supported AReno example from the CI working copy.
# $1 is the manifest entry path (relative to the examples root); it is only
# checked for existence. Everything example-specific travels in the manifest's
# overlay_args (OVERLAY_ARGS, a JSON array): dataset path, loader, reward,
# agent entry, algo and CI-scaling flags. The shared single-card NPU smoke
# harness (native attention, Qwen3-0.6B via ModelScope, one checkpoint per
# profile) lives here, so adding an example only touches the manifest.
# Never git add/commit/push.
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
  # with shell quoting intact, then allow CI paths such as ${FIXTURE_DIR} /
  # ${EXAMPLES_ROOT} to resolve only in this job's environment.
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

# Shared single-card NPU smoke harness. --attn-backend native avoids the
# flash-attn-npu kvcache path that is incompatible with NPU graph capture.
# Per-example flags follow in overlay_args and override any of these because
# argparse keeps the last occurrence of a repeated flag.
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

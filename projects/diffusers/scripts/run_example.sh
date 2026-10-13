#!/usr/bin/env bash
# Run one diffusers example from a CI working copy of the target tree.
# $1 is the manifest entry path. EXEC, when set, names the launchable
# file relative to the target root; otherwise path itself must be a
# launchable file. Overlay CLI args come from OVERLAY_ARGS (JSON array,
# possibly []).
#
# Python examples are launched with `accelerate launch` (the upstream
# entrypoint for the training scripts). Shell examples that do not pass
# "$@" get it attached in this working copy only (last command line).
# Never git add/commit/push.
#
# Contract: docs/guarding-examples.md "项目运行脚本契约".
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"

EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"
[[ -e "$EXAMPLE_PATH" ]] || { echo "example not found: $EXAMPLE_PATH" >&2; exit 1; }

# Resolve the launchable file: EXEC (relative to the target root) when
# set, otherwise path itself.
if [[ -n "${EXEC:-}" ]]; then
  LAUNCH_PATH="$TARGET_ROOT/$EXEC"
else
  LAUNCH_PATH="$EXAMPLE_PATH"
fi
if [[ ! -f "$LAUNCH_PATH" ]]; then
  echo "launchable file not found: $LAUNCH_PATH (directory examples need an exec field)" >&2
  exit 1
fi

mkdir -p "$CI_OUTPUT_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c "import torch, torch_npu; print('NPU available:', torch.npu.is_available(), 'devices:', torch.npu.device_count())"

expand_overlay() {
  # The workflow serializes manifest.overlay_args as JSON. Expand each
  # item with shell quoting intact, then allow CI paths such as
  # ${CI_OUTPUT_DIR} / ${SDXL_BASE_PATH} to resolve only in this job's
  # environment.
  "$PYTHON" - <<'PY'
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

echo "running $LAUNCH_PATH with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

ensure_passthrough() {
  # Shell examples that already forward "$@" need no patch; otherwise
  # attach it in this CI working copy only, on the last non-comment
  # line (the tail of the example's main command). Python entry points
  # take the overlay args directly on their own command line.
  local script="$1"
  [[ "$script" == *.sh ]] || return 0
  if grep -qE '"\$@"' "$script"; then
    echo "example already has \"\$@\"; skipping patch"
    return 0
  fi
  "$PYTHON" - "$script" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
for i in range(len(lines) - 1, -1, -1):
    stripped = lines[i].strip()
    if stripped and not stripped.startswith('#'):
        raw = lines[i]
        newline = ''
        if raw.endswith('\r\n'):
            newline = '\r\n'
            raw = raw[:-2]
        elif raw.endswith('\n'):
            newline = '\n'
            raw = raw[:-1]
        lines[i] = raw.rstrip() + ' "$@"' + newline
        path.write_text(''.join(lines), encoding='utf-8')
        print(f'patched {path} to pass "$@" on last command line')
        raise SystemExit(0)
raise SystemExit(f'{path}: cannot find a command line to attach "$@"')
PY
}

ensure_passthrough "$LAUNCH_PATH"

# Visible NPU count for the multi-card launcher.
count_npus() {
  "$PYTHON" -c "import torch, torch_npu; print(torch.npu.device_count())"
}

# Materialize an accelerate+DeepSpeed launch config whose num_processes follows
# the visible NPU count. ZeRO-2 (configs/ds_zero2.json): gradients sharded,
# optimizer states CPU-offloaded, parameters NOT sharded. Do not go back to
# ZeRO-3: with stage 3 transformers auto-constructs every from_pretrained model
# with partitioned (zero.Init) params, and frozen modules that are never wrapped
# by a DeepSpeedEngine - the text encoders - stay broken: the position-embedding
# table reads shape[0]=0 and CLIPTextModel.forward raises "Sequence length must
# be less than max_position_embeddings" (verified on a2-2, 2026-10-13;
# zero3_init_flag is NOT the trigger - only the stage is).
# LAUNCHER=accelerate-deepspeed.
prepare_deepspeed_configs() {
  local ds_json="${PROJECT_ROOT:?PROJECT_ROOT is required}/configs/ds_zero2.json"
  local cfg_dir="$GITHUB_WORKSPACE/ci_patch"
  mkdir -p "$cfg_dir"
  local npus
  npus=$(count_npus)
  cat > "$cfg_dir/accelerate-deepspeed.yml" <<EOF
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: $ds_json
  zero3_init_flag: false
distributed_type: DEEPSPEED
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: $npus
rdzv_backend: static
same_network: true
use_cpu: false
EOF
  echo "deepspeed launch config: $cfg_dir/accelerate-deepspeed.yml (ds: $ds_json, $npus npu(s))"
}

# Shell examples run with cwd = the example's directory; Python entry
# points run with cwd = the target root (diffusers examples resolve
# their own paths relative to the repo root).
case "$LAUNCH_PATH" in
  *.sh)
    cd "$(dirname "$LAUNCH_PATH")"
    bash "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    ;;
  *)
    cd "$TARGET_ROOT"
    # Some examples import a package that sits next to them (e.g.
    # examples/research_projects/ip_adapter imports `ip_adapter.ip_adapter`);
    # add the example's own directory to PYTHONPATH.
    export PYTHONPATH="$(dirname "$LAUNCH_PATH")${PYTHONPATH:+:$PYTHONPATH}"
    case "${LAUNCHER:-}" in
      accelerate-deepspeed)
        prepare_deepspeed_configs
        "$PYTHON" -m accelerate.commands.launch \
          --config_file "$GITHUB_WORKSPACE/ci_patch/accelerate-deepspeed.yml" \
          "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
        ;;
      "")
        # Training examples expect the upstream entrypoint. Single process;
        # the script's own --mixed_precision controls the precision, so the
        # launcher's is pinned to `no` to avoid a conflict.
        accelerate launch --num_processes 1 --num_machines 1 \
          --mixed_precision no "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
        ;;
      *)
        echo "unknown LAUNCHER: $LAUNCHER (supported: accelerate-deepspeed)" >&2
        exit 2
        ;;
    esac
    ;;
esac

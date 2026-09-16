#!/usr/bin/env bash
# Run one torchtitan example from a CI working copy of the target tree.
# $1 is the manifest entry path. Overlay CLI args come from OVERLAY_ARGS
# (JSON array, possibly []). Shell launchers already forward "$@"; this
# script never patches them. Never git add/commit/push.
#
# The manifest stores two fields per supported entry:
#   path: the upstream file we want to verify exists (used by
#         manifest-check, never executed)
#   exec: the .sh launcher setup writes under $TARGET_ROOT/scripts/
#         (execs `torchrun -m torchtitan.train "$@"` so LOCAL_RANK
#         is set by torchrun — running `python torchtitan/train.py`
#         directly crashes on `LOCAL_RANK must be set`)
# When EXEC is set we run the launcher; otherwise we fall back to
# the path field (ad-hoc Python entry points).
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"
# The engine exports EXEC = manifest.entry.exec when set; run_example
# scripts for projects that have no exec field (ad-hoc .py entries)
# silently keep the legacy behaviour.
EXEC_REL="${EXEC:-}"

EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"
[[ -e "$EXAMPLE_PATH" ]] || { echo "example not found: $EXAMPLE_PATH" >&2; exit 1; }

if [[ -n "$EXEC_REL" ]]; then
  LAUNCH_PATH="$TARGET_ROOT/$EXEC_REL"
else
  LAUNCH_PATH="$EXAMPLE_PATH"
fi
if [[ ! -f "$LAUNCH_PATH" ]]; then
  echo "launchable file not found: $LAUNCH_PATH" >&2
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
  # ${CI_OUTPUT_DIR} to resolve only in this job's environment.
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

# sitecustomize.py injects the CUDA->NPU transfer at interpreter
# startup. torchtitan imports torch but does not pin device="cuda"
# directly; however, downstream deps (datasets, accelerate's plugin
# loader, fbgemm-like helpers) sometimes do, and the c10d backend map
# test in run_example.sh checks npu.is_available(). The transfer
# itself is a no-op if no torch.cuda call has been made yet.
prepare_shims() {
  local shim_dir="$GITHUB_WORKSPACE/ci_patch"
  mkdir -p "$shim_dir"
  cat > "$shim_dir/sitecustomize.py" <<'PY'
from torch_npu.contrib import transfer_to_npu  # noqa: F401  (cuda->npu)
PY
  export PYTHONPATH="$shim_dir:${PYTHONPATH:-}"
}

prepare_shims

# .sh launchers in scripts/ (written by setup_example.sh) cd to
# $TARGET_ROOT internally and exec torchrun -m torchtitan.train;
# we still cd to the launcher's dir for logging consistency.
case "$LAUNCH_PATH" in
  *.sh)
    cd "$(dirname "$LAUNCH_PATH")"
    bash "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    ;;
  *)
    cd "$TARGET_ROOT"
    python "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    ;;
esac

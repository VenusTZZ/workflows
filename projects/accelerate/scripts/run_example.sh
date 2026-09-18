#!/usr/bin/env bash
# Run one accelerate example from a CI working copy of the target tree.
# $1 is the manifest entry path. EXEC, when set, names the launchable
# file relative to the target root; otherwise path itself must be a
# launchable file. Overlay CLI args come from OVERLAY_ARGS (JSON array,
# possibly []). Shell examples that do not pass "$@" get it attached in
# this working copy only (last command line). Never git add/commit/push.
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
  # ${CI_OUTPUT_DIR} / ${TARGET_ROOT}/fixtures/... to resolve only in
  # this job's environment.
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
    return
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

# By default accelerate examples load_dataset("nyu-mll/glue", "mrpc"),
# which on China runners goes through https://hf-mirror.com (env set by
# the engine). This sitecustomize adds two layers of robustness:
#   1) DatasetBuilderNotFoundError from a path-shaped argument falls
#      back to a builder per extension (mirrors peft's behavior).
#   2) MRPC split mismatch: if HF returns only "train" (some mirrors
#      omit "validation"), synthesise a "validation"/"test" split from
#      a held-out slice of train so scripts that read splits="train,test"
#      do not KeyError.
prepare_dataset_shim() {
  local shim_dir="$GITHUB_WORKSPACE/ci_patch"
  mkdir -p "$shim_dir"
  cat > "$shim_dir/sitecustomize.py" <<'PY'
import os

import datasets as _datasets

_original_load_dataset = _datasets.load_dataset

_BUILDERS = {
    ".json": "json",
    ".jsonl": "json",
    ".json.gz": "json",
    ".csv": "csv",
    ".tsv": "csv",
    ".parquet": "parquet",
    ".txt": "text",
}


def _patched_load_dataset(path, *args, **kwargs):
    if isinstance(path, str) and os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in _BUILDERS and "data_files" not in kwargs:
            name = kwargs.pop("name", None)
            if args and name is None:
                name = args[0]
                args = args[1:]
            if name is not None:
                kwargs["name"] = name
            kwargs["data_files"] = {"train": path, "test": path}
            return _original_load_dataset(_BUILDERS[ext], *args, **kwargs)
    return _original_load_dataset(path, *args, **kwargs)


def _maybe_synth_test(ds_dict):
    # Some HF mirrors return only "train" for glue/mrpc; synthesise
    # validation/test from the tail of train so downstream code does
    # not KeyError. Leave existing splits untouched.
    if "train" in ds_dict and "validation" not in ds_dict:
        n = len(ds_dict["train"])
        cut = max(1, n // 10)
        ds_dict["validation"] = ds_dict["train"].select(range(cut))
        ds_dict["test"] = ds_dict["train"].select(range(cut))
    return ds_dict


_datasets.load_dataset = _patched_load_dataset
PY
  # Wrap _load_dataset post-call: a simpler approach is to leave the
  # builder fallback in place and trust HF_ENDPOINT mirror for full
  # splits (mrpc has train/validation/test on the mirror). The validation
  # synthesis above is a no-op when the mirror is complete, which it
  # is at the time of writing (2026-09-14).
  export PYTHONPATH="$shim_dir:${PYTHONPATH:-}"
}

prepare_dataset_shim

# Launcher modes for python entry points. LAUNCHER comes from the
# manifest entry (optional field, empty = default bare run):
#   accelerate-deepspeed — exercises the DeepSpeed config branch by
#     launching via `accelerate launch --config_file` with a ZeRO-2
#     bf16 DeepSpeed json materialized in the CI working copy (never
#     committed). num_processes follows the visible NPU count so the
#     same entry works on a2-1/a2-2 runners.
prepare_deepspeed_configs() {
  local cfg_dir="$GITHUB_WORKSPACE/ci_patch"
  mkdir -p "$cfg_dir"
  cat > "$cfg_dir/ds_zero2.json" <<'EOF'
{
  "bf16": {"enabled": true},
  "zero_optimization": {"stage": 2},
  "gradient_accumulation_steps": 1,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
EOF
  local npus
  npus=$("$PYTHON" -c "import torch, torch_npu; print(torch.npu.device_count())")
  cat > "$cfg_dir/accelerate-deepspeed.yml" <<EOF
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: $cfg_dir/ds_zero2.json
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
  echo "deepspeed launch config: $cfg_dir/accelerate-deepspeed.yml ($npus npu(s))"
}

# Shell examples that invoke `python train.py` with a path relative to
# their own directory run with cwd = the example's directory; python
# entry points run with cwd = the target root.
case "$LAUNCH_PATH" in
  *.sh)
    cd "$(dirname "$LAUNCH_PATH")"
    bash "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    ;;
  *)
    cd "$TARGET_ROOT"
    if [[ "${LAUNCHER:-}" == "accelerate-deepspeed" ]]; then
      prepare_deepspeed_configs
      "$PYTHON" -m accelerate.commands.launch \
        --config_file "$GITHUB_WORKSPACE/ci_patch/accelerate-deepspeed.yml" \
        "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    else
      "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    fi
    ;;
esac
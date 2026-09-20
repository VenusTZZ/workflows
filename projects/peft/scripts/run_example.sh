#!/usr/bin/env bash
# Run one peft example from a CI working copy of the target tree.
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
  # ${CI_OUTPUT_DIR} / ${SFT_MODEL_PATH} to resolve only in this job's
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


# One sitecustomize.py, three patches, injected at interpreter startup:
# 1. CUDA->NPU: most peft examples hardcode device="cuda";
#    torch_npu's transfer_to_npu maps torch.cuda onto npu (also covers
#    `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
#    device-branch examples such as lora_ga: verified npu-5, the branch
#    resolves to npu).
# 2. Dataset: the sft example calls load_dataset(path) with a local
#    .jsonl fixture and reads BOTH "train" and "test" splits
#    (splits="train,test"); datasets 3.x cannot infer a builder from a
#    single file path - rewrite local data files to
#    load_dataset("<builder>", data_files={"train": path, "test": path}).
# 3. fp16 neutralizer: several finetuning scripts hardcode
#    TrainingArguments(fp16=True, ...). accelerate 1.15's fp16 chain
#    (optimizer.step(grad_scaler=scaler)) is incompatible with torch_npu
#    2.9 amp.GradScaler -> TypeError "Adam.step got unexpected keyword
#    argument grad_scaler" (gralora/lily/peanut/alora/dora/road, all
#    repro'd pre-shim; coder npu-5 verified the neutralizer makes them
#    exit 0). No supported entry relies on fp16=True, so forcing
#    fp16=False globally is a no-op for every other entry.
prepare_shims() {
  local shim_dir="$GITHUB_WORKSPACE/ci_patch"
  mkdir -p "$shim_dir"
  cat > "$shim_dir/sitecustomize.py" <<'PY'
from torch_npu.contrib import transfer_to_npu  # noqa: F401  (cuda->npu)

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
            # Map the single file onto both splits: the sft example
            # loads split="train" and split="test" from the same
            # dataset_name.
            kwargs["data_files"] = {"train": path, "test": path}
            return _original_load_dataset(_BUILDERS[ext], *args, **kwargs)
    return _original_load_dataset(path, *args, **kwargs)


_datasets.load_dataset = _patched_load_dataset

import transformers as _tf

_orig_training_args_init = _tf.TrainingArguments.__init__


def _fp16_neutral_init(self, *args, **kwargs):
    if kwargs.get("fp16"):
        kwargs["fp16"] = False
    _orig_training_args_init(self, *args, **kwargs)


_tf.TrainingArguments.__init__ = _fp16_neutral_init
PY
  export PYTHONPATH="$shim_dir:${PYTHONPATH:-}"
}

prepare_shims

# run_peft.sh invokes `python train.py` with a path relative to its own
# directory, so shell examples run with cwd = the example's directory;
# python entry points run with cwd = the target root.
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

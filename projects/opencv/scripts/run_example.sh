#!/usr/bin/env bash
# Run one opencv example from a CI working copy of the target tree.
# $1 is the manifest entry path (under TARGET_ROOT). EXEC, when set,
# names the launchable file relative to the target root; otherwise
# path itself must be a launchable file. Overlay CLI args come from
# OVERLAY_ARGS (JSON array, possibly []). Shell examples that do not
# pass "$@" get it attached in this working copy only (last command
# line). Never git add/commit/push.
#
# opencv-specific bits beyond the peft contract:
#   - Source nnal/atb BEFORE the ascend-toolkit on top of the peft
#     order, because the opencv_test_cannops gtest binary links
#     against ATB libs and atb's set_env.sh must come AFTER hdc env
#     (which already sources atb under set -u, breaking on $ZSH_VERSION
#     unset). We re-source under set +u explicitly.
#   - Inject PYTHONPATH for the source-built cv2 if the script looks
#     like a Python example (peft only handles torch + datasets shims;
#     we have no equivalent here, but PYTHONPATH must already be set
#     by setup_example.sh; we just verify it is).

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
    echo "launchable file not found: $LAUNCH_PATH" >&2
    exit 1
fi

mkdir -p "$CI_OUTPUT_DIR"

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
else
    PYTHON=python
fi

# Source CANN env (with the set -u workaround for nnal/atb — its
# set_env.sh references $ZSH_VERSION which is unset under bash + set -u
# and triggers an unbound variable exit). The hdc env.sh already
# sources both ascend-toolkit AND nnal/atb, so we just do it once
# here under set +u and skip a redundant re-source.
unset ZSH_VERSION
set +u
source /home/coder/.hdc/env.sh 2>/dev/null || true
set -u

# PYTHONPATH for the source-built cv2 — must be set in this step as
# well, because the examples-template workflow gives each step a
# fresh shell and the PYTHONPATH exported by setup_example.sh does
# not carry across (CI step boundary, not a same-shell export).
# setup_example.sh builds into this prefix; if it's missing the
# source build never ran and the example would fail on import anyway.
OPENCV_INSTALL=/usr/local/opencv-cann
PP="$OPENCV_INSTALL/lib/python3.12/site-packages"
if [[ -d "$PP" ]] && [[ ":${PYTHONPATH:-}:" != *":$PP:"* ]]; then
    export PYTHONPATH="$PP:${PYTHONPATH:-}"
fi
echo "run: PYTHONPATH -> $PYTHONPATH"

# Sanity guard: source-built cv2 must be importable for the script to
# do anything useful (DNN_BACKEND_CANN == 0 means the pip wheel was
# loaded instead). Print a clear error rather than failing at the
# assert in the example script.
$PYTHON -c "import cv2; assert cv2.dnn.DNN_BACKEND_CANN != 0, 'cv2 has no CANN backend; source build not in PYTHONPATH?'" 2>&1 | head -3

expand_overlay() {
    # The workflow serializes manifest.overlay_args as JSON. Expand
    # each item with shell quoting intact, then allow CI paths such
    # as ${CI_OUTPUT_DIR} / ${FIXTURE_DIR} to resolve only in this
    # job's environment.
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
    # Shell examples that already forward "$@" need no patch;
    # otherwise attach it in this CI working copy only, on the last
    # non-comment line (the tail of the example's main command).
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
            newline = '\r\n'; raw = raw[:-2]
        elif raw.endswith('\n'):
            newline = '\n'; raw = raw[:-1]
        lines[i] = raw.rstrip() + ' "$@"' + newline
        path.write_text(''.join(lines), encoding='utf-8')
        print(f'patched {path} to pass "$@" on last command line')
        raise SystemExit(0)
raise SystemExit(f'{path}: cannot find a command line to attach "$@"')
PY
}

ensure_passthrough "$LAUNCH_PATH"

# Change to the target root so relative paths in the scripts
# (opencv/samples/data/baboon.jpg etc.) resolve against the upstream
# checkout. Use a subshell so the cd doesn't leak to the caller.
(
    cd "$TARGET_ROOT"
    if [[ "$LAUNCH_PATH" == *.py ]]; then
        exec "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    else
        # .sh: the patched "$@" is on the last command line
        exec bash "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    fi
)

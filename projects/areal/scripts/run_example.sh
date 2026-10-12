#!/usr/bin/env bash
# Run one AReaL example from a CI working copy of the target tree.
# $1 is the example path relative to TARGET_ROOT. Overlay CLI args come from
# OVERLAY_ARGS (JSON array, serialized by the workflow from
# manifest.overlay_args); ${...} items are expanded from the job environment.
#
# AReaL examples are plain `python examples/.../foo.py <overrides>` entrypoints
# (Hydra-style key=value overrides), so no launcher is involved.
set -euo pipefail

EXAMPLE_REL="${1:?example path is required}"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
LAUNCH_PATH="$TARGET_ROOT/$EXAMPLE_REL"
[[ -f "$LAUNCH_PATH" ]] || { echo "no such example: $LAUNCH_PATH" >&2; exit 2; }

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

expand_overlay() {
  "$PYTHON" - <<'PY'
import json
import os
import shlex

raw = os.environ.get('OVERLAY_ARGS', '').strip()
if not raw or raw in ('null', '""'):
    raise SystemExit(0)
items = json.loads(raw)
if items in (None, ''):
    raise SystemExit(0)
tokens = []
for item in items:
    tokens.extend(shlex.split(os.path.expandvars(item), posix=True))
print(' '.join(shlex.quote(token) for token in tokens))
PY
}

eval "EXTRA_ARGS=( $(expand_overlay) )"
echo "running $LAUNCH_PATH with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

cd "$TARGET_ROOT"
# Repo root first so dotted module paths used by workflow kwargs resolve
# (e.g. boba_grpo.py's reward_fn "examples.math.boba_grpo.boba_reward_fn");
# the example dir covers sibling modules imported by the script itself.
export PYTHONPATH="$TARGET_ROOT:$(dirname "$LAUNCH_PATH")${PYTHONPATH:+:$PYTHONPATH}"
# examples/scaffolding/*.py use package-relative imports (from ._compat
# import ...), so they must run as modules, not as plain scripts.
#
# Forked runtime logs (data-worker / data-router / data-gateway under
# <fileroot>/logs/) are NOT part of this job's stdout, so a crash there is
# otherwise invisible; dump them when the example fails.
dump_forked_logs() {
  local root="${AREAL_LOG_ROOT:-/tmp/areal/experiments/logs}"
  [[ -d "$root" ]] || return 0
  local f
  while IFS= read -r f; do
    echo "===== $f (tail -n 200) ====="
    tail -n 200 "$f" || true
  done < <(find "$root" -type f -name 'data-*.log' 2>/dev/null)
}

run_rc=0
case "$EXAMPLE_REL" in
  examples/scaffolding/*.py)
    MODULE="${EXAMPLE_REL%.py}"
    MODULE="${MODULE//\//.}"
    "$PYTHON" -m "$MODULE" "${EXTRA_ARGS[@]}" || run_rc=$?
    ;;
  examples/hermes/train.py)
    # Online RL: train.py blocks waiting for an externally-driven session
    # lifecycle, so a plain foreground run would hang until timeout.
    # Orchestrate one full loop instead: background trainer (starts the
    # vLLM rollout engine + proxy gateway) -> Agent Service (HermesAgent)
    # -> start_session (mint sk-sess key) -> one piped conversation via
    # hermes_loop (EOF exits cleanly) -> set_reward -> start_session
    # refresh (ends the session, exports the trajectory) -> the trainer's
    # _OnlineAgent unblocks, trains its single step and exits.
    HERMES_ADMIN_KEY="areal-agent-ci"
    TRAIN_LOG="$PWD/hermes_train.log"
    AGENT_LOG="$PWD/hermes_agent.log"

    hermes_cleanup() {
      areal agent stop >/dev/null 2>&1 || true
      if [[ -n "${TRAINER_PID:-}" ]] && kill -0 "$TRAINER_PID" 2>/dev/null; then
        kill "$TRAINER_PID" 2>/dev/null || true
      fi
    }
    trap hermes_cleanup EXIT

    # 1) Trainer in the background; its log is the source of truth for
    #    the proxy/inference gateway address (rl_trainer logs
    #    "Proxy gateway available at http://host:port" once online mode
    #    is up - engine load takes minutes).
    "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" >"$TRAIN_LOG" 2>&1 &
    TRAINER_PID=$!

    GW=""
    for i in $(seq 1 120); do
      # Strict URL charset: the areal log line can carry ANSI colour bytes
      # right after the URL, and [^[:space:]] would swallow the ESC byte;
      # requests then dies with "Failed to parse: http://...<ESC>/...".
      GW=$(grep 'Proxy gateway available at' "$TRAIN_LOG" 2>/dev/null | tail -1 \
        | grep -oE 'https?://[A-Za-z0-9._:-]+' | tail -1 || true)
      [[ -n "$GW" ]] && break
      if ! kill -0 "$TRAINER_PID" 2>/dev/null; then
        echo "trainer exited before the gateway came up; tail of $TRAIN_LOG:"
        tail -n 120 "$TRAIN_LOG"
        exit 1
      fi
      echo "[wait $((i * 10))s] trainer: $(tail -n 1 "$TRAIN_LOG" 2>/dev/null | cut -c1-200)"
      sleep 10
    done
    if [[ -z "$GW" ]]; then
      echo "proxy gateway did not come up in 20min; tail of $TRAIN_LOG:"
      tail -n 120 "$TRAIN_LOG"
      exit 1
    fi
    echo "inference gateway: $GW"

    # 2) Agent Service with the Hermes agent (run blocks until the stack
    #    is healthy, then exits leaving the service up).
    if ! areal agent run --agent examples.hermes.hermes.HermesAgent \
        --admin-api-key "$HERMES_ADMIN_KEY" --force >"$AGENT_LOG" 2>&1; then
      echo "areal agent run failed; tail of $AGENT_LOG:"
      tail -n 120 "$AGENT_LOG"
      exit 1
    fi
    AGENT_GW=$(grep -oE 'gateway=[A-Za-z0-9._:/-]+' "$AGENT_LOG" | tail -1 | cut -d= -f2 || true)
    if [[ -z "$AGENT_GW" ]]; then
      AGENT_GW=$(areal agent status 2>/dev/null | grep -oE 'https?://[A-Za-z0-9._:-]+' | tail -1 || true)
    fi
    if [[ -z "$AGENT_GW" ]]; then
      echo "agent gateway not found; tail of $AGENT_LOG:"
      tail -n 120 "$AGENT_LOG"
      exit 1
    fi
    echo "agent gateway: $AGENT_GW"

    # 3) Mint the session key on the inference gateway (admin key is the
    #    rollout.admin_api_key overlay value). start_session.py prints the
    #    machine-readable "SESSION_API_KEY=..." line on STDERR (the stdout
    #    side is ANSI-coloured prose), and it can return HTTP 429 until the
    #    trainer grants capacity - retry briefly.
    SESS_KEY=""
    SESS_OUT=""
    for attempt in $(seq 1 12); do
      SESS_OUT=$("$PYTHON" examples/hermes/start_session.py "$GW" \
        --admin-key sk-areal-ci 2>&1 || true)
      SESS_KEY=$(printf '%s\n' "$SESS_OUT" \
        | sed -n 's/^SESSION_API_KEY=//p' | tail -1)
      [[ -n "$SESS_KEY" ]] && break
      echo "[session attempt $attempt] $(printf '%s\n' "$SESS_OUT" \
        | grep -E 'HTTP|Error|✘' | tail -1)"
      sleep 5
    done
    if [[ -z "$SESS_KEY" ]]; then
      echo "start_session produced no session key; raw output:"
      printf '%s\n' "$SESS_OUT"
      tail -n 40 "$TRAIN_LOG"
      exit 1
    fi
    echo "session: $SESS_KEY"

    # 4) One piped conversation; hermes_loop exits cleanly on EOF. The
    #    inf-* flags route the agent's LLM calls through the inference
    #    gateway under the session key (self-evolution capture). The model
    #    id is "default" - the name InferenceDataProxy registers the served
    #    model under (log: "Model registered: name=default").
    if ! echo "Say hello and introduce yourself in one sentence." | \
        "$PYTHON" examples/hermes/hermes_loop.py "$AGENT_GW" \
          --admin-api-key "$HERMES_ADMIN_KEY" \
          --inf-base-url "$GW" \
          --inf-model default \
          --session-api-key "$SESS_KEY"; then
      echo "hermes_loop failed; tails:"
      tail -n 60 "$AGENT_LOG"
      tail -n 40 "$TRAIN_LOG"
      exit 1
    fi

    # 5) Reward, then refresh the session: ends it and exports the
    #    trajectory, which unblocks the trainer.
    "$PYTHON" examples/hermes/set_reward.py "$GW" --api-key "$SESS_KEY" --reward 1.0
    "$PYTHON" examples/hermes/start_session.py "$GW" --admin-key sk-areal-ci \
      --api-key "$SESS_KEY" >/dev/null

    # 6) Trainer trains its single step and exits.
    if wait "$TRAINER_PID"; then
      echo "hermes online-rl loop completed"
    else
      run_rc=$?
      echo "trainer exited rc=$run_rc; tail of $TRAIN_LOG:"
      tail -n 120 "$TRAIN_LOG"
    fi
    ;;
  *.sh)
    # Shell examples dispatch with bash (generic guard contract: .sh -> bash,
    # .py -> python).
    bash "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" || run_rc=$?
    ;;
  *)
    "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" || run_rc=$?
    ;;
esac

if ((run_rc != 0)); then
  echo "example exited rc=$run_rc; dumping forked runtime logs"
  dump_forked_logs
fi
exit "$run_rc"

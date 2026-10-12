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
    # -> start_session over HTTP (mint session key; see step 3 for why the
    # CLI is bypassed) -> one piped conversation via hermes_loop (EOF exits
    # cleanly) -> set_reward -> start_session refresh (ends the session,
    # exports the trajectory) -> the trainer's _OnlineAgent unblocks, trains
    # its single step and exits.
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

    # 3) Mint the session key on the inference gateway. Talk HTTP directly
    #    instead of examples/hermes/start_session.py: that CLI hardcodes the
    #    V2 gateway contract (HTTP 201 + {"sessions":[{"session_api_key":..}]})
    #    while the v1 proxy gateway returns 200 + flat {"api_key","session_id"}
    #    (proxy_gateway.py NEW path returns the worker response as-is) - the
    #    CLI dies on a perfectly successful session, the retry loop then
    #    spawns zombie sessions until all 3 workers' capacity is exhausted
    #    and everything 429s. 429 ("no capacity") is retried: workers only
    #    register in the readiness queue after the trainer is fully up.
    gw_post() {
      # $1 = path, $2 = bearer key, $3 = json body; prints body on HTTP 200,
      # nothing otherwise (status goes to stderr for the retry log).
      "$PYTHON" - "$GW" "$1" "$2" "$3" <<'PY'
import sys, urllib.request, urllib.error

_, base, path, key, body = sys.argv
req = urllib.request.Request(
    f"{base}/{path}",
    data=body.encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=180) as resp:
        print(resp.read().decode())
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}", file=sys.stderr)
except OSError as e:
    print(f"{type(e).__name__}: {e}", file=sys.stderr)
PY
    }

    SESS_KEY=""
    for attempt in $(seq 1 24); do
      BODY=$(gw_post rl/start_session sk-areal-ci '{"task_id":"ci-hermes-smoke"}')
      SESS_KEY=$(printf '%s' "$BODY" | "$PYTHON" -c \
        'import json,sys; print(json.load(sys.stdin)["api_key"])' 2>/dev/null || true)
      [[ -n "$SESS_KEY" ]] && break
      echo "[session attempt $attempt] no session yet (capacity/registering)"
      sleep 5
    done
    if [[ -z "$SESS_KEY" ]]; then
      echo "start_session never returned a session key"
      tail -n 40 "$TRAIN_LOG"
      exit 1
    fi
    echo "session: $SESS_KEY"

    # 4) One piped conversation; hermes_loop exits cleanly on EOF. The
    #    inf-* flags route the agent's LLM calls through the inference
    #    gateway under the session key (self-evolution capture). The model
    #    id is "default" - the name InferenceDataProxy registers the served
    #    model under (log: "Model registered: name=default").
    #    TOOLSETS: hermes-agent enables 30+ toolsets by default (web/search/
    #    browser/terminal/vision/...); their definitions ride EVERY request
    #    and overflow vllm's max_model_len=8192 - the agent then death-loops
    #    its context compressor ("Context length exceeded (20 tokens).
    #    Cannot compress further.") because tool definitions cannot be
    #    compressed away, producing an error-message-only trajectory that
    #    the trainer rejects. Disable everything except the minimal
    #    "clarify" toolset so a plain greeting fits comfortably.
    if ! echo "Say hello and introduce yourself in one sentence." | \
        HERMES_DISABLED_TOOLSETS="web,search,x_search,vision,video,image_gen,video_gen,computer_use,terminal,skills,browser,cronjob,file,tts,todo,memory,context_engine,session_search,project,code_execution,delegation,homeassistant,kanban,discord,discord_admin,yuanbao,feishu_doc,feishu_drive,spotify,debugging,safe,coding" \
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

    # 5) Reward the session, then refresh it: the refresh call (start_session
    #    with the existing api_key) ends the old session, waits for the
    #    training pipeline to cycle and starts a new one with the same key -
    #    the trajectory export unblocks the trainer's _OnlineAgent. Both go
    #    through gw_post (v1 contract: 200 on success). Refresh can 429 while
    #    the pipeline cycles (refresh_timeout on the gateway side), so retry.
    REWARD_RESP=$(gw_post rl/set_reward "$SESS_KEY" '{"reward":1.0}')
    if [[ -z "$REWARD_RESP" ]]; then
      echo "set_reward failed; tail of $TRAIN_LOG:"
      tail -n 40 "$TRAIN_LOG"
      exit 1
    fi
    echo "reward applied: $REWARD_RESP"

    REFRESHED=""
    for attempt in $(seq 1 12); do
      BODY=$(gw_post rl/start_session sk-areal-ci \
        "{\"task_id\":\"ci-hermes-smoke\",\"api_key\":\"$SESS_KEY\"}")
      NEW_KEY=$(printf '%s' "$BODY" | "$PYTHON" -c \
        'import json,sys; print(json.load(sys.stdin)["api_key"])' 2>/dev/null || true)
      if [[ -n "$NEW_KEY" ]]; then
        REFRESHED=1
        echo "session refreshed (key: $NEW_KEY)"
        break
      fi
      echo "[refresh attempt $attempt] pipeline still cycling"
      sleep 10
    done
    if [[ -z "$REFRESHED" ]]; then
      echo "session refresh never succeeded; the trainer may be stuck waiting"
      tail -n 40 "$TRAIN_LOG"
      exit 1
    fi

    # 6) Trainer trains its single step and exits. Bounded wait: a rejected
    #    trajectory (empty/error-only session) leaves the trainer waiting
    #    for the NEXT session forever - instead of holding the runner until
    #    the job timeout, fail after 15 min with the trainer log dumped.
    TRAIN_WAIT_S=900
    waited=0
    train_timed_out=0
    while kill -0 "$TRAINER_PID" 2>/dev/null; do
      if (( waited >= TRAIN_WAIT_S )); then
        echo "trainer still running after ${TRAIN_WAIT_S}s - likely waiting for another session (rejected trajectory?); killing and dumping"
        train_timed_out=1
        kill "$TRAINER_PID" 2>/dev/null || true
        sleep 5
        kill -9 "$TRAINER_PID" 2>/dev/null || true
        break
      fi
      sleep 15
      waited=$((waited + 15))
    done
    if (( train_timed_out )); then
      run_rc=1
      echo "trainer wait timed out; tail of $TRAIN_LOG:"
      tail -n 120 "$TRAIN_LOG"
    elif wait "$TRAINER_PID"; then
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

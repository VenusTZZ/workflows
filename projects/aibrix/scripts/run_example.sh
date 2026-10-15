#!/usr/bin/env bash
# Run aibrix local-mode gateway against a vLLM-Ascend backend.
# After vLLM starts and before run-local.sh, VLLM_LOG must contain backend=hccl.
set -euo pipefail

export PYTHONNOUSERSITE=1

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"

if [[ "${EXAMPLE_REL}" != "deployment/local/run-local.sh" ]]; then
  echo "unsupported aibrix guard path: ${EXAMPLE_REL}" >&2
  exit 1
fi

EXAMPLE_PATH="${TARGET_ROOT}/${EXAMPLE_REL}"
if [[ ! -f "${EXAMPLE_PATH}" ]]; then
  echo "upstream example not found: ${EXAMPLE_PATH}" >&2
  exit 1
fi

# CI-only patch on the temporary checkout: upstream waits only 10s for the
# Envoy listener, which the shared ARC runner exceeded while envoy was alive
# and healthy (config parsed, no error). Widen to 60s. Never committed back.
# Fail loudly if upstream rewrote this loop, so the patch gets re-reviewed
# instead of silently dropping back to 10s. Done before vLLM starts so a
# stale patch costs seconds, not a model load.
if ! grep -qF 'seq 1 10' "${EXAMPLE_PATH}"; then
  echo "upstream ${EXAMPLE_REL} no longer has the 10s Envoy readiness loop; re-review the CI patch" >&2
  exit 1
fi
sed -i 's/seq 1 10/seq 1 60/; s/-eq 10 \]/-eq 60 ]/; s/not ready after 10s/not ready after 60s/' "${EXAMPLE_PATH}"
grep -qF 'seq 1 60' "${EXAMPLE_PATH}"

mkdir -p "${CI_OUTPUT_DIR}"
RUN_LOG="${CI_OUTPUT_DIR}/run.log"
VLLM_LOG="${CI_OUTPUT_DIR}/vllm.log"
VLLM_PID_FILE="${CI_OUTPUT_DIR}/vllm.pid"
ENDPOINTS="${CI_OUTPUT_DIR}/endpoints.yaml"
TOOLS="${AIBRIX_TOOLS_DIR:-${HF_HOME:-${HOME}/.cache/huggingface}/aibrix/tools}"

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:${TOOLS}/bin:${TOOLS}/toolchain/go/bin:${PATH}"
set +u
# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# ATB env: accept both the legacy bind-mount layout (atb/latest/atb)
# and the CANN image layout (atb) - see setup_example.sh.
if [[ -f /usr/local/Ascend/nnal/atb/latest/atb/set_env.sh ]]; then
  source /usr/local/Ascend/nnal/atb/latest/atb/set_env.sh
elif [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
else
  echo "ATB env script not found under /usr/local/Ascend/nnal" >&2
  exit 1
fi
set -euo pipefail

stop_backend() {
  if [[ -f "${VLLM_PID_FILE}" ]]; then
    pid=$(cat "${VLLM_PID_FILE}" || true)
    if [[ -n "${pid}" ]]; then
      kill "${pid}" 2>/dev/null || true
    fi
  fi
  if [[ -x "${TARGET_ROOT}/deployment/local/stop-local.sh" ]]; then
    bash "${TARGET_ROOT}/deployment/local/stop-local.sh" >/dev/null 2>&1 || true
  fi
}
trap stop_backend EXIT

export PYTHONUNBUFFERED=1
export VLLM_USE_MODELSCOPE=True
: > "${VLLM_LOG}"
setsid bash -c "
  echo \$\$ > '${VLLM_PID_FILE}'
  exec vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --served-model-name Qwen/Qwen2.5-0.5B-Instruct \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 2048 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.2
" </dev/null >> "${VLLM_LOG}" 2>&1 &

HEALTH_URL='http://127.0.0.1:8000/health'
for attempt in $(seq 1 180); do
  pid=$(cat "${VLLM_PID_FILE}" 2>/dev/null || true)
  if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
    echo "vLLM exited before /health succeeded" >&2
    cat "${VLLM_LOG}" >&2
    exit 1
  fi
  if [[ -n "${pid}" ]] \
      && curl -sf --connect-timeout 2 -- "${HEALTH_URL}" >/dev/null \
      && ss -ltnp 'sport = :8000' | grep -q "pid=${pid},"; then
    break
  fi
  if [[ "${attempt}" -eq 180 ]]; then
    echo "timed out waiting for vLLM /health" >&2
    cat "${VLLM_LOG}" >&2
    exit 1
  fi
  sleep 2
done

if ! grep -F 'backend=hccl' "${VLLM_LOG}" >/dev/null; then
  echo "vLLM log has no backend=hccl; refusing silent CPU fallback" >&2
  cat "${VLLM_LOG}" >&2
  exit 1
fi

cat > "${ENDPOINTS}" <<'YAML'
models:
  - name: "Qwen/Qwen2.5-0.5B-Instruct"
    engine: "vllm"
    endpoints:
      - "127.0.0.1:8000"
YAML

rc=0
bash "${PROJECT_ROOT}/scripts/with_free_pprof.sh" \
  bash "${EXAMPLE_PATH}" -e "${ENDPOINTS}" \
  2>&1 | tee "${RUN_LOG}" || rc=$?
if [[ "${rc}" -ne 0 ]]; then
  echo "run-local.sh exited ${rc}; listener snapshot:" >&2
  ss -ltnp 2>/dev/null | grep -E ':(10080|9901|50052|8080) ' || true
  echo "envoy.log tail:" >&2
  tail -n 50 "${TARGET_ROOT}/deployment/local/logs/envoy.log" 2>/dev/null || true
  exit "${rc}"
fi

if ! grep -F 'AIBrix gateway is running!' "${RUN_LOG}" >/dev/null; then
  echo "run-local.sh did not print the ready banner" >&2
  exit 1
fi

python - <<'PY'
import json
import urllib.request

payload = {
    'model': 'Qwen/Qwen2.5-0.5B-Instruct',
    'messages': [{'role': 'user', 'content': 'Say hi in one sentence.'}],
    'max_tokens': 32,
    'temperature': 0,
}
request = urllib.request.Request(
    'http://127.0.0.1:10080/v1/chat/completions',
    data=json.dumps(payload).encode(),
    headers={'Content-Type': 'application/json'},
    method='POST',
)
with urllib.request.urlopen(request, timeout=120) as response:
    body = json.load(response)
content = (body['choices'][0]['message']['content'] or '').strip()
if body.get('model') != 'Qwen/Qwen2.5-0.5B-Instruct':
    raise SystemExit(f'unexpected model {body.get("model")!r}')
if not content:
    raise SystemExit('empty completion content')
print('completion_ok', body['usage']['completion_tokens'])
PY

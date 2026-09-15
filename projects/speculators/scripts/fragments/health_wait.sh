echo "Waiting for vLLM server to be ready..."
VLLM_READY=0
for _i in $(seq 1 360); do
  if curl -sf "__HEALTH_URL__" > /dev/null 2>&1; then
    VLLM_READY=1
    break
  fi
  sleep 2
done
if [ "$VLLM_READY" != "1" ]; then
  echo "vLLM server failed to come up within 12 min" >&2
  exit 1
fi
echo "vLLM server ready."

#!/usr/bin/env bash
# Prepare the CI environment for xllm examples on NPU.
# $1 is the manifest profile. Unknown profiles fail before any install.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
ASCEND_MIRROR_PIP_INDEX=https://mirrors.huaweicloud.com/ascend/repos/pypi
ASCEND_VARIANT_PIP_INDEX=https://mirrors.huaweicloud.com/ascend/repos/pypi/variant
CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"
FALLBACK_PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

pip_ascend() {
  python -m pip install --extra-index-url "$ASCEND_PIP_INDEX" "$@"
}

pip_ascend_variant() {
  python -m pip install \
    --extra-index-url "$ASCEND_VARIANT_PIP_INDEX" \
    --extra-index-url "$ASCEND_MIRROR_PIP_INDEX" \
    "$@"
}

select_pip_index() {
  if python -c "
import os
import urllib.error
import urllib.request
try:
    urllib.request.urlopen(os.environ['CLUSTER_PIP_INDEX'], timeout=3)
except urllib.error.HTTPError:
    pass
" 2>/dev/null; then
    export PIP_INDEX_URL="$CLUSTER_PIP_INDEX"
    export PIP_TRUSTED_HOST="$CLUSTER_PIP_HOST"
  else
    export PIP_INDEX_URL="$FALLBACK_PIP_INDEX"
    unset PIP_TRUSTED_HOST
  fi
  echo "pip index: $PIP_INDEX_URL"
}

ensure_torch_stack() {
  # The CANN base image usually ships torch/torch_npu; verify the versions
  # match what xllm expects (2.9.0 / 2.9.0.post2) and reinstall if not.
  if python -c "
import torch, torch_npu
print('found torch', torch.__version__, 'torch_npu', torch_npu.__version__)
raise SystemExit(0 if torch.__version__.startswith('2.9.0') and torch_npu.__version__.startswith('2.9.0') else 1)
"; then
    echo "reusing image torch stack"
    return
  fi
  echo "installing torch==2.9.0 torch_npu==2.9.0.post2"
  pip_ascend torch==2.9.0 torch_npu==2.9.0.post2
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

# Resolve one example model into $2 (shell var + GITHUB_ENV) from the
# shared runner cache. The cache-seed workflow plants the seed assets
# from ModelScope into the HF hub-cache layout (refs/main = real HF
# sha, real files - see cache-seed/xllm/ms_seeds.yaml); the warm path
# is zero-network. Cold-cache self-heal (seed not dispatched yet, or a
# model added since): pull once with huggingface_hub - no local_dir,
# so it lands back in the shared hub-cache layout and every later run
# is a hit. If HF/hf-mirror is down too, fall back to ModelScope into
# the per-run workspace: discarded with the workspace, so the durable
# fix is dispatching the cache-seed workflow (projects=xllm).
# Replaces the legacy /data/ci-cache/modelscope bind-mount +
# in-setup modelscope download (the engine mounts no project volumes).
resolve_model() {
  local hf_id="$1" var="$2"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  local resolved
  resolved=$(python3 - "$hf_id" <<'PY' | tee /dev/stderr | sed -n "s/^RESOLVED=//p" | tail -n 1
import os
import subprocess
import sys

model_id = sys.argv[1]
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
repo_dir = os.path.join(hf_home, "hub", "models--" + model_id.replace("/", "--"))
workspace_dir = os.path.join(
    os.environ["GITHUB_WORKSPACE"], "xllm_models", model_id.split("/")[-1]
)


def cached_snapshot():
    refs = os.path.join(repo_dir, "refs", "main")
    if not os.path.isfile(refs):
        return None
    with open(refs) as fh:
        sha = fh.read().strip()
    snap = os.path.join(repo_dir, "snapshots", sha)
    if os.path.isdir(snap) and os.listdir(snap):
        return snap
    return None


snapshot = cached_snapshot()
if snapshot is not None:
    print(f"{model_id}: shared-cache hit -> {snapshot}", flush=True)
else:
    print(f"{model_id}: not in the shared cache ({repo_dir}); filling it via HF",
          flush=True)
    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(model_id)
        print(f"filled shared cache: {snapshot}", flush=True)
    except Exception as e:
        print(f"HF fill failed ({type(e).__name__}: {e}); trying ModelScope",
              flush=True)
        snapshot = None

    if snapshot is None:
        try:
            from modelscope import snapshot_download as ms_snapshot
        except ImportError:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q",
                 "modelscope==1.37.0"],
                check=True,
            )
            from modelscope import snapshot_download as ms_snapshot
        os.makedirs(workspace_dir, exist_ok=True)
        ms_snapshot(model_id, local_dir=workspace_dir)
        snapshot = workspace_dir
        print(f"downloaded to ephemeral {snapshot}", flush=True)

print(f"RESOLVED={snapshot}", flush=True)
PY
)
  [[ -n "$resolved" ]] || { echo "resolve_model($hf_id) printed no path" >&2; exit 2; }
  # Shell var for same-script consumers; GITHUB_ENV for the run-example
  # step's overlay_args expansion (${XLLM_MODEL_PATH} in the manifest).
  export "$var"="$resolved"
  echo "$var=$resolved" >> "$GITHUB_ENV"
}

ensure_vlm_images() {
  # generate_vlm.py reads ./images/3.jpg and ./images/4.jpg relative to the
  # run cwd (/tmp); upstream ships no images, so synthesize tiny JPEGs.
  python -m pip install -q pillow 2>/dev/null || true
  python - <<'PY'
import os
from PIL import Image
os.makedirs('/tmp/images', exist_ok=True)
for name, color in (('3.jpg', (196, 74, 60)), ('4.jpg', (58, 96, 180))):
    path = os.path.join('/tmp/images', name)
    if os.path.exists(path):
        continue
    Image.new('RGB', (256, 256), color).save(path, 'JPEG')
print('vlm fixture images ready at /tmp/images')
PY
}

# Default profile: resolve the LLM example model (xllm is pre-installed in
# the official release image, no source build needed).
setup_default() {
  echo "profile=default: resolving example model"
  resolve_model Qwen/Qwen2-7B-Instruct XLLM_MODEL_PATH
}

# VLM profile: vision model + image fixtures for generate_vlm.py.
# Currently dormant (generate_vlm.py sits in unsupported until the
# upstream Qwen2VLPromptProcessor crash is fixed); kept so the move-back
# is a manifest-only change.
setup_vlm() {
  echo "profile=vlm: resolving VLM model and image fixtures"
  resolve_model Qwen/Qwen2.5-VL-7B-Instruct XLLM_VLM_MODEL_PATH
  ensure_vlm_images
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

# Later steps are a new shell; do not rely on a previous workflow step.
source /usr/local/Ascend/ascend-toolkit/set_env.sh

select_pip_index
python -m pip install -U pip
ensure_torch_stack

# Verify NPU is available
python -c "import torch, torch_npu; print('torch:', torch.__version__, 'torch_npu:', torch_npu.__version__, 'npu_count:', torch.npu.device_count())"
npu-smi info

# Verify xllm from the official release image (pre-installed, no source build)
python -c "import xllm; print('xllm import ok:', xllm.__file__)"

"setup_${PROFILE}"

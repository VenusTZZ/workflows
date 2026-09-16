#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"
# A profile selects the smallest dependency set needed by one example family;
# it is not a transformers feature flag. Keep unknown profiles failing before
# any package is installed so a manifest typo cannot run a partial setup.
case "$PROFILE" in
  generation)
    DEPS=(accelerate)
    ;;
  glue)
    DEPS=(accelerate datasets evaluate scikit-learn)
    ;;
  small-training)
    DEPS=(accelerate datasets evaluate seqeval)
    ;;
  *)
    echo "unknown profile: $PROFILE (supported: generation glue small-training)" >&2
    exit 1
    ;;
esac

: "${TARGET_ROOT:?TARGET_ROOT is required}"
: "${GITHUB_ENV:?GITHUB_ENV is required}"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_TRUSTED_HOST="repo.huaweicloud.com"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python -m pip install -U pip
# The Ascend image normally contains a compatible torch/torch_npu pair. Reuse
# it when possible because these wheels are large; install only on a missing
# or unusable image stack.
if python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.device_count())"; then
  echo "reusing image torch stack"
else
  python -m pip install --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi \
    torch==2.9.0 torch_npu==2.9.0.post2
fi
# Install the current target checkout. Its dependencies (regex, tokenizers,
# huggingface-hub, safetensors, ...) are resolved by pip from the index; this is
# safe for the NPU torch stack because transformers does not depend on torch_npu
# and its torch requirement is already satisfied by the image build.
python -m pip install -e "$TARGET_ROOT"
python -m pip install "${DEPS[@]}"

# Pre-download example model weights from ModelScope (China-reachable) so the
# examples load them from a local path instead of the blocked HuggingFace CDN.
# The returned local snapshot dirs are exported as env vars for the example
# (overlay_args in examples_manifest.yaml reference them via ${VAR}).
# Pinned: modelscope>=1.38 splits the hub code into modelscope-hub, and the
# fresh 1.40.1 wheel's loose ">=0.4.2" floor breaks import when the mirror
# lags on hub 0.4.3. 1.37.0 is the last pre-split line.
python -m pip install "modelscope==1.37.0"
python - <<'PY'
import os
from modelscope import snapshot_download

MODEL_CACHE = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))
mapping = {
    "DISTILBERT_PATH": "distilbert/distilbert-base-uncased",
    "TINYGPT2_PATH": "sshleifer/tiny-gpt2",
}
for env_name, model_id in mapping.items():
    local = snapshot_download(model_id, cache_dir=MODEL_CACHE)
    with open(os.environ["GITHUB_ENV"], "a") as fh:
        fh.write(f"{env_name}={local}\n")
PY

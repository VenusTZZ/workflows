#!/usr/bin/env bash
# Prepare the CI environment for one supported example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# DeepSpeed source is installed from the main-repo checkout (TARGET_ROOT in
# split mode; DEEPSPEED_SOURCE_ROOT overrides). Examples run from the
# DeepSpeedExamples checkout (EXAMPLES_ROOT).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
FALLBACK_PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"

pip_ascend() {
  python -m pip install --extra-index-url "$ASCEND_PIP_INDEX" "$@"
}

select_pip_index() {
  if python -c "
import urllib.error
import urllib.request
try:
    urllib.request.urlopen('${CLUSTER_PIP_INDEX}', timeout=3)
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

# Install DeepSpeed from the main-repo checkout (TARGET_ROOT).
install_deepspeed_source() {
  local src="${DEEPSPEED_SOURCE_ROOT:-$TARGET_ROOT}"
  echo "installing DeepSpeed from source at $src"
  python -m pip install -e "$src"
  python -c "
import deepspeed
print('DeepSpeed version:', deepspeed.__version__)
"
  ds_report 2>&1 | grep -i 'npu' || {
    echo 'WARNING: ds_report did not list npu accelerator'
  }
  echo "installing MPI runtime for deepspeed.initialize distributed discovery"
  apt-get update && apt-get install -y libopenmpi-dev numactl
  python -m pip install mpi4py
}

# Download models from ModelScope and expose their local paths to the run step
# via GITHUB_ENV (same pattern as projects/trl). overlay_args reference
# ${OPT_125M_PATH} so the example gets a concrete local dir.
ms_download_models() {
  # modelscope>=1.38 splits hub code into modelscope-hub; pin the last pre-split
  # release because the runner mirror may only expose an older hub for the latest wheel.
  python -m pip install -q "modelscope==1.37.0"
  TQDM_MININTERVAL="${TQDM_MININTERVAL:-15}" python - "$@" <<'PY'
import os, sys
from modelscope import snapshot_download
MODEL_CACHE = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))
for pair in sys.argv[1:]:
    env_name, model_id = pair.split("=", 1)
    local = snapshot_download(model_id, cache_dir=MODEL_CACHE)
    with open(os.environ["GITHUB_ENV"], "a") as fh:
        fh.write(f"{env_name}={local}\n")
PY
}

patch_hello_wikitext() {
  local target="$EXAMPLES_ROOT/training/HelloDeepSpeed/train_bert_ds.py"
  python3 - "$target" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text(encoding='utf-8')
old = 'datasets.load_dataset("wikitext",'
new = 'datasets.load_dataset("Salesforce/wikitext",'
if old not in text:
    raise SystemExit(f'wikitext load_dataset not found in {path}')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
print(f'patched {path}: wikitext -> Salesforce/wikitext')
PY
}

# Copy a fixture into the DeepSpeed-Chat data/ dir that local/jsonfile reads.
# $1 = fixture name under $FIXTURE_DIR.
plant_chat_fixture() {
  local fixture="$1"
  local chat_data="$EXAMPLES_ROOT/applications/DeepSpeed-Chat/data"
  mkdir -p "$chat_data"
  cp "$FIXTURE_DIR/$fixture" "$chat_data/train.json"
  cp "$FIXTURE_DIR/$fixture" "$chat_data/eval.json"
  echo "planted chat fixture $fixture -> $chat_data/{train,eval}.json"
}

setup_deepspeed() {
  install_deepspeed_source
  echo "installing HelloDeepSpeed dependencies"
  PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple" \
    python -m pip install "tokenizers>=0.22.0,<0.23" datasets transformers fire loguru "sh==1.14.2" tqdm pytz tensorboard torchvision
  patch_hello_wikitext
}

# Shared DeepSpeed-Chat setup: DS source + transformers + opt-125m + fixture.
setup_ds_chat() {
  local fixture="$1"
  install_deepspeed_source
  python -m pip install modelscope transformers datasets accelerate
  # DeepSpeed-Chat steps import the top-level dschat package (sibling of the
  # step dirs); install it editable so 'from dschat.utils... import' resolves.
  # --no-deps: its setup.py pins deepspeed/torch/transformers which we already
  # provide (source install / image); only tensorboard is genuinely missing.
  python -m pip install --no-deps -e "$EXAMPLES_ROOT/applications/DeepSpeed-Chat"
  python -m pip install tensorboard sentencepiece
  ms_download_models "OPT_125M_PATH=facebook/opt-125m"
  plant_chat_fixture "$fixture"
}

setup_ds_chat_sft()  { setup_ds_chat ci_sft_8.json; }
setup_ds_chat_rw()   { setup_ds_chat ci_rw_8.json; }
setup_ds_chat_dpo()  { setup_ds_chat ci_dpo_8.json; }
setup_ds_chat_rlhf() { setup_ds_chat ci_rlhf_8.json; }

setup_ds_infer() {
  install_deepspeed_source
  python -m pip install modelscope transformers accelerate
  ms_download_models "OPT_125M_PATH=facebook/opt-125m"
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
EXAMPLES_ROOT="${EXAMPLES_ROOT:-$TARGET_ROOT}"
FIXTURE_DIR="${FIXTURE_DIR:-$GITHUB_WORKSPACE/workflows/projects/deepspeed/fixtures}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

select_pip_index
python -m pip install -U pip setuptools wheel
ensure_torch_stack

"setup_${PROFILE}"

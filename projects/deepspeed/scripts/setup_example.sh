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

# Download a model from ModelScope and plant it into the HF hub cache so the
# example's hardcoded HF id resolves offline. $1 = HF id, $2 = ModelScope id.
modelscope_plant() {
  local hf_id="$1" ms_id="$2"
  python - "$hf_id" "$ms_id" <<'PY'
import os, shutil, sys
from modelscope import snapshot_download
hf_id, ms_id = sys.argv[1], sys.argv[2]
src = snapshot_download(ms_id)
dst = os.path.join(os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')),
                   'hub', 'models--' + hf_id.replace('/', '--'))
os.makedirs(dst, exist_ok=True)
print(f'plant {ms_id} -> {dst}')
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
    python -m pip install "tokenizers>=0.22.0,<0.23" datasets transformers fire loguru "sh==1.14.2" tqdm pytz tensorboard
  patch_hello_wikitext
}

# Shared DeepSpeed-Chat setup: DS source + transformers + opt-125m + fixture.
setup_ds_chat() {
  local fixture="$1"
  install_deepspeed_source
  python -m pip install modelscope transformers datasets accelerate
  modelscope_plant facebook/opt-125m facebook/opt-125m
  plant_chat_fixture "$fixture"
}

setup_ds_chat_sft()  { setup_ds_chat ci_sft_8.json; }
setup_ds_chat_rw()   { setup_ds_chat ci_rw_8.json; }
setup_ds_chat_dpo()  { setup_ds_chat ci_dpo_8.json; }
setup_ds_chat_rlhf() { setup_ds_chat ci_rlhf_8.json; }

setup_ds_infer() {
  install_deepspeed_source
  python -m pip install modelscope transformers accelerate
  modelscope_plant facebook/opt-125m facebook/opt-125m
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

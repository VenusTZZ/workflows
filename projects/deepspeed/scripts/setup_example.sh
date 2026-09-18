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
# via GITHUB_ENV (same pattern as projects/trl). overlay_args reference these
# variables so examples receive concrete local directories.
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

# Redirect the renamed wikitext dataset at interpreter startup without editing
# the DeepSpeedExamples checkout. The shim is scoped to this CI job through
# GITHUB_ENV and leaves every other datasets.load_dataset call unchanged.
install_hello_dataset_shim() {
  local shim_dir="$GITHUB_WORKSPACE/.ci/deepspeed-sitecustomize"
  mkdir -p "$shim_dir"
  cat > "$shim_dir/sitecustomize.py" <<'PY'
import datasets as _datasets

_original_load_dataset = _datasets.load_dataset


def _patched_load_dataset(path, *args, **kwargs):
    if path == "wikitext":
        path = "Salesforce/wikitext"
    return _original_load_dataset(path, *args, **kwargs)


_datasets.load_dataset = _patched_load_dataset
PY
  echo "PYTHONPATH=$shim_dir${PYTHONPATH:+:$PYTHONPATH}" >> "$GITHUB_ENV"
  echo "installed wikitext runtime redirect in $shim_dir"
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

# DeepSpeed-Chat's setup.py uses find_packages(include=['dschat']), while the
# dschat root is a PEP 420 namespace package without __init__.py. pip therefore
# installs distribution metadata but does not make the source package
# importable when the launcher changes into a training/step* directory. Expose
# the upstream source root explicitly without editing the examples checkout.
install_chat_source_path() {
  local chat_root="$EXAMPLES_ROOT/applications/DeepSpeed-Chat"
  case ":${PYTHONPATH:-}:" in
    *":$chat_root:"*) ;;
    *) export PYTHONPATH="$chat_root${PYTHONPATH:+:$PYTHONPATH}" ;;
  esac
  echo "PYTHONPATH=$PYTHONPATH" >> "$GITHUB_ENV"
  python - <<'PY'
import dschat

paths = list(dschat.__path__)
if not paths:
    raise SystemExit("dschat namespace package has no source path")
print("runtime dschat namespace:", paths)
PY
}

setup_deepspeed() {
  install_deepspeed_source
  echo "installing dependencies from the HelloDeepSpeed and CIFAR requirements baselines"
  PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple" \
    python -m pip install "tokenizers>=0.22.0,<0.23" "transformers<5" datasets \
      fire loguru "sh==1.14.2" tqdm pytz tensorboard \
      "torchvision==0.24.0" "pillow>=7.1.0" matplotlib
  install_hello_dataset_shim
}

# Pre-stage CIFAR-10 from a pinned ModelScope dataset revision. Both the
# single-card and MoE examples use torchvision with root=./data and
# download=True; a complete local tree makes torchvision skip its unreachable
# Toronto download URL. Verify both the archive SHA-256 and torchvision's
# official per-file MD5 list.
plant_cifar10() {
  local data_dir="$EXAMPLES_ROOT/training/cifar/data"
  CIFAR10_DATA_DIR="$data_dir" python - <<'PY'
import hashlib
import os
from pathlib import Path
import time
import urllib.request
import zipfile

from torchvision.datasets import CIFAR10

URL = (
    "https://modelscope.cn/api/v1/datasets/studyhard1/"
    "cifar10-dataset/repo?Revision="
    "9231e736fd8d53f7158165a07d801429b4414993&"
    "FilePath=cifar-10-batches-py.zip"
)
EXPECTED_SHA256 = "4f287e6733e987d5c1ab2af557413cf0f5bc78f293765d87dc5633788246e5d7"
data_dir = Path(os.environ["CIFAR10_DATA_DIR"]).resolve()
archive = data_dir / "cifar-10-batches-py.zip"
partial = archive.with_suffix(".zip.part")
data_dir.mkdir(parents=True, exist_ok=True)


def complete() -> bool:
    dataset = CIFAR10.__new__(CIFAR10)
    dataset.root = str(data_dir)
    return dataset._check_integrity()


if complete():
    print("CIFAR-10 fixture already passes torchvision integrity checks:", data_dir)
    raise SystemExit(0)

for attempt in range(1, 4):
    digest = hashlib.sha256()
    try:
        request = urllib.request.Request(URL, headers={"User-Agent": "cosdt-ci/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != EXPECTED_SHA256:
            raise RuntimeError(
                f"CIFAR-10 archive sha256 mismatch: expected {EXPECTED_SHA256}, got {actual}"
            )
        partial.replace(archive)
        break
    except Exception:
        partial.unlink(missing_ok=True)
        if attempt == 3:
            raise
        print(f"CIFAR-10 download attempt {attempt}/3 failed; retrying", flush=True)
        time.sleep(2 * attempt)

with zipfile.ZipFile(archive) as bundle:
    for member in bundle.infolist():
        destination = (data_dir / member.filename).resolve()
        if destination != data_dir and data_dir not in destination.parents:
            raise RuntimeError(f"unsafe CIFAR-10 archive member: {member.filename}")
    bundle.extractall(data_dir)
archive.unlink()

if not complete():
    raise SystemExit("extracted CIFAR-10 data failed torchvision integrity checks")
print("planted verified CIFAR-10 fixture in", data_dir)
PY
}

setup_ds_cifar() {
  setup_deepspeed
  plant_cifar10
}

# Shared DeepSpeed-Chat setup: DS source + transformers + opt-125m + fixture.
setup_ds_chat() {
  local fixture="$1"
  install_deepspeed_source
  # Install the declared dependencies explicitly so pip cannot replace the NPU
  # torch stack or the source DeepSpeed under test. The upstream editable
  # package is intentionally not used: its find_packages() call produces an
  # empty package for the PEP 420 dschat namespace.
  python -m pip install "transformers>=4.31.0,<5,!=4.33.2" \
    "datasets>=2.8.0" "accelerate>=0.15.0" "sentencepiece>=0.1.97" \
    "protobuf==3.20.3" tensorboard
  install_chat_source_path
  ms_download_models "OPT_125M_PATH=facebook/opt-125m"
  plant_chat_fixture "$fixture"
}

setup_ds_chat_sft()  { setup_ds_chat ci_sft_8.json; }
setup_ds_chat_rw()   { setup_ds_chat ci_rw_8.json; }
setup_ds_chat_dpo()  { setup_ds_chat ci_dpo_8.json; }
setup_ds_chat_rlhf() { setup_ds_chat ci_rlhf_8.json; }

setup_ds_infer() {
  install_deepspeed_source
  python -m pip install "transformers<5" accelerate
  ms_download_models "OPT_125M_PATH=facebook/opt-125m"
}

setup_ds_autotp_equivalence() {
  install_deepspeed_source
  # Qwen3 support is present in current 4.x transformers; keep the upper bound
  # below the next major release to avoid unreviewed API changes.
  python -m pip install "transformers>=4.51.0,<5" safetensors
  ms_download_models "QWEN3_06B_PATH=Qwen/Qwen3-0.6B"
}

verify_installed_runtime() {
  python - <<'PY'
import os
from pathlib import Path

import deepspeed
import torch
import torch_npu

source_root = Path(os.environ["TARGET_ROOT"]).resolve()
deepspeed_file = Path(deepspeed.__file__).resolve()
print("runtime torch:", torch.__version__)
print("runtime torch_npu:", torch_npu.__version__)
print("runtime deepspeed:", deepspeed.__version__, deepspeed_file)
try:
    deepspeed_file.relative_to(source_root)
except ValueError as exc:
    raise SystemExit(
        f"DeepSpeed was not imported from target source {source_root}: "
        f"{deepspeed_file}") from exc
PY
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
verify_installed_runtime

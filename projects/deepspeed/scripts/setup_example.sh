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
  # GITHUB_ENV is applied in the next step; expose only requested model names
  # in this setup process so fixture checks can use the downloaded tokenizer.
  local pair env_name local_path
  for pair in "$@"; do
    env_name="${pair%%=*}"
    [[ "$env_name" =~ ^[A-Z_][A-Z0-9_]*$ ]] || {
      echo "invalid model environment name: $env_name" >&2
      return 1
    }
    local_path="$(awk -v key="$env_name" '
      index($0, key "=") == 1 { value = substr($0, length(key) + 2) }
      END { print value }
    ' "$GITHUB_ENV")"
    [[ -d "$local_path" ]] || {
      echo "downloaded model directory missing for $env_name: $local_path" >&2
      return 1
    }
    export "$env_name=$local_path"
  done
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
  # CIFAR profiles reuse these packages but never load wikitext.
  if [[ "$PROFILE" == deepspeed ]]; then
    install_hello_dataset_shim
  fi
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

# Evaluation entries use the same upstream Chat baseline with protected runtime
# dependencies. Leave the already-validated training profiles unchanged.
setup_ds_chat_eval_dependencies() {
  install_deepspeed_source
  install_example_dependencies "transformers>=4.31.0,<5,!=4.33.2" \
    "datasets>=2.8.0" "accelerate>=0.15.0" "sentencepiece>=0.1.97" \
    "protobuf==3.20.3" tensorboard
  install_chat_source_path
  ms_download_models "OPT_125M_PATH=facebook/opt-125m"
  python - <<'PY'
import importlib.util
import os
from pathlib import Path
import torch_npu
from transformers import AutoTokenizer

root = Path(os.environ["EXAMPLES_ROOT"]) / "applications/DeepSpeed-Chat/training"
for step, filename in (
    ("step1_supervised_finetuning", "prompt_eval.py"),
    ("step2_reward_model_finetuning", "rw_eval.py"),
):
    spec = importlib.util.spec_from_file_location(f"ds_ci_{filename[:-3]}", root / step / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
tokenizer = AutoTokenizer.from_pretrained(os.environ["OPT_125M_PATH"])
print("Chat evaluation local tokenizer:", type(tokenizer).__name__)
PY
}

setup_ds_chat_prompt_eval() {
  setup_ds_chat_eval_dependencies
  # The runner trains a genuine small SFT checkpoint before prompt comparison.
  plant_chat_fixture ci_sft_8.json
}

setup_ds_chat_reward_eval() {
  setup_ds_chat_eval_dependencies
  # rw_eval's default create_critic_model() constructs a fresh value head; this
  # entry is an honest reward-forward smoke, not trained checkpoint restoration.
}

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

# Install only each example's non-runtime requirements. Constraints also stop
# transitive dependencies from replacing the image NPU stack or source DS.
install_example_dependencies() {
  local constraint_dir="$GITHUB_WORKSPACE/.ci/deepspeed-dependencies"
  local constraint_file="$constraint_dir/protected-stack.txt"
  mkdir -p "$constraint_dir"
  python - "$constraint_file" <<'PY'
from importlib.metadata import version
from pathlib import Path
import sys

protected = ("torch", "torch-npu", "deepspeed")
constraints = "".join(f"{name}=={version(name)}\n" for name in protected)
Path(sys.argv[1]).write_text(constraints)
print("preserving installed runtime dependencies:\n" + constraints)
PY
  python -m pip install --constraint "$constraint_file" "$@"
}

# New examples accept explicit fixture paths, so stage their data outside the
# examples checkout instead of changing any upstream source or dataset loader.
plant_ci_fixture() {
  local fixture="$1"
  local env_name="$2"
  local ci_dir="$GITHUB_WORKSPACE/.ci/deepspeed-fixtures"
  mkdir -p "$ci_dir"
  cp "$FIXTURE_DIR/$fixture" "$ci_dir/$fixture"
  export "$env_name=$ci_dir/$fixture"
  echo "$env_name=$ci_dir/$fixture" >> "$GITHUB_ENV"
  echo "staged $fixture -> $ci_dir/$fixture ($env_name)"
}

setup_ds_hf_autotp() {
  install_deepspeed_source
  # train.py imports Trainer directly; the extra evaluation/OpenAI packages in
  # the upstream requirements belong to utils.py, which this entry never imports.
  install_example_dependencies "transformers>=4.51.0,<5" "accelerate>=1.10.1,<2" \
    sentencepiece psutil numpy safetensors
  ms_download_models "OPT_125M_PATH=facebook/opt-125m"
  plant_ci_fixture ci_alpaca_16.json ALPACA_CI_PATH
  python - <<'PY'
import accelerate
import importlib.util
import os
from pathlib import Path
import psutil
import sys
import sentencepiece
import torch_npu
import transformers
from transformers import AutoTokenizer, Trainer

entry = Path(os.environ["EXAMPLES_ROOT"]) / "training/tensor_parallel/hf_integration/train.py"
spec = importlib.util.spec_from_file_location("ds_ci_hf_autotp", entry)
upstream = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = upstream
spec.loader.exec_module(upstream)
tokenizer = AutoTokenizer.from_pretrained(os.environ["OPT_125M_PATH"],
                                         model_max_length=128, padding_side="right", use_fast=False)
special_tokens = {}
for kind in ("pad", "eos", "bos", "unk"):
    if getattr(tokenizer, f"{kind}_token") is None:
        special_tokens[f"{kind}_token"] = getattr(upstream, f"DEFAULT_{kind.upper()}_TOKEN")
tokenizer.add_special_tokens(special_tokens)
# SupervisedDataset writes a fixed dataset_dict.pkl in cwd. Precompute it once
# before eight ranks start, in the output directory used by the project runner.
# The engine supplies CI_OUTPUT_DIR only to run, whose default is workspace/output.
output_root = Path(os.environ.get("CI_OUTPUT_DIR", str(Path(os.environ["GITHUB_WORKSPACE"]) / "output")))
work_dir = output_root / "hf-autotp-work"
work_dir.mkdir(parents=True, exist_ok=True)
cache = work_dir / "dataset_dict.pkl"
# A runner may retain output from a previous job. This file is our generated
# tokenization cache; rebuild it from this run's fixture before workers read it.
cache.unlink(missing_ok=True)
previous_cwd = Path.cwd()
try:
    os.chdir(work_dir)
    dataset = upstream.SupervisedDataset(os.environ["ALPACA_CI_PATH"], tokenizer)
finally:
    os.chdir(previous_cwd)
counts = [(labels != upstream.IGNORE_INDEX).sum().item() for labels in dataset.labels]
if len(dataset) != 16 or any(count == 0 for count in counts):
    raise SystemExit(f"HF AutoTP fixture lost training labels at length 128: {counts}")
if not cache.is_file() or cache.stat().st_size == 0:
    raise SystemExit(f"HF AutoTP pre-tokenized cache missing: {cache}")
print("HF AutoTP shared pre-tokenized cache:", cache)
print("HF AutoTP non-masked target tokens per fixture row:", counts)

print("HF AutoTP dependencies:", "transformers", transformers.__version__,
      "accelerate", accelerate.__version__, "psutil", psutil.__version__)
PY
}

setup_ds_hf_bench_length() {
  install_deepspeed_source
  # The sibling utils.py imports the legacy OpenAIObject API at module import.
  # This dependency enables ordinary imports only; this recipe makes no API call.
  install_example_dependencies "transformers>=4.51.0,<5" "accelerate>=1.10.1,<2" \
    "openai==0.28.1" sentencepiece psutil numpy safetensors
  ms_download_models "OPT_125M_PATH=facebook/opt-125m"
  plant_ci_fixture ci_alpaca_16.json ALPACA_CI_PATH
  python - <<'PY'
import accelerate
import importlib.util
import os
from pathlib import Path
import sys

import numpy
import openai
from openai import openai_object
import torch_npu
import transformers
from transformers import AutoTokenizer

entry = Path(os.environ["EXAMPLES_ROOT"]) / "training/tensor_parallel/hf_integration/train_bench_length.py"
sys.path.insert(0, str(entry.parent))
spec = importlib.util.spec_from_file_location("ds_ci_hf_bench_length", entry)
upstream = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = upstream
spec.loader.exec_module(upstream)
tokenizer = AutoTokenizer.from_pretrained(os.environ["OPT_125M_PATH"],
                                         model_max_length=128, padding_side="right", use_fast=False)
special_tokens = {}
for kind in ("pad", "eos", "bos", "unk"):
    if getattr(tokenizer, f"{kind}_token") is None:
        special_tokens[f"{kind}_token"] = getattr(upstream, f"DEFAULT_{kind.upper()}_TOKEN")
tokenizer.add_special_tokens(special_tokens)
output_root = Path(os.environ.get("CI_OUTPUT_DIR", str(Path(os.environ["GITHUB_WORKSPACE"]) / "output")))
work_dir = output_root / "hf-bench-length-work"
work_dir.mkdir(parents=True, exist_ok=True)
cache = work_dir / "dataset_dict128.pkl"
# Our generated shared cache is rebuilt before the eight readers start.
cache.unlink(missing_ok=True)
previous_cwd = Path.cwd()
try:
    os.chdir(work_dir)
    dataset = upstream.SupervisedDataset(os.environ["ALPACA_CI_PATH"], tokenizer)
finally:
    os.chdir(previous_cwd)
counts = [(labels != upstream.IGNORE_INDEX).sum().item() for labels in dataset.labels]
if len(dataset) != 16 or any(count == 0 for count in counts):
    raise SystemExit(f"fixed-length HF fixture lost training labels: {counts}")
for ids, labels in zip(dataset.input_ids, dataset.labels):
    if ids.numel() != 128 or labels.numel() != 128:
        raise SystemExit("HF benchmark cache does not contain fixed 128-token rows")
    if not bool((labels[ids == tokenizer.pad_token_id] == upstream.IGNORE_INDEX).all()):
        raise SystemExit("HF benchmark cache contains unmasked padding targets")
if not cache.is_file() or cache.stat().st_size == 0:
    raise SystemExit(f"fixed-length HF pre-tokenized cache missing: {cache}")
print("HF benchmark shared pre-tokenized cache:", cache)
print("HF benchmark non-masked target tokens:", counts)
print("HF benchmark dependencies:", "transformers", transformers.__version__,
      "accelerate", accelerate.__version__, "openai", openai.__version__, "numpy", numpy.__version__)
PY
}

setup_ds_superoffload() {
  install_deepspeed_source
  # finetune_zero3 imports wandb unconditionally, but the runner never enables
  # its network logging. CPU Adam is compiled up front rather than on every rank.
  install_example_dependencies "transformers>=4.56.1,<5" "accelerate>=1.10.1,<2" \
    "datasets>=4,<5" "numpy>=1.21.0" packaging psutil pyarrow \
    sentencepiece safetensors wandb ninja
  apt-get install -y build-essential
  ms_download_models "OPT_125M_PATH=facebook/opt-125m"
  plant_ci_fixture ci_alpaca_16.json ALPACA_CI_PATH
  export ALPACA_DATASET_DIR="$GITHUB_WORKSPACE/.ci/deepspeed-fixtures/alpaca-parquet"
  mkdir -p "$ALPACA_DATASET_DIR"
  echo "ALPACA_DATASET_DIR=$ALPACA_DATASET_DIR" >> "$GITHUB_ENV"
  python - <<'PY'
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys

import datasets
import torch_npu
import transformers
import wandb
from datasets import Dataset, load_dataset
from deepspeed.ops.op_builder import CPUAdamBuilder

rows = json.loads(Path(os.environ["ALPACA_CI_PATH"]).read_text())
if len(rows) != 16 or any(set(row) != {"instruction", "input", "output"} for row in rows):
    raise SystemExit("SuperOffload fixture must contain 16 Alpaca instruction/input/output rows")
dataset_dir = Path(os.environ["ALPACA_DATASET_DIR"])
Dataset.from_list(rows).to_parquet(str(dataset_dir / "train.parquet"))
# The upstream entry passes a directory directly to load_dataset(), not the
# JSON loader or load_from_disk(). Validate precisely that public loading path.
dataset = load_dataset(str(dataset_dir))
if len(dataset["train"]) != 16 or set(dataset["train"].column_names) != {"instruction", "input", "output"}:
    raise SystemExit("local parquet directory did not resolve to the 16-row train split")
entry = Path(os.environ["EXAMPLES_ROOT"]) / "training/DeepSpeed-SuperOffload/finetune_zero3.py"
spec = importlib.util.spec_from_file_location("ds_ci_superoffload", entry)
upstream = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = upstream
spec.loader.exec_module(upstream)
tokenizer = upstream.load_tokenizer(os.environ["OPT_125M_PATH"], logging.getLogger("ci-superoffload"))
for row in dataset["train"]:
    batch = upstream.preprocess_alpaca_example(row, tokenizer, max_length=128)
    if len(batch["input_ids"]) != 128 or len(batch["labels"]) != 128:
        raise SystemExit("SuperOffload fixture does not produce 128-token training rows")
    if sum(batch["attention_mask"]) < 2 or batch["labels"] != batch["input_ids"]:
        raise SystemExit("SuperOffload fixture lost the upstream causal language-model targets")
builder = CPUAdamBuilder()
print("checking SuperOffload CPU Adam extension:", type(builder).__name__)
builder.load(verbose=True)
print("SuperOffload CPU Adam extension loaded successfully")
print("SuperOffload local parquet train split:", len(dataset["train"]), dataset_dir)
print("SuperOffload upstream optimizer LR:", upstream.DEFAULT_OPTIMIZER_LR)
print("SuperOffload dependencies:", "datasets", datasets.__version__,
      "transformers", transformers.__version__, "wandb", wandb.__version__)
PY
}

setup_ds_variable_batch() {
  install_deepspeed_source
  # DeepSpeed's DataAnalyzer/data-sampling path uses these packages. The
  # example itself constructs its small model and synthetic dataset locally.
  install_example_dependencies numpy pandas
  python - <<'PY'
import numpy
import pandas
import torch_npu
from deepspeed.runtime.data_pipeline.data_sampling.variable_batch_size_and_lr import (
    get_dataloader_and_lr_scheduler_for_variable_batch_size_deepspeed,
)

print("dynamic batch dependencies:", "numpy", numpy.__version__,
      "pandas", pandas.__version__)
PY
}

setup_ds_zenflow() {
  install_deepspeed_source
  # benchmark/requirements.txt without its torch/deepspeed entries. CPU Adam
  # is imported here; its actual NPU/offload execution is checked by the run.
  install_example_dependencies "datasets>=2.14.1" "transformers>=4.37.2,<5" \
    "numpy>=1.21.0" tabulate pandas ninja
  echo "installing C++ build tools for upstream ZenFlow CPU Adam"
  apt-get install -y build-essential
  python - <<'PY'
import datasets
import numpy
import pandas
import tabulate
import torch_npu
import transformers
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.op_builder import CPUAdamBuilder

builder = CPUAdamBuilder()
print("checking ZenFlow CPU Adam extension:", type(builder).__name__)
builder.load(verbose=True)
print("ZenFlow CPU Adam extension loaded successfully")

print("ZenFlow dependencies:", "datasets", datasets.__version__,
      "transformers", transformers.__version__, "numpy", numpy.__version__,
      "pandas", pandas.__version__, "tabulate", tabulate.__version__)
PY
}

setup_ds_rac_prune() {
  install_deepspeed_source
  # Local prompt JSONL takes the standard-library branch of the calibration
  # loader. No datasets, vLLM, evaluation backend or runtime patch is needed.
  install_example_dependencies "transformers>=4.45.0,<5" "accelerate>=0.30" \
    sentencepiece safetensors
  ms_download_models "OPT_125M_PATH=facebook/opt-125m"
  plant_ci_fixture ci_rac_8.jsonl RAC_CI_PATH
  python - <<'PY'
import os
from pathlib import Path
import sys

import accelerate
import torch_npu
import transformers

sys.path.insert(0, str(Path(os.environ["EXAMPLES_ROOT"]) /
                       "compression/reasoning_aware_compression"))
from rac import build_calibration_samples, sequential_prune
from rac.wanda import Wanda
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(os.environ["OPT_125M_PATH"])
samples = build_calibration_samples(
    "prompt", tokenizer, nsamples=4, seqlen=32, seed=0,
    dataset_name=os.environ["RAC_CI_PATH"], prompt_column="prompt",
    use_chat_template=False,
)
if len(samples) != 4 or any(sample.numel() != 32 for sample in samples):
    raise SystemExit("RAC fixture cannot fill the required 4 x 32 token windows")
print("RAC calibration fixture:", len(samples), "windows x", samples[0].numel(), "tokens")

print("Wanda pruning dependencies:", "transformers", transformers.__version__,
      "accelerate", accelerate.__version__)
PY
}

setup_ds_pin_memory() {
  install_deepspeed_source
  install_example_dependencies numpy ninja
  apt-get install -y build-essential
  # Adjust the shell so its Python/compiler subprocesses inherit the allowed
  # limit. The later run step independently makes the same bounded adjustment.
  local memlock_hard
  memlock_hard="$(ulimit -Hl)"
  ulimit -Sl "$memlock_hard"
  echo "pin-memory shell memlock soft/hard KiB: $(ulimit -Sl)/$memlock_hard"
  python - <<'PY'
import os
import resource

import torch
import torch_npu
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import CPUAdamBuilder, PinMemoryBuilder
from deepspeed.utils.pin_memory import get_native_pinned_memory

# Never increase the container's hard limit or require extra privileges.
soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
print("pin-memory RLIMIT_MEMLOCK soft/hard bytes:", soft, hard)
required = 4 * 1024 * 1024
if soft != resource.RLIM_INFINITY and soft < required:
    raise SystemExit(f"small native pin-memory CI recipes need at least {required} locked bytes; container limit is {soft}")
for builder in (CPUAdamBuilder(), PinMemoryBuilder()):
    print("preloading pin-memory benchmark extension:", type(builder).__name__)
    builder.load(verbose=True)
os.environ["DS_PIN_MEMORY_BACKEND"] = "native"
os.environ["DS_PIN_MEMORY_REGISTER_DEVICE"] = "0"
manager = get_native_pinned_memory()
raw = torch.arange(1024 * 1024 // 4, dtype=torch.float32)
locked = manager.pin(raw)
try:
    if not manager.is_pinned(locked) or not torch.equal(locked, raw):
        raise SystemExit("native page-locked allocation failed the pin/content preflight")
    accelerator = get_accelerator()
    device = locked.to(accelerator.current_device_name())
    accelerator.synchronize()
    if not torch.equal(device.cpu(), raw):
        raise SystemExit("native page-locked host/NPU copy lost tensor contents")
finally:
    manager.unpin(locked)
print("native mlock host allocation and NPU copy verified; this is not CUDA host registration")
PY
}

setup_ds_zenflow_finetune() {
  install_deepspeed_source
  install_example_dependencies "transformers==4.57.6" "datasets>=4,<5" \
    "accelerate>=1.10.1,<2" numpy pyarrow sentencepiece safetensors ninja
  apt-get install -y build-essential
  ms_download_models "OPT_125M_PATH=facebook/opt-125m"
  plant_ci_fixture ci_alpaca_16.json ALPACA_CI_PATH
  export ZENFLOW_WORK_DIR="${CI_OUTPUT_DIR:-$GITHUB_WORKSPACE/output}/zenflow-finetune-work"
  mkdir -p "$ZENFLOW_WORK_DIR/tatsu-lab/alpaca"
  echo "ZENFLOW_WORK_DIR=$ZENFLOW_WORK_DIR" >> "$GITHUB_ENV"
  echo "HF_DATASETS_OFFLINE=1" >> "$GITHUB_ENV"
  HF_DATASETS_OFFLINE=1 python - <<'PY'
import importlib.util
import json
import os
from pathlib import Path
import sys

import datasets
import torch_npu
import transformers
from datasets import Dataset, load_dataset
from deepspeed.ops.op_builder import CPUAdamBuilder
from transformers import AutoTokenizer, default_data_collator

rows = json.loads(Path(os.environ["ALPACA_CI_PATH"]).read_text())
if len(rows) != 16 or any(set(row) != {"instruction", "input", "output"} for row in rows):
    raise SystemExit("ZenFlow fixture must contain 16 original Alpaca-schema rows")
work = Path(os.environ["ZENFLOW_WORK_DIR"]).resolve()
Dataset.from_list(rows).to_parquet(str(work / "tatsu-lab/alpaca/train.parquet"))
os.chdir(work)
# Match the unmodified entry's hardcoded public API call precisely. The local
# directory is resolved natively; no dataset shim or source rewrite is used.
dataset = load_dataset("tatsu-lab/alpaca")
if len(dataset["train"]) != 16 or set(dataset["train"].column_names) != {"instruction", "input", "output"}:
    raise SystemExit("ZenFlow's exact hardcoded dataset name did not resolve to the local fixture")
entry = Path(os.environ["EXAMPLES_ROOT"]) / "training/DeepSpeed-ZenFlow/finetuning/finetune_llama.py"
spec = importlib.util.spec_from_file_location("ds_ci_zenflow_finetune", entry)
upstream = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = upstream
spec.loader.exec_module(upstream)
tokenizer = AutoTokenizer.from_pretrained(os.environ["OPT_125M_PATH"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Keep the original string columns just as the entry does. The official
# collator ignores strings, so the engine receives only model input tensors.
tokenized = dataset["train"].map(lambda row: upstream.preprocess_alpaca(row, tokenizer), batched=False)
for row in tokenized:
    if len(row["input_ids"]) != 512 or row["labels"] != row["input_ids"] or sum(row["attention_mask"]) < 2:
        raise SystemExit("ZenFlow's original preprocessing produced invalid causal training targets")
batch = default_data_collator([tokenized[0], tokenized[1]])
if set(batch) != {"input_ids", "attention_mask", "labels"} or batch["input_ids"].shape != (2, 512):
    raise SystemExit(f"ZenFlow's original collator did not drop auxiliary strings: {list(batch)}")
CPUAdamBuilder().load(verbose=True)
print("ZenFlow local offline name resolution:", work / "tatsu-lab/alpaca", "rows:", len(tokenized))
print("ZenFlow finetuning dependencies:", "transformers", transformers.__version__, "datasets", datasets.__version__)
PY
}

setup_ds_opsd() {
  install_deepspeed_source
  install_example_dependencies "transformers==4.57.6" "accelerate>=1.10.1,<2" \
    numpy sentencepiece safetensors ninja
  apt-get install -y build-essential
  ms_download_models "QWEN25_STUDENT_PATH=Qwen/Qwen2.5-0.5B-Instruct" \
    "QWEN25_TEACHER_PATH=Qwen/Qwen2.5-0.5B"
  plant_ci_fixture ci_rac_8.jsonl OPSD_CI_PATH
  python - <<'PY'
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch_npu
import transformers
from deepspeed.ops.op_builder import CPUAdamBuilder
from deepspeed.runtime.rollout import RolloutConfig, build_rollout
from transformers import AutoConfig, AutoTokenizer

student = AutoTokenizer.from_pretrained(os.environ["QWEN25_STUDENT_PATH"], padding_side="left")
teacher = AutoTokenizer.from_pretrained(os.environ["QWEN25_TEACHER_PATH"], padding_side="left")
# KL compares the same token index in each logit tensor. Validate the entire
# mapping, including added tokens, rather than merely comparing its size.
if student.get_vocab() != teacher.get_vocab() or student.get_added_vocab() != teacher.get_added_vocab():
    raise SystemExit("OPSD student/teacher token IDs differ; token-wise distillation is invalid")
student_config = AutoConfig.from_pretrained(os.environ["QWEN25_STUDENT_PATH"])
teacher_config = AutoConfig.from_pretrained(os.environ["QWEN25_TEACHER_PATH"])
if student_config.vocab_size != teacher_config.vocab_size or max(student.get_vocab().values()) >= student_config.vocab_size:
    raise SystemExit("OPSD model logit vocabularies do not align with their token IDs")
if not student.chat_template:
    raise SystemExit("OPSD instruction student must provide its own chat template")
entry = Path(os.environ["EXAMPLES_ROOT"]) / "training/opsd/main.py"
sys.path.insert(0, str(entry.parent))
spec = importlib.util.spec_from_file_location("ds_ci_opsd", entry)
upstream = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = upstream
spec.loader.exec_module(upstream)
dataset = upstream.PromptDataset(os.environ["OPSD_CI_PATH"], tokenizer=student,
                                max_prompt_length=128, prompt_field="prompt")
if len(dataset) != 8 or any(not isinstance(dataset[index], str) or not dataset[index] for index in range(len(dataset))):
    raise SystemExit("OPSD fixture must produce eight nonempty native chat-templated prompts")
collator = upstream.LeftPaddedPromptCollator(student, max_prompt_length=128)
batch = collator([dataset[0], dataset[1]])
if batch["prompt_ids"].shape[0] != 2 or not bool((batch["prompt_attention_mask"].sum(dim=1) > 0).all()):
    raise SystemExit("OPSD native prompt collator lost valid prompt tokens")
CPUAdamBuilder().load(verbose=True)
print("OPSD shared token mapping verified:", len(student), "model vocab size:", student_config.vocab_size)
print("OPSD student/teacher use different checkpoint weights; student's chat/EOS policy drives rollout")
print("OPSD rollout API available:", RolloutConfig.__name__, build_rollout.__name__)
print("OPSD local prompt fixture:", len(dataset), "transformers:", transformers.__version__)
PY
}

verify_installed_runtime() {
  python - "$PROFILE" <<'PY'
import os
from pathlib import Path
import sys

import deepspeed
import torch
import torch_npu
from deepspeed.accelerator import get_accelerator

source_root = Path(os.environ["TARGET_ROOT"]).resolve()
deepspeed_file = Path(deepspeed.__file__).resolve()
print("runtime torch:", torch.__version__)
print("runtime torch_npu:", torch_npu.__version__)
print("runtime deepspeed:", deepspeed.__version__, deepspeed_file)
if sys.argv[1] in {"ds_cifar", "ds_hf_autotp", "ds_variable_batch", "ds_zenflow", "ds_rac_prune",
                  "ds_chat_prompt_eval", "ds_chat_reward_eval", "ds_hf_bench_length", "ds_superoffload",
                  "ds_pin_memory", "ds_zenflow_finetune", "ds_opsd", "ds_finetune_demo",
                  "ds_sd_distil", "ds_opsd_decode", "ds_legacy_inference_bench", "ds_hf_ds_compare",
                  "ds_fill_mask_bert", "ds_fill_mask_electra", "ds_fill_mask_roberta",
                  "ds_t5_translation", "ds_hybrid_rollout", "ds_asr_ctc"}:
    accelerator = get_accelerator()._name
    available = torch.npu.is_available()
    print("runtime accelerator:", accelerator, "NPU available:", available)
    if accelerator != "npu" or not available:
        raise SystemExit(f"new DeepSpeed NPU example requires available npu accelerator, got {accelerator}")
try:
    deepspeed_file.relative_to(source_root)
except ValueError as exc:
    raise SystemExit(
        f"DeepSpeed was not imported from target source {source_root}: "
        f"{deepspeed_file}") from exc
PY
}

# Optional profiles use the same protected dependency and source-install helpers.
source "$(dirname "${BASH_SOURCE[0]}")/setup_inference_examples.sh"
source "$(dirname "${BASH_SOURCE[0]}")/setup_training_expansion.sh"

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

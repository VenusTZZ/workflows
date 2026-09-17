#!/usr/bin/env bash
# Prepare the CI environment for one supported torchtune example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# torchtune itself is installed from TARGET_ROOT (the release checkout
# under test), so the guarded tag is exactly the code that runs.
#
# Dependency line (mirror Quick-start-Ascend.md lines 85-88, 114-117, 168-172;
# quick-start is the canonical install path for torchtune on this image):
#   torch 2.11.0+cpu + torch_npu 2.11.0 + transformers 4.57.1
#   + omegaconf + tokenizers + safetensors + modelscope 1.37.0
# - torch 2.11.0 + torch_npu 2.11.0: pin explicit in ensure_torch_stack()
#   below (was: "reuse whatever image ships", which was 2.9.0+cpu). The +cpu
#   wheel comes from aliyun's mirror of pytorch.org/whl/cpu; the cluster pip
#   cache ships only CUDA torch wheels (Requires-Dist: cuda-toolkit), which
#   collide with constraints-npu.txt's cuda-toolkit<0 — see
#   torchtitan-cuda-torch-wheel-trap memory. torch 2.11.0 is also the floor
#   for `from torch.nn.functional import ScalingType` that torchao 0.18
#   needs, so upgrading makes the main HEAD path in the dynamic case below
#   work too (instead of being a known-broken documented limitation).
# - transformers 4.57.1: matches torchtune v0.6.x's LlamaModel / Qwen2
#   forward signatures; 5.x has renamed some attributes.
# - torchao pin: still decided dynamically inside setup_torchtune() because
#   the NF4Tensor import path in torchtune/modules/common_utils.py:19 has
#   moved three times (0.10-0.13 use torchao.dtypes.nf4tensor, 0.14-0.17
#   don't have the symbol, 0.18+ re-expose it under torchao.quantization).
#   v0.6.x release tag hits the first path (pin 0.13.0); main HEAD hits the
#   third (pin 0.18.0, now compatible thanks to the torch 2.11.0 upgrade
#   above). The case statement at the bottom of setup_torchtune() does
#   the probe.
# - non-editable torchtune install: PEP 660 editable makes
#   `torchtune.__file__` = None, which breaks `torchtune/_cli/cp.py:15`'s
#   `Path(torchtune.__file__).parent.parent`. Quick-start line 171 uses
#   `uv pip install .` for the same reason; we use `pip install .` here
#   (recipes are run as `python recipes/<x>.py`, so no editable hot-reload
#   is needed).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"
ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
ALIYUN_PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/

pip_ascend() {
  python -m pip install --extra-index-url "$ASCEND_PIP_INDEX" "$@"
}

select_pip_index() {
  # Runners live in mainland China: prefer the cluster pip cache, fall
  # back to the Aliyun mirror. The ascend index stays available via
  # PIP_EXTRA_INDEX_URL (set by the engine) for torch_npu wheels.
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
    export PIP_INDEX_URL="$ALIYUN_PIP_INDEX"
    unset PIP_TRUSTED_HOST
  fi
  echo "pip index: $PIP_INDEX_URL"
}

ensure_torch_stack() {
  # Mirror Quick-start-Ascend.md lines 85-88: pin torch==2.11.0+cpu +
  # torch_npu==2.11.0 explicitly so the setup is independent of what the
  # image happens to ship. Quick-start verified end-to-end on this same
  # CANN 9.1.0 image; reusing that line lets the dynamic torchao case
  # below work for both v0.6.x release (torchao 0.13.0) and main HEAD
  # (torchao 0.18.0, needs ScalingType from torch.nn.functional which is
  # torch 2.11+).
  #
  # The +cpu torch wheel is fetched from aliyun's mirror of
  # pytorch.org/whl/cpu via --find-links. Pure-Python deps of the +cpu
  # torch (filelock / typing-extensions / sympy / networkx / jinja2 /
  # fsspec / mpmath / markupsafe / setuptools) are resolved through
  # PIP_INDEX_URL (cluster cache, set by select_pip_index above). None
  # of those need CUDA, so constraints-npu.txt's cuda-toolkit<0 doesn't
  # fire. (Contrast with `pip install -i aliyun torch==X.Y.Z` from
  # torchtitan-cuda-torch-wheel-trap memory: that pulls the CUDA torch
  # wheel directly, whose Requires-Dist: cuda-toolkit collides with
  # constraints — --find-links to a +cpu-only directory sidesteps it.)
  echo "installing torch==2.11.0+cpu (aliyun pytorch-wheels/cpu find-links, deps from PIP_INDEX_URL)"
  python -m pip install \
    --find-links https://mirrors.aliyun.com/pytorch-wheels/cpu \
    torch==2.11.0
  echo "installing torch_npu==2.11.0 (Huawei ascend index)"
  pip_ascend torch_npu==2.11.0
}

# Copy CI fixture data files into the target root so that example
# recipes can load them via a local path under $TARGET_ROOT/fixtures/
# (engine contract: overlay_args uses ${TARGET_ROOT}/fixtures/...;
# same decoupling from the workflows-checkout subtree as peft).
prepare_fixtures() {
  local src="${FIXTURE_DIR:?FIXTURE_DIR is required}"
  local dst="$TARGET_ROOT/fixtures"
  echo "preparing fixtures from $src to $dst"
  # Fail-soft: we don't want to hard-fail if the project only ships a
  # custom-task YAML and no JSON/JSONL — the eleuther_eval recipe, for
  # example, ships fixtures/eleuther_tasks/*.yaml but no top-level JSON.
  # The actual presence/validity of the file is checked downstream when
  # the recipe opens it; a missing file at setup is a config bug that
  # surfaces as a real stack trace, not a setup-script one.
  if ! compgen -G "$src/*" >/dev/null; then
    echo "FATAL: fixture dir $src is empty" >&2
    exit 1
  fi
  mkdir -p "$dst"
  # Mirror the entire tree: flat *.json / *.jsonl for dataset recipes,
  # plus *.yaml subdirs (e.g. fixtures/eleuther_tasks/) for lm_eval
  # custom-task definitions consumed by recipes/eleuther_eval.py via
  # cfg.include_path = ${TARGET_ROOT}/fixtures/eleuther_tasks.
  cp -r "$src"/. "$dst"/
  echo "copied $(find "$dst" -type f | wc -l) fixture file(s) to $dst"
}

setup_torchtune() {
  # torchtune from the guarded checkout. The torchao pin is decided
  # dynamically because the NF4Tensor import path in
  # torchtune/modules/common_utils.py:19 has moved three times:
  #
  #   torchao 0.10-0.13 : torchao.dtypes.nf4tensor.NF4Tensor  (v0.6.x)
  #   torchao 0.14-0.17 : dtypes submodule deleted; quantization
  #                       renamed NF4Tensor -> Int4Tensor. Anything
  #                       importing NF4Tensor breaks here.
  #   torchao 0.18+     : torchao.quantization.NF4Tensor
  #                       re-exposed (main HEAD post PR #2960).
  #
  # Picking the wrong line is fatal — v0.6.x + torchao 0.18 throws
  # ModuleNotFoundError on import; main + torchao 0.13 throws the
  # same. There is no single torchao that satisfies both import paths,
  # so we must probe the actual checkout before pinning.
  echo "installing torchtune from $TARGET_ROOT (non-editable, see Quick-start-Ascend.md:164-167)"
  python -m pip install "$TARGET_ROOT"
  python -m pip install "transformers==4.57.1" "omegaconf>=2.3,<3" \
    tokenizers safetensors tqdm pyyaml

  # Probe which NF4Tensor import path the torchtune checkout uses, then
  # install the matching torchao exact pin. Probe runs against the
  # installed source (non-editable, see Quick-start-Ascend.md:164-167),
  # so it reflects whatever ref the engine checked out (release tag
  # OR main HEAD).
  local import_path torchao_pin
  import_path="$(python -c "
import importlib.util, pathlib
p = pathlib.Path('$TARGET_ROOT') / 'torchtune' / 'modules' / 'common_utils.py'
src = p.read_text() if p.exists() else ''
for line in src.splitlines():
    s = line.strip()
    if s.startswith('from torchao') and 'NF4Tensor' in s:
        print(s)
        break
else:
    print('NF4Tensor_NOT_IMPORTED')
")"
  echo "torchtune common_utils.py NF4Tensor import: $import_path"
  case "$import_path" in
    "from torchao.dtypes.nf4tensor import NF4Tensor")
      torchao_pin="torchao==0.13.0"
      ;;
    "from torchao.quantization import NF4Tensor")
      # 0.18.0 re-exposed NF4Tensor under torchao.quantization; main
      # HEAD depends on the exact 0.18 series (later 0.18.x keep the
      # symbol but pin to whatever the upstream test grid currently
      # passes — 0.18.0 is the first stable release with the re-add).
      torchao_pin="torchao==0.18.0"
      ;;
    "NF4Tensor_NOT_IMPORTED")
      # Newer torchtune may drop NF4Tensor entirely. Default to a
      # neutral recent torchao and let setup proceed; recipe failures
      # downstream will surface real reasons rather than setup noise.
      echo "WARN: common_utils.py does not import NF4Tensor; skipping torchao pin"
      torchao_pin=""
      ;;
    *)
      echo "FATAL: unexpected NF4Tensor import line: $import_path" >&2
      exit 1
      ;;
  esac
  if [ -n "$torchao_pin" ]; then
    echo "pinning $torchao_pin (matched import path)"
    python -m pip install "$torchao_pin"
  fi

  # importlib.metadata.version returns the real install tag for both
  # editable installs (where __version__ is empty string) and wheel
  # installs. torchtune.__version__ is "" by default in the source
  # tree; reading it directly would print a confusing blank.
  python -c "
import importlib.metadata as md
import torchao, omegaconf, transformers
import torchtune  # noqa: just to confirm the import chain
print('torchtune', md.version('torchtune'), '/ torchao', torchao.__version__, '/ transformers', transformers.__version__)
"

  # Pre-download the example model from ModelScope (China-reachable)
  # because runners cannot reach HuggingFace. The local snapshot dir
  # is exported as TT_MODEL_PATH for overlay_args to reference.
  # Pinned to the doc's verified line: modelscope>=1.38 splits the hub
  # code into modelscope-hub, and the fresh 1.40.1 wheel's loose
  # ">=0.4.2" floor breaks import when the mirror lags on hub 0.4.3.
  python -m pip install "modelscope==1.37.0"
  # lm_eval is only needed for the eleuther_eval recipe (not declared as
  # an upstream dep of torchtune). The recipe's __init__ checks
  # `version("lm-eval") < "0.4.5"` (recipes/eleuther_eval.py:446) using
  # STRING comparison (importlib.metadata.version returns a string, no
  # version coercion), not packaging.version. So "0.4.13" < "0.4.5" is
  # True ("0.4.1" prefix beats "0.4.5" lexicographically) and 0.4.10-0.4.49
  # all get rejected — only 0.4.5-0.4.9 and 0.4.50+ pass. Pin exact 0.4.5
  # (lowest acceptable) to keep the floor obvious; the import chain only
  # uses evaluator/models/tasks/utils which is stable across 0.4.x.
  python -m pip install "lm-eval==0.4.5"
  python - <<'PY'
import os
# Non-TTY CI logs: throttle tqdm refreshes instead of disabling.
os.environ.setdefault("TQDM_MININTERVAL", "15")
from modelscope import snapshot_download

MODEL_CACHE = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))
# CI runner containers start with no /root/.cache/modelscope. The
# default cache path returned by os.path.expanduser is not created by
# modelscope itself — snapshot_download fails mid-transfer when the
# ._____temp staging dir cannot be opened (FileDownloadError on the
# *.safetensors file). mkdir -p is a no-op on coder where env.sh
# already exports MODELSCOPE_CACHE to /home/coder/work/modelscope-cache.
os.makedirs(MODEL_CACHE, exist_ok=True)
local = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", cache_dir=MODEL_CACHE)
with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"TT_MODEL_PATH={local}\n")
print("TT_MODEL_PATH=", local)
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
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

HERE=$(cd "$(dirname "$0")" && pwd)
export PIP_CONSTRAINT="$(cd "$HERE/.." && pwd)/constraints-npu.txt"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

select_pip_index
python -m pip install -U pip setuptools wheel
ensure_torch_stack
prepare_fixtures

"setup_${PROFILE}"

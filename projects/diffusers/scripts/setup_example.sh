#!/usr/bin/env bash
# Prepare the CI environment for one supported diffusers example.
# $1 is the manifest profile. Unknown profiles fail before any install.
#
# diffusers itself is installed from TARGET_ROOT (the release checkout
# under test), so the guarded tag is exactly the code that runs.
#
# Contract: docs/guarding-examples.md "项目运行脚本契约".
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

# Validate the profile before installing anything (contract: unknown
# profile must exit non-zero before any install).
SUPPORTED_PROFILES="diffusers-sdxl diffusers-sd15 diffusers-dreambooth diffusers-instruct-pix2pix diffusers-kandinsky diffusers-amused diffusers-cogvideo diffusers-lcm diffusers-lcm-sdxl diffusers-controlnet diffusers-controlnet-sdxl diffusers-llada2"
case "$PROFILE" in
  diffusers-sdxl|diffusers-sd15|diffusers-dreambooth|diffusers-instruct-pix2pix|diffusers-kandinsky|diffusers-amused|diffusers-cogvideo|diffusers-lcm|diffusers-lcm-sdxl|diffusers-controlnet|diffusers-controlnet-sdxl|diffusers-llada2) ;;
  *)
    echo "unknown profile: ${PROFILE} (supported: ${SUPPORTED_PROFILES})" >&2
    exit 1
    ;;
esac

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
  # Same torch line as diffusers quick-start (CANN 9.1.0 pairing):
  # reuse the image stack when it already matches, otherwise install.
  if python -c "
import torch, torch_npu
raise SystemExit(
    0 if torch.__version__.startswith('2.9.0')
    and torch_npu.__version__.startswith('2.9.0') else 1)
"; then
    echo "reusing image torch stack ($(python -c 'import torch; print(torch.__version__)'))"
    return
  fi
  echo "installing torch==2.9.0 torch_npu==2.9.0.post2"
  pip_ascend torch==2.9.0 torch_npu==2.9.0.post2
}

prepare_fixtures() {
  # Optional: copy project fixtures into the target root so examples can
  # read them under $TARGET_ROOT/fixtures/. No-op when FIXTURE_DIR is
  # absent or empty (diffusers examples currently pull their datasets
  # from ModelScope / hf-mirror during setup instead).
  local src="${FIXTURE_DIR:-}"
  local dst="$TARGET_ROOT/fixtures"
  if [[ -z "$src" || ! -d "$src" ]]; then
    echo "no fixtures to prepare ($src)"
    return
  fi
  shopt -s nullglob
  local files=("$src"/*)
  shopt -u nullglob
  if ((${#files[@]} == 0)); then
    echo "no fixtures to prepare (empty $src)"
    return
  fi
  echo "preparing fixtures from $src to $dst"
  mkdir -p "$dst"
  cp -r "${files[@]}" "$dst/"
}

# ---------------------------------------------------------------
# Shared steps
# ---------------------------------------------------------------

install_example_stack() {
  echo "installing diffusers from $TARGET_ROOT"
  python -m pip install -e "$TARGET_ROOT"
  # Example stack. transformers 5.x is required by diffusers 0.40+
  # (huggingface-hub>=1.23 vs transformers 4.x's hub<1.0 conflict);
  # modelscope pinned to 1.37.0 (hub split started at 1.38 and 1.40.1's
  # "modelscope-hub>=0.4.2" floor dies on DEFAULT_CREDENTIALS_PATH);
  # torchvision pinned to the torch 2.9.0 pairing; prodigyopt is imported
  # by the advanced scripts' --optimizer=prodigy and is not in
  # examples/advanced_diffusion_training/requirements.txt.
  # datasets<4.0.0: diffusers examples (e.g. controlnet's load_dataset path)
  # break on datasets 4.x; the whole project stays below 4.
  # wandb: every training script does `if is_wandb_available(): import wandb`
  # and several default --report_to to wandb; installed once here, kept
  # offline by the global WANDB_MODE=disabled.
  python -m pip install \
    "transformers>=5.0,<6.0" "accelerate>=1.0,<2.0" "peft>=0.6" \
    "datasets<4.0.0" ftfy tensorboard Jinja2 sentencepiece "torchvision==0.24.0" \
    prodigyopt "modelscope==1.37.0" wandb
  python -c "import diffusers, transformers, accelerate, peft; print('diffusers', diffusers.__version__, '/ transformers', transformers.__version__, '/ accelerate', accelerate.__version__, '/ peft', peft.__version__)"
}

# download_assets <what>: comma-separated tokens from
# {sdxl, sdxl-vae, sd15, 3d-icon, cogvideo}. Each token exports a path to
# GITHUB_ENV under the name overlay_args reference:
#   sdxl      -> SDXL_BASE_PATH    (AI-ModelScope/stable-diffusion-xl-base-1.0)
#   sdxl-vae  -> SDXL_VAE_PATH     (AI-ModelScope/sdxl-vae-fp16-fix)
#   sd15      -> SD15_MODEL_PATH   (AI-ModelScope/stable-diffusion-v1-5)
#   3d-icon   -> THREE_D_ICON_PATH (hf-mirror linoyts/3d_icon)
#   cogvideo  -> COGVIDEOX_MODEL_PATH + COGVIDEOX_DATASET_DIR
download_assets() {
  DIFFUSERS_DOWNLOAD="$1" python - <<'PY'
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_MININTERVAL", "15")

from modelscope import snapshot_download

WORKSPACE = Path(os.environ["GITHUB_WORKSPACE"])
ENV_FILE = os.environ["GITHUB_ENV"]
MODEL_CACHE = Path(os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope")))
WANT = {item.strip() for item in os.environ.get("DIFFUSERS_DOWNLOAD", "").split(",") if item.strip()}

exports: dict[str, str] = {}
failures: list[str] = []


def export(name: str, path: str) -> None:
    exports[name] = path
    print(f"{name}={path}", flush=True)


def snapshot(name: str, ms_id: str, **kwargs) -> None:
    try:
        local = Path(snapshot_download(ms_id, cache_dir=str(MODEL_CACHE), **kwargs))
        export(name, str(local))
    except Exception as exc:  # noqa: BLE001 - report and continue
        failures.append(f"{ms_id}: {type(exc).__name__}: {exc}")
        print(f"FAIL {ms_id}: {exc}", flush=True)


# SDXL base components only. The repo also carries sd_xl_base_1.0.safetensors
# (6.9 GB, ComfyUI single-file) and fp16 duplicates (~11 GB) that
# from_pretrained does not need; the script loads the fp32 component weights
# and casts to bf16 itself.
if "sdxl" in WANT:
    snapshot(
        "SDXL_BASE_PATH",
        "AI-ModelScope/stable-diffusion-xl-base-1.0",
        allow_file_pattern=[
            "*.json", "*.txt", "*.model",
            "unet/*", "vae/*",
            "text_encoder/*", "text_encoder_2/*",
            "tokenizer/*", "tokenizer_2/*", "scheduler/*",
        ],
        # .bin is the duplicate of .safetensors; skip both fp16 and .bin to
        # halve the download (and the .bin download is what fails).
        ignore_file_pattern=["*.fp16.*", "*.bin"],
    )

# SDXL fp16-safe VAE. Only entries that pass
# --pretrained_vae_model_name_or_path need it (the advanced SDXL entry does
# not; the LCM SDXL distillation does).
if "sdxl-vae" in WANT:
    snapshot("SDXL_VAE_PATH", "AI-ModelScope/sdxl-vae-fp16-fix")

# SD 1.5 diffusers components only; skip the single-file checkpoints
# (v1-5-pruned*.safetensors, ~4 GB each). Also the LCM SD teacher.
if "sd15" in WANT:
    snapshot(
        "SD15_MODEL_PATH",
        "AI-ModelScope/stable-diffusion-v1-5",
        allow_file_pattern=[
            "*.json", "*.txt", "*.model",
            "unet/*", "vae/*",
            "text_encoder/*", "tokenizer/*", "scheduler/*",
        ],
        # Same as SDXL: the repo carries both .bin and .safetensors; only the
        # latter is needed.
        ignore_file_pattern=["*.fp16.*", "*.bin"],
    )

# SD 1.4 (the dreambooth examples' README base model). Same component-only
# filter as SD1.5.
if "sd14" in WANT:
    snapshot(
        "SD14_MODEL_PATH",
        "AI-ModelScope/stable-diffusion-v1-4",
        allow_file_pattern=[
            "*.json", "*.txt", "*.model",
            "unet/*", "vae/*",
            "text_encoder/*", "tokenizer/*", "scheduler/*",
        ],
        ignore_file_pattern=["*.fp16.*", "*.bin"],
    )

# CogVideoX-2b (transformer + T5 text_encoder + VAE). The ModelScope repo
# layout is already clean (component dirs only, no single-file/fp16 dupes).
if "cogvideo" in WANT:
    snapshot(
        "COGVIDEOX_MODEL_PATH",
        "AI-ModelScope/CogVideoX-2b",
        allow_file_pattern=[
            "*.json", "*.model",
            "scheduler/*", "text_encoder/*", "tokenizer/*",
            "transformer/*", "vae/*",
        ],
    )
    # Dataset: Wild-Heart/Disney-VideoGeneration-Dataset (69 videos, ~25 MB,
    # Steamboat Willie clips) already in the README's first format:
    # prompt.txt + videos.txt + videos/. No ModelScope mirror, so pull via
    # hf-mirror (the engine sets HF_ENDPOINT + HF_HUB_DISABLE_XET).
    #
    # NOTE: the videos are Xet-backed. If hf-mirror 302s them to
    # cas-bridge.xethub.hf.co and the runner cannot reach it, this download
    # will fail and the dataset must be delivered via cache-seed/diffusers/
    # instead (same treatment as peft's Xet-backed fixtures).
    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download

        dataset_dir = WORKSPACE / "datasets" / "disney"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        hf_snapshot_download(
            "Wild-Heart/Disney-VideoGeneration-Dataset",
            repo_type="dataset",
            local_dir=str(dataset_dir),
        )
        export("COGVIDEOX_DATASET_DIR", str(dataset_dir))
    except Exception as exc:  # noqa: BLE001
        failures.append(f"Wild-Heart/Disney-VideoGeneration-Dataset: {type(exc).__name__}: {exc}")
        print(f"FAIL Wild-Heart/Disney-VideoGeneration-Dataset: {exc}", flush=True)

# 3d_icon dataset (only the advanced dreambooth entries use it): no
# ModelScope mirror (205 MB imagefolder + metadata.jsonl), so pull it via
# hf-mirror into the workspace and point --dataset_name at the local directory.
if "3d-icon" in WANT:
    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download

        dataset_dir = WORKSPACE / "datasets" / "3d_icon"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        hf_snapshot_download(
            "linoyts/3d_icon",
            repo_type="dataset",
            local_dir=str(dataset_dir),
            ignore_patterns=[".gitattributes"],
        )
        export("THREE_D_ICON_PATH", str(dataset_dir))
    except Exception as exc:  # noqa: BLE001
        failures.append(f"linoyts/3d_icon: {type(exc).__name__}: {exc}")
        print(f"FAIL linoyts/3d_icon: {exc}", flush=True)

if exports:
    with open(ENV_FILE, "a", encoding="utf-8") as handle:
        for key, value in exports.items():
            handle.write(f"{key}={value}\n")

if failures:
    print(f"setup incomplete: {failures}", file=sys.stderr, flush=True)
    raise SystemExit(1)
PY
}

# ---------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------
#
# Convention: install_example_stack() is the common stack. It already covers
# everything every example dir's requirements.txt asks for EXCEPT the
# per-dir extras below (transformers/accelerate/peft/datasets/torchvision/
# ftfy/tensorboard/Jinja2/sentencepiece are all in the base; datasets is
# pinned <4.0.0 project-wide). Each setup_diffusers_<profile>() adds only the
# delta for its example dir, then pre-downloads that dir's model/dataset:
#
#   advanced_diffusion_training : base only
#   amused                      : base only (no requirements.txt upstream)
#   cogvideo                    : + decord2, imageio, imageio-ffmpeg
#   consistency_distillation    : + webdataset, braceexpand
#   controlnet                  : base only (datasets already pinned)
#
# NOTE: when adding a new example, diff its requirements.txt against the base
# and add the delta here; never `pip install -r` it verbatim (it pins to git
# main / loose versions and pulls CUDA-only deps).

# diffusers-sdxl: SDXL advanced dreambooth/LoRA training example. Uses the
# tiny HF model at run time, so no ModelScope pre-download.
setup_diffusers_sdxl() {
  install_example_stack
}

# diffusers-sd15: SD1.5 advanced dreambooth/LoRA training example.
setup_diffusers_sd15() {
  install_example_stack
  download_assets sd15,3d-icon
}

# diffusers-dreambooth: SD1.4 dreambooth / dreambooth-LoRA training examples.
# The dataset is the repo's own docs/source/en/imgs, so no dataset download.
setup_diffusers_dreambooth() {
  install_example_stack
  download_assets sd14
}

# diffusers-instruct-pix2pix: SD1.5 InstructPix2Pix. The dataset
# (fusing/instructpix2pix-1000-samples) is fetched by the example at run time.
setup_diffusers_instruct_pix2pix() {
  install_example_stack
  download_assets sd15
}

# diffusers-kandinsky: Kandinsky 2.2 decoder training. Base only; the model
# (kandinsky-community/kandinsky-2-2-decoder, no ModelScope mirror) and the
# dataset are fetched by the example at run time.
setup_diffusers_kandinsky() {
  install_example_stack
}

# diffusers-amused: Amused-256 finetuning. ModelScope has neither
# amused/amused-256 nor the m1guelpf/nouns dataset, so nothing is
# pre-downloaded: the example fetches both via hf-mirror at run time
# (the engine sets HF_ENDPOINT + HF_HUB_DISABLE_XET).
# amused's --report_to defaults to wandb (wandb is in the base stack; the
# global WANDB_MODE=disabled keeps it offline). Do NOT use
# --report_to tensorboard: amused passes list-valued args to init_trackers,
# which tensorboard's add_hparams rejects.
setup_diffusers_amused() {
  install_example_stack
}

# diffusers-controlnet: ControlNet training with an SD1.5 base. The dataset
# (fusing/fill50k) is pulled by the example itself via hf-mirror at run time.
# fill50k is a script-based dataset: it needs datasets<4.0 (base pin) and
# trust_remote_code. Persist the env var to the run step via GITHUB_ENV.
setup_diffusers_controlnet() {
  install_example_stack
  echo "HF_DATASETS_TRUST_REMOTE_CODE=1" >> "$GITHUB_ENV"
  download_assets sd15
}

# diffusers-controlnet-sdxl: same, with an SDXL base.
setup_diffusers_controlnet_sdxl() {
  install_example_stack
  echo "HF_DATASETS_TRUST_REMOTE_CODE=1" >> "$GITHUB_ENV"
  download_assets sdxl
}

# diffusers-lcm: LCM consistency-distillation webdataset examples with an
# SD1.5 teacher. Needs webdataset + braceexpand. The training shards are
# streamed at run time (overlay URL), not pre-downloaded.
setup_diffusers_lcm() {
  install_example_stack
  python -m pip install webdataset braceexpand
  download_assets sd15
}

# diffusers-lcm-sdxl: same, but with an SDXL teacher + SDXL fp16-safe VAE.
setup_diffusers_lcm_sdxl() {
  install_example_stack
  python -m pip install webdataset braceexpand
  download_assets sdxl,sdxl-vae
}

# diffusers-cogvideo: CogVideoX-2b LoRA finetuning. decord + imageio /
# imageio-ffmpeg are the example's own requirements (video decode / export).
# Model from ModelScope; dataset (Wild-Heart/Disney-VideoGeneration-Dataset)
# from hf-mirror.
setup_diffusers_cogvideo() {
  install_example_stack
  # decord2 is the Ascend/aarch64 build of decord (plain decord does not work
  # on NPU); it provides the `decord` import the example uses. imageio /
  # imageio-ffmpeg are the example's own requirements (video export).
  python -m pip install decord2 imageio imageio-ffmpeg
  download_assets cogvideo
}

# diffusers-llada2: LLaDA2 block-refinement training smoke. Base stack is
# enough (datasets is already pinned there); Qwen2.5-0.5B is fetched at run
# time and --use_dummy_data avoids any dataset download.
setup_diffusers_llada2() {
  install_example_stack
}

if [[ -z "${TARGET_ROOT:-}" || -z "${GITHUB_WORKSPACE:-}" || -z "${GITHUB_ENV:-}" ]]; then
  echo "TARGET_ROOT / GITHUB_WORKSPACE / GITHUB_ENV must be set" >&2
  exit 2
fi

HERE=$(cd "$(dirname "$0")" && pwd)
export PIP_CONSTRAINT="$(cd "$HERE/.." && pwd)/constraints-npu.txt"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

select_pip_index
python -m pip install -U pip setuptools wheel
ensure_torch_stack
prepare_fixtures

# All examples: keep wandb offline. Some entrypoints default --report_to to
# wandb; WANDB_MODE=disabled prevents any network reporting and is harmless
# for the ones that use tensorboard. Persisted to the run step via GITHUB_ENV.
echo "WANDB_MODE=disabled" >> "$GITHUB_ENV"

# All examples: let HCCL pick a free socket port range (required on some
# Ascend setups to avoid "address already in use" on multi-process init).
echo "HCCL_NPU_SOCKET_PORT_RANGE=auto" >> "$GITHUB_ENV"

# Profile names map to setup_<profile with '-' -> '_'> functions.
"setup_${PROFILE//-/_}"

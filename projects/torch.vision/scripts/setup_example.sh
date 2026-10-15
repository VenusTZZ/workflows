#!/usr/bin/env bash
# Prepare one example from the exact release checked out by examples-template.
set -euo pipefail

profile="${1:-}"
case "$profile" in
  vision_gallery_npu|vision_classification_npu|vision_segmentation_npu|\
  vision_flow_npu|vision_video_npu) ;;
  *) echo "unknown TorchVision example profile: ${profile:-<missing>}" >&2; exit 2 ;;
esac

: "${TARGET_ROOT:?TARGET_ROOT is required}"
: "${PROJECT_ROOT:?PROJECT_ROOT is required}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

resolved_tag="$(git -C "$TARGET_ROOT" describe --tags --exact-match 2>/dev/null || true)"
echo "TorchVision tested release: ${resolved_tag:-<tag unavailable in shallow checkout>}"
echo "TorchVision tested commit: $(git -C "$TARGET_ROOT" rev-parse HEAD)"

# The existing torch.vision quick-start has verified this source-build
# compatibility patch on the NPU runner. It changes the checked-out library
# implementation only; the example scripts themselves remain untouched.
# Let git apply report a normal failure if a later release no longer matches.
git -C "$TARGET_ROOT" apply "$PROJECT_ROOT/patches/torch-2.12-stable-api-permute.patch"

# Use the same public Ascend-compatible wheel sources as this project's
# quick-start. The source install below uses --no-deps and FORCE_CUDA=0 so
# torchvision cannot replace torch_npu's matching PyTorch with a CUDA wheel.
python -m pip install uv
uv pip install --system -f https://mirrors.aliyun.com/pytorch-wheels/cpu 'torch==2.12.0'
uv pip install --system --extra-index-url https://mirrors.aliyun.com/pypi/simple 'torch_npu==2.12.0'
uv pip install --system 'pillow>=10.0' numpy
FORCE_CUDA=0 uv pip install --system --no-build-isolation --no-deps -e "$TARGET_ROOT"

python - <<'PY'
import os
from pathlib import Path

import torch
import torch_npu
import torchvision

target = Path(os.environ["TARGET_ROOT"]).resolve()
loaded = Path(torchvision.__file__).resolve()
declared = (target / "version.txt").read_text(encoding="utf-8").strip()
print(f"torch={torch.__version__} torch_npu={torch_npu.__version__}")
print(f"torchvision={torchvision.__version__} source={loaded}")
if not torch.__version__.startswith("2.12.0") or not torch_npu.__version__.startswith("2.12.0"):
    raise SystemExit("torch and torch_npu must match the declared CANN 9.1 stack")
if not loaded.is_relative_to(target) or not torchvision.__version__.startswith(declared):
    raise SystemExit("torchvision must be installed from this release checkout")
if not torch.npu.is_available() or torch.npu.device_count() < 1:
    raise SystemExit("NPU unavailable; refusing CPU fallback")
PY

if [[ "$profile" == vision_classification_npu ]]; then
  : "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
  export VISION_DATA_ROOT="$GITHUB_WORKSPACE/vision-classification-fixture"
  python - <<'PY'
import os
from pathlib import Path
from PIL import Image

root = Path(os.environ["VISION_DATA_ROOT"])
for split, count in (("train", 2), ("val", 1)):
    for cls in range(5):  # upstream evaluation always calculates top-5
        folder = root / split / f"class_{cls}"
        folder.mkdir(parents=True, exist_ok=True)
        for item in range(count):
            image = Image.new("RGB", (80, 80), (25 + cls * 35, 15 + item * 80, 180 - cls * 20))
            image.save(folder / f"{item}.png")
print(f"five-class ImageFolder fixture: {root}")
PY
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    printf 'VISION_DATA_ROOT=%s\n' "$VISION_DATA_ROOT" >> "$GITHUB_ENV"
  fi
fi

if [[ "$profile" == vision_gallery_npu ]]; then
  uv pip install --system matplotlib
  export MPLBACKEND=Agg
  [[ -z "${GITHUB_ENV:-}" ]] || printf 'MPLBACKEND=Agg\n' >> "$GITHUB_ENV"
fi

if [[ "$profile" == vision_flow_npu || "$profile" == vision_video_npu ]]; then
  : "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
  export VISION_REFERENCE_DATA_ROOT="$GITHUB_WORKSPACE/vision-$profile-fixture"
  kind=flow
  if [[ "$profile" == vision_video_npu ]]; then
    kind=video
    uv pip install --system 'av>=14,<17'
  fi
  python "$PROJECT_ROOT/scripts/prepare_ci_assets.py" "$kind" "$VISION_REFERENCE_DATA_ROOT"
  [[ -z "${GITHUB_ENV:-}" ]] || printf 'VISION_REFERENCE_DATA_ROOT=%s\n' "$VISION_REFERENCE_DATA_ROOT" >> "$GITHUB_ENV"
fi

if [[ "$profile" == vision_segmentation_npu ]]; then
  # train.py imports coco_utils at module load even for --dataset voc;
  # coco_utils imports pycocotools unconditionally. Keep this optional
  # reference-script dependency scoped to the segmentation profile.
  uv pip install --system pycocotools
  python -c 'from pycocotools import mask; print("pycocotools mask import OK")'

  : "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
  export VISION_SEG_DATA_ROOT="$GITHUB_WORKSPACE/vision-segmentation-fixture"
  python - <<'PY'
import os
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(os.environ["VISION_SEG_DATA_ROOT"]) / "VOCdevkit" / "VOC2012"
images = root / "JPEGImages"
masks = root / "SegmentationClass"
splits = root / "ImageSets" / "Segmentation"
for folder in (images, masks, splits):
    folder.mkdir(parents=True, exist_ok=True)

for split, names in (("train", ("ci_train_0", "ci_train_1")), ("val", ("ci_val_0",))):
    (splits / f"{split}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    for idx, name in enumerate(names):
        Image.new("RGB", (80, 80), (40 + idx * 50, 90, 130)).save(images / f"{name}.jpg")
        mask = Image.new("L", (80, 80), 0)
        ImageDraw.Draw(mask).rectangle((16, 16, 63, 63), fill=idx + 1)
        mask.save(masks / f"{name}.png")
print(f"generated three-image VOC segmentation fixture at {root}")
PY
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    printf 'VISION_SEG_DATA_ROOT=%s\n' "$VISION_SEG_DATA_ROOT" >> "$GITHUB_ENV"
  fi
fi

#!/usr/bin/env bash
# Sourced by setup_example.sh. Keep each upstream recipe's dependencies isolated.

setup_ds_finetune_demo() {
  install_deepspeed_source
  # The upstream rotary reset requires base/dim but sets max_seq_len_cached=None.
  # Llama 4.42 uses base/dim and position_ids without a cached-length comparison;
  # newer Llama/Qwen and older Qwen rotary implementations do not meet both rules.
  install_example_dependencies 'transformers==4.42.4' 'accelerate>=1.0,<2' \
    'datasets>=4,<5' safetensors sentencepiece wandb
  ms_download_models 'SMOLLM2_135M_PATH=HuggingFaceTB/SmolLM2-135M'
  plant_ci_fixture ci_alpaca_16.json ALPACA_CI_PATH
  export ALPACA_FINETUNE_DIR="$GITHUB_WORKSPACE/.ci/deepspeed-fixtures/finetune-demo"
  echo "ALPACA_FINETUNE_DIR=$ALPACA_FINETUNE_DIR" >> "$GITHUB_ENV"
  python - <<'PY'
import importlib.util
import json
import os
from pathlib import Path
import sys

from datasets import Dataset, load_dataset
import torch
import torch_npu
import transformers
from transformers import AutoConfig, AutoTokenizer, LlamaConfig, LlamaForCausalLM

if transformers.__version__ != '4.42.4':
    raise SystemExit('finetune demo requires Transformers 4.42.4 rotary semantics')
config = AutoConfig.from_pretrained(os.environ['SMOLLM2_135M_PATH'], local_files_only=True)
if config.model_type != 'llama' or config.rope_scaling is not None:
    raise SystemExit('finetune demo requires the original SmolLM2 Llama/default-RoPE config')
entry = Path(os.environ['EXAMPLES_ROOT']) / 'training/deepspeed_finetune_demo/finetune_llama.py'
sys.path.insert(0, str(entry.parent))
spec = importlib.util.spec_from_file_location('ds_ci_finetune_demo', entry)
upstream = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = upstream
spec.loader.exec_module(upstream)
# Execute the exact upstream reset, then a real CPU forward/backward. This
# preflight detects deterministic dependency/API incompatibilities before NPU launch.
with torch.device('cpu'):
    tiny = LlamaForCausalLM(LlamaConfig(vocab_size=128, hidden_size=32,
                                    intermediate_size=64, num_hidden_layers=2,
                                    num_attention_heads=4, num_key_value_heads=2,
                                    rope_scaling=None, max_position_embeddings=128))
upstream._reset_rotary_embeddings(tiny)
ids = torch.arange(16, device='cpu').reshape(1, 16)
loss = tiny(input_ids=ids, labels=ids).loss
if not torch.isfinite(loss):
    raise SystemExit('upstream rotary-reset CPU preflight produced nonfinite loss')
loss.backward()
print('real upstream Llama rotary-reset CPU forward/backward:', loss.item())
tokenizer = AutoTokenizer.from_pretrained(os.environ['SMOLLM2_135M_PATH'], local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
rows = json.loads(Path(os.environ['ALPACA_CI_PATH']).read_text())
directory = Path(os.environ['ALPACA_FINETUNE_DIR'])
directory.mkdir(parents=True, exist_ok=True)
Dataset.from_list(rows).to_parquet(directory / 'train.parquet')
dataset = load_dataset(str(directory))['train']
if len(dataset) != 16:
    raise SystemExit('finetune demo local dataset must contain exactly 16 rows')
for row in dataset:
    encoded = upstream.preprocess_alpaca(row, tokenizer, max_length=128)
    if len(encoded['input_ids']) != 128 or not any(x != -100 for x in encoded['labels'][1:]):
        raise SystemExit('finetune demo fixture lost shifted training labels at length 128')
print('finetune demo native local Parquet and labels:', directory, len(dataset))
PY
}

download_sd15_ci_model() {
  install_example_dependencies 'modelscope==1.37.0'
  python - <<'PY'
import os
from pathlib import Path
from modelscope import snapshot_download

# Full Diffusers components, but not duplicate .bin/root .ckpt weights. The
# final unchanged entry reloads the safety checker when constructing its pipeline.
patterns = [
    'model_index.json', 'feature_extractor/preprocessor_config.json',
    'scheduler/scheduler_config.json', 'tokenizer/*',
    'text_encoder/config.json', 'text_encoder/model.safetensors',
    'unet/config.json', 'unet/diffusion_pytorch_model.safetensors',
    'vae/config.json', 'vae/diffusion_pytorch_model.safetensors',
    'safety_checker/config.json', 'safety_checker/model.safetensors',
]
path = snapshot_download('AI-ModelScope/stable-diffusion-v1-5',
                         allow_patterns=patterns,
                         cache_dir=os.environ.get('MODELSCOPE_CACHE', os.path.expanduser('~/.cache/modelscope')))
required = [name for name in patterns if '*' not in name]
required += ['tokenizer/vocab.json', 'tokenizer/merges.txt', 'tokenizer/tokenizer_config.json']
for name in required:
    asset = Path(path) / name
    if not asset.is_file() or not asset.stat().st_size:
        raise SystemExit(f'SD15 complete local Diffusers asset missing: {asset}')
with open(os.environ['GITHUB_ENV'], 'a') as handle:
    handle.write(f'SD15_PATH={path}\n')
print('complete ModelScope SD15 Diffusers components:', path)
PY
  local model_path
  model_path="$(awk 'index($0,"SD15_PATH=")==1 { value=substr($0,11) } END {print value}' "$GITHUB_ENV")"
  [[ -d "$model_path" ]] || { echo "SD15 download did not expose a local directory" >&2; return 1; }
  export SD15_PATH="$model_path"
}

setup_ds_sd_distil() {
  install_deepspeed_source
  # Teacher UNet is an unwrapped FP32 module. Run the entire recipe in FP32;
  # BF16 VAE output would otherwise meet FP32 teacher convolution weights.
  install_example_dependencies 'transformers==4.44.2' 'diffusers==0.30.3' \
    'accelerate==1.10.1' 'datasets>=4,<5' 'pillow>=10' 'torchvision==0.24.0' \
    safetensors sentencepiece numpy
  download_sd15_ci_model
  python - <<'PY'
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

from datasets import Dataset, Features, Image as DatasetImage, Value, load_dataset
from PIL import Image, ImageDraw
import torch_npu
from transformers import AutoTokenizer

output = Path(os.environ.get('CI_OUTPUT_DIR', str(Path(os.environ['GITHUB_WORKSPACE']) / 'output')))
work = output / 'sd-distil-work'
directory = work / 'poloclub/diffusiondb'
directory.mkdir(parents=True, exist_ok=True)
prompts, images = [], []
for index in range(8):
    image = Image.new('RGB', (64, 64), (16 * index, 64, 128))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8 + index, 8, 48, 48), fill=(192, 32 * index, 64))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    images.append({'bytes': buffer.getvalue(), 'path': None})
    prompts.append(f'a colorful geometric rectangle number {index}')
features = Features({'image': DatasetImage(), 'prompt': Value('string')})
Dataset.from_dict({'image': images, 'prompt': prompts}, features=features).to_parquet(directory / 'train.parquet')
(directory / 'README.md').write_text(
    '---\nconfigs:\n- config_name: 2m_first_10k\n  data_files:\n'
    '  - split: train\n    path: train.parquet\n---\nLocal eight-image CI fixture.\n'
)
previous = Path.cwd()
try:
    os.chdir(work)
    # Native local-directory resolution, not a datasets.load_dataset wrapper.
    dataset = load_dataset('poloclub/diffusiondb', '2m_first_10k')['train']
finally:
    os.chdir(previous)
if len(dataset) != 8 or any(row['image'].mode != 'RGB' or row['image'].size != (64, 64)
                           or not row['prompt'] for row in dataset):
    raise SystemExit('SD native local dataset must decode eight RGB 64x64 images/prompts')
entry = Path(os.environ['EXAMPLES_ROOT']) / 'training/stable_diffusion/train_sd_distil_lora.py'
sys.path.insert(0, str(entry.parent))
spec = importlib.util.spec_from_file_location('ds_ci_sd_distil', entry)
upstream = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = upstream
spec.loader.exec_module(upstream)
# The original dataset reads the original entry's args global, as its normal
# __main__ does. Populate it through the original parser, without changing APIs.
upstream.args = upstream.parse_args([
    '--pretrained_model_name_or_path', os.environ['SD15_PATH'],
    '--default_prompt', 'a colorful geometric shape', '--train_batch_size', '1',
    '--resolution', '64', '--max_train_steps', '3', '--mixed_precision', 'no', '--report_to', 'none',
])
tokenizer = AutoTokenizer.from_pretrained(os.environ['SD15_PATH'], subfolder='tokenizer', local_files_only=True)
native = upstream.DreamBoothDataset(dataset['prompt'], dataset['image'], tokenizer, size=64, center_crop=True)
batch = upstream.collate_fn([native[0]], with_prior_preservation=False)
if tuple(batch['pixel_values'].shape) != (1, 3, 64, 64) or batch['input_ids'].shape[0] != 1:
    raise SystemExit('SD original DreamBooth dataset/collator produced invalid CI tensors')
model_index = json.loads((Path(os.environ['SD15_PATH']) / 'model_index.json').read_text())
if model_index['_class_name'] != 'StableDiffusionPipeline':
    raise SystemExit('SD CI requires a complete StableDiffusionPipeline snapshot')
print('SD native fixture/collator verified:', directory.resolve(), len(native), tuple(batch['pixel_values'].shape))
print('SD teacher CFG distillation trains full UNet, not LoRA; dtype remains FP32')
PY
}

setup_ds_opsd_decode() {
  install_deepspeed_source
  install_example_dependencies 'transformers==4.57.6' 'accelerate>=1.10.1,<2' safetensors numpy
  ms_download_models 'QWEN3_06B_PATH=Qwen/Qwen3-0.6B'
  python - <<'PY'
import os
import torch
import torch_npu
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig
from transformers import AutoTokenizer

index = get_accelerator().current_device()
# Torch 2.9 maps integer devices to the registered accelerator (PrivateUse1
# first). Verify the exact tensor protocols the unchanged benchmark uses.
tensor = torch.empty(1, device='cpu').to(index)
random = torch.randint(10, 1000, (1, 2), device=index)
if tensor.device.type != 'npu' or random.device.type != 'npu':
    raise SystemExit(f'OPSD decode integer device did not select NPU: {tensor.device}, {random.device}')
tokenizer = AutoTokenizer.from_pretrained(os.environ['QWEN3_06B_PATH'], local_files_only=True)
print('OPSD decode integer-device preflight:', index, tensor.device, random.device, type(tokenizer).__name__)
print('OPSD rollout APIs:', HybridEngineRollout.__name__, HybridEngineRolloutConfig.__name__)
PY
}

#!/usr/bin/env bash
# Optional inference profiles. Sourced by setup_example.sh; no top-level installs.

setup_inference_dependencies() {
  install_deepspeed_source
  # 4.51+ resets pipeline.device to a CPU-loaded model.device when distributed
  # is already initialized; these older recipes require the pre-change behavior.
  install_example_dependencies 'transformers==4.44.2' 'accelerate>=0.30,<2' \
    sentencepiece protobuf safetensors numpy
}

plant_inference_alias() {
  local snapshot="$1" alias="$2"
  local work="$GITHUB_WORKSPACE/.ci/deepspeed-inference/$PROFILE"
  [[ -d "$snapshot" && "$alias" != /* && "$alias" != *'..'* ]] || {
    echo "invalid local inference alias: $alias -> $snapshot" >&2
    return 1
  }
  mkdir -p "$work/$(dirname "$alias")"
  if [[ -e "$work/$alias" || -L "$work/$alias" ]]; then
    [[ -L "$work/$alias" && "$(readlink "$work/$alias")" == "$snapshot" ]] || {
      echo "refusing to replace existing inference asset: $work/$alias" >&2
      return 1
    }
  else
    ln -s "$snapshot" "$work/$alias"
  fi
  export INFERENCE_CI_WORK="$work"
  printf 'INFERENCE_CI_WORK=%s\nHF_HUB_OFFLINE=1\nTRANSFORMERS_OFFLINE=1\n' "$work" >> "$GITHUB_ENV"
}

setup_ds_legacy_inference_bench() {
  setup_inference_dependencies
  ms_download_models 'BERT_BASE_CASED_PATH=AI-ModelScope/bert-base-cased' 'OPT_125M_PATH=facebook/opt-125m'
}

setup_ds_hf_ds_compare() {
  setup_inference_dependencies
  ms_download_models 'OPT_125M_PATH=facebook/opt-125m'
}

setup_ds_fill_mask_bert() {
  setup_inference_dependencies
  ms_download_models 'BERT_LARGE_CASED_PATH=AI-ModelScope/bert-large-cased'
  plant_inference_alias "$BERT_LARGE_CASED_PATH" 'bert-large-cased'
}

setup_ds_fill_mask_electra() {
  setup_inference_dependencies
  ms_download_models 'ELECTRA_GENERATOR_PATH=google/electra-base-generator'
  plant_inference_alias "$ELECTRA_GENERATOR_PATH" 'google/electra-base-generator'
}

setup_ds_fill_mask_roberta() {
  setup_inference_dependencies
  ms_download_models 'ROBERTA_LARGE_PATH=AI-ModelScope/roberta-large'
  plant_inference_alias "$ROBERTA_LARGE_PATH" 'roberta-large'
}

setup_ds_t5_translation() {
  setup_inference_dependencies
  ms_download_models 'T5_BASE_PATH=AI-ModelScope/t5-base'
  # A separate asset view bounds generation, without writing ModelScope cache
  # or changing the upstream script. Weights/tokenizer files remain symlinks.
  local work="$GITHUB_WORKSPACE/.ci/deepspeed-inference/$PROFILE"
  mkdir -p "$work/t5-base"
  python - "$T5_BASE_PATH" "$work/t5-base" <<'PY'
import json
import sys
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, GenerationConfig

snapshot, view = map(Path, sys.argv[1:])
for source in snapshot.iterdir():
    if source.name in ('generation_config.json', 'config.json'):
        continue
    target = view / source.name
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise SystemExit(f'unexpected T5 asset alias: {target}')
    elif target.exists():
        raise SystemExit(f'refusing to replace T5 asset: {target}')
    else:
        target.symlink_to(source, target_is_directory=source.is_dir())
config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
if config.model_type != 't5' or config.num_heads % 2:
    raise SystemExit('translation requires a T5 model with heads divisible by TP=2')
with torch.device('meta'):
    model = AutoModelForSeq2SeqLM.from_config(config)
names = [name for name, _ in model.named_modules()]
for suffix in ('SelfAttention.o', 'EncDecAttention.o', 'DenseReluDense.wo'):
    if not any(name.endswith(suffix) for name in names):
        raise SystemExit(f'T5 injection target missing: {suffix}')
generation = GenerationConfig.from_model_config(config)
generation.max_new_tokens = 8
generation.do_sample = False
# Pipeline applies legacy task_specific_params after generation_config loading.
# Keep its translation task semantics, but supply a legal short min_length.
generation.min_length = 0
for name, task in (config.task_specific_params or {}).items():
    if name.startswith('translation'):
        task['min_length'] = 0
        task['max_length'] = 16
config.save_pretrained(view)
generation.save_pretrained(view)
print('T5 TP policy checked; local generation_config limits output to 8 new tokens')
PY
  export INFERENCE_CI_WORK="$work"
  printf 'INFERENCE_CI_WORK=%s\nHF_HUB_OFFLINE=1\nTRANSFORMERS_OFFLINE=1\n' "$work" >> "$GITHUB_ENV"
}

setup_ds_hybrid_rollout() {
  setup_inference_dependencies
  ms_download_models 'OPT_125M_PATH=facebook/opt-125m'
  python - <<'PY'
import torch_npu
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import InferenceBuilder

if get_accelerator().device_name() != 'npu':
    raise SystemExit('HybridEngine requires the real NPU accelerator')
implementation = InferenceBuilder().load()
if implementation.__name__ != 'NPUInference':
    raise SystemExit(f'expected native NPU inference implementation, got {implementation}')
for name in ('qkv_gemm_bf16', 'softmax_context_bf16', 'mlp_gemm_bf16', 'vector_matmul_bf16', 'residual_add_bias_bf16'):
    if not callable(getattr(implementation, name, None)):
        raise SystemExit(f'native NPU inference operation missing: {name}')
print('verified native NPU HybridEngine inference operations')
PY
}

setup_ds_asr_ctc() {
  setup_inference_dependencies
  # datasets 4 returns Column objects for result['text']; this original script
  # and jiwer 3 expect ordinary lists. Keep the supported list-valued API.
  install_example_dependencies 'datasets==3.6.0' 'jiwer>=3,<4' soundfile pyarrow
  ms_download_models 'WAV2VEC2_PATH=AI-ModelScope/wav2vec2-base-960h'
  plant_inference_alias "$WAV2VEC2_PATH" 'facebook/wav2vec2-base-960h'
  # Native local-directory resolution satisfies the upstream's fixed
  # load_dataset("librispeech_asr", "clean", split="test") call. These are
  # synthetic waveforms, deliberately a CTC forward smoke, not LibriSpeech WER.
  python - "$INFERENCE_CI_WORK" <<'PY'
import json
import math
from pathlib import Path
import struct
import sys
import wave
from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq

work = Path(sys.argv[1])
dataset = work / 'librispeech_asr'
dataset.mkdir(exist_ok=True)
rows = []
for index in range(2):
    audio = dataset / f'ci-wave-{index}.wav'
    values = [int(2500 * math.sin(2 * math.pi * (220 + index * 110) * i / 16000)) for i in range(16000)]
    with wave.open(str(audio), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(struct.pack('<' + 'h' * len(values), *values))
    rows.append({'file': str(audio), 'text': 'HELLO WORLD'})
pq.write_table(pa.Table.from_pylist(rows), dataset / 'test.parquet')
(dataset / 'README.md').write_text('---\nconfigs:\n- config_name: clean\n  data_files:\n  - split: test\n    path: test.parquet\n---\nSynthetic CTC CI inputs, not LibriSpeech.\n')
# Validate the exact original loader contract, not an alternative builder.
import os
os.chdir(work)
loaded = load_dataset('librispeech_asr', 'clean', split='test')
if len(loaded) != 2 or not all(Path(row['file']).is_file() for row in loaded):
    raise SystemExit('native clean/test CTC fixture loader contract failed')
if not isinstance(loaded['text'], list):
    raise SystemExit('CTC original jiwer call requires a list-valued dataset column')
print('prepared two synthetic 16 kHz CTC inputs; not a LibriSpeech accuracy benchmark')
PY
}

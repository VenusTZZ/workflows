#!/usr/bin/env bash
# Optional inference dispatch. Sourced by run_example.sh; no launch side effects.

is_new_inference_entry() {
  case "$entry_key" in
    benchmarks/inference/bert-bench.py|benchmarks/inference/gpt-bench.py|\
    inference/huggingface/text-generation/ds-hf-compare.py|\
    inference/huggingface/fill-mask/test-bert.py|\
    inference/huggingface/fill-mask/test-electra.py|\
    inference/huggingface/fill-mask/test-roberta.py|\
    inference/huggingface/translation/test-t5-base.py|\
    benchmarks/opsd/benchmark_hybrid_engine_rollout.py|\
    inference/huggingface/automatic-speech-recognition/test-wav2vec2.py) return 0 ;;
    *) return 1 ;;
  esac
}

write_inference_bootstrap() {
  local output="$1"
  "$PYTHON" - "$output" <<'PY'
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(r'''import json
import math
import os
from pathlib import Path
import runpy
import sys
import torch
import torch_npu
from deepspeed.accelerator import get_accelerator

kind = os.environ['DS_INFERENCE_KIND']
source = os.environ['DS_INFERENCE_SOURCE']
if get_accelerator().device_name() != 'npu' or not torch.npu.is_available():
    raise SystemExit('inference smoke requires real NPU execution')
get_accelerator().set_device(int(os.environ.get('LOCAL_RANK', '0')))
sys.argv[0] = source
sys.path.insert(0, str(Path(source).parent))
state = runpy.run_path(source, run_name='__main__')

def require(condition, message):
    if not condition:
        raise SystemExit(message)

def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)

if kind == 'asr':
    model = state['model']
    require(next(model.parameters()).device.type == 'npu', 'CTC model weights are not NPU')
    result = state['result']
    require(len(result) == 2 and all(isinstance(x['transcription'], str) for x in result), 'CTC did not process both synthetic waveforms')
    score = state['wer'](result['text'], result['transcription'])
    require(finite(score) and score >= 0, 'CTC fixture WER must be finite, not an accuracy assertion')
    # The original maps argmax predictions; additionally reject NaN/Inf logits
    # on one original waveform without replacing its dataset/model functions.
    speech = state['librispeech_eval'][0]['speech']
    values = state['processor'](speech, sampling_rate=16000, return_tensors='pt', padding='longest').input_values
    with torch.no_grad():
        logits = model(values.to(state['device'])).logits
    require(logits.numel() > 0 and bool(torch.isfinite(logits).all()), 'CTC forward logits must be nonempty and finite')
elif kind == 'hybrid':
    flag = sys.argv.index('--output')
    result = json.loads(Path(sys.argv[flag + 1]).read_text())
    require(result['device'].startswith('npu'), 'HybridEngine ran on a non-NPU device')
    require(result['iterations'] == 2 and len(result['cases']) == 1, 'unexpected rollout case count')
    case = result['cases'][0]
    require(case['returned_response_length'] == 4, 'HybridEngine response token count mismatch')
    require(len(case['profiles']) == 2, 'HybridEngine did not complete two profiled generations')
    for profile in case['profiles']:
        for name in ('prompt_expansion_ms', 'generation_ms', 'post_processing_ms', 'total_ms', 'tokens_per_second'):
            require(finite(profile[name]) and profile[name] >= 0, f'invalid rollout metric: {name}')
else:
    pipe = state.get('pipe', state.get('translator'))
    require(pipe is not None and pipe.device.type == 'npu', 'pipeline device is not NPU')
    require(next(pipe.model.parameters()).device.type == 'npu', 'model weights are not on NPU')
    if kind in ('bert-bench', 'gpt-bench'):
        require(len(state['times']) == 4 and all(finite(x) and x > 0 for x in state['times']), 'missing/invalid four NPU timings')
        require(state['mtimes'] and all(finite(x) and x >= 0 for x in state['mtimes']), 'missing/invalid DS model timings')
        require(len(state['responses']) == 4 and all(state['responses']), 'missing benchmark predictions')
    elif kind == 'compare':
        require(state['match_count'] == 2 and state['mismatch_count'] == 0, 'HF/DS outputs must match for both CI prompts')
    elif kind == 'translation':
        require(state['translation'] and state['translation'][0].get('translation_text', '').strip(), 'empty T5 translation')
    elif kind == 'fill-mask':
        predictions = state['output']
        require(predictions and all(finite(x['score']) and x['token_str'].strip() for x in predictions), 'missing/invalid fill-mask predictions')
    else:
        raise SystemExit(f'unknown inference assertion kind: {kind}')
print(f'CI inference assertions passed: {kind}, device=npu')
''', encoding='utf-8')
PY
}

run_new_inference() {
  local kind cards=1 alias='' offset=70
  case "$entry_key" in
    benchmarks/inference/bert-bench.py)
      kind=bert-bench; require_overlay_path --model directory ;;
    benchmarks/inference/gpt-bench.py)
      kind=gpt-bench; offset=71; require_overlay_path --model directory ;;
    inference/huggingface/text-generation/ds-hf-compare.py)
      kind=compare; offset=72; require_overlay_path --model directory ;;
    inference/huggingface/fill-mask/test-bert.py)
      kind=fill-mask; alias=bert-large-cased; offset=73 ;;
    inference/huggingface/fill-mask/test-electra.py)
      kind=fill-mask; alias=google/electra-base-generator; cards=2; offset=74 ;;
    inference/huggingface/fill-mask/test-roberta.py)
      kind=fill-mask; alias=roberta-large; cards=2; offset=75 ;;
    inference/huggingface/translation/test-t5-base.py)
      kind=translation; alias=t5-base; cards=2; offset=76 ;;
    benchmarks/opsd/benchmark_hybrid_engine_rollout.py)
      kind=hybrid; offset=77; require_overlay_path --model directory ;;
    inference/huggingface/automatic-speech-recognition/test-wav2vec2.py)
      kind=asr; alias=facebook/wav2vec2-base-960h; offset=78 ;;
    *) return 1 ;;
  esac
  require_visible_devices "$cards" "$([[ "$cards" == 2 ]] && printf '0,1' || printf '0')"
  local devices bootstrap
  devices="$(first_visible_devices "$cards")"
  if [[ -n "$alias" ]]; then
    if [[ -z "${INFERENCE_CI_WORK:-}" || ! -f "$INFERENCE_CI_WORK/$alias/config.json" ]]; then
      echo "local inference alias missing for $entry_key: $alias" >&2
      exit 1
    fi
    cd "$INFERENCE_CI_WORK"
  else
    mkdir -p "$CI_OUTPUT_DIR/inference-work"
    cd "$CI_OUTPUT_DIR/inference-work"
  fi
  bootstrap="$CI_OUTPUT_DIR/inference-entry.py"
  write_inference_bootstrap "$bootstrap" || exit "$?"
  # no_local_rank works with every native entry, including those with no parser.
  # LOCAL_RANK/WORLD_SIZE are still injected by DeepSpeed's launcher.
  # Explicit || exit is essential: this function is called from an if-condition,
  # which otherwise disables bash errexit throughout the function body.
  DS_INFERENCE_KIND="$kind" DS_INFERENCE_SOURCE="$LAUNCH_PATH" \
  ASCEND_RT_VISIBLE_DEVICES="$devices" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  deepspeed --master_port "$(master_port_for "$offset")" --num_nodes 1 \
    --num_gpus "$cards" --no_local_rank "$bootstrap" "${EXTRA_ARGS[@]}" || exit "$?"
  return 0
}

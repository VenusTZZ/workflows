"""Inference recipe contracts and mock dispatch; never simulate a real NPU pass."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
SETUP = PROJECT / 'scripts/setup_inference_examples.sh'
RUN = PROJECT / 'scripts/run_inference_examples.sh'
ENTRIES = {
    'benchmarks/inference/bert-bench.py': (1, None, 'bert-bench'),
    'benchmarks/inference/gpt-bench.py': (1, None, 'gpt-bench'),
    'inference/huggingface/text-generation/ds-hf-compare.py': (1, None, 'compare'),
    'inference/huggingface/fill-mask/test-bert.py': (1, 'bert-large-cased', 'fill-mask'),
    'inference/huggingface/fill-mask/test-electra.py': (2, 'google/electra-base-generator', 'fill-mask'),
    'inference/huggingface/fill-mask/test-roberta.py': (2, 'roberta-large', 'fill-mask'),
    'inference/huggingface/translation/test-t5-base.py': (2, 't5-base', 'translation'),
    'benchmarks/opsd/benchmark_hybrid_engine_rollout.py': (1, None, 'hybrid'),
    'inference/huggingface/automatic-speech-recognition/test-wav2vec2.py': (1, 'facebook/wav2vec2-base-960h', 'asr'),
}


def bootstrap_source() -> str:
    source = RUN.read_text(encoding='utf-8')
    match = re.search(r"write_text\(r'''(.*?)''', encoding", source, re.S)
    assert match
    return match.group(1)


class InferenceContracts(unittest.TestCase):
    def test_pinned_pipeline_and_local_asset_aliases(self):
        source = SETUP.read_text(encoding='utf-8')
        self.assertIn("'transformers==4.44.2'", source)
        for _, alias, _ in ENTRIES.values():
            if alias:
                self.assertIn(alias, source)
        self.assertIn("'ELECTRA_GENERATOR_PATH=google/electra-base-generator'", source)
        self.assertNotIn('transfer_to_npu', source)
        self.assertNotIn('pip install torch', source)

    def test_every_entry_has_dispatch_and_predicate(self):
        source = RUN.read_text(encoding='utf-8')
        for entry in ENTRIES:
            self.assertGreaterEqual(source.count(entry), 2)
        self.assertIn('--no_local_rank', source)
        self.assertIn('|| exit "$?"', source)

    def test_generated_bootstrap_compiles_and_asserts_real_execution(self):
        source = bootstrap_source()
        compile(source, 'inference-entry.py', 'exec')
        self.assertIn('not torch.npu.is_available()', source)
        self.assertIn("pipe.device.type == 'npu'", source)
        self.assertIn("state['mismatch_count'] == 0", source)
        self.assertIn("len(state['times']) == 4", source)
        self.assertIn("case['returned_response_length'] == 4", source)
        self.assertIn("len(result) == 2", source)

    def test_all_setup_heredocs_compile(self):
        source = SETUP.read_text(encoding='utf-8')
        bodies = re.findall(r"<<'PY'\n(.*?)\nPY", source, re.S)
        self.assertGreaterEqual(len(bodies), 3)
        for index, body in enumerate(bodies):
            compile(body, f'inference-setup-{index}.py', 'exec')

    def test_t5_asset_metadata_is_separate_and_bounded(self):
        source = SETUP.read_text(encoding='utf-8')
        self.assertIn("source.name in ('generation_config.json', 'config.json')", source)
        self.assertIn('generation.max_new_tokens = 8', source)
        self.assertIn('config.save_pretrained(view)', source)
        self.assertIn("task['min_length'] = 0", source)
        self.assertIn("'DenseReluDense.wo'", source)

    def test_ctc_is_honest_native_fixture_not_loader_patch(self):
        source = SETUP.read_text(encoding='utf-8')
        self.assertIn("load_dataset('librispeech_asr', 'clean', split='test')", source)
        self.assertIn("'datasets==3.6.0'", source)
        self.assertIn('not a LibriSpeech accuracy benchmark', source)
        self.assertNotIn('load_dataset =', source)


class MockBootstrapAssertions(unittest.TestCase):
    """Fake framework objects only test rejection logic, not NPU compatibility."""

    def evaluate(self, kind, state):
        accelerator = SimpleNamespace(device_name=lambda: 'npu', set_device=lambda _: None)
        modules = {
            'torch': SimpleNamespace(npu=SimpleNamespace(is_available=lambda: True)),
            'torch_npu': SimpleNamespace(),
            'deepspeed': SimpleNamespace(),
            'deepspeed.accelerator': SimpleNamespace(get_accelerator=lambda: accelerator),
            'runpy': SimpleNamespace(run_path=lambda *args, **kwargs: state),
        }
        previous_path = sys.path[:]
        try:
            with patch.dict(sys.modules, modules), patch.dict(os.environ, {
                'DS_INFERENCE_KIND': kind, 'DS_INFERENCE_SOURCE': '/mock/example.py',
                'LOCAL_RANK': '0',
            }), patch.object(sys, 'argv', ['/mock/bootstrap.py']):
                exec(compile(bootstrap_source(), 'mock-inference-entry', 'exec'), {})
        finally:
            sys.path[:] = previous_path

    def pipe(self, device='npu'):
        parameter = SimpleNamespace(device=SimpleNamespace(type=device))
        return SimpleNamespace(device=SimpleNamespace(type=device),
                               model=SimpleNamespace(parameters=lambda: iter([parameter])))

    def test_cpu_pipeline_cannot_pass_as_npu(self):
        with self.assertRaisesRegex(SystemExit, 'pipeline device is not NPU'):
            self.evaluate('compare', {'pipe': self.pipe('cpu'), 'match_count': 2, 'mismatch_count': 0})

    def test_compare_mismatch_is_a_failure_even_after_successful_program_exit(self):
        with self.assertRaisesRegex(SystemExit, 'outputs must match'):
            self.evaluate('compare', {'pipe': self.pipe(), 'match_count': 1, 'mismatch_count': 1})

    def test_nonfinite_latency_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, 'NPU timings'):
            self.evaluate('gpt-bench', {'pipe': self.pipe(), 'times': [1, 1, 1, float('nan')],
                                      'mtimes': [1], 'responses': [['response']] * 4})

    def test_nonfinite_fill_mask_score_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, 'fill-mask predictions'):
            self.evaluate('fill-mask', {'pipe': self.pipe(), 'output': [{'score': float('inf'), 'token_str': 'word'}]})

    def test_valid_mock_comparison_checks_both_prompts(self):
        self.evaluate('compare', {'pipe': self.pipe(), 'match_count': 2, 'mismatch_count': 0})


@unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'requires POSIX bash')
class InferenceDispatchTests(unittest.TestCase):
    def launch(self, entry, *, insufficient=False, missing_alias=False, failure=False):
        with tempfile.TemporaryDirectory(prefix='ds inference ') as directory:
            root = Path(directory)
            output, model, work, binary = (root / name for name in ('output', 'model with spaces', 'work', 'bin'))
            for path in (output, model, work, binary):
                path.mkdir()
            cards, alias, kind = ENTRIES[entry]
            if alias and not missing_alias:
                local = work / alias
                local.mkdir(parents=True)
                (local / 'config.json').write_text('{}')
            recorder = binary / 'deepspeed'
            recorder.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
from pathlib import Path
Path(os.environ['RECORD']).write_text(json.dumps({'args':sys.argv[1:], 'cwd':os.getcwd(),
'devices':os.environ['ASCEND_RT_VISIBLE_DEVICES'], 'kind':os.environ['DS_INFERENCE_KIND'],
'source':os.environ['DS_INFERENCE_SOURCE']}))
raise SystemExit(int(os.environ.get('FAIL_LAUNCH', '0')))
''')
            recorder.chmod(0o755)
            shell = '''set -euo pipefail
source "$LIBRARY"
entry_key="$ENTRY"
LAUNCH_PATH="$MODEL/source.py"
PYTHON="$REAL_PYTHON"
EXTRA_ARGS=(--model "$MODEL")
require_overlay_path() { [[ -d "$MODEL" ]]; }
require_visible_devices() {
 local -a values; IFS=, read -r -a values <<< "$ASCEND_RT_VISIBLE_DEVICES"
 if ((${#values[@]} < $1)); then echo 'insufficient NPU devices' >&2; exit 1; fi
}
first_visible_devices() {
 local -a values; IFS=, read -r -a values <<< "$ASCEND_RT_VISIBLE_DEVICES"
 local selected="${values[0]}"; for ((i=1;i<$1;i++)); do selected+=",${values[i]}"; done
 printf '%s' "$selected"
}
master_port_for() { printf '%s' "$((21000+$1))"; }
if is_new_inference_entry; then run_new_inference; else exit 3; fi
'''
            env = dict(os.environ, LIBRARY=str(RUN), ENTRY=entry, MODEL=str(model),
                       REAL_PYTHON=sys.executable, CI_OUTPUT_DIR=str(output),
                       INFERENCE_CI_WORK=str(work), RECORD=str(root / 'record.json'),
                       ASCEND_RT_VISIBLE_DEVICES='3' if insufficient else '3,4,5,6',
                       FAIL_LAUNCH='17' if failure else '0', PATH=f'{binary}:{os.environ["PATH"]}')
            result = subprocess.run(['bash', '-c', shell], env=env, text=True, capture_output=True)
            record = json.loads((root / 'record.json').read_text()) if (root / 'record.json').exists() else None
            generated = (output / 'inference-entry.py').read_text() if (output / 'inference-entry.py').exists() else None
            return result, record, generated

    def test_all_nine_native_recipes(self):
        for entry, (cards, alias, kind) in ENTRIES.items():
            with self.subTest(entry=entry):
                result, record, generated = self.launch(entry)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(record['kind'], kind)
                self.assertEqual(record['devices'], '3,4' if cards == 2 else '3')
                self.assertEqual(record['args'][record['args'].index('--num_gpus') + 1], str(cards))
                self.assertIn('--no_local_rank', record['args'])
                self.assertEqual(generated, bootstrap_source())

    def test_two_card_recipe_rejects_insufficient_cards(self):
        result, record, _ = self.launch('inference/huggingface/fill-mask/test-electra.py', insufficient=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(record)

    def test_hardcoded_model_alias_must_exist(self):
        result, record, _ = self.launch('inference/huggingface/translation/test-t5-base.py', missing_alias=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('local inference alias missing', result.stderr)
        self.assertIsNone(record)

    def test_launch_failure_is_not_swallowed_by_dispatch(self):
        result, _, _ = self.launch('benchmarks/inference/gpt-bench.py', failure=True)
        self.assertEqual(result.returncode, 17)


if __name__ == '__main__':
    unittest.main()

"""Training recipe contracts and real shell dispatch with recording launchers.

CPU/NPU training and model downloads are not simulated as hardware successes.
"""
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
import types
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
SETUP = (PROJECT / 'scripts/setup_training_expansion.sh').read_text(encoding='utf-8')
RUN = (PROJECT / 'scripts/run_training_expansion.sh').read_text(encoding='utf-8')
MAIN = (PROJECT / 'scripts/run_example.sh').read_text(encoding='utf-8')
FINETUNE = 'training/deepspeed_finetune_demo/finetune_llama.py'
SD = 'training/stable_diffusion/train_sd_distil_lora.py'
DECODE = 'training/opsd/benchmarks/bench_decode_1p1r.py'


def function(source, name):
    match = re.search(r'^' + re.escape(name) + r'\(\) \{\n(.*?)(?=^[a-zA-Z_][a-zA-Z_0-9]*\(\) \{|\Z)', source, re.S | re.M)
    # Python dict literals inside heredocs also have a standalone closing brace.
    # The final closing brace before the next shell function belongs to bash.
    return match.group(1).rsplit('\n}', 1)[0]


def blocks(source, name):
    return re.findall(r"<<'PY'\n(.*?)\nPY", function(source, name), re.S)


class TrainingExpansionContracts(unittest.TestCase):
    def test_every_heredoc_compiles(self):
        for index, code in enumerate(re.findall(r"<<'PY'\n(.*?)\nPY", SETUP + RUN, re.S)):
            compile(code, f'training expansion block {index}', 'exec')

    def test_independent_legacy_llama_profile_and_native_reset(self):
        body = function(SETUP, 'setup_ds_finetune_demo')
        self.assertIn("'transformers==4.42.4'", body)
        self.assertIn('HuggingFaceTB/SmolLM2-135M', body)
        self.assertIn('upstream._reset_rotary_embeddings(tiny)', body)
        self.assertIn('loss.backward()', body)
        self.assertIn("config.rope_scaling is not None", body)
        self.assertIn("load_dataset(str(directory))['train']", body)
        self.assertIn("encoded['labels'][1:]", body)

    def test_sd_full_component_download_is_whitelisted(self):
        code = blocks(SETUP, 'download_sd15_ci_model')[0]
        tree = ast.parse(code)
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'patterns' for target in node.targets))
        patterns = ast.literal_eval(assignment.value)
        self.assertIn('safety_checker/model.safetensors', patterns)
        self.assertIn('unet/diffusion_pytorch_model.safetensors', patterns)
        self.assertFalse(any(pattern in {'*', '*.safetensors', '**/*.safetensors'} for pattern in patterns))
        self.assertFalse(any('.bin' in pattern or '.ckpt' in pattern for pattern in patterns))
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name in [x for x in patterns if '*' not in x] + ['tokenizer/vocab.json', 'tokenizer/merges.txt', 'tokenizer/tokenizer_config.json']:
                file = directory / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b'asset')
            module = types.ModuleType('modelscope')
            def snapshot_download(name, **kwargs):
                calls.append((name, kwargs))
                return str(directory)
            module.snapshot_download = snapshot_download
            envfile = directory / 'github-env'
            with patch.dict(sys.modules, {'modelscope': module}), patch.dict(os.environ, {'GITHUB_ENV': str(envfile)}):
                exec(compile(code, 'SD model materialization', 'exec'), {})
            self.assertEqual(calls[0][0], 'AI-ModelScope/stable-diffusion-v1-5')
            self.assertEqual(calls[0][1]['allow_patterns'], patterns)
            self.assertEqual(envfile.read_text(), f'SD15_PATH={directory}\n')

    def test_native_sd_dataset_not_api_wrapper(self):
        body = function(SETUP, 'setup_ds_sd_distil')
        self.assertIn('range(8)', body)
        self.assertIn("load_dataset('poloclub/diffusiondb', '2m_first_10k')", body)
        self.assertIn('config_name: 2m_first_10k', body)
        self.assertIn('upstream.parse_args([', body)
        self.assertIn('upstream.DreamBoothDataset(', body)
        self.assertIn('upstream.collate_fn(', body)
        self.assertNotIn('datasets.load_dataset =', SETUP)
        self.assertNotIn('monkeypatch', RUN)

    def test_sd_deepspeed_config_is_fp32_and_uses_original_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            with patch.object(sys, 'argv', ['-', str(path)]):
                exec(compile(blocks(RUN, 'run_sd_distil_ci')[0], 'SD config', 'exec'), {})
            config = json.loads(path.read_text())
            self.assertEqual(config['zero_optimization']['stage'], 2)
            self.assertFalse(config['fp16']['enabled'])
            self.assertFalse(config['bf16']['enabled'])
            self.assertEqual(config['train_batch_size'], 2)
            self.assertNotIn('optimizer', config)
            self.assertNotIn('scheduler', config)

    def test_decode_integer_device_preflight_is_normal_backend_use(self):
        self.assertIn("torch.empty(1, device='cpu').to(index)", SETUP)
        self.assertIn('torch.randint(10, 1000, (1, 2), device=index)', SETUP)
        self.assertIn('runpy.run_path(sys.argv[0], run_name="__main__")', RUN)
        self.assertNotIn('torch.cuda =', RUN + SETUP)
        self.assertNotIn('--graph-capture', RUN)

    def validate(self, name, text, *, saved=True):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / 'log'
            log.write_text(text)
            report = root / 'report.json'
            if name == 'run_sd_distil_ci':
                model = root / 'pipeline'
                if saved:
                    for name in ('model_index.json', 'unet/config.json', 'unet/diffusion_pytorch_model.safetensors'):
                        file = model / name
                        file.parent.mkdir(parents=True, exist_ok=True)
                        file.write_bytes(b'trained')
                args = ['-', str(log), str(model), str(report)]
                code = blocks(RUN, 'run_sd_distil_ci')[-1]
            else:
                args = ['-', str(log), str(report)]
                code = blocks(RUN, name)[-1]
            with patch.object(sys, 'argv', args):
                exec(compile(code, 'acceptance validator', 'exec'), {})
            return json.loads(report.read_text())

    def test_finetune_validator_requires_actual_three_losses(self):
        text = ('First batch valid shifted labels: 8, finite params: True, finite logits: True, '
                'logits abs max: 2.0, finite loss: True\n'
                'Step 1, Loss: 3.0, Time: 1ms\nStep 2, Loss: 2.0, Time: 1ms\n'
                'Step 3, Loss: 1.0, Time: 1ms\nTraining complete!\n')
        self.assertEqual(len(self.validate('run_finetune_demo_ci', text)), 3)
        for changed in (text.replace('Loss: 1.0', 'Loss: nan'), text.replace('Step 3', 'Step 4'),
                        text.replace('finite loss: True', 'finite loss: False')):
            with self.assertRaises(SystemExit):
                self.validate('run_finetune_demo_ci', changed)

    def test_sd_validator_requires_finite_progress_and_saved_pipeline(self):
        text = '\rSteps: 1/3 [00:00<00:01, loss=2.0, lr=5e-6]\rSteps: 2/3 [loss=1.5]\rSteps: 3/3 [loss=1.0]\n'
        self.assertEqual(len(self.validate('run_sd_distil_ci', text)), 3)
        for changed, saved in ((text.replace('loss=1.0', 'loss=inf'), True),
                               (text.replace('3/3', '2/3'), True), (text, False)):
            with self.assertRaises(SystemExit):
                self.validate('run_sd_distil_ci', changed, saved=saved)

    def test_decode_validator_requires_both_paths_without_speed_threshold(self):
        self.assertEqual(self.validate('run_opsd_decode_ci',
                                      'Raw decode loop: 2.0 ms\nHybridEngine rollout: 10.0 ms\n'),
                         {'Raw decode loop': 2.0, 'HybridEngine rollout': 10.0})
        for text in ('Raw decode loop: 2.0 ms\n',
                     'Raw decode loop: nan ms\nHybridEngine rollout: 1.0 ms\n',
                     'Raw decode loop: 0 ms\nHybridEngine rollout: 1.0 ms\n'):
            with self.assertRaises(SystemExit):
                self.validate('run_opsd_decode_ci', text)


@unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'real shell tests require POSIX bash')
class TrainingExpansionLaunchers(unittest.TestCase):
    def launch(self, entry, *, devices=None, fail=False, missing_data=False, dtype='no'):
        with tempfile.TemporaryDirectory(prefix='ds training recipes ') as tmp:
            root = Path(tmp)
            model = root / 'model with spaces'
            model.mkdir()
            dataset = root / 'data with spaces'
            dataset.mkdir()
            output = root / 'output'
            output.mkdir()
            example = root / 'examples' / entry
            example.parent.mkdir(parents=True)
            example.touch()
            if not missing_data:
                data = output / 'sd-distil-work/poloclub/diffusiondb/train.parquet'
                data.parent.mkdir(parents=True)
                data.write_bytes(b'setup generated fixture')
            bins = root / 'bin'
            bins.mkdir()
            record = root / 'record.jsonl'
            recorder = f'#!{sys.executable}\nREAL_PYTHON={sys.executable!r}\n' + r'''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
command = Path(sys.argv[0]).name
if command == 'python3' and (not args or args[0] != '-c'):
    os.execv(REAL_PYTHON, [REAL_PYTHON, *args])
with open(os.environ['COMMAND_RECORD'], 'a') as handle:
    handle.write(json.dumps({'command': command, 'args': args, 'cwd': os.getcwd(),
                            'devices': os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}) + chr(10))
if command == 'deepspeed':
    print('First batch valid shifted labels: 8, finite params: True, finite logits: True, logits abs max: 2.0, finite loss: True')
    for step in (1, 2, 3):
        print(f'Step {step}, Loss: {3.0 / step}, Time: 1ms')
    print('Training complete!')
elif command == 'accelerate':
    for step in (1, 2, 3):
        print(f'Steps: {step}/3 [loss={3.0 / step}]', end=chr(13))
    path = Path(args[args.index('--output_dir') + 1])
    for name in ('model_index.json', 'unet/config.json', 'unet/diffusion_pytorch_model.safetensors'):
        file = path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b'trained mock')
else:
    print('Raw decode loop: 2.0 ms\nHybridEngine rollout: 3.0 ms')
if os.environ.get('MOCK_FAIL') == '1':
    raise SystemExit(17)
'''
            for command in ('python3', 'deepspeed', 'accelerate'):
                file = bins / command
                file.write_text(recorder)
                file.chmod(0o755)
            helper_names = ('expand_overlay', 'require_visible_devices', 'first_visible_devices',
                            'master_port_for', 'require_overlay_path', 'overlay_value')
            harness = '#!/usr/bin/env bash\nset -euo pipefail\nPYTHON=python3\n'
            harness += '\n'.join(f'{name}() {{' + function(MAIN, name) + '\n}' for name in helper_names)
            harness += '\neval "EXTRA_ARGS=( $(expand_overlay) )"\n'
            harness += f'\nsource {json.dumps(str(PROJECT / "scripts/run_training_expansion.sh"))}\n'
            harness += 'is_training_expansion_entry || exit 3\nrun_training_expansion_executor\n'
            shell = root / 'harness.sh'
            shell.write_text(harness)
            overlays = {
                FINETUNE: [f'--model_name "{model}"', f'--dataset_name "{dataset}"',
                           f'--output_dir "{output / "finetune-demo"}"', '--max_steps 3'],
                SD: [f'--pretrained_model_name_or_path "{model}"', f'--output_dir "{output / "sd-distil"}"',
                     '--max_train_steps 3', f'--mixed_precision {dtype}'],
                DECODE: [f'--model "{model}"', '--prompt-len 16', '--max-new-tokens 4', '--num-warmup 1', '--num-iters 2'],
            }[entry]
            env = dict(os.environ, PATH=f'{bins}:{os.environ["PATH"]}', entry_key=entry,
                       LAUNCH_PATH=str(example), CI_OUTPUT_DIR=str(output), OVERLAY_ARGS=json.dumps(overlays),
                       COMMAND_RECORD=str(record), MOCK_FAIL=str(int(fail)))
            env.pop('ASCEND_RT_VISIBLE_DEVICES', None)
            if devices is not None:
                env['ASCEND_RT_VISIBLE_DEVICES'] = devices
            result = subprocess.run(['bash', str(shell)], env=env, capture_output=True, text=True)
            commands = [json.loads(line) for line in record.read_text().splitlines()] if record.exists() else []
            config = output / ('sd-distil-config.json' if entry == SD else 'finetune-demo-config.json')
            return result, commands, json.loads(config.read_text()) if config.exists() else None

    def test_finetune_real_launcher_and_visible_subset(self):
        result, commands, config = self.launch(FINETUNE, devices='4,6,7')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0]['command'], 'deepspeed')
        self.assertEqual(commands[0]['devices'], '4,6')
        self.assertIn('--deepspeed_config', commands[0]['args'])
        self.assertNotIn('--no_local_rank', commands[0]['args'])
        self.assertEqual(config['zero_optimization']['stage'], 2)

    def test_sd_uses_accelerate_deepspeed_fp32_and_native_data_cwd(self):
        result, commands, config = self.launch(SD)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(commands[0]['command'], 'accelerate')
        self.assertIn('--use_deepspeed', commands[0]['args'])
        self.assertEqual(commands[0]['args'][commands[0]['args'].index('--mixed_precision') + 1], 'no')
        self.assertTrue(commands[0]['cwd'].endswith('/sd-distil-work'))
        self.assertFalse(config['bf16']['enabled'])

    def test_decode_runs_normal_runpy_and_selects_one_card(self):
        result, commands, _ = self.launch(DECODE, devices='4,6')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(commands[0]['devices'], '4')
        self.assertIn('import torch, torch_npu', commands[0]['args'][1])
        self.assertIn('runpy.run_path', commands[0]['args'][1])

    def test_multicard_recipes_reject_insufficient_devices(self):
        for entry in (FINETUNE, SD):
            result, commands, _ = self.launch(entry, devices='0')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('insufficient NPU devices', result.stderr)
            self.assertFalse(commands)

    def test_sd_missing_local_fixture_stops_before_launch(self):
        result, commands, _ = self.launch(SD, missing_data=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('dataset missing', result.stderr)
        self.assertFalse(commands)

    def test_sd_rejects_teacher_dtype_mismatch_before_launch(self):
        result, commands, _ = self.launch(SD, dtype='bf16')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires the FP32', result.stderr)
        self.assertFalse(commands)

    def test_launcher_exit_codes_not_hidden_by_log_capture(self):
        for entry in (FINETUNE, SD, DECODE):
            result, commands, _ = self.launch(entry, fail=True)
            self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
            self.assertEqual(len(commands), 1)


if __name__ == '__main__':
    unittest.main()

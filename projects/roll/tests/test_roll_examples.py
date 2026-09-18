"""Project-level tests for projects/roll: manifest ledger, fixture schema, CI config constraints."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import unittest

import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent.parent.parent
PROJECT = REPO / 'projects' / 'roll'
ENGINE_SCRIPT = REPO / 'scripts' / 'check_supported_entries.py'

# Pinned upstream snapshot used for the ledger assertion.
# Local-only conformance checkout (git clone --depth 1 --filter=blob:none
# --sparse https://github.com/alibaba/ROLL.git + sparse-checkout examples).
# Ledger and engine-check tests require it; they skip when absent.
UPSTREAM_EXAMPLES = pathlib.Path('F:/work/tmp/ROLL-plan-2/examples')
HAVE_UPSTREAM = UPSTREAM_EXAMPLES.is_dir()

SUPPORTED = {
    'examples/qwen2.5-0.5B-agentic/agentic_rollout_sokoban.yaml',
}
CHECKOUT_EXCLUDED = {
    'examples/qwen2.5-0.5B-agentic/agentic_val_webshop.yaml',
}
FORBIDDEN = ['/data/oss_', '/data/cpfs_', '/home/', 'megatron', 'sglang',
             'nccl', 'cuda', 'flash_attn']


def load_manifest():
    return yaml.safe_load(
        (PROJECT / 'examples_manifest.yaml').read_text(encoding='utf-8'))


class RollProjectTests(unittest.TestCase):
    @unittest.skipUnless(
        HAVE_UPSTREAM,
        'upstream ROLL examples checkout not present; ledger test skipped')
    def test_manifest_ledger_matches_upstream_snapshot(self) -> None:
        manifest = load_manifest()
        all_yaml = {
            str(p.relative_to(UPSTREAM_EXAMPLES.parent)).replace('\\', '/')
            for p in UPSTREAM_EXAMPLES.rglob('*.yaml')
        }
        self.assertEqual(len(all_yaml), 117)
        excluded = set(CHECKOUT_EXCLUDED)
        excluded |= {
            p for p in all_yaml if p.startswith('examples/config/')
        }
        candidates = all_yaml - excluded
        self.assertEqual(len(candidates), 107)
        supported = [entry['path'] for entry in manifest['supported']]
        unsupported = list(manifest['unsupported'])
        self.assertEqual(set(supported) - candidates, set())
        self.assertEqual(set(unsupported) - candidates, set())
        self.assertEqual(candidates - set(supported) - set(unsupported), set())
        self.assertEqual(candidates, set(supported) | set(unsupported))
        self.assertEqual(len(supported), 1)
        self.assertEqual(len(unsupported), 106)

    def test_manifest_scan_reflects_ledger_semantics(self) -> None:
        manifest = load_manifest()
        self.assertEqual(manifest['scan']['root'], 'examples')
        self.assertIn('.yaml', manifest['scan']['include_extensions'])
        self.assertIn('examples/config', manifest['scan']['exclude'])
        self.assertIn(
            'examples/qwen2.5-0.5B-agentic/agentic_val_webshop.yaml',
            manifest['scan']['exclude'],
        )

    def test_supported_shape(self) -> None:
        manifest = load_manifest()
        for entry in manifest['supported']:
            for field in ('path', 'profile', 'runner', 'image', 'exec',
                          'overlay_args', 'timeout_minutes'):
                self.assertTrue(entry.get(field), (entry['path'], field))
            self.assertEqual(
                entry['image'],
                'swr.cn-south-1.myhuaweicloud.com/ascendhub/'
                'cann:9.1.0-910b-ubuntu22.04-py3.12',
                entry['path'])
            self.assertIn('.yaml', entry['path'])

    @unittest.skipUnless(
        HAVE_UPSTREAM,
        'upstream ROLL examples checkout not present; engine test skipped')
    def test_engine_manifest_check_runs(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ENGINE_SCRIPT),
             '--target-root', str(UPSTREAM_EXAMPLES.parent),
             '--manifest', str(PROJECT / 'examples_manifest.yaml')],
            capture_output=True, text=True, check=False)
        self.assertEqual(
            result.returncode, 0,
            f'engine check failed: {result.stderr}')
        self.assertIn('manifest ok: 1 supported entry(ies)', result.stdout)

    def test_fixture_schema(self) -> None:
        rows = [
            json.loads(line)
            for line in (PROJECT / 'fixtures' / 'ci_math_8.jsonl')
            .read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({row['id'] for row in rows}), 8)
        for row in rows:
            self.assertEqual(
                set(row),
                {'id', 'source', 'difficulty', 'prompt', 'messages',
                 'ground_truth', 'case_type', 'test_case_function',
                 'test_cases', 'tag'})
            self.assertEqual(row['tag'], 'math_rule')
            messages = json.loads(row['messages'])
            self.assertEqual([m['role'] for m in messages],
                             ['system', 'user'])
            self.assertIn('\\boxed{}', messages[0]['content'])

    def test_ci_config_constraints(self) -> None:
        for name in ('ci_agentic_rollout',):
            path = PROJECT / 'configs' / f'{name}.yaml'
            text = path.read_text(encoding='utf-8')
            self.assertNotIn('${CI_OUTPUT_DIR}', text, name)
            self.assertNotIn('${FIXTURE_DIR}', text, name)
            cfg = yaml.safe_load(text)
            self.assertEqual(cfg['max_steps'], 1, name)
            lowered = text.lower()
            for token in FORBIDDEN:
                self.assertNotIn(token, lowered, (name, token))
            self.assertIn('vllm', lowered, name)
            self.assertIn('ROLL_MODEL_PATH', text, name)
            self.assertIn('${oc.env:CI_OUTPUT_DIR}', text, name)

        rollout = yaml.safe_load(
            (PROJECT / 'configs/ci_agentic_rollout.yaml').read_text(
                encoding='utf-8'))
        self.assertEqual(rollout['num_gpus_per_node'], 1)
        self.assertNotIn('actor_train', rollout)
        self.assertEqual(rollout['actor_infer']['strategy_args']
                         ['strategy_name'], 'vllm')

    def test_phase_one_setup_uses_domestic_runtime_sources(self) -> None:
        text = (PROJECT / 'scripts/setup_example.sh').read_text(
            encoding='utf-8')
        for token in (
            'torch==2.10.0',
            'torch-npu==2.10.0.post4',
            'vllm==0.23.0',
            'vllm-ascend==0.23.0rc1',
            'triton-ascend==3.2.1',
            'modelscope==1.37.0',
            'reasoning-gym==0.1.23',
            'repo.huaweicloud.com/repository/pypi/simple',
        ):
            self.assertIn(token, text)
        self.assertIn(
            'ms_download_model "Qwen/Qwen2.5-0.5B-Instruct"', text)
        self.assertNotIn('quay.io', text)

    def test_run_entry_clears_vllm_incompatible_allocator(self) -> None:
        text = (PROJECT / 'scripts/run_example.sh').read_text(
            encoding='utf-8')
        self.assertIn('unset PYTORCH_NPU_ALLOC_CONF', text)


if __name__ == '__main__':
    unittest.main()

"""Static checks; these do not execute NPU training."""
from pathlib import Path
import re
import unittest
import yaml

PROJECT = Path(__file__).resolve().parents[1]


class AuditContractTests(unittest.TestCase):
    def test_five_recipes_and_only_three_real_unsupported_entries(self):
        manifest = yaml.safe_load((PROJECT / 'examples_manifest.yaml').read_text(encoding='utf-8'))
        self.assertEqual(len(manifest['supported']), 5)
        by_path = {entry['path']: entry for entry in manifest['supported']}
        orpo = by_path['examples/alignment/run_orpo.py']
        self.assertEqual(orpo['runner'], 'linux-aarch64-a2-2')
        self.assertEqual(orpo['profile'], 'liger_orpo')
        self.assertEqual(orpo['overlay_args'], [])
        self.assertEqual(set(manifest['unsupported']), {
            'examples/lightning/training.py', 'examples/megatron/run_mode1_monkey_patch.py',
            'examples/megatron/run_mode2_hand_spec.py'})

    def test_orpo_preserves_fixed_steps_and_uses_real_fsdp(self):
        run = (PROJECT / 'scripts/run_orpo.sh').read_text(encoding='utf-8')
        self.assertIn('trainer.state.global_step != 100', run)
        self.assertIn('FullyShardedDataParallel', run)
        self.assertIn('trainer.args.device.type != "npu"', run)
        self.assertIn("config['num_processes'] = 2", run)
        self.assertIn('TORCH_COMPILE_DISABLE=1', run)
        self.assertNotIn('torch.compile =', run)
        self.assertNotIn('transfer_to_npu', run)
        self.assertNotIn('ORPOConfig(', run)

    def test_local_loader_and_no_foreign_hub_fallback(self):
        setup = (PROJECT / 'scripts/setup_example.sh').read_text(encoding='utf-8')
        new = setup.split('setup_liger_orpo()', 1)[1].split('supported_profiles()', 1)[0]
        self.assertIn('LLM-Research/Llama-3.2-1B-Instruct', new)
        self.assertIn("load_dataset('trl-lib/tldr-preference', split='train')", new)
        self.assertIn("len(data) != 8", new)
        self.assertNotIn('sitecustomize', new)
        run = (PROJECT / 'scripts/run_orpo.sh').read_text(encoding='utf-8')
        self.assertIn('HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1', run)

    def test_embedded_python_compiles(self):
        for path in (PROJECT / 'scripts').glob('*.sh'):
            for index, block in enumerate(re.findall(r"<<'PY'\n(.*?)\nPY", path.read_text(encoding='utf-8'), re.S)):
                compile(block, f'{path.name}:{index}', 'exec')


if __name__ == '__main__':
    unittest.main()

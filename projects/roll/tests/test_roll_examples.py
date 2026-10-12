"""Project-level tests for projects/roll: manifest ledger, fixture schema, CI config constraints."""
from __future__ import annotations

import json
import importlib.util
import os
import pathlib
import subprocess
import sys
import unittest
import tempfile
from unittest.mock import patch

import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent.parent.parent
PROJECT = REPO / 'projects' / 'roll'
ENGINE_SCRIPT = REPO / 'scripts' / 'check_supported_entries.py'

# 分类账本使用历史快照，配置测试使用受测 release；通过环境变量分别指定。
UPSTREAM_EXAMPLES = pathlib.Path(os.environ.get('ROLL_LEDGER_ROOT', '')) / 'examples'
HAVE_UPSTREAM = bool(os.environ.get('ROLL_LEDGER_ROOT')) and UPSTREAM_EXAMPLES.is_dir()
CONFIG_ROOT = pathlib.Path(os.environ.get('ROLL_UPSTREAM_ROOT', ''))
HAVE_CONFIG = (bool(os.environ.get('ROLL_UPSTREAM_ROOT'))
               and (CONFIG_ROOT / 'examples').is_dir()
               and importlib.util.find_spec('hydra') is not None)

SUPPORTED = {
    'examples/qwen2.5-0.5B-agentic/agentic_rollout_sokoban.yaml',
    'examples/qwen2.5-0.5B-agentic/agentic_val_sokoban.yaml',
    'examples/ascend_examples/qwen3_8b_rlvr_fsdp2.yaml',
    'examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake.yaml',
    'examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake-pg_var.yaml',
    'examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake-pg_var_is_correct.yaml',
    'examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake_gigpo.yaml',
    'examples/qwen2.5-0.5B-agentic/agentic_val_sokoban_gigpo.yaml',
    'examples/qwen2.5-0.5B-agentic/agentic_val_sokoban_lora.yaml',
    'examples/qwen2.5-0.5B-agentic/agentic_val_sokoban_dynamic_batching.yaml',
    'examples/qwen2.5-0.5B-agentic/agentic_val_sokoban_native.yaml',
    'examples/qwen2.5-0.5B-agentic/agentic_val_sokoban_ppo.yaml',
    'examples/qwen2.5-0.5B-agentic/agentic_val_sokoban_sao.yaml',
    'examples/qwen2.5-0.5B-agentic/agentic_val_sokoban_agent_runner.yaml',
    'examples/docs_examples/example_grpo.yaml',
    'examples/docs_examples/example_gspo.yaml',
    'examples/docs_examples/example_ppo.yaml',
    'examples/docs_examples/example_topr.yaml',
    'examples/qwen2.5-7B-rlvr_megatron/rlvr_config_dynamic_batching.yaml',
    'examples/qwen2.5-7B-rlvr_megatron/rlvr_lora_fsdp2.yaml',
}


def load_manifest():
    return yaml.safe_load(
        (PROJECT / 'examples_manifest.yaml').read_text(encoding='utf-8'))


def load_ci_config(name):
    from omegaconf import OmegaConf
    from projects.roll.scripts.prepare_config import compose_config

    profile = {'ci_agentic_rollout': 'agentic_rollout_npu',
               'ci_agentic_train': 'agentic_train_npu',
               'ci_rlvr': 'rlvr_npu'}[name]
    entry = next(e for e in load_manifest()['supported'] if e['profile'] == profile)
    args = entry['overlay_args']
    with patch.dict(os.environ, {'CI_OUTPUT_DIR': '/ci/output',
                                'ROLL_MODEL_PATH': '/ci/models/qwen',
                                'FIXTURE_DIR': str(PROJECT / 'fixtures')}):
        cfg = compose_config(CONFIG_ROOT, args[1], args[3], args[4:])
        return OmegaConf.to_container(cfg, resolve=True)


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
        self.assertEqual(len(all_yaml), 146)
        # 共享配置由一个目录条目登记，WebShop 则仍单独归入 unsupported。
        fragments = {p for p in all_yaml if p.startswith('examples/config/')}
        self.assertEqual(len(fragments), 9)
        candidates = all_yaml - fragments
        self.assertEqual(len(candidates), 137)
        supported = [entry['path'] for entry in manifest['supported']]
        unsupported = list(manifest['unsupported'])
        self.assertIn('examples/config', unsupported)
        unsupported_files = [p for p in unsupported if p != 'examples/config']
        for path in supported + unsupported_files:
            self.assertFalse(
                path.startswith('examples/config/'),
                f'{path}: examples/config/* must remain covered by the '
                'directory-level aggregate entry, not re-enumerated',
            )
        self.assertEqual(set(supported) - candidates, set())
        self.assertEqual(set(unsupported_files) - candidates, set())
        self.assertEqual(candidates, set(supported) | set(unsupported_files))
        self.assertEqual(len(supported), 20)
        self.assertEqual(len(unsupported), 118)

    def test_manifest_scan_reflects_ledger_semantics(self) -> None:
        manifest = load_manifest()
        self.assertEqual(manifest['scan']['root'], 'examples')
        self.assertEqual(
            set(manifest['scan'].keys()),
            {'root', 'include_extensions'},
            'scan schema is now {root, include_extensions}; exclude was '
            'retired 2026-09-20',
        )
        self.assertIn('.yaml', manifest['scan']['include_extensions'])
        # The two former scan.exclude items are now in `unsupported`
        # ledger entries (directory aggregate for examples/config + the
        # single webshop file).
        unsupported = list(manifest['unsupported'])
        self.assertIn('examples/config', unsupported)
        self.assertIn(
            'examples/qwen2.5-0.5B-agentic/agentic_val_webshop.yaml',
            unsupported,
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
        self.assertIn('manifest ok: 20 supported entry(ies)', result.stdout)

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

    @unittest.skipUnless(HAVE_CONFIG, 'set ROLL_UPSTREAM_ROOT and install hydra-core')
    def test_ci_config_constraints(self) -> None:
        # 所有 supported 条目的 overlay 都必须能被 v0.4.0 树真实 compose。
        from omegaconf import OmegaConf
        from projects.roll.scripts.prepare_config import compose_config
        expected_adv = {
            'agent_val_frozen_lake': 'grpo',
            'agent_val_frozen_lake-pg_var': 'grpo',
            'agent_val_frozen_lake-pg_var_is_correct': 'grpo',
            'agent_val_frozen_lake_gigpo': 'gigpo',
            'agentic_val_sokoban_gigpo': 'gigpo',
            'agentic_val_sokoban_lora': 'grpo',
            'agentic_val_sokoban_dynamic_batching': 'grpo',
            'agentic_val_sokoban_native': 'step_reinforce',
            'agentic_val_sokoban_ppo': 'gae',
            'agentic_val_sokoban_sao': 'skip_obs_gae',
            'agentic_val_sokoban_agent_runner': 'grpo',
        }
        for entry in load_manifest()['supported']:
            name = pathlib.PurePosixPath(entry['path']).stem
            if name not in expected_adv:
                continue
            with self.subTest(config=name):
                args = entry['overlay_args']
                with patch.dict(os.environ, {'CI_OUTPUT_DIR': '/ci/output',
                                            'ROLL_MODEL_PATH': '/ci/models/qwen',
                                            'FIXTURE_DIR': str(PROJECT / 'fixtures')}):
                    cfg = compose_config(CONFIG_ROOT, args[1], args[3], args[4:])
                    resolved = OmegaConf.to_container(cfg, resolve=True)
                self.assertEqual(resolved['max_steps'], 1)
                self.assertEqual(resolved['pretrain'], '/ci/models/qwen')
                self.assertEqual(resolved['num_gpus_per_node'], 2)
                self.assertEqual(resolved['track_with'], 'stdout')
                self.assertIsNone(resolved['transfer_backend']['backend_name'])
                self.assertEqual(resolved['adv_estimator'], expected_adv[name])
                self.assertEqual(
                    resolved['actor_train']['strategy_args']['strategy_name'],
                    'fsdp2_train')
                self.assertEqual(
                    resolved['actor_train']['device_mapping'], 'list(range(0,1))')
                self.assertEqual(
                    resolved['actor_infer']['device_mapping'], 'list(range(1,2))')
                if name in ('agentic_val_sokoban_ppo', 'agentic_val_sokoban_sao'):
                    self.assertEqual(
                        resolved['critic']['device_mapping'], 'list(range(0,1))')
                    self.assertEqual(resolved['critic_warmup'], 0)
                    self.assertEqual(resolved['critic_epochs'], 1)
                if name == 'agentic_val_sokoban_lora':
                    self.assertEqual(
                        resolved['actor_train']['model_args']['lora_target'],
                        'all-linear')
                if name == 'agentic_val_sokoban_dynamic_batching':
                    self.assertTrue(resolved['actor_train']['use_dynamic_batching_in_train'])
                    self.assertEqual(
                        resolved['reference']['strategy_args']['strategy_name'],
                        'hf_infer')
                if name == 'agentic_val_sokoban_native':
                    self.assertIsNone(resolved['async_generation_ratio'])
                    self.assertEqual(
                        resolved['train_env_manager']['tags'], ['SokobanNativeEnv'])
                if name == 'agentic_val_sokoban_agent_runner':
                    self.assertEqual(
                        resolved['custom_envs']['SimpleSokoban']['agent_runner_cls'],
                        'roll.pipeline.agentic.agent_runner.gem_runner.GEMRunner')
                serialized = json.dumps(resolved)
                for forbidden in ('/data/oss_', '/data/cpfs_', 'megatron_train'):
                    self.assertNotIn(forbidden, serialized)
        rollout = load_ci_config('ci_agentic_rollout')
        self.assertEqual(rollout['num_gpus_per_node'], 1)
        self.assertNotIn('actor_train', rollout)
        self.assertEqual(rollout['rollout_batch_size'], 1)
        self.assertEqual(rollout['max_actions_per_traj'], 2)
        train = load_ci_config('ci_agentic_train')
        self.assertEqual(train['num_gpus_per_node'], 2)
        self.assertEqual(train['actor_train']['strategy_args']['strategy_name'],
                         'fsdp2_train')
        self.assertEqual(train['actor_train']['strategy_args']['strategy_config'],
                         {'fsdp_size': 1, 'param_dtype': 'bf16',
                          'reduce_dtype': 'bf16', 'reshard_after_forward': True,
                          'offload_policy': False})
        self.assertEqual(train['actor_train']['device_mapping'], 'list(range(0,1))')
        self.assertEqual(train['actor_infer']['device_mapping'], 'list(range(1,2))')
        self.assertEqual(train['reference']['device_mapping'], 'list(range(0,1))')
        self.assertEqual(train['train_env_manager']['group_size'], 2)
        rlvr = load_ci_config('ci_rlvr')
        self.assertEqual(rlvr['num_gpus_per_node'], 4)
        self.assertEqual(rlvr['actor_train']['device_mapping'], 'list(range(0,2))')
        self.assertEqual(rlvr['actor_infer']['device_mapping'], 'list(range(2,3))')
        self.assertEqual(rlvr['reference']['device_mapping'], 'list(range(3,4))')
        self.assertEqual(rlvr['actor_train']['strategy_args']['strategy_config']
                         ['fsdp_size'], 2)
        self.assertEqual(rlvr['rewards']['math_rule']['world_size'], 1)
        self.assertEqual(rlvr['actor_train']['data_args']['file_name'],
                         [str(PROJECT / 'fixtures') + '/ci_math_8.jsonl'])
        self.assertNotIn('dataset_dir', rlvr['actor_train']['data_args'])
        self.assertNotIn('validation', rlvr)
        self.assertNotIn('response_length', rlvr)
        for section in ('actor_train', 'actor_infer', 'reference'):
            self.assertEqual(rlvr[section]['model_args']['flash_attn'], 'fa2')
        for cfg in (rollout, train, rlvr):
            self.assertEqual(cfg['max_steps'], 1)
            self.assertEqual(cfg['pretrain'], '/ci/models/qwen')
            self.assertEqual(cfg['track_with'], 'stdout')
            self.assertEqual(cfg['output_dir'], '/ci/output/output')
            text = json.dumps(cfg)
            for forbidden in ('/data/oss_', '/data/cpfs_', 'megatron_train'):
                self.assertNotIn(forbidden, text)
            self.assertEqual(cfg['actor_infer']['strategy_args']['strategy_config']
                             ['max_model_len'], 256)
        for cfg in (rollout, train):
            self.assertEqual(set(cfg['custom_envs']), {'SimpleSokoban'})
            self.assertEqual(cfg['custom_envs']['SimpleSokoban']['max_steps'], 2)
            self.assertEqual(cfg['custom_envs']['SimpleSokoban']
                             ['max_tokens_per_step'], 32)
            self.assertFalse(cfg['render_save_dir'])
        for cfg in (train, rlvr):
            self.assertEqual(cfg['eval_steps'], 1000)
            self.assertEqual(cfg['actor_train']['training_args']
                             ['gradient_accumulation_steps'], 1)

    def test_manifest_selects_upstream_configs(self) -> None:
        self.assertEqual({e['path'] for e in load_manifest()['supported']}, SUPPORTED)
        for entry in load_manifest()['supported']:
            args = entry['overlay_args']
            self.assertEqual(args[0], '--config_path')
            self.assertEqual(args[2], '--config_name')
            selected = pathlib.PurePosixPath('examples') / args[1] / (args[3] + '.yaml')
            self.assertEqual(str(selected), entry['path'])
            self.assertIn('++max_steps=1', args)
            self.assertIn('++transfer_backend.backend_name=null', args)
        self.assertEqual(list((PROJECT / 'configs').glob('*.yaml')), [])
        setup = (PROJECT / 'scripts/setup_example.sh').read_text(encoding='utf-8')
        self.assertNotIn('prepare_ci_configs', setup)

    @unittest.skipUnless(HAVE_CONFIG, 'set ROLL_UPSTREAM_ROOT and install hydra-core')
    def test_added_rlvr_configs_preserve_algorithm_features(self) -> None:
        from omegaconf import OmegaConf
        from projects.roll.scripts.prepare_config import compose_config
        for entry in load_manifest()['supported']:
            if entry['profile'] != 'rlvr_npu' or '/ascend_examples/' in entry['path']:
                continue
            args = entry['overlay_args']
            name = pathlib.PurePosixPath(entry['path']).stem
            with self.subTest(config=name), patch.dict(os.environ, {
                'CI_OUTPUT_DIR': '/ci/output', 'ROLL_MODEL_PATH': '/ci/models/qwen',
                'FIXTURE_DIR': str(PROJECT / 'fixtures'),
            }):
                cfg = OmegaConf.to_container(
                    compose_config(CONFIG_ROOT, args[1], args[3], args[4:]), resolve=True)
                self.assertEqual(cfg['max_steps'], 1)
                self.assertEqual(cfg['num_gpus_per_node'], 4)
                self.assertEqual(set(cfg['rewards']), {'math_rule'})
                self.assertEqual(cfg['actor_train']['data_args']['domain_interleave_probs'], {'math_rule': 1.0})
                self.assertEqual(cfg['actor_train']['strategy_args']['strategy_name'], 'fsdp2_train')
                self.assertEqual(cfg['reference']['strategy_args']['strategy_name'], 'hf_infer')
                self.assertIsNone(cfg['reference']['strategy_args']['strategy_config'])
                self.assertEqual(cfg['pretrain'], '/ci/models/qwen')
                self.assertNotIn('validation', cfg)
                if name == 'example_gspo':
                    self.assertEqual(cfg['importance_sampling'], 'seq')
                if name in ('example_ppo', 'example_topr'):
                    self.assertEqual(cfg['adv_estimator'], 'gae')
                    self.assertEqual(cfg['critic_warmup'], 0)
                    self.assertEqual(cfg['critic']['device_mapping'], 'list(range(0,2))')
                if name == 'example_topr':
                    self.assertEqual(cfg['actor_train']['worker_cls'], 'roll.pipeline.rlvr.actor_pg_worker.ActorPGWorker')
                    self.assertEqual(cfg['actor_train']['pg_variant'], 'topr')
                    self.assertEqual(cfg['actor_train']['topr_positive_weight'], 1.0)
                    self.assertEqual(cfg['actor_train']['topr_negative_weight'], 1.0)
                    self.assertEqual(cfg['postive_loss_coef'], 1.0)
                    self.assertEqual(cfg['use_topr_neg_loss_coef'], 1.0)
                    self.assertEqual(cfg['rl_loss_coef'], 1.0)
                    self.assertNotIn('positive_loss_coef', cfg)
                if name == 'rlvr_config_dynamic_batching':
                    self.assertTrue(cfg['actor_train']['use_dynamic_batching_in_train'])
                    self.assertTrue(cfg['reference']['use_dynamic_batching_in_infer'])
                if name == 'rlvr_config_sequence_packing':
                    self.assertTrue(cfg['actor_train']['use_sequence_packing'])
                    self.assertTrue(cfg['reference']['use_sequence_packing'])
                if name == 'rlvr_lora_fsdp2':
                    self.assertEqual(cfg['adv_estimator'], 'reinforce')
                    self.assertEqual(cfg['actor_train']['model_args']['lora_rank'], 32)
                    self.assertEqual(cfg['actor_infer']['model_args']['lora_rank'], 32)
                for forbidden in ('/data/oss_', '/data/cpfs_', 'megatron_train', 'megatron_infer'):
                    self.assertNotIn(forbidden, json.dumps(cfg))

    @unittest.skipUnless(importlib.util.find_spec('hydra'), 'install hydra-core')
    def test_overlay_preserves_hydra_types_quotes_and_interpolation(self) -> None:
        from projects.roll.scripts.prepare_config import expand_overlay
        items = ['--config_path recipe', '--config_name', 'original',
                 '++device_mapping="list(range(0,1))"',
                 '++file_name=["${oc.env:FIXTURE_DIR}/ci_math_8.jsonl"]',
                 '++pretrain="${oc.env:ROLL_MODEL_PATH}"',
                 '~validation']
        tokens = expand_overlay(json.dumps(items))
        self.assertEqual(tokens[:4], ['--config_path', 'recipe', '--config_name', 'original'])
        self.assertEqual(tokens[4:], items[3:])
        for bad in ('{}', '[42]', '[""]', 'not json'):
            with self.subTest(raw=bad), self.assertRaises(ValueError):
                expand_overlay(bad)

    @unittest.skipUnless(importlib.util.find_spec('hydra'), 'install hydra-core')
    def test_config_adapter_preserves_launcher_contract_and_exit_code(self) -> None:
        from projects.roll.scripts.prepare_config import main
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            examples = root / 'examples'
            upstream = examples / 'recipe'
            upstream.mkdir(parents=True)
            original = upstream / 'original.yaml'
            original.write_text('max_steps: 100\ndevice_mapping: "list(range(0,8))"\n',
                                encoding='utf-8')
            before = original.read_bytes()
            launcher = examples / 'start.py'
            # 模拟上游仅接受两个 config 参数的接口，实际启动独立 Python 进程。
            launcher.write_text(
                'import argparse\n'
                'from hydra import compose, initialize\n'
                'p=argparse.ArgumentParser()\n'
                'p.add_argument("--config_path")\n'
                'p.add_argument("--config_name")\n'
                'a=p.parse_args()\n'
                'with initialize(config_path=a.config_path, version_base="1.1"):\n'
                ' c=compose(config_name=a.config_name)\n'
                ' assert c.max_steps == 1\n'
                ' assert c.device_mapping == "list(range(0,1))"\n'
                'raise SystemExit(7)\n', encoding='utf-8')
            with patch.dict(os.environ, {'TARGET_ROOT': str(root),
                                        'CI_OUTPUT_DIR': str(root / 'output')}):
                status = main(['--launcher', 'examples/start.py',
                               '--config_path', 'recipe', '--config_name', 'original',
                               '++max_steps=1',
                               '++device_mapping="list(range(0,1))"'])
            self.assertEqual(status, 7)
            self.assertEqual(before, original.read_bytes())
            self.assertFalse(list(examples.glob('.ci-roll-*')))
            resolved = yaml.safe_load(
                (root / 'output/resolved_config.yaml').read_text(encoding='utf-8'))
            self.assertEqual(resolved['max_steps'], 1)

    @unittest.skipUnless(importlib.util.find_spec('hydra'), 'install hydra-core')
    def test_config_adapter_rejects_paths_outside_examples(self) -> None:
        from projects.roll.scripts.prepare_config import compose_config
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                compose_config(pathlib.Path(tmp), '../outside', 'config', [])

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

    @unittest.skipUnless(HAVE_CONFIG, 'set ROLL_UPSTREAM_ROOT and install hydra-core')
    def test_npu_configs_disable_remote_batch_transfer(self) -> None:
        # v0.4.0 enables TransferQueue by default, while DataProto rejects
        # RemoteBatch on NPU. A null backend name selects DummyClient and
        # makes DataProto.to_remote return the original local batch.
        for name in ('ci_agentic_rollout', 'ci_agentic_train', 'ci_rlvr'):
            with self.subTest(config=name):
                cfg = load_ci_config(name)
                self.assertIn('transfer_backend', cfg)
                self.assertIn('backend_name', cfg['transfer_backend'])
                self.assertIsNone(cfg['transfer_backend']['backend_name'])

    @unittest.skipUnless(HAVE_CONFIG, 'set ROLL_UPSTREAM_ROOT and install hydra-core')
    def test_rlvr_fixture_tags_are_routed_to_configured_domains(self) -> None:
        cfg = load_ci_config('ci_rlvr')
        tag_to_domain = {}
        for domain, reward in cfg['rewards'].items():
            for tag in reward['tag_included']:
                self.assertNotIn(tag, tag_to_domain, 'ambiguous reward tag')
                tag_to_domain[tag] = domain
        rows = [json.loads(line) for line in
                (PROJECT / 'fixtures/ci_math_8.jsonl').read_text(
                    encoding='utf-8').splitlines() if line.strip()]
        domain_counts = dict.fromkeys(
            cfg['actor_train']['data_args']['domain_interleave_probs'], 0)
        for row in rows:
            self.assertIn(row['tag'], tag_to_domain, row['id'])
            domain = tag_to_domain[row['tag']]
            self.assertIn(domain, domain_counts, row['id'])
            domain_counts[domain] += 1
        for domain, count in domain_counts.items():
            self.assertGreater(count, 0, f'domain dataset {domain} has no data')

    def test_setup_injects_device_lists_per_profile(self) -> None:
        text = (PROJECT / 'scripts/setup_example.sh').read_text(
            encoding='utf-8')
        self.assertIn('agentic_train_npu', text)
        self.assertIn('rlvr_npu', text)
        self.assertIn('ASCEND_RT_VISIBLE_DEVICES="0,1"', text)
        self.assertIn('ASCEND_RT_VISIBLE_DEVICES="0,1,2,3"', text)
        self.assertIn(
            'echo "ASCEND_RT_VISIBLE_DEVICES=', text)

    def test_actionlint_registers_four_card_runner(self) -> None:
        text = (REPO / '.github' / 'actionlint.yaml').read_text(
            encoding='utf-8')
        self.assertIn('linux-aarch64-a2-4', text)

    def test_thin_trigger_calls_examples_engine(self) -> None:
        text = (REPO / '.github/workflows/roll-examples.yml').read_text(
            encoding='utf-8')
        self.assertIn('uses: ./.github/workflows/examples-template.yml', text)
        self.assertIn('project: roll', text)
        self.assertIn('upstream_repo: alibaba/ROLL', text)


if __name__ == '__main__':
    unittest.main()

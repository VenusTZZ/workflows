"""Tests for scripts/write_example_result.py (result publishing)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent.parent / 'scripts'
sys.path.insert(0, str(SCRIPTS))

import write_example_result

ENV = {
    'GH_TOKEN': 't',
    'GITHUB_REPOSITORY': 'org/workflows',
    'GITHUB_RUN_ID': '123',
    'EXPECTED_JOB_NAME': 'run-example (examples/sft/run_peft.sh)',
    'TRIGGER': 'schedule',
    'TARGET_REPO': 'huggingface/peft',
    'TARGET_REF': 'v0.20.0',
    'EXAMPLE_PATH': 'examples/sft/run_peft.sh',
    'IMAGE': 'swr.example/cann:tag',
}


class ConclusionMappingTests(unittest.TestCase):

    def test_success_and_cancelled_pass_through(self) -> None:
        self.assertEqual(write_example_result.conclusion_to_status('success'),
                         'success')
        self.assertEqual(
            write_example_result.conclusion_to_status('cancelled'),
            'cancelled')

    def test_everything_else_is_failure(self) -> None:
        for conclusion in ('failure', 'startup_failure', 'skipped', ''):
            self.assertEqual(
                write_example_result.conclusion_to_status(conclusion),
                'failure', conclusion)


class MainTests(unittest.TestCase):

    def test_writes_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'result.json'
            with mock.patch.dict(os.environ, ENV), \
                    mock.patch.object(write_example_result, 'fetch_conclusion',
                                      return_value='failure'), \
                    mock.patch.object(sys, 'argv',
                                      ['prog', '--output', str(out)]):
                write_example_result.main()
            data = json.loads(out.read_text(encoding='utf-8'))
        self.assertEqual(data, {
            'trigger': 'schedule',
            'target_repo': 'huggingface/peft',
            'target_ref': 'v0.20.0',
            'path': 'examples/sft/run_peft.sh',
            'image': 'swr.example/cann:tag',
            'job_status': 'failure',
        })

    def test_missing_conclusion_exits_1(self) -> None:
        with mock.patch.dict(os.environ, ENV), \
                mock.patch.object(write_example_result, 'fetch_conclusion',
                                  return_value=None), \
                mock.patch.object(sys, 'argv',
                                  ['prog', '--output', '/tmp/x.json']), \
                self.assertRaises(SystemExit) as ctx:
            write_example_result.main()
        self.assertEqual(ctx.exception.code, 1)

    def test_missing_env_exits_1(self) -> None:
        env = {k: v for k, v in ENV.items() if k != 'IMAGE'}
        with mock.patch.dict(os.environ, env, clear=False), \
                mock.patch.object(sys, 'argv', ['prog']), \
                self.assertRaises(SystemExit) as ctx:
            write_example_result.main()
        self.assertEqual(ctx.exception.code, 1)


if __name__ == '__main__':
    unittest.main()

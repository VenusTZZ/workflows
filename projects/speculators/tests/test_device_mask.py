"""Device-mask and vLLM health-wait behavior for the speculators guard."""
from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITECUSTOMIZE = ROOT / 'scripts' / 'sitecustomize_dir' / 'sitecustomize.py'
HEALTH_WAIT = ROOT / 'scripts' / 'fragments' / 'health_wait.sh'


def _load_sitecustomize():
    spec = importlib.util.spec_from_file_location(
        'speculators_sitecustomize_under_test',
        SITECUSTOMIZE,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load {SITECUSTOMIZE}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeviceMaskTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_sitecustomize()

    def test_parent_vllm_mask_replaces_job_slice(self) -> None:
        env = {
            'CUDA_VISIBLE_DEVICES': '0,1',
            'ASCEND_RT_VISIBLE_DEVICES': '0,1,2,3',
        }
        self.mod.sync_ascend_visible_devices(env)
        self.assertEqual(env['ASCEND_RT_VISIBLE_DEVICES'], '0,1')

    def test_trainer_mask_replaces_job_slice(self) -> None:
        env = {
            'CUDA_VISIBLE_DEVICES': '2,3',
            'ASCEND_RT_VISIBLE_DEVICES': '0,1,2,3',
        }
        self.mod.sync_ascend_visible_devices(env)
        self.assertEqual(env['ASCEND_RT_VISIBLE_DEVICES'], '2,3')

    def test_dp_child_keeps_single_card(self) -> None:
        for card in ('0', '1'):
            with self.subTest(card=card):
                env = {
                    'CUDA_VISIBLE_DEVICES': '0,1',
                    'ASCEND_RT_VISIBLE_DEVICES': card,
                }
                self.mod.sync_ascend_visible_devices(env)
                self.assertEqual(env['ASCEND_RT_VISIBLE_DEVICES'], card)

    def test_unset_ascend_copies_cuda(self) -> None:
        env = {'CUDA_VISIBLE_DEVICES': '0,1'}
        self.mod.sync_ascend_visible_devices(env)
        self.assertEqual(env['ASCEND_RT_VISIBLE_DEVICES'], '0,1')

    def test_unset_cuda_leaves_ascend(self) -> None:
        env = {'ASCEND_RT_VISIBLE_DEVICES': '0,1,2,3'}
        self.mod.sync_ascend_visible_devices(env)
        self.assertEqual(env['ASCEND_RT_VISIBLE_DEVICES'], '0,1,2,3')

    def test_autoimport_keeps_narrow_child_mask(self) -> None:
        env = os.environ.copy()
        env['PYTHONPATH'] = str(SITECUSTOMIZE.parent)
        env['CUDA_VISIBLE_DEVICES'] = '0,1'
        env['ASCEND_RT_VISIBLE_DEVICES'] = '1'
        env.pop('PYTHONNOUSERSITE', None)
        completed = subprocess.run(
            [
                'python3',
                '-c',
                'import os; print(os.environ["ASCEND_RT_VISIBLE_DEVICES"])',
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(completed.stdout.strip(), '1')


class HealthWaitTests(unittest.TestCase):

    def _script(self, body: str) -> str:
        wait = HEALTH_WAIT.read_text().replace(
            '__HEALTH_URL__',
            'http://127.0.0.1:9/nope',
        )
        return textwrap.dedent(body) + '\n' + wait

    def test_dead_server_exits_immediately(self) -> None:
        dead = subprocess.Popen(['sleep', '0'])
        dead.wait(timeout=5)
        completed = subprocess.run(
            ['bash', '-c', self._script(f'VLLM_PID={dead.pid}')],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertIn(
            f'vLLM server process {dead.pid} exited before becoming ready',
            completed.stderr,
        )
        self.assertNotIn('within 12 min', completed.stderr)

    def test_living_server_is_not_treated_as_exited(self) -> None:
        script = self._script(
            textwrap.dedent(
                """
                sleep 30 &
                VLLM_PID=$!
                """
            )
        )
        proc = subprocess.Popen(
            ['bash', '-c', script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=1)
            stderr = proc.stderr.read() if proc.stderr is not None else ''
            self.fail(
                f'health wait exited early: rc={proc.returncode} stderr={stderr}'
            )
        except subprocess.TimeoutExpired:
            pass
        finally:
            if proc.stderr is not None:
                proc.stderr.close()
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)

    def test_ready_endpoint_succeeds(self) -> None:
        bindir = self.id().replace('.', '_')
        fake = Path('/tmp') / bindir
        fake.mkdir(exist_ok=True)
        curl = fake / 'curl'
        curl.write_text('#!/bin/sh\nexit 0\n')
        curl.chmod(0o755)
        env = os.environ.copy()
        env['PATH'] = f'{fake}{os.pathsep}{env.get("PATH", "")}'
        completed = subprocess.run(
            ['bash', '-c', self._script('VLLM_PID=999999')],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env=env,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('vLLM server ready.', completed.stdout)

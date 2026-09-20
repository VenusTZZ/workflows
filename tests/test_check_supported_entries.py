"""Tests for scripts/check_supported_entries.py (engine manifest check)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / 'scripts' / 'check_supported_entries.py'

VALID_MANIFEST = """\
version: 1
scan:
  root: examples
  include_extensions: ['.sh', '.py']
supported:
  - path: examples/sft/run.sh
    profile: p
    runner: linux-aarch64-a2-1
    image: img:tag
    overlay_args: ['--max_steps 1']
    timeout_minutes: 90
unsupported:
  - examples/other/thing.py
"""


class CheckSupportedEntriesTests(unittest.TestCase):
    """Run the script as a subprocess against a temp target tree."""

    def _run(self, manifest_text: str, *, remove_supported: bool = False
             ) -> tuple[int, dict, str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'target'
            (target / 'examples' / 'sft').mkdir(parents=True)
            (target / 'examples' / 'brand_new').mkdir(parents=True)
            (target / 'examples' / 'sft' / 'run.sh').write_text('#!/bin/sh\n')
            # An unclassified new file: the check must ignore it.
            (target / 'examples' / 'brand_new' / 'new.py').write_text('')
            if remove_supported:
                (target / 'examples' / 'sft' / 'run.sh').unlink()
            manifest_path = root / 'examples_manifest.yaml'
            manifest_path.write_text(manifest_text, encoding='utf-8')
            env = dict(os.environ, GITHUB_OUTPUT=str(root / 'github_output'))
            proc = subprocess.run(
                [sys.executable, str(SCRIPT),
                 '--target-root', str(target),
                 '--manifest', str(manifest_path)],
                capture_output=True, text=True, env=env, check=False)
            outputs: dict = {}
            out_path = root / 'github_output'
            if out_path.exists():
                raw = out_path.read_text(encoding='utf-8')
                if 'supported_matrix<<EOF\n' in raw:
                    payload = raw.split(
                        'supported_matrix<<EOF\n')[1].split('\nEOF\n')[0]
                    outputs['supported_matrix'] = json.loads(payload)
                if 'has_supported=' in raw:
                    outputs['has_supported'] = raw.split(
                        'has_supported=')[1].splitlines()[0]
            return proc.returncode, outputs, proc.stderr

    def test_valid_entry_passes_and_ignores_unclassified(self) -> None:
        code, outputs, _ = self._run(VALID_MANIFEST)
        self.assertEqual(code, 0)
        self.assertEqual(outputs['has_supported'], 'true')
        matrix = outputs['supported_matrix']
        self.assertEqual(len(matrix), 1)
        self.assertEqual(matrix[0]['path'], 'examples/sft/run.sh')
        self.assertEqual(matrix[0]['overlay_args'], ['--max_steps 1'])

    def test_missing_supported_fails(self) -> None:
        code, _, stderr = self._run(VALID_MANIFEST, remove_supported=True)
        self.assertEqual(code, 1)
        self.assertIn('missing from target tree: examples/sft/run.sh', stderr)

    def test_bad_overlay_args_fails(self) -> None:
        manifest = VALID_MANIFEST.replace(
            "overlay_args: ['--max_steps 1']", 'overlay_args: [512]')
        code, _, stderr = self._run(manifest)
        self.assertEqual(code, 1)
        self.assertIn('overlay_args must be a list of non-empty strings',
                      stderr)

    def test_launcher_field_passes_through(self) -> None:
        manifest = VALID_MANIFEST.replace(
            '    timeout_minutes: 90\n',
            '    timeout_minutes: 90\n    launcher: accelerate-deepspeed\n')
        code, outputs, _ = self._run(manifest)
        self.assertEqual(code, 0)
        matrix = outputs['supported_matrix']
        self.assertEqual(matrix[0]['launcher'], 'accelerate-deepspeed')

    def test_bad_launcher_fails(self) -> None:
        manifest = VALID_MANIFEST.replace(
            '    timeout_minutes: 90\n', '    timeout_minutes: 90\n'
            '    launcher: 512\n')
        code, _, stderr = self._run(manifest)
        self.assertEqual(code, 1)
        self.assertIn('launcher must be a non-empty string', stderr)

    def test_missing_required_field_fails(self) -> None:
        manifest = VALID_MANIFEST.replace('    image: img:tag\n', '')
        code, _, stderr = self._run(manifest)
        self.assertEqual(code, 1)
        self.assertIn("missing required field(s): ['image']", stderr)

    def test_empty_supported_passes_with_false(self) -> None:
        manifest = VALID_MANIFEST.split('supported:')[0] + (
            'supported: []\nunsupported: []\n')
        code, outputs, _ = self._run(manifest)
        self.assertEqual(code, 0)
        self.assertEqual(outputs['has_supported'], 'false')
        self.assertEqual(outputs['supported_matrix'], [])

    def test_mixed_sources_keep_relative_paths_in_job_names(self) -> None:
        manifest = VALID_MANIFEST.replace(
            '    timeout_minutes: 90\n',
            '    timeout_minutes: 90\n'
            '  - source: project\n'
            '    path: example/run.py\n'
            '    profile: npu\n'
            '    runner: linux-aarch64-a2-1\n'
            '    image: img:tag\n'
            '    timeout_minutes: 60\n',
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'target'
            upstream = target / 'examples' / 'sft' / 'run.sh'
            upstream.parent.mkdir(parents=True)
            upstream.write_text('#!/bin/sh\n', encoding='utf-8')
            project_case = root / 'example' / 'run.py'
            project_case.parent.mkdir()
            project_case.write_text('def test_npu(): pass\n', encoding='utf-8')
            manifest_path = root / 'examples_manifest.yaml'
            manifest_path.write_text(manifest, encoding='utf-8')
            env = dict(os.environ, GITHUB_OUTPUT=str(root / 'out'))

            proc = subprocess.run(
                [sys.executable, str(SCRIPT), '--target-root', str(target),
                 '--manifest', str(manifest_path)],
                capture_output=True, text=True, env=env, check=False,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = (root / 'out').read_text(encoding='utf-8').split(
                'supported_matrix<<EOF\n', 1)[1].split('\nEOF\n', 1)[0]
            matrix = json.loads(payload)
            self.assertEqual(matrix[0]['name'], 'examples/sft/run')
            self.assertEqual(matrix[1]['name'], 'example/run')
            self.assertEqual(matrix[1]['source'], 'project')

    def test_missing_project_case_fails_before_runner(self) -> None:
        manifest = VALID_MANIFEST.replace(
            '    timeout_minutes: 90\n',
            '    timeout_minutes: 90\n'
            '  - source: project\n'
            '    path: example/missing.py\n'
            '    profile: npu\n'
            '    runner: linux-aarch64-a2-1\n'
            '    image: img:tag\n'
            '    timeout_minutes: 60\n',
        )
        code, _, stderr = self._run(manifest)
        self.assertEqual(code, 1)
        self.assertIn('supported project example missing', stderr)

    def test_project_case_path_cannot_escape_manifest_directory(self) -> None:
        manifest = VALID_MANIFEST.replace(
            '    timeout_minutes: 90\n',
            '    timeout_minutes: 90\n'
            '  - source: project\n'
            '    path: ../outside.py\n'
            '    profile: npu\n'
            '    runner: linux-aarch64-a2-1\n'
            '    image: img:tag\n'
            '    timeout_minutes: 60\n',
        )
        code, _, stderr = self._run(manifest)
        self.assertEqual(code, 1)
        self.assertIn('path must be relative', stderr)

if __name__ == '__main__':
    unittest.main()

"""Check that CI launches the user-facing script and propagates failures."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "scripts/run_example.py"


class TestExampleLaunch(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(RUNNER.is_file(), "The example launcher is missing")
        spec = importlib.util.spec_from_file_location("fla_example_launch", RUNNER)
        self.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.runner)

    def test_launches_project_example_with_normal_cli_arguments(self) -> None:
        command = self.runner.build_command(
            PROJECT, "example/train_text.py", Path("/tmp/run output"),
            json.dumps(["--steps 20", "--prompt 'The small model'", "--max-new-tokens 32"]),
        )
        self.assertEqual(command, [
            sys.executable, str(PROJECT / "example/train_text.py"),
            "--steps", "20", "--prompt", "The small model", "--max-new-tokens", "32",
            "--output-dir", "/tmp/run output",
        ])

    def test_rejects_invalid_paths_and_overlay_types(self) -> None:
        for path in ("../train_text.py", "tests/ops/test_gdn.py", "/tmp/train_text.py"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.runner.build_command(PROJECT, path, Path("/tmp/output"), "[]")
        for raw in ('{"steps": 20}', '[20]', 'invalid-json'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.runner.build_command(PROJECT, "example/train_text.py", Path("/tmp/output"), raw)

    def test_nonzero_exit_and_missing_outputs_cannot_pass_ci(self) -> None:
        for body in ("raise SystemExit(17)\n", "print('no training performed')\n"):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as tmp:
                project = Path(tmp) / "project"
                script = project / "example/train_text.py"
                script.parent.mkdir(parents=True)
                script.write_text(body, encoding="utf-8")
                result = subprocess.run(
                    [sys.executable, str(RUNNER), "example/train_text.py"],
                    env={
                        **os.environ, "PROJECT_ROOT": str(project), "TARGET_ROOT": tmp,
                        "CI_OUTPUT_DIR": str(Path(tmp) / "output"),
                        "EXAMPLE_SOURCE": "project", "OVERLAY_ARGS": "[]",
                    },
                    capture_output=True, text=True, check=False,
                )
                self.assertNotEqual(result.returncode, 0, result.stdout)
                if "17" in body:
                    self.assertIn("exit status 17", result.stderr)
                else:
                    self.assertIn("metrics.json", result.stderr)


if __name__ == "__main__":
    unittest.main()

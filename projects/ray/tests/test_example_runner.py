"""Behavior tests for dispatching upstream and project-owned Ray examples."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RUNNER = _PROJECT_ROOT / "scripts" / "run_example.sh"


class TestRayExampleRunner(unittest.TestCase):
    def _run(self, source: str, relative: str, *, create_file: bool = True):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        target = root / "target"
        project = root / "project"
        selected_root = project if source == "project" else target
        if create_file:
            selected = selected_root / relative
            selected.parent.mkdir(parents=True, exist_ok=True)
            selected.write_text("def test_sample(): pass\n", encoding="utf-8")

        cann_env = root / "set_env.sh"
        cann_env.write_text("export RAY_CANN_TEST=ready\n", encoding="utf-8")
        fake_bin = root / "bin"
        fake_bin.mkdir()
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/usr/bin/env bash\n"
            'printf "%s\\n" "$*" >> "$FAKE_PYTHON_LOG"\n',
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        log_path = root / "python.log"
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{fake_bin}:{env['PATH']}",
                "TARGET_ROOT": str(target),
                "PROJECT_ROOT": str(project),
                "EXAMPLE_SOURCE": source,
                "OVERLAY_ARGS": "[]",
                "RAY_CANN_SET_ENV": str(cann_env),
                "FAKE_PYTHON_LOG": str(log_path),
            }
        )
        completed = subprocess.run(
            ["bash", str(_RUNNER), relative],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed, log_path, selected_root

    def test_configurable_cann_env_is_used_in_local_harness(self) -> None:
        relative = "python/ray/tests/accelerators/test_npu.py"
        completed, log_path, root = self._run("upstream", relative)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(str(root / relative), log_path.read_text())

    def test_project_case_runs_from_project_directory(self) -> None:
        relative = "example/test_npu_discovery.py"
        completed, log_path, root = self._run("project", relative)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(str(root / relative), log_path.read_text())

    def test_project_case_rejects_directory_traversal(self) -> None:
        completed, _, _ = self._run("project", "../escape.py", create_file=False)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("invalid example path", completed.stderr)


if __name__ == "__main__":
    unittest.main()

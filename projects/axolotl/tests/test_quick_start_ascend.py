"""End-to-end test for the axolotl Ascend Quick Start."""

from __future__ import annotations

import os
import subprocess
import unittest

from workflows.markdown_doc_test_base import MarkdownDocTestBase
from workflows.model_cache import (
    ensure_safetensors,
    purge_modelscope_corrupt,
    resolve_modelscope_cache,
)


def _is_truthy(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() == "true"


def _e2e_enabled() -> bool:
    return _is_truthy(os.environ.get("NPU_READY"))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    DEFAULT_COMMAND_TIMEOUT = 3600
    USER_AGENT = "cosdt-ci-test/quick-start"
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        "applicaiton exception",
        "ERR99999",
    )
    _CANN_SET_ENV = "/usr/local/Ascend/ascend-toolkit/set_env.sh"

    @classmethod
    def prepare_environment(cls) -> None:
        path_dirs = "/usr/local/sbin:/usr/local/bin"
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{path_dirs}:{current_path}"
        os.environ["PYTHONNOUSERSITE"] = "1"
        os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

        if not os.path.isfile(cls._CANN_SET_ENV):
            raise RuntimeError(
                f"CANN environment script is missing: {cls._CANN_SET_ENV}"
            )

        merged = subprocess.run(
            ["bash", "-c", f"source {cls._CANN_SET_ENV} >/dev/null 2>&1; env"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in merged.stdout.splitlines():
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ[key] = value
        print("setup: sourced CANN environment")

        ensure_safetensors()
        purge_modelscope_corrupt(resolve_modelscope_cache())

    @classmethod
    def setUpClass(cls) -> None:
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        "end-to-end requires an Ascend NPU runner; set NPU_READY=true",
    )
    def test_runs_doc(self) -> None:
        self.run_template()


if __name__ == "__main__":
    unittest.main()

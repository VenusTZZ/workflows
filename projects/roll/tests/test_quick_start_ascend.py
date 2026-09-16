"""Quick-start-Ascend documentation test (MarkdownDocTestBase contract).

Document under test: projects/roll/docs/Quick-start-Ascend.md.
The doc is the executable test case: CI replays every labeled block in order,
so anything a user must do lives in the doc, and anything only CI needs lives
in prepare_environment below.

Doc flow mirrored here (install discipline only, doc content untouched):
  1. pip install -r ROLL/requirements_common.txt
     -> ray[default,cgraph]==2.48.0, peft==0.12.0, trl==0.9.6,
        datasets==3.1.0, hydra-core, omegaconf, math-verify==0.9.0,
        latex2sympy2==1.5.4, latex2sympy2_extended==1.10.1 ...
     pinned jointly with antlr4-python3-runtime==4.9.3 by hydra/omegaconf;
     math-verify 0.9.0 accepts latex2sympy2_extended 1.10.1 (its 1.11 pin
     predates ROLL's pin), so the resolver stays green without overrides.
  2. pip install reasoning-gym==0.1.23 (pure python).
  3. pip install --no-deps gem-llm==0.0.4
     -> the doc does this on purpose: gem-llm pulls
        math-verify[antlr4_13_2] == antlr4 4.13.2, which conflicts with
        omegaconf/hydra's antlr4==4.9.* pin. gem only needs the runtime
        package for roll.pipeline.agentic.env registration.

Env vars (injected by .github/workflows/roll-quick-start.yml):
  NPU_READY=true          gate, the class is skipped otherwise
  MONITORED_DOC_URL       engine monitor input (not used by the test,
                          the doc is read from the local checkout)
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from workflows.markdown_doc_test_base import MarkdownDocTestBase
from workflows.model_cache import (
    ensure_safetensors,
    purge_modelscope_corrupt,
    resolve_modelscope_cache,
)


def _is_truthy(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    DEFAULT_COMMAND_TIMEOUT = 1800
    USER_AGENT = 'cosdt-ci-test/quick-start'
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',
        'ERR99999',
    )

    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    def pre_process(self) -> str:
        doc_path = (
            Path(__file__).resolve().parent.parent
            / 'docs'
            / 'Quick-start-Ascend.md'
        )
        if not doc_path.is_file():
            raise RuntimeError(
                f'doc not found in local checkout: {doc_path}'
            )
        return doc_path.read_text(encoding='utf-8')

    @classmethod
    def prepare_environment(cls) -> None:
        """CANN env merge, card pin, PYTHONNOUSERSITE, PATH prefix, and
        model-cache sanity. Dependency installs live in the doc itself
        (that is the user path); only CI-side hygiene happens here."""
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '0')

        if not os.path.isfile(cls._CANN_SET_ENV):
            raise RuntimeError(
                f'CANN environment script is missing: {cls._CANN_SET_ENV}'
            )

        merged = subprocess.run(
            ['bash', '-c', f'source {cls._CANN_SET_ENV} >/dev/null 2>&1; env'],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in merged.stdout.splitlines():
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            os.environ[key] = value
        path_dirs = '/usr/local/sbin:/usr/local/bin'
        venv_bin = os.path.dirname(sys.executable)
        os.environ['PATH'] = f'{venv_bin}:{path_dirs}:{os.environ.get("PATH", "")}'
        print('setup: sourced CANN environment')

        ensure_safetensors()
        purge_modelscope_corrupt(resolve_modelscope_cache())

    @classmethod
    def setUpClass(cls) -> None:
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end requires NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        self.run_template()


if __name__ == '__main__':
    unittest.main()

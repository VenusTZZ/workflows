"""Quick-start-Ascend documentation test: end-to-end case built on top
of the ``MarkdownDocTestBase`` contract.

Document under test: ``projects/areno/docs/Quick-start-Ascend.md``
(follows the ``docs/markdown_doc_test_label.md`` contract: every
``shell`` / ``python`` code block carries one of the ``#test`` /
``#test-setup`` / ``#test-result`` labels plus ``id=`` / ``store=`` /
``load='x>>y'`` / ``fuzzy='xxx'`` parameters).

Run: ``python -m unittest tests.test_quick_start_ascend -v 2>&1``

Environment variables (injected by the quick-start engine workflow
``quick-start-template.yml``, triggered by ``areno-quick-start.yml``):
    ``MONITORED_DOC_URL``         Required; raw URL of the document under test.
    ``UPSTREAM_REF``              Required; bash reads ``$UPSTREAM_REF`` to get
                                  the latest release tag. The value is
                                  captured into ``captures`` via the
                                  ``#test-setup store="upstream_ref"`` block's
                                  stdout, then substituted into the doc
                                  command body where ``<ref>`` appears.
    ``NPU_READY=true``            Required, otherwise the class is skipped.
                                  End-to-end tests only run on the NPU runner:
                                  local dev machines / normal ubuntu runners
                                  have no ``/dev/davinci*`` device, and the
                                  hard run would fail on ``import torch_npu``.
"""

from __future__ import annotations

import os
import subprocess
import unittest

from workflows.markdown_doc_test_base import MarkdownDocTestBase


def _is_truthy(value: str | None) -> bool:
    """``'true'`` -> True (case-insensitive); anything else (including unset) -> False."""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """Return True when ``NPU_READY=true`` is set, releasing the skip."""
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """``Quick-start-Ascend.md`` end-to-end test: fetch doc -> validate
    contract -> run ``#test-setup`` / ``#test`` in order -> compare against
    ``#test-result``.

    Scope: NPU stack check (torch / torch_npu / device count) + source
    install at upstream main (temporary: the latest release tag v0.0.8
    predates the NPU backend — same situation as wenet; revert to the
    engine-resolved release tag once a release ships NPU. The installer
    detects ``torch_npu``, switches to ``requirements/npu.txt`` and
    compiles the ``areno.accel._areno_accel_npu`` Ascend C extension —
    no fallback, so the build must succeed on the CANN 9.0.0 dev image)
    + CLI entry smoke
    + GSPO-on-GSM8K 1-step RLVR smoke (Qwen3-0.6B via ModelScope, rollout
    + reward + train + checkpoint save) + OpenAI-compatible serve of the
    saved checkpoint (background start, /health poll, /v1/models and
    /v1/chat/completions checks, teardown).
    """

    # 60 min: the AscendC kernel + torch_npu extension build inside
    # ``uv pip install -e . --no-build-isolation`` dominates; the 1-step
    # GSPO smoke (0.6B model, 8 x 256-token rollouts) adds a few minutes
    # after the cold model/dataset download.
    DEFAULT_COMMAND_TIMEOUT = 3600
    USER_AGENT = 'cosdt-ci-test/quick-start'  # monitored source is the fork under cosdt-ci-test org
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # generic [ERROR] + Traceback
        'applicaiton exception',  # CANN toolkit emits this typo (sic) in its Python driver
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the container image pinned by the
    # ``image:`` input of ``areno-quick-start.yml``.
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # ----------------------------------------------------------
    # prepare_environment: CANN env + uv install
    # (torch / torch_npu / AReno installs are owned by the doc's
    #  `#test-setup` / `#test` blocks)
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env and install uv.

        ``torch / torch_npu`` and ``areno`` are owned by the doc's
        ``#test-setup`` / ``#test`` blocks; this method only does the
        runner-side scaffolding that has to happen before those blocks run.

        Class-level setup, triggered by ``setUpClass`` (not
        ``unittest.TestCase.setUp``, which fires per test method).
        """
        # 0) CANN env: source set_env.sh and merge into os.environ.
        # Workflow-level jobs.env / steps.env wins over CANN's defaults.
        # Every doc block subprocess inherits this env, so the AscendC
        # build inside ``uv pip install -e .`` sees ASCEND_HOME_PATH even
        # before the doc's own ``source`` lines run.
        if os.path.isfile(cls._CANN_SET_ENV):
            merged = subprocess.run(
                ['bash', '-c', f'source {cls._CANN_SET_ENV} >/dev/null 2>&1; env'],
                capture_output=True, text=True, check=True,
            )
            for line in merged.stdout.splitlines():
                if '=' not in line:
                    continue
                key, _, value = line.partition('=')
                os.environ.setdefault(key, value)
            print('setup: sourced CANN env from set_env.sh')
        else:
            print(
                f'setup: skipping CANN env source ({cls._CANN_SET_ENV} not present)'
            )

        # 1) uv: the doc's install block calls ``uv pip install -e .``
        # which handles PEP 517 build deps (and the git direct reference
        # for flash-linear-attention) more reliably than pip. Inherit
        # ``PIP_INDEX_URL`` + ``PIP_TRUSTED_HOST`` from the yml job-level
        # env (cluster cache path + trusted-host).
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class: CANN env + uv.

        ``areno`` is NOT installed here — see ``prepare_environment`` for why.

        ``@unittest.skipIf`` only skips the test *method* — ``setUpClass``
        itself always runs. The ``if _e2e_enabled()`` body guard below is
        what actually keeps heavy setup from firing when ``NPU_READY``
        is unset.
        """
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end requires NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        """Template-method entry point. The base class
        ``run_template()`` runs the full ``pre_process`` -> ``parse`` ->
        ``execute`` -> ``post_process`` flow. ``prepare_environment`` is
        triggered by ``setUpClass`` once, not from ``run_template``."""

        self.run_template()


if __name__ == '__main__':
    unittest.main()

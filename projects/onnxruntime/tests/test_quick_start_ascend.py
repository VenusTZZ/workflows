"""Quick-start-Ascend documentation test: end-to-end case built on top
of the ``MarkdownDocTestBase`` contract.

Document under test: ``projects/onnxruntime/docs/Quick-start-Ascend.md``
(follows the ``docs/markdown_doc_test_label.md`` contract: every
``shell`` code block carries one of the ``#test`` / ``#test-setup`` /
``#test-result`` labels plus ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` parameters).

Run: ``python -m unittest tests.test_quick_start_ascend -v 2>&1``

Environment variables (injected by GitHub workflow
``onnxruntime-quick-start.yml``):
    ``MONITORED_DOC_URL``         Required; raw URL of the document under test.
    ``NPU_READY=true``            Required, otherwise the class is skipped.
                                  End-to-end tests only run on the NPU runner.
    ``UPSTREAM_REF``              Latest GitHub Release tag, captured by the
                                  hidden ``#test-setup store="upstream_ref"``
                                  block and substituted where
                                  ``<UPSTREAM_REF>`` appears.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import unittest

from workflows.markdown_doc_test_base import MarkdownDocTestBase

_CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'


def _is_truthy(value: str | None) -> bool:
    """``'true'`` -> True (case-insensitive); anything else (including unset) -> False."""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """Return True when ``NPU_READY=true`` is set, releasing the skip."""
    return _is_truthy(os.environ.get('NPU_READY'))


def _merge_sourced_env(*scripts: str) -> None:
    """Source the scripts in a child bash and adopt the resulting environment.

    Overwrites (not setdefault): container images may pre-set PATH-like vars
    (e.g. LD_LIBRARY_PATH), and the CANN additions from set_env.sh must win.
    The child inherits os.environ, so untouched vars are rewritten with their
    own values — lossless. ``env -0`` keeps multi-line values in one entry.
    """
    sourced = ' && '.join(f'source {shlex.quote(script)}' for script in scripts)
    merged = subprocess.run(
        ['bash', '-c', f'set +u; {{ {sourced}; }} >/dev/null 2>&1; env -0'],
        capture_output=True,
        text=True,
        check=True,
    )
    for entry in merged.stdout.split('\0'):
        if not entry or '=' not in entry:
            continue
        key, _, value = entry.partition('=')
        os.environ[key] = value


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """``Quick-start-Ascend.md`` end-to-end test: fetch doc -> validate
    contract -> run ``#test-setup`` / ``#test`` in order -> compare against
    ``#test-result``."""

    DEFAULT_COMMAND_TIMEOUT = 10800  # source build of onnxruntime-cann
    USER_AGENT = 'cosdt-ci-test/quick-start'
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'CANN failure',
        'aclgrphBuildInitialize',
        'aclopCompileAndExecute',
        'ACL_ERROR_FAILURE',
    )

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env once so later ``bash -c`` blocks inherit it.

        Class-level setup: run once per test class, triggered by
        ``setUpClass``. Each labeled fence is a new subprocess, so a
        ``source set_env.sh`` block in the document does not persist.
        """
        if not os.path.isfile(_CANN_SET_ENV):
            raise RuntimeError(f'required Ascend env script missing: {_CANN_SET_ENV}')
        _merge_sourced_env(_CANN_SET_ENV)

        # CANN set_env.sh rewrites PATH. Put the active venv first so later
        # ``python -m pip`` / ``cmake`` land in that interpreter, not
        # /usr/local/bin's copy. npu-smi lives under /usr/local/sbin.
        venv_bin = ''
        if os.environ.get('VIRTUAL_ENV'):
            venv_bin = os.path.join(os.environ['VIRTUAL_ENV'], 'bin')
        prefix_parts = [
            p for p in (
                venv_bin,
                '/usr/local/sbin',
                '/usr/local/bin',
                os.path.expanduser('~/.local/bin'),
            ) if p
        ]
        current_path = os.environ.get('PATH', '')
        os.environ['PATH'] = ':'.join(prefix_parts + [current_path])
        print('setup: sourced CANN env from set_env.sh')

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class.

        ``@unittest.skipIf`` only skips the test *method* — ``setUpClass``
        itself always runs. The ``if _e2e_enabled()`` body guard below is
        what actually keeps heavy setup from firing when ``NPU_READY`` is
        unset.
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

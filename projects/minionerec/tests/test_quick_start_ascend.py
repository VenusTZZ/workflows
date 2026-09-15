"""Quick-start-Ascend documentation test: end-to-end case built on top
of the ``MarkdownDocTestBase`` contract.

Document under test: ``projects/minionerec/docs/Quick-start-Ascend.md``
(follows the ``docs/markdown_doc_test_label.md`` contract: every
``shell`` code block carries one of the ``#test`` / ``#test-setup`` /
``#test-result`` labels plus ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` parameters).

Smoke scope (see the doc): single-card MiniOneRec SFT + constrained-
decoding evaluation on Ascend NPU — ModelScope base model (Qwen2.5-0.5B)
+ HF-mirror preprocessed Amazon Industrial_and_Scientific subset (incl.
the info/ SID-item list txt for evaluation) + ``transfer_to_npu`` patch
on both sft.py and evaluate.py (unused ``bitsandbytes`` import stripped;
aarch64 wheel availability is uncertain) + 1-epoch HF-Trainer SFT
(sample=600/dataset, cutoff 256) + beam-20 evaluation over a 200-line
test slice; calc.py asserts the tail CC is 0 (every generated SID is a
legal trie path — the key NPU-correctness signal). HR/NDCG magnitudes
float with training state and are matched with ``...`` placeholders.
RL (GRPO) is pointer-only: its vLLM path is CUDA-only and the
transformers-native fallback is too slow for a smoke.

Run: ``python -m unittest tests.test_quick_start_ascend -v 2>&1``

Environment variables (injected by GitHub workflow
``minionerec-quick-start.yml``):
    ``MONITORED_DOC_URL``         Required; raw URL of the document under test.
    ``UPSTREAM_REF``              Required; bash reads ``$UPSTREAM_REF`` to get
                                  the target ref. The value is captured into
                                  ``captures`` via the
                                  ``#test-setup store="upstream_ref"`` block's
                                  stdout, then substituted into the doc
                                  command body where ``<UPSTREAM_REF>`` appears.
    ``NPU_READY=true``            Required, otherwise the class is skipped.
                                  End-to-end tests only run on the NPU runner.
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
    ``#test-result``."""

    DEFAULT_COMMAND_TIMEOUT = 3600  # the 1-epoch SFT fence dominates; the eval fence (beam-20 over 199 samples) adds ~1 min
    USER_AGENT = 'cosdt-ci-test/quick-start'
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # Process-level CUDA exclusion list to prevent NVIDIA/CUDA packages
    # from being pulled in as transitive dependencies. MiniOneRec's own
    # requirements.txt is never installed (CUDA-only pins: torchrec /
    # fbgemm_gpu / nvidia-*), but this guards any future transitive pull.
    _CUDA_CONSTRAINTS = (
        'cuda-toolkit<0',
        'cuda-python<0',
        'cuda-bindings<0',
        'nvidia-cublas<0',
        'nvidia-cuda-runtime<0',
        'nvidia-cudnn<0',
        'nvidia-nccl<0',
    )
    _CONSTRAINTS_FILE = '/tmp/minionerec_npu_constraints.txt'

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env + CUDA exclusion list + torch stack probe.

        The doc's ``## 4. 安装 NPU 依赖`` section installs the business
        dependencies as the canonical user path. This method only handles
        infrastructure bootstrap: CANN env, CUDA exclusion, and a
        probe-and-fallback for torch/torch-npu to avoid redundant
        installation when the image already has the correct versions
        (CANN 9.1.0 pairing: torch 2.9.0 + torch_npu 2.9.0.post2, the
        same line dgl / xdit guards run on this image).

        Class-level setup: run once per test class, triggered by
        ``setUpClass``. Each labeled fence is a new subprocess, so a
        ``source set_env.sh`` block in the document does not persist.
        """
        path_dirs = '/usr/local/sbin:/usr/local/bin'
        current_path = os.environ.get('PATH', '')
        if path_dirs not in current_path:
            os.environ['PATH'] = f'{path_dirs}:{current_path}'

        # 0) CANN env: source set_env.sh and merge env into os.environ
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

        # 1) CUDA exclusion list
        with open(cls._CONSTRAINTS_FILE, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(cls._CUDA_CONSTRAINTS) + '\n')
        os.environ['PIP_CONSTRAINT'] = cls._CONSTRAINTS_FILE

        # 2) torch stack probe + install: when version matches the image's
        # pre-installed wheels, reuse them to avoid redundant installation.
        _PROBE_SCRIPT = (
            'import torch, torch_npu\n'
            "raise SystemExit(0 if "
            "torch.__version__.startswith('2.9') "
            "and torch_npu.__version__.startswith('2.9') "
            "else 1)"
        )
        probe = subprocess.run(
            ['python', '-c', _PROBE_SCRIPT],
            capture_output=True,
            check=False,
        )
        if probe.returncode == 0:
            _VERSIONS_SCRIPT = (
                'import torch, torch_npu; '
                'print(torch.__version__, torch_npu.__version__)'
            )
            versions = subprocess.run(
                ['python', '-c', _VERSIONS_SCRIPT],
                capture_output=True, text=True, check=True,
            )
            print(f'setup: reusing image torch stack ({versions.stdout.strip()})')
        else:
            print('setup: installing torch==2.9.0 torch-npu==2.9.0.post2')
            subprocess.run(
                [
                    'python', '-m', 'pip', 'install',
                    'torch==2.9.0', 'torch-npu==2.9.0.post2',
                ],
                check=True,
            )

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class.

        ``@unittest.skipIf`` only skips the test *method* -- ``setUpClass``
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

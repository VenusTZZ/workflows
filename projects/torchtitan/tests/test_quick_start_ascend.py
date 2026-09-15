"""Quick-start-Ascend documentation test: end-to-end case built on top
of the ``MarkdownDocTestBase`` contract.

Document under test: ``projects/torchtitan/docs/Quick-start-Ascend.md``
(follows the ``docs/markdown_doc_test_label.md`` contract: every
``shell`` code block carries one of the ``#test`` / ``#test-setup`` /
``#test-result`` labels plus ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` parameters).

Run: ``python -m unittest tests.test_quick_start_ascend -v 2>&1``

Environment variables (injected by GitHub workflow
``torchtitan-quick-start.yml``):
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
    """``Quick-start-Ascend.md`` end-to-end test: fetch doc → validate
    contract → run ``#test-setup`` / ``#test`` in order → compare against
    ``#test-result``.

    The test subclass itself does not own any ``test_*`` method beyond the
    template-method entry; the doc body is the spec. ``prepare_environment``
    only does runner-side scaffolding (CANN env + CUDA exclusion list +
    ``uv`` install); ``torch / torch_npu / torchtitan`` are all installed by
    the doc's ``#test-setup`` / ``#test`` blocks against a ``<ref>`` checkout,
    so the test exercises the exact install path users get.
    """

    # torchtitan's smallest meaningful command — `python -m torchtitan.train
    # --module llama3 --config llama3_debugmodel --training.steps 1
    # --comm.mode fake_backend` — needs to: clone torchtitan (~5 MB
    # sparse), `uv pip install -r requirements.txt` (torchdata / datasets /
    # tensorboard / wandb / fsspec / tyro / tokenizers / safetensors /
    # einops / pillow on v0.2.x; cluster cache resolves most in well under
    # a minute on a warm runner), `uv pip install -e . --no-deps`, then run
    # a 2-step fake-backend training that builds the Llama 3 debug model
    # (~256 dim / 6 layers) on NPU. 60 minutes: the first train step on a
    # cold cache compiles the flex-attention triton kernel through the
    # bisheng compiler on NPU (tens of seconds per config), dwarfing the
    # install chain; watch runs showed 1300s is not enough for that.
    DEFAULT_COMMAND_TIMEOUT = 3600

    # Monitored source is the cosdt-ci-test/workflows fork (this repo): the
    # doc lives at projects/torchtitan/docs/Quick-start-Ascend.md and the
    # engine sets MONITORED_DOC_URL to the raw.githubusercontent.com URL
    # for the same path. Upstream torchtitan's quickstart lives at
    # pytorch/torchtitan/README.md and is referenced from inside the doc
    # body, but it is NOT the file under test.
    USER_AGENT = 'cosdt-ci-test/quick-start'

    # Extend the base ERROR_MARKERS with CANN's typo + sentinel so a CANN
    # failure surfaces a full stderr dump (head/tail by default would hide
    # the line that names the failure).
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',  # CANN toolkit emits this typo (sic)
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    # Process-level CUDA exclusion list. Same rationale as the ms-swift
    # test: write to /tmp and export, so subprocesses (subprocess.run
    # inherits parent env by default) see it. torchtitan's transitive
    # deps (datasets) occasionally pull CUDA-marked wheels if
    # the resolver sees the cluster cache's CUDA index first.
    # Also blocks community `triton` (CUDA-only backend): on NPU it must
    # be replaced by triton-ascend, and installing both clobbers the
    # same triton/ tree into a broken mix.
    _CUDA_CONSTRAINTS = (
        'triton<0',
        'cuda-toolkit<0',
        'cuda-python<0',
        'cuda-bindings<0',
        'cuda-core<0',
        'cuda-pathfinder<0',
        'flashinfer-python<0',
        'nvidia-cublas<0',
        'nvidia-cuda-runtime<0',
        'nvidia-cuda-nvrtc<0',
        'nvidia-cuda-cupti<0',
        'nvidia-cudnn<0',
        'nvidia-cudnn-frontend<0',
        'nvidia-cufft<0',
        'nvidia-curand<0',
        'nvidia-cusolver<0',
        'nvidia-cusparse<0',
        'nvidia-cutlass-dsl<0',
        'nvidia-cutlass-dsl-libs-base<0',
        'nvidia-cutlass-dsl-libs-core<0',
        'nvidia-cutlass-dsl-libs-cu12<0',
        'nvidia-ml-py<0',
        'nvidia-nccl<0',
        'nvidia-nvjitlink<0',
        'nvidia-nvtx<0',
        'nvidia-cublas-cu12<0',
        'nvidia-cuda-nvdisasm<0',
        'nvidia-cuda-runtime-cu12<0',
        'nvidia-cuda-nvrtc-cu12<0',
        'nvidia-cuda-cupti-cu12<0',
        'nvidia-cudnn-cu12<0',
        'nvidia-cufft-cu12<0',
        'nvidia-curand-cu12<0',
        'nvidia-cusolver-cu12<0',
        'nvidia-cusparse-cu12<0',
        'nvidia-cusparselt-cu12<0',
        'nvidia-nccl-cu12<0',
        'nvidia-nvjitlink-cu12<0',
        'nvidia-nvtx-cu12<0',
    )
    _CONSTRAINTS_FILE = '/tmp/torchtitan_npu_constraints.txt'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the GitHub workflow container image.
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # ----------------------------------------------------------
    # prepare_environment: CANN env + CUDA constraints + uv
    # (torch / torch_npu / torchtitan install are owned by the doc's
    #  #test-setup / #test blocks)
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Install CANN env + CUDA constraints + uv.

        torchtitan itself is intentionally NOT pre-installed here: the doc
        body runs ``git clone`` + ``uv pip install -r requirements.txt &&
        uv pip install -e . --no-deps`` against the upstream release tag
        injected by the workflow, so the test exercises the exact install
        path users get.

        What this hook does pre-install:
            * CANN env (so torch_npu is importable);
            * CUDA exclusion list (so an accidental ``nvidia-cudnn`` pull
              doesn't shadow the NPU build).

        ``torch / torch_npu`` and ``torchtitan`` are owned by the doc's
        ``#test-setup`` / ``#test`` blocks; this method only does the
        runner-side scaffolding that has to happen before those blocks run.
        """
        # 0) CANN env: source set_env.sh and merge the env stream into
        # os.environ
        if os.path.isfile(cls._CANN_SET_ENV):
            merged = subprocess.run(
                ['bash', '-c', f'source {cls._CANN_SET_ENV} >/dev/null 2>&1; env'],
                capture_output=True, text=True, check=True,
            )
            for line in merged.stdout.splitlines():
                if '=' not in line:
                    continue
                key, _, value = line.partition('=')
                # Don't overwrite envs explicitly injected by the
                # workflow; only fill in CANN keys that are missing.
                os.environ.setdefault(key, value)
            print('setup: sourced CANN env from set_env.sh')
        else:
            print(
                f'setup: skipping CANN env source ({cls._CANN_SET_ENV} not present)'
            )

        # 1) CUDA exclusion list + process-level env
        with open(cls._CONSTRAINTS_FILE, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(cls._CUDA_CONSTRAINTS) + '\n')
        os.environ['PIP_CONSTRAINT'] = cls._CONSTRAINTS_FILE
        os.environ['UV_CONSTRAINT'] = cls._CONSTRAINTS_FILE

        # 2) uv: the doc body's install step is
        # ``uv pip install -r requirements.txt && uv pip install -e .
        # --no-deps``, which gives clean stdout (no pip's "Obtaining /
        # Installing collected packages" preamble) so the #test-result
        # fuzzy match against ``torchtitan xxx`` isn't polluted. Inherit
        # ``PIP_INDEX_URL`` + ``PIP_TRUSTED_HOST`` from the yml job-level
        # env (cluster cache path + trusted-host).
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

        # 3) torchtitan transitive deps (tyro / tokenizers / safetensors /
        # datasets / tensorboard / wandb / torchdata / fsspec / einops /
        # pillow) are all declared in the release's `requirements.txt`. The
        # doc body's `uv pip install -r requirements.txt` block installs
        # them itself; pre-installing here would just be a redundant
        # `uv pip install` on top of that.
        #
        # The only doc-time env-setup that's NOT inside requirements.txt
        # is `uv` itself, which step (2) above already installs.

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

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
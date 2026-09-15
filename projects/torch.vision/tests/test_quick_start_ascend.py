"""Quick-start-Ascend documentation test: end-to-end case built on top
of the ``MarkdownDocTestBase`` contract.

Document under test: ``projects/torch.vision/docs/Quick-start-Ascend.md``
(follows the ``docs/markdown_doc_test_label.md`` contract: every
``shell`` code block carries one of the ``#test`` / ``#test-setup`` /
``#test-result`` labels plus ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` parameters).

Run: ``python -m unittest tests.test_quick_start_ascend -v 2>&1``

Environment variables (injected by GitHub workflow
``torch.vision-quick-start.yml``):
    ``MONITORED_DOC_URL``         Required; raw URL of the document under test.
    ``NPU_READY=true``            Required, otherwise the class is skipped.
                                  End-to-end tests only run on the NPU runner:
                                  local dev machines / normal ubuntu runners
                                  have no ``/dev/davinci*`` device, and the
                                  hard run would fail on ``import torch_npu``.

    Note: ``UPSTREAM_REF`` IS consumed by the doc — the (HTML-commented,
    still runner-executed) ``#test-setup store="upstream_ref"`` block
    captures it, and the source-build ``#test`` block loads it into the
    ``git clone --branch <ref>`` placeholder. The engine resolves it to
    the latest pytorch/vision release tag; the doc no longer clones the
    Ascend/vision fork (the fork is a ``torchvision_npu`` patch package
    whose ops don't intersect transforms.v2).

Scope note: the doc body covers the smoke path for **stock torchvision**
running under ``torch_npu`` PrivateUse1 dispatch. ``torch`` /
``torch_npu`` are installed by the doc body via ``uv pip install`` from
Aliyun PyPI mirror / Huawei Cloud ascend pypi
(versions per [Ascend PyTorch Compatibility 矩阵](https://gitcode.com/Ascend/pytorch/blob/main/COMPATIBILITY.en.md): torch==2.12.0 / torch_npu==2.12.0 / CANN==9.1.0).
``torchvision`` is installed from source: `git clone
github.com/pytorch/vision` at the latest release tag + apply the
Stable ABI compat patch ``patches/torch-2.12-stable-api-permute.patch``
(v0.29.0 targets torch 2.14's stable API surface and uses
``torch::stable::permute``, which torch 2.12/2.13 lack — the patch
swaps the two permute call sites for an equivalent transpose chain)
+ ``uv pip install -e .`` with ``FORCE_CUDA=0``, mirroring ms-swift's
source build pattern. No Ascend/vision fork
build: the fork is a ``torchvision_npu`` patch package whose ops
(``deform_conv`` / ``roi_pool``) are not exercised by the transforms.v2 smoke
path, and its ``csrc`` includes a ``npu_decode_video_kernel.{cpp,hpp}`` that
needs CANN DVPP dev headers not present in the base image.
The doc verifies:

* package + sub-module imports (``torchvision``, ``torchvision.transforms``,
  ``torchvision.transforms.v2``, ``torchvision.io``, ``torchvision.models``)
* transforms.v2 pipeline (synthetic PIL image -> tensor) with NPU
  dispatch (``tensor.to('npu:0')``), which fails fast if the C++
  extensions did not register NPU kernels

Model download / ModelScope cache is **not** in scope — torchvision
transforms work on plain PIL images, no checkpoints are fetched. So the
test does **not** call ``purge_modelscope_corrupt`` or ``ensure_safetensors``
(those helpers in ``workflows.model_cache`` are only meaningful when
a ``snapshot_download`` is in the pipeline).
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

    The test subclass itself does not own any ``test_*`` method beyond the
    template-method entry; the doc body is the spec. ``prepare_environment``
    makes sure CANN env is sourced + CUDA exclusion list is written + ``uv``
    is bootstrapped before the framework starts executing doc commands
    (the doc body itself installs ``torch`` / ``torch_npu`` from Aliyun
    pytorch-wheels + Huawei Cloud ascend pypi, then torchvision via
    patched source build of the latest pytorch/vision release, then 9
    v2 transforms smoke tests that exercise ``torch_npu`` PrivateUse1
    dispatch — no Ascend/vision fork source build, see module
    docstring).
    """

    # 60 min per command: the doc installs torch / torch_npu from the
    # Aliyun pytorch-wheels + Huawei Cloud ascend dual-source, then the
    # stock torchvision cpu wheel from Aliyun PyPI mirror, then 9 v2
    # transforms smoke tests on NPU. The torch_npu wheel is the slowest
    # piece (~30 min cold cache on a busy runner); 60 min leaves room
    # for the installs + transforms / NPU-dispatch tests.
    DEFAULT_COMMAND_TIMEOUT = 3600

    # Monitored source is the cosdt-ci-test/workflows fork (this repo):
    # the doc lives at projects/torch.vision/docs/Quick-start-Ascend.md
    # and the engine sets MONITORED_DOC_URL to the api.github.com URL for
    # the same path. ``upstream_repo`` is Ascend/vision (kept for monitoring
    # parity with other projects, even though this test no longer reads
    # UPSTREAM_REF — stock torchvision cpu wheel is what the doc installs).
    USER_AGENT = 'cosdt-ci-test/quick-start'

    # Extend the base ERROR_MARKERS with CANN's typo + sentinel so a CANN
    # failure surfaces a full stderr dump (head/tail by default would hide
    # the line that names the failure). Same pattern as peft / slime /
    # torchtune / torchtitan / diffusers.
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # generic [ERROR] + Traceback
        'applicaiton exception',  # CANN toolkit emits this typo (sic)
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    # Process-level CUDA exclusion list. Same rationale as the other
    # projects' tests: write to /tmp and export, so subprocesses
    # (subprocess.run inherits parent env by default) see it.
    # torchvision's own wheel resolver can transitively drag in CUDA
    # wheels without this constraint, and we never want a
    # `nvidia-cublas-cu12` etc. to sneak into a NPU run.
    _CUDA_CONSTRAINTS = (
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
    _CONSTRAINTS_FILE = '/tmp/torch_vision_npu_constraints.txt'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the GitHub workflow container image
    # (`image:` input of `torch.vision-quick-start.yml`).
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # ----------------------------------------------------------
    # prepare_environment: CANN env + CUDA constraints + uv
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env + write CUDA exclusion list + install uv.

        The doc body is the single source of truth for ``torch`` /
        ``torch_npu`` / ``torchvision`` installs (versions per
        [COMPATIBILITY.en.md](https://gitcode.com/Ascend/pytorch/blob/main/COMPATIBILITY.en.md)):
        ``torch`` / ``torch_npu`` via the doc's ``#test-setup`` block
        (Aliyun pytorch-wheels + Huawei Cloud ascend dual-source);
        ``torchvision`` via the doc's ``## 安装 torchvision`` block
        (patched source build of the latest pytorch/vision release).
        This class only owns
        env-level concerns that aren't doc-visible: CANN env sourcing,
        defensive CUDA exclusion list, ``uv`` bootstrap.

        Class-level setup: run once per test class, triggered by
        ``setUpClass``. Not the same as ``unittest.TestCase.setUp`` —
        that lifecycle hook fires before every test method, which is
        wrong for a one-shot setup.
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
                # workflow (jobs.env / steps.env); only fill in CANN
                # keys that are missing, to avoid conflicts.
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

        # 2) uv: the doc body's install steps use ``uv pip install``
        # (torch / torch_npu / pillow / torchvision). Inherit
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
        """Run env setup once per test class: CANN env + CUDA constraints + uv.

        ``torch`` / ``torch_npu`` / ``torchvision`` / ``pillow`` are NOT
        installed here — the doc's ``#test-setup`` blocks install them
        from Aliyun PyPI mirror / Huawei Cloud ascend pypi, so a broken
        install block fails loudly instead of being masked by a
        pre-installed copy.

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
"""Quick-start-Ascend documentation test: end-to-end case built on top
of the ``MarkdownDocTestBase`` contract.

Document under test: ``projects/peft/docs/Quick-start-Ascend.md``
(follows the ``docs/markdown_doc_test_label.md`` contract: every
``shell`` code block carries one of the ``#test`` / ``#test-setup`` /
``#test-result`` labels plus ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` parameters).

Run: ``python -m unittest tests.test_quick_start_ascend -v 2>&1``

Environment variables (injected by the quick-start engine workflow
``quick-start-template.yml``, triggered by ``peft-quick-start.yml``):
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
import sys
import unittest

from workflows.markdown_doc_test_base import MarkdownDocTestBase
from workflows.model_cache import (
    ensure_safetensors,
    purge_modelscope_corrupt,
    resolve_modelscope_cache,
)


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

    DEFAULT_COMMAND_TIMEOUT = 1200  # 20 min: long enough for model download + LoRA apply/save/load
    USER_AGENT = 'cosdt-ci-test/quick-start'  # monitored source is the fork under cosdt-ci-test org
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # generic [ERROR] + Traceback
        'applicaiton exception',  # CANN toolkit emits this typo (sic) in its Python driver
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    # Process-level CUDA exclusion list. Originally written inside the
    # workflow step as a child-process env passed through to pip / uv /
    # peft's own wheel resolver. Moved to the test layer: write to /tmp
    # and export; subprocesses (subprocess.run inherits parent env by
    # default) see it the same way.
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
    _CONSTRAINTS_FILE = '/tmp/peft_npu_constraints.txt'

    # Cluster-internal nginx PyPI cache + Huawei Cloud ascend dual-source.
    _CLUSTER_INDEX = 'http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple'
    _ASCEND_EXTRA = 'https://repo.huaweicloud.com/ascend/repos/pypi'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the container image pinned by the
    # ``image:`` input of ``peft-quick-start.yml``.
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # ----------------------------------------------------------
    # prepare_environment: CANN env + CUDA constraints + torch stack probe
    # + safetensors + modelscope cache purge (transformers / modelscope /
    # peft are installed by the doc's `### 前置安装` / `## 安装 PEFT` blocks)
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env + write CUDA exclusion list + install uv + torch stack probe
        + safetensors + purge stale modelscope cache shards.

        The doc's ``### 前置安装`` and ``## 安装 PEFT`` sections are the
        single source of truth for which packages + versions get
        installed; this class only handles ``torch`` / ``torch_npu``
        here (via the cluster cache + Huawei ascend dual-source), the
        defensive ``safetensors`` install, and the modelscope cache
        purge. All other packages — ``transformers`` / ``modelscope``
        (via ``check-ml-deps``) and ``peft`` (via
        ``peft-install-binary`` / ``peft-install-source``) — install
        themselves in document order via the ``#test`` machinery.

        Class-level setup: run once per test class, triggered by
        ``setUpClass``. Not the same as ``unittest.TestCase.setUp`` —
        that lifecycle hook fires before every test method, which is
        wrong for a one-shot install.
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

        # 2) uv: the doc's ``peft-install-source`` block calls
        # ``uv pip install -e .`` which handles PEP 517 build deps more
        # reliably than pip. Inherit ``PIP_INDEX_URL`` + ``PIP_TRUSTED_HOST``
        # from the yml job-level env (cluster cache path + trusted-host).
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

        # 3) torch stack probe + install: when version matches the image's
        # pre-installed wheels, reuse them to avoid the cluster cache
        # triggering ``+cpu`` resolution.
        _PROBE_SCRIPT = (
            'import torch, torch_npu\n'
            "raise SystemExit(0 if "
            "torch.__version__.startswith('2.12.0') "
            "and torch_npu.__version__.startswith('2.12.0') "
            "else 1)"
        )
        probe = subprocess.run(
            ['python', '-c', _PROBE_SCRIPT],
            capture_output=True,
            check=False,  # probe's success/failure is the branch signal — don't raise
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
            # torch 2.12.0 (upgraded 2026-09-20 from 2.9.0 to match the
            # examples guard; unblocks shira's sparse-COO tuner): the
            # cluster cache / aliyun only host the CUDA torch wheel whose
            # METADATA declares cuda-toolkit (conflicts with
            # PIP_CONSTRAINT cuda-toolkit<0). Fetch the CPU wheel directly
            # from pytorch.org and pull torch's pure-Python deps from the
            # cluster index. torch_npu 2.12.0 comes from the ascend extra.
            print('setup: installing torch==2.12.0+cpu (direct URL) + torch_npu==2.12.0')
            cp_abi = f'cp{sys.version_info.major}{sys.version_info.minor}'
            subprocess.run(
                [
                    'python', '-m', 'pip', 'install', '--no-deps',
                    'https://download.pytorch.org/whl/cpu/'
                    f'torch-2.12.0%2Bcpu-{cp_abi}-{cp_abi}-manylinux_2_28_aarch64.whl',
                ],
                check=True,
            )
            subprocess.run(
                [
                    'python', '-m', 'pip', 'install',
                    '--index-url', cls._CLUSTER_INDEX,
                    'filelock', 'typing-extensions>=4.10.0', 'setuptools<82',
                    'sympy>=1.13.3', 'networkx>=2.5.1', 'jinja2', 'fsspec>=0.8.5',
                ],
                check=True,
            )
            subprocess.run(
                [
                    'python', '-m', 'pip', 'install',
                    '--index-url', cls._CLUSTER_INDEX,
                    '--extra-index-url', cls._ASCEND_EXTRA,
                    'torch_npu==2.12.0',
                ],
                check=True,
            )

        # 4) safetensors: native loader used by the cache validation
        # step below. Pulled in transitively by torch on most images;
        # install defensively in case the CANN base ships without it.
        ensure_safetensors()

        # 5) Cache validation: persistent host-side bind mount can hold
        # truncated safetensors from interrupted runs. Walk every shard
        # under each model dir and purge it on failure; modelscope
        # will re-download cleanly on next access. Implementation
        # lives in workflows.model_cache; see that module's
        # docstring for the full rationale.
        purge_modelscope_corrupt(resolve_modelscope_cache())

        # 6) transformers / modelscope / peft are NOT installed here —
        # see the docstring above for why (the doc's own blocks install
        # them in document order).
        #
        # accelerate is intentionally NOT installed anywhere: this
        # quickstart uses explicit ``.to("npu:0")`` placement and never
        # touches ``device_map`` / ``Accelerator``; skipping accelerate
        # keeps the install footprint aligned with what the doc exercises.

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class: CANN env + CUDA constraints + uv +
        torch stack + safetensors + modelscope cache purge.

        ``transformers`` / ``modelscope`` / ``peft`` are NOT installed
        here — see ``prepare_environment`` for why.

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
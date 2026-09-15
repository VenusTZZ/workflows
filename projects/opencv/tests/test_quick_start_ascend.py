"""Quick-start-Ascend documentation test: end-to-end case built on top
of the ``MarkdownDocTestBase`` contract.

Document under test: ``projects/opencv/docs/Quick-start-Ascend.md``
(follows the ``docs/markdown_doc_test_label.md`` contract: every
``shell`` code block carries one of the ``#test`` / ``#test-setup`` /
``#test-result`` labels plus ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` parameters).

Run: ``python -m unittest tests.test_quick_start_ascend -v 2>&1``

Environment variables (injected by the quick-start engine workflow
``quick-start-template.yml``, triggered by ``opencv-quick-start.yml``):
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

    The OpenCV quickstart's heaviest steps are:
        1. ``apt-get install`` build deps (cmake / gcc / libjpeg / libpng / libtiff);
        2. ``git clone`` opencv + opencv_contrib (sparse, --depth 1);
        3. ``cmake -DWITH_HUAWEI_NPU=ON ...`` configure (DNN samples
           + opencv_contrib modules);
        4. ``make -j$(nproc)`` build (10-20 min on a 32-core runner);
        5. ``make install`` to /usr/local/opencv-cann;
        6. ``opencv_test_dnn`` + ``opencv_version`` smoke + HUAWEI
           backend gtest filter;
        7. Python binding smoke (cv2 import + HUAWEI backend enum) +
           MobileNetV2 ONNX forward on NPU (classic DNN engine forced
           via OPENCV_FORCE_DNN_ENGINE=1 inside the doc block).
    """

    # 150 min per command: `make -j2` is the long pole — Debug -O0 at
    # parallelism 2 on the a2-4 runner. With BUILD_LIST (8 modules
    # instead of main + ~35 contrib) and the all_ops.h -> narrow-header
    # patch (dnn .text 516MB -> 47MB, per-TU preprocessed size down
    # ~80%), the full build is ~50-70 min at -j2 (measured locally in
    # the CANN image on a comparable ARM core). 9000s keeps generous
    # headroom under the engine's 240-min job budget for install +
    # cannops gtests + quickstart + first ACL graph build (~30 min
    # cold AOE cache); a pre-patch build needed 8208s just to die at
    # the dnn link (CI 33244401262, R_AARCH64_CALL26 overflow).
    DEFAULT_COMMAND_TIMEOUT = 9000

    USER_AGENT = 'cosdt-ci-test/quick-start'  # monitored source is the fork under cosdt-ci-test org
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # generic [ERROR] + Traceback
        'applicaiton exception',  # CANN toolkit emits this typo (sic) in its Python driver
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    # Process-level CUDA exclusion list. Write to /tmp and export;
    # subprocesses (subprocess.run inherits parent env by default) see
    # it the same way. OpenCV itself does not pull CUDA wheels, but
    # pip's resolver on a cold cluster cache occasionally surfaces
    # nvidia-* markers via transitive deps of utility libs (numpy
    # etc.) that can creep in; pinning them to <0 keeps NPU wheels
    # authoritative.
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
    _CONSTRAINTS_FILE = '/tmp/opencv_npu_constraints.txt'

    # Cluster-internal nginx PyPI cache + Huawei Cloud ascend dual-source.
    _CLUSTER_INDEX = 'http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple'
    _ASCEND_EXTRA = 'https://repo.huaweicloud.com/ascend/repos/pypi'

    # CANN toolkit env (ASCEND_HOME / LD_LIBRARY_PATH / PATH) is sourced
    # from inside the doc — see the ``load-cann`` block in
    # ``Quick-start-Ascend.md``. The doc body is the single source of
    # truth for which commands set up the runner env; this class only
    # handles what the doc can't (PYTHONPATH for the source-built cv2,
    # CUDA exclusion list, uv install, torch stack probe).

    # Source-built OpenCV installs Python bindings under
    # ``/usr/local/opencv-cann/lib/python3.12/site-packages``. Each
    # ``#test`` / ``#test-setup`` block in the doc body is its own
    # ``subprocess.run(env=env)`` call (the env dict is captured once
    # at the top of ``MarkdownDocTestBase.execute`` and reused across
    # all blocks), so a one-shot ``export PYTHONPATH=...`` inside the
    # doc would only live for that single subprocess. The runner sets
    # it here in ``os.environ`` so every block's ``python`` child sees
    # the source-built ``cv2`` instead of the system pip wheel.
    _OPENCV_CANN_PYTHONPATH = (
        '/usr/local/opencv-cann/lib/python3.12/site-packages'
    )

    # ----------------------------------------------------------
    # prepare_environment: PYTHONPATH + CUDA constraints + build deps
    # (CANN env is sourced inside the doc — see the module-level
    # note above and the ``load-cann`` block in Quick-start-Ascend.md)
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Set PYTHONPATH to source-built cv2 + write CUDA exclusion list +
        install uv + torch stack probe (CANN env is sourced inside the
        doc's ``load-cann`` block; the OpenCV build itself is handled
        by the doc's ``## 安装 OpenCV`` blocks).

        The doc's ``## 安装 OpenCV（源码，启用 CANN 后端）`` section is
        the single source of truth for which apt packages install the
        OpenCV compile-time deps + which CMake flags turn on
        ``WITH_HUAWEI_NPU`` + where the source build is installed +
        which commands set up the runner env (e.g. CANN toolkit env).
        This class only handles:
            * PYTHONPATH pointing at the source-built cv2 (each #test
              block is its own ``subprocess.run(env=env)`` call, so a
              doc-level ``export PYTHONPATH=...`` would only live for
              that single subprocess — we set it once on os.environ
              here and the base class copies it into every block);
            * CUDA exclusion list;
            * torch stack probe — OpenCV's wheel doesn't need torch,
              but the image's torch stack probe (mirror reuse) is
              inherited from the peft pattern for runner consistency;
            * OpenCV itself is intentionally NOT pre-installed here —
              the doc's ``opencv-cmake-configure`` / ``opencv-build`` /
              ``opencv-install`` blocks install it, so a broken
              install surfaces as a fuzzy mismatch in ``opencv-
              cmake-configure`` / ``opencv-cann-run-tests`` rather than
              being masked by a pre-installed copy.
        """
        # 1) PYTHONPATH: PREPEND the source-built OpenCV site-packages so
        # the doc's Python blocks import the cv2 with HUAWEI_NPU backend.
        # PREPEND, not setdefault: the CANN image itself exports
        # PYTHONPATH=/usr/local/Ascend/cann-9.1.0/python/site-packages:...
        # (its TBE/ACL python bits), so setdefault would silently keep the
        # image value and every subprocess would import nothing (CI
        # 33286049447: "ModuleNotFoundError: No module named 'cv2'" on the
        # first quickstart block even though the install log showed
        # cv2.cpython-312-*.so landing in the right place).
        _pp = cls._OPENCV_CANN_PYTHONPATH
        _existing = os.environ.get('PYTHONPATH', '')
        if _pp not in _existing:
            os.environ['PYTHONPATH'] = (
                f'{_pp}:{_existing}' if _existing else _pp
            )
        print(
            f'setup: PYTHONPATH -> {os.environ["PYTHONPATH"]} '
            '(prepended source-built cv2; kept image entries)'
        )

        # 2) CUDA exclusion list + process-level env
        with open(cls._CONSTRAINTS_FILE, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(cls._CUDA_CONSTRAINTS) + '\n')
        os.environ['PIP_CONSTRAINT'] = cls._CONSTRAINTS_FILE
        os.environ['UV_CONSTRAINT'] = cls._CONSTRAINTS_FILE

        # 3) uv: kept for parity with other projects' env, even though
        # the OpenCV doc body uses `apt-get` + `make` rather than
        # `uv pip install`. The doc's source-install block does not
        # call uv, but having it on PATH mirrors the other projects
        # and lets future doc revisions drop in `uv pip install` lines
        # without requiring env changes.
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

        # 4) torch stack probe — OpenCV's wheel does not need torch,
        # but the CANN 9.1.0 image ships torch==2.9.0+cpu +
        # torch_npu==2.9.0.post2 as a baseline; probing confirms the
        # image is consistent (otherwise the doc's `cmake` configure
        # can pick up a half-installed `__init__.py` and fail). The
        # probe also exercises that CANN env sourcing didn't break
        # `import torch_npu` — a CANN path that breaks PrivateUse1
        # registration is hard to surface otherwise.
        _PROBE_SCRIPT = (
            'import torch, torch_npu\n'
            "raise SystemExit(0 if "
            "torch.__version__.startswith('2.9.0') "
            "and torch_npu.__version__.startswith('2.9.0') "
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
            print('setup: installing torch==2.9.0 torch_npu==2.9.0.post2')
            subprocess.run(
                [
                    'python', '-m', 'pip', 'install',
                    '--index-url', cls._CLUSTER_INDEX,
                    '--extra-index-url', cls._ASCEND_EXTRA,
                    'torch==2.9.0', 'torch_npu==2.9.0.post2',
                ],
                check=True,
            )

        # 5) OpenCV itself is intentionally NOT pre-installed here: the
        # doc body does ``git clone`` + ``cmake`` + ``make`` + ``make
        # install`` against the upstream release tag injected by the
        # workflow, so the test exercises the exact install path users
        # get (source build with CANN backend via
        # ``opencv-cmake-configure`` / ``opencv-build`` /
        # ``opencv-install``).
        #
        # ``cmake`` is assumed pre-installed on the CANN 9.1.0 image;
        # if a CI runner ever drops it, the doc's ``check-toolchain``
        # block surfaces the missing binary as a fuzzy mismatch.

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class: PYTHONPATH + CUDA
        constraints + uv + torch stack probe (CANN env is sourced
        inside the doc's ``load-cann`` block).

        ``opencv`` is NOT installed here — see ``prepare_environment``
        for why (the doc's own blocks install it in document order).

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

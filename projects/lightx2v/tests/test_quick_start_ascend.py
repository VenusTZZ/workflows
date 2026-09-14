"""Quick-start-Ascend documentation test (MarkdownDocTestBase contract).

Document under test: `projects/lightx2v/docs/Quick-start-Ascend.md`
(follows the `docs/markdown_doc_test_label.md` contract: every
`shell` code block carries one of the `#test` / `#test-setup` /
`#test-result` labels plus `id=` / `store=` / `load='x>>y'` /
`fuzzy='xxx'` parameters).

Run: `python -m unittest tests.test_quick_start_ascend -v 2>&1`

Environment variables (injected by the quick-start engine workflow
`quick-start-template.yml`, triggered by `lightx2v-quick-start.yml`):
    `MONITORED_DOC_URL`   Required; raw URL of the document under test.
    `UPSTREAM_REF`        Injected by the engine but NOT consumed by the
                            doc body: the doc's clone block just clones the
                            default branch, exactly what a user gets. The
                            trigger pins `fixed_ref: main` so the monitor
                            polls `/commits/main` as its change key, which
                            matches the doc following rolling main.
    `NPU_READY=true`      Required, otherwise the class is skipped.
                            End-to-end tests only run on the NPU runner:
                            local dev machines / normal ubuntu runners have
                            no `/dev/davinci*` device, and the hard run
                            would fail on `import torch_npu`.

The doc body is cwd-relative ("wherever you run it is the project root"):
the upstream clone lands in `./LightX2V`, outputs in `./save_results`,
and the Wan2.1-T2V-1.3B weights go to the default ModelScope cache via the
`snapshot_download` embedded in the doc's generation script. CI pins the
execution cwd in `prepare_environment` via `os.chdir('/root/lightx2v-test')`.

Everything a user does NOT have to do lives here, not in the doc: CANN
`set_env.sh` sourcing, the CUDA exclusion list, card pinning, the torch
stack probe, the ModelScope cache validation and the doc cwd.
"""

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
    """`'true'` -> True (case-insensitive); anything else (including unset) -> False."""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """Return True when `NPU_READY=true` is set, releasing the skip."""
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """`Quick-start-Ascend.md` end-to-end test: fetch doc -> validate
    contract -> run `#test-setup` / `#test@@ in order -> compare against
    `#test-result`.

    Scope (single card):
      * The doc installs the pinned torch stack itself (torch 2.9.0 +
        torchvision 0.24.* + torch_npu 2.9.0.post2 + triton 3.5.*, the
        torch 2.9 official line) via the `lightx2v-install-torch`
        `#test-setup` block, clones upstream and installs it with
        `uv pip install --no-deps` plus an explicit NPU dependency list
        (`lightx2v-install-deps`), then verifies the import chain, the
        version printout and the NPU platform dispatch
        (`PLATFORM=ascend_npu` -> `platform: ascend_npu npu` and
        `npu available: True`) in `lightx2v-install-verify`.
      * Smoke: the official `LightX2VPipeline` Python API generates one
        Wan2.1 t2v video using the repo's own NPU config
        (`configs/platforms/ascend_npu/wan_t2v.json`: `npu_flash_attn`
        / 50 steps / 480x832 / 81 frames / `cpu_offload`), with the
        ~17.6 GB `Wan-AI/Wan2.1-T2V-1.3B` weights pulled automatically
        into the default ModelScope cache by the script's embedded
        `snapshot_download`. The output is validated structurally (mp4
        `ftyp` header + `moov` box + size floor).
      * Multi-card parallelism, quantization and service deployment are
        pointer-only (one closing line linking upstream `examples/`).

    Dependency list rationale (`lightx2v-install-deps`): upstream's
    `pyproject.toml` declares packages with no aarch64 wheel (notably
    `decord`, x86_64-only on PyPI), so a full dependency resolution
    cannot succeed on this runner. The doc therefore installs the code
    with `--no-deps` and then the packages the wan2.1 t2v path actually
    imports at module level — established by walking the import closure of
    `lightx2v` + `lightx2v.common.ops` +
    `lightx2v.models.runners.wan.wan_runner`. Every remaining
    third-party import on that closure is function-level and guarded by
    `try` / `except ImportError` (CUDA and other-accelerator
    attention / quantization backends), and `cv2` / `decord` /
    `torchaudio` are not on the closure at all — so no stub packages
    are needed here.
    """

    DEFAULT_COMMAND_TIMEOUT = 1800  # 30 min per block: the cold ~17.6 GB weight pull rides on the yml-level budget
    USER_AGENT = 'cosdt-ci-test/quick-start'  # monitored source is the fork under cosdt-ci-test org
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # generic [ERROR] + Traceback
        'applicaiton exception',  # CANN toolkit emits this typo (sic) in its Python driver
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
        'RuntimeError: Failed to load the backend extension: torch_npu',  # torch_npu loaded outside the CANN env
    )

    # Process-level CUDA exclusion list: written to /tmp and exported so
    # every doc subprocess (subprocess.run inherits the parent env) sees
    # it. torch's aarch64 wheels and the resolver otherwise reach for
    # nvidia-* CUDA packages that have no aarch64 + torch_npu ABI match;
    # `<0` forces resolution-impossible early so the resolver settles on
    # the CPU / torch_npu wheels.
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
    _CONSTRAINTS_FILE = '/tmp/lightx2v_npu_constraints.txt'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the container image pinned by the
    # `image:` input of `lightx2v-quick-start.yml`.
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # Doc execution cwd for CI: the doc body is cwd-relative (clone to
    # ./LightX2V, outputs to ./save_results) so "wherever the user runs
    # it" is the project root. CI chdirs to /root/lightx2v-test to keep
    # the clone and the generated video out of the checkout dir. Model
    # weights land in the default ModelScope cache (~/.cache/modelscope),
    # which the workflow yml bind-mounts from the host
    # (/data/ci-cache/modelscope/lightx2v) so they persist across runs.
    _PROJECT_ROOT = '/root/lightx2v-test'

    # ----------------------------------------------------------
    # prepare_environment: CANN env + CUDA constraints + uv +
    # torch stack probe + doc execution cwd + card pin + cache
    # validation (the lightx2v install and the weight pull live
    # in the doc body)
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env + write the CUDA exclusion list + install uv +
        probe the torch stack + chdir to the doc cwd + validate the
        ModelScope cache.

        `lightx2v` itself is NOT installed here — the doc's
        `lightx2v-install-source` / `lightx2v-install-deps` blocks
        exercise the clone + source install path, so a broken install
        surfaces as a fuzzy mismatch against `lightx2v: xxx` in the
        doc's verify block rather than being masked by a pre-installed
        copy.

        The doc body is cwd-relative; `os.chdir(_PROJECT_ROOT)` here
        makes every doc command run under /root/lightx2v-test (the engine
        executes each block with `cwd=Path.cwd()`).

        No ModelScope cache path is set here either — the workflow yml
        declares the bind at `container_options` time, which makes it
        visible to `resolve_modelscope_cache()` through the standard
        `~/.cache/modelscope` path that `snapshot_download` writes to
        when no local dir is requested.
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
                # workflow (jobs.env / steps.env); only fill in the CANN
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

        # 2) uv: the doc's install blocks call `uv pip install`, which
        # handles PEP 517 build deps more reliably than pip for a source
        # tree. Inherits PIP_INDEX_URL / PIP_TRUSTED_HOST / UV_* from the
        # engine's job-level env (cluster cache path + trusted host).
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

        # 3) torch stack probe: when a pre-installed stack imports and
        # sees the NPU, reuse it (bare metal / images that ship torch).
        # The plain CANN base image ships none, so the probe fails and the
        # doc's `lightx2v-install-torch` block installs the pinned stack
        # — which is then what we test against. triton rides along in that
        # same block: the real 3.5.x wheel matches torch 2.9's official
        # triton line, ships cp312 aarch64 manylinux wheels with zero
        # runtime deps, and an empty import-time stub would only push
        # torch._inductor into real triton code paths it cannot survive.
        _PROBE_SCRIPT = (
            'import torch, torch_npu\n'
            'raise SystemExit(0 if torch.npu.is_available() else 1)\n'
        )
        probe = subprocess.run(
            ['python', '-c', _PROBE_SCRIPT],
            capture_output=True,
            check=False,  # the probe's success/failure is the branch signal
        )
        if probe.returncode == 0:
            _VERSIONS_SCRIPT = (
                'import torch, torch_npu; '
                'print(torch.__version__, torch_npu.__version__)'
            )
            versions = subprocess.run(
                ['python', '-c', _VERSIONS_SCRIPT], capture_output=True,
                text=True, check=True,
            )
            print(f'setup: reusing image torch stack ({versions.stdout.strip()})')
        else:
            print(
                'setup: torch stack probe failed, the doc install-torch '
                'block will install the pinned stack'
            )

        # 4) execution cwd: chdir to /root/lightx2v-test — the doc body is
        # cwd-relative and the engine runs every block with
        # cwd=Path.cwd(). Pre-create the dir first (a bind-mount onto a
        # missing target dir fails on some kernels).
        try:
            os.makedirs(cls._PROJECT_ROOT, exist_ok=True)
        except OSError as exc:
            print(f'setup: doc cwd mkdir failed: {exc}')
        os.chdir(cls._PROJECT_ROOT)
        print(f'setup: cwd -> {os.getcwd()}')

        # 4.5) ASCEND_RT_VISIBLE_DEVICES=0: the NPU runner label is
        # `linux-aarch64-a2-2` (2 cards) and the cluster device-plugin
        # passes both /dev/davinci* into the container. The doc's smoke is
        # single-card, so pin card 0 at process level here — every doc
        # subprocess inherits it.
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'

        # 5) cache validation: a persistent host-side bind mount can hold
        # truncated safetensors from an interrupted download. Walk every
        # shard under the ModelScope cache and purge on failure;
        # `snapshot_download` re-downloads cleanly on next access.
        # ensure_safetensors pulls in safetensors (a torch transitive);
        # install it defensively in case the CANN base image rolls forward.
        ensure_safetensors()
        try:
            purge_modelscope_corrupt(resolve_modelscope_cache())
        except Exception as exc:
            # purge_modelscope_corrupt is best-effort: a permission error
            # or a missing dir shouldn't abort the test. Log and continue;
            # the doc's snapshot_download surfaces real download failures
            # through its own rc.
            print(f'setup: cache purge skipped ({exc})')

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class.

        `@unittest.skipIf` only skips the test *method* —
        `setUpClass` itself always runs. The `if _e2e_enabled()` guard
        below is what actually keeps heavy setup from firing when
        `NPU_READY` is unset.
        """
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end requires NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        """Template-method entry point. The base class `run_template()`
        runs the full `pre_process` -> `parse` -> `execute` ->
        `post_process` flow. `prepare_environment` is triggered by
        `setUpClass` once, not from `run_template`."""

        self.run_template()


if __name__ == '__main__':
    unittest.main()


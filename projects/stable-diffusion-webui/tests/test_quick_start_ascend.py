"""Quick-start-Ascend documentation test: end-to-end case built on top
of the MarkdownDocTestBase contract.

Document under test: projects/stable-diffusion-webui/docs/Quick-start-Ascend.md
"""

from __future__ import annotations

import os
import subprocess
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from workflows.markdown_doc_test_base import (
    MarkdownDocTestBase,
    SetupCommand,
    TestCommand,
)
from workflows.model_cache import (
    ensure_safetensors,
    purge_modelscope_corrupt,
    resolve_modelscope_cache,
)


def _is_truthy(value):
    if not value:
        return False
    return value.strip().lower() == "true"


def _e2e_enabled():
    return _is_truthy(os.environ.get("NPU_READY"))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    DEFAULT_COMMAND_TIMEOUT = 1800
    USER_AGENT = "cosdt-ci-test/quick-start"

    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        "applicaiton exception",
        "ERR99999",
    )

    _CUDA_CONSTRAINTS = (
        "cuda-toolkit<0",
        "cuda-python<0",
        "cuda-bindings<0",
        "cuda-core<0",
        "cuda-pathfinder<0",
        "flashinfer-python<0",
        "nvidia-cublas<0",
        "nvidia-cuda-runtime<0",
        "nvidia-cuda-nvrtc<0",
        "nvidia-cuda-cupti<0",
        "nvidia-cudnn<0",
        "nvidia-cudnn-frontend<0",
        "nvidia-cufft<0",
        "nvidia-curand<0",
        "nvidia-cusolver<0",
        "nvidia-cusparse<0",
        "nvidia-cutlass-dsl<0",
        "nvidia-cutlass-dsl-libs-base<0",
        "nvidia-cutlass-dsl-libs-core<0",
        "nvidia-cutlass-dsl-libs-cu12<0",
        "nvidia-ml-py<0",
        "nvidia-nccl<0",
        "nvidia-nvjitlink<0",
        "nvidia-nvtx<0",
        "nvidia-cublas-cu12<0",
        "nvidia-cuda-nvdisasm<0",
        "nvidia-cuda-runtime-cu12<0",
        "nvidia-cuda-nvrtc-cu12<0",
        "nvidia-cuda-cupti-cu12<0",
        "nvidia-cudnn-cu12<0",
        "nvidia-cufft-cu12<0",
        "nvidia-curand-cu12<0",
        "nvidia-cusolver-cu12<0",
        "nvidia-cusparse-cu12<0",
        "nvidia-cusparselt-cu12<0",
        "nvidia-nccl-cu12<0",
        "nvidia-nvjitlink-cu12<0",
        "nvidia-nvtx-cu12<0",
    )
    _CONSTRAINTS_FILE = "/tmp/sd-webui_npu_constraints.txt"
    _CLUSTER_INDEX = "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
    _ASCEND_EXTRA = "https://repo.huaweicloud.com/ascend/repos/pypi"
    _CANN_SET_ENV = "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    _API_READY_ENDPOINT = "http://127.0.0.1:7861/docs"
    _API_READY_ATTEMPTS = 120
    _API_READY_INTERVAL_S = 5

    def _wait_for_api(self):
        """Poll the --nowebui docs endpoint until it answers.

        The pending check runs here (CI-side concern) instead of in the doc,
        so the visible ""wait-ready"" step only asserts readiness via curl.
        """
        for attempt in range(1, self._API_READY_ATTEMPTS + 1):
            try:
                with urllib.request.urlopen(
                    self._API_READY_ENDPOINT, timeout=10
                ) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            self.log(
                "waiting for api: "
                f"attempt {attempt}/{self._API_READY_ATTEMPTS}"
            )
            time.sleep(self._API_READY_INTERVAL_S)
        log = Path("/tmp/sdwebui.log")
        tail = ""
        if log.is_file():
            try:
                tail = "\n".join(
                    log.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()[-100:]
                )
            except OSError:
                tail = ""
        self.log("--- tail /tmp/sdwebui.log ---")
        self.log(tail or "(log file missing or empty)")
        raise RuntimeError(
            f"api not ready after {self._API_READY_ATTEMPTS} attempts"
        )

    def _run_one(self, cmd, results, env, cwd, timeout, idx):
        if isinstance(cmd, SetupCommand) and "nohup python launch.py" in cmd.cmd and "sdwebui.log" not in cmd.cmd:
            actual_cmd = self.substitute_placeholders(cmd.cmd, cmd.load, self._captures)
            launch_cmd = actual_cmd.rstrip()
            if launch_cmd.endswith("&"):
                launch_cmd = launch_cmd[:-1].rstrip()
            rc, out, err = self.run_command(launch_cmd + " > /tmp/sdwebui.log 2>&1 &", env, cwd, timeout)
            if rc != 0:
                raise AssertionError(f"setup command failed (rc={rc}); CMD stderr:\n{err.rstrip() or '(empty)'}")
            return
        if (
            isinstance(cmd, TestCommand)
            and getattr(cmd, "id", None) == "wait-ready"
        ):
            self._wait_for_api()
        return super()._run_one(cmd, results, env, cwd, timeout, idx)

    def pre_process(self):
        doc_path = (
            Path(__file__).resolve().parent.parent
            / "docs"
            / "Quick-start-Ascend.md"
        )
        if not doc_path.is_file():
            raise RuntimeError(
                f"doc not found in local checkout: {doc_path}"
            )
        return doc_path.read_text(encoding="utf-8")

    @classmethod
    def prepare_environment(cls):
        if os.path.isfile(cls._CANN_SET_ENV):
            merged = subprocess.run(
                ["bash", "-c", "source " + cls._CANN_SET_ENV + " >/dev/null 2>&1; env"],
                capture_output=True, text=True, check=True,
            )
            for line in merged.stdout.splitlines():
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key, value)
            print("setup: sourced CANN env from set_env.sh")
        else:
            print("setup: skipping CANN env source")

        with open(cls._CONSTRAINTS_FILE, "w", encoding="utf-8") as fh:
            fh.write("\n".join(cls._CUDA_CONSTRAINTS) + "\n")
        os.environ["PIP_CONSTRAINT"] = cls._CONSTRAINTS_FILE
        os.environ["UV_CONSTRAINT"] = cls._CONSTRAINTS_FILE

        os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(
            [
                "apt-get", "install", "-y", "-qq", "--no-install-recommends",
                "libgl1", "libglib2.0-0",
            ],
            check=True,
        )
        print("setup: installed libgl1 libglib2.0-0")

        _PROBE = (
            "import torch, torch_npu\n"
            + 'raise SystemExit(0 if torch.__version__.startswith("2.9.0") and torch_npu.__version__.startswith("2.9.0") else 1)'
        )
        probe = subprocess.run(
            ["python", "-c", _PROBE],
            capture_output=True,
            check=False,
        )
        if probe.returncode == 0:
            v = subprocess.run(
                ["python", "-c",
                 "import torch, torch_npu; "
                 "print(torch.__version__, torch_npu.__version__)"],
                capture_output=True, text=True, check=True,
            )
            print(f"setup: reusing image torch stack ({v.stdout.strip()})")
        else:
            print("setup: installing torch==2.9.0 torch_npu==2.9.0.post2")
            subprocess.run(
                [
                    "python", "-m", "pip", "install",
                    "--index-url", cls._CLUSTER_INDEX,
                    "--extra-index-url", cls._ASCEND_EXTRA,
                    "torch==2.9.0", "torch_npu==2.9.0.post2",
                ],
                check=True,
            )

        ensure_safetensors()
        purge_modelscope_corrupt(resolve_modelscope_cache())

    @classmethod
    def setUpClass(cls):
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        "end-to-end requires NPU runner; set NPU_READY=true",
    )
    def test_runs_doc(self):
        self._kill_webui()
        try:
            self.run_template()
        finally:
            self._kill_webui()

    @staticmethod
    def _kill_webui():
        subprocess.run(["pkill", "-f", "launch.py --nowebui"], capture_output=True, timeout=10)


if __name__ == "__main__":
    unittest.main()

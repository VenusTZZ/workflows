"""Static contract for the open_clip Ascend Quick Start guard."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
for _path in (_SRC, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from workflows.markdown_doc_test_base import MarkdownDocTestBase  # noqa: E402


class _Parser(MarkdownDocTestBase):
    pass


class TestOpenClipProjectContract(unittest.TestCase):
    def test_registry_points_to_one_card_quick_start(self) -> None:
        registry = yaml.safe_load(
            (_REPO_ROOT / "projects.yaml").read_text(encoding="utf-8")
        )
        projects = [
            item for item in registry["projects"] if item["name"] == "open_clip"
        ]
        self.assertEqual(len(projects), 1)
        project = projects[0]
        self.assertEqual(project["upstream_repo"], "mlfoundations/open_clip")
        self.assertEqual(project["runner"], "linux-aarch64-a2-1")
        self.assertEqual(project["dir"], "projects/open_clip")
        self.assertEqual(
            project["workflows"],
            {"quick_start": ".github/workflows/open-clip-quick-start.yml"},
        )

    def test_workflow_uses_shared_template_and_hf_cache(self) -> None:
        workflow_path = (
            _REPO_ROOT / ".github" / "workflows" / "open-clip-quick-start.yml"
        )
        self.assertTrue(workflow_path.is_file(), workflow_path)
        text = workflow_path.read_text(encoding="utf-8")
        self.assertIn("uses: ./.github/workflows/quick-start-template.yml", text)
        self.assertIn("project: open_clip", text)
        self.assertIn("test_runner: '[\"linux-aarch64-a2-1\"]'", text)
        self.assertIn("cann:9.1.0-910b-ubuntu22.04-py3.12", text)
        self.assertIn("upstream_repo: mlfoundations/open_clip", text)
        self.assertIn(
            "--volume=/data/ci-cache/huggingface/open-clip:/root/.cache/huggingface",
            text,
        )
        self.assertNotIn("--device=/dev/davinci", text)
        self.assertIn("workflow_dispatch:", text)

    def test_doc_has_pretrained_inference_and_upstream_training_smoke(self) -> None:
        doc_path = (
            _REPO_ROOT
            / "projects"
            / "open_clip"
            / "docs"
            / "Quick-start-Ascend.md"
        )
        self.assertTrue(doc_path.is_file(), doc_path)
        text = doc_path.read_text(encoding="utf-8")
        commands, results = _Parser().parse(text)
        test_ids = {command.id for command in commands if hasattr(command, "id")}
        self.assertEqual(
            test_ids,
            {
                "check-python",
                "check-torch",
                "install-open-clip",
                "npu-inference",
                "npu-training",
            },
        )
        self.assertEqual(set(results), test_ids)

        required = (
            "## 前置条件",
            "### 本文档示例使用的版本",
            "## 安装 open_clip",
            "## Quick Start：单卡预训练图文推理",
            "## 单卡最小训练验证",
            "mlfoundations/open_clip",
            "requirements-training.txt",
            'pretrained="laion2b_s34b_b79k"',
            'device="npu:0"',
            '.to("npu:0")',
            'top_label == "a diagram"',
            "NPU inference PASSED",
            "tests/test_training_simple.py",
            "--dataset-type synthetic",
            "--train-num-samples 4",
            "--device npu:0",
            "--precision fp32",
            "--logs none",
            "Running with a single process. Device npu:0.",
            "Train Epoch: 0",
            "NPU training PASSED",
            'load="upstream_ref>>ref"',
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)

        quick_start = text.split("## Quick Start：单卡预训练图文推理", 1)[1]
        self.assertEqual(quick_start.count('#test id="npu-inference"'), 1)
        self.assertEqual(quick_start.count('#test id="npu-training"'), 1)
        training = quick_start.split("## 单卡最小训练验证", 1)[1]
        self.assertIn("set -euo pipefail", training)
        self.assertNotIn("torchrun", quick_start)
        self.assertNotIn("--dist-backend", quick_start)

    def test_e2e_test_prepares_the_npu_and_hf_environment(self) -> None:
        test_path = (
            _REPO_ROOT
            / "projects"
            / "open_clip"
            / "tests"
            / "test_quick_start_ascend.py"
        )
        self.assertTrue(test_path.is_file(), test_path)
        text = test_path.read_text(encoding="utf-8")
        required = (
            "MarkdownDocTestBase",
            "NPU_READY",
            "/usr/local/Ascend/ascend-toolkit/set_env.sh",
            "torch==2.9.0",
            "torch_npu==2.9.0.post2",
            "torchvision==0.24.0",
            "HF_ENDPOINT",
            "https://hf-mirror.com",
            "HF_HUB_DISABLE_XET",
            "PIP_CONSTRAINT",
            "UV_CONSTRAINT",
            "ASCEND_RT_VISIBLE_DEVICES",
            "ERR99999",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)

    def test_environment_setup_defers_torchvision_import_to_doc(self) -> None:
        test_path = (
            _REPO_ROOT
            / "projects"
            / "open_clip"
            / "tests"
            / "test_quick_start_ascend.py"
        )
        text = test_path.read_text(encoding="utf-8")
        self.assertNotIn(
            "import torch, torch_npu, torchvision",
            text,
            "torchvision imports Pillow; defer its import until the document "
            "has installed the complete open_clip dependency set",
        )


if __name__ == "__main__":
    unittest.main()

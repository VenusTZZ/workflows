"""Static contract for the flash-linear-attention Ascend Quick Start guard."""

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


class TestFlashLinearAttentionProjectContract(unittest.TestCase):
    def test_registry_points_to_quick_start_workflow(self) -> None:
        registry = yaml.safe_load(
            (_REPO_ROOT / "projects.yaml").read_text(encoding="utf-8")
        )
        projects = [
            item
            for item in registry["projects"]
            if item["name"] == "flash-linear-attention"
        ]
        self.assertEqual(len(projects), 1)
        project = projects[0]
        self.assertEqual(project["upstream_repo"], "fla-org/flash-linear-attention")
        self.assertEqual(project["runner"], "linux-aarch64-a2-1")
        self.assertEqual(
            project["workflows"],
            {
                "quick_start": (
                    ".github/workflows/flash-linear-attention-quick-start.yml"
                ),
                "examples": ".github/workflows/flash-linear-attention-examples.yml",
            },
        )

    def test_workflow_uses_shared_template_and_swr_image(self) -> None:
        workflow_path = (
            _REPO_ROOT
            / ".github"
            / "workflows"
            / "flash-linear-attention-quick-start.yml"
        )
        self.assertTrue(workflow_path.is_file(), workflow_path)
        workflow = workflow_path.read_text(encoding="utf-8")
        self.assertIn("uses: ./.github/workflows/quick-start-template.yml", workflow)
        self.assertIn("project: flash-linear-attention", workflow)
        self.assertIn("linux-aarch64-a2-1", workflow)
        self.assertIn(
            "swr.cn-south-1.myhuaweicloud.com/ascendhub/"
            "cann:9.0.0-910b-ubuntu22.04-py3.11",
            workflow,
        )
        self.assertNotIn("ascendai/cann:9.0.0", workflow)
        self.assertIn("container_options: '--shm-size=16g'", workflow)
        self.assertIn("upstream_repo: fla-org/flash-linear-attention", workflow)

    def test_doc_executes_release_install_and_real_gdn_backward(self) -> None:
        doc_path = (
            _REPO_ROOT
            / "projects"
            / "flash-linear-attention"
            / "docs"
            / "Quick-start-Ascend.md"
        )
        self.assertTrue(doc_path.is_file(), doc_path)
        commands, results = _Parser().parse(doc_path.read_text(encoding="utf-8"))
        test_ids = {command.id for command in commands if hasattr(command, "id")}
        self.assertEqual(
            test_ids,
            {"check-cann", "install-fla", "check-npu", "gdn-forward-backward"},
        )
        self.assertEqual(set(results), test_ids)

        text = doc_path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# Quick Start (Ascend NPU)\n"))
        self.assertIn("## 前置条件", text)
        self.assertIn("### 硬件", text)
        self.assertIn("### 基础软件", text)
        self.assertIn("### 本文档示例使用的版本", text)
        self.assertIn("### 检查前置是否满足", text)
        self.assertIn("## 安装 flash-linear-attention", text)
        self.assertIn("## 使用样例", text)
        self.assertIn("| Python | 3.11 |", text)
        self.assertIn("| CANN | 9.0.0 |", text)
        self.assertNotIn("# Quick Start：在昇腾 NPU 上运行 GatedDeltaNet", text)
        self.assertIn("fla-org/flash-linear-attention/blob/main/INSTALL.md", text)
        self.assertIn('store="upstream_ref"', text)
        self.assertIn('load="upstream_ref>>UPSTREAM_REF"', text)
        self.assertIn("git checkout <UPSTREAM_REF>", text)
        self.assertIn('python -m pip install -q ".[npu]"', text)
        self.assertIn("triton-ascend.osinfra.cn/pypi/simple", text)
        self.assertIn("GatedDeltaNet", text)
        self.assertIn("loss.backward()", text)
        self.assertIn("torch.npu.synchronize()", text)
        self.assertIn("assert IS_NPU", text)
        self.assertNotIn("cuda:0", text)


if __name__ == "__main__":
    unittest.main()

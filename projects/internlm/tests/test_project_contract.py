"""Static contract for the InternLM Ascend Quick Start guard."""

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


class TestInternLMProjectContract(unittest.TestCase):
    def test_registry_points_to_main_branch_quick_start(self) -> None:
        registry = yaml.safe_load(
            (_REPO_ROOT / "projects.yaml").read_text(encoding="utf-8")
        )
        projects = [
            item for item in registry["projects"] if item["name"] == "internlm"
        ]
        self.assertEqual(len(projects), 1)
        project = projects[0]
        self.assertEqual(project["upstream_repo"], "InternLM/InternLM")
        self.assertEqual(project["runner"], "linux-aarch64-a2-1")
        self.assertEqual(project["dir"], "projects/internlm")
        self.assertEqual(
            project["workflows"],
            {"quick_start": ".github/workflows/internlm-quick-start.yml"},
        )

    def test_workflow_uses_shared_template_main_and_modelscope_cache(self) -> None:
        workflow_path = (
            _REPO_ROOT / ".github" / "workflows" / "internlm-quick-start.yml"
        )
        self.assertTrue(workflow_path.is_file(), workflow_path)
        text = workflow_path.read_text(encoding="utf-8")
        required = (
            "uses: ./.github/workflows/quick-start-template.yml",
            "project: internlm",
            "test_runner: '[\"linux-aarch64-a2-1\"]'",
            "cann:9.1.0-910b-ubuntu22.04-py3.12",
            "upstream_repo: InternLM/InternLM",
            "fixed_ref: main",
            "--volume=/data/ci-cache/modelscope/internlm:/root/.cache/modelscope",
            "timeout_minutes: 180",
            "workflow_dispatch:",
            "projects/internlm/docs/Quick-start-Ascend.md?ref=${{ github.sha }}",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)
        self.assertNotIn("--device=/dev/davinci", text)
        self.assertNotIn("/usr/local/Ascend/driver", text)

    def test_doc_tracks_upstream_npu_example_and_runs_one_npu_inference(self) -> None:
        doc_path = (
            _REPO_ROOT
            / "projects"
            / "internlm"
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
                "checkout-upstream",
                "install-deps",
                "download-model",
                "npu-inference",
            },
        )
        self.assertEqual(set(results), test_ids)

        required = (
            "## 前置条件",
            "### 本文档示例使用的版本",
            "## 获取 InternLM 上游源码",
            "## 安装推理依赖",
            "## 下载 InternLM3-8B-Instruct",
            "## Quick Start：单卡 NPU 推理",
            "InternLM/InternLM",
            "ecosystem/README_npu.md",
            "Shanghai_AI_Laboratory/internlm3-8b-instruct",
            "transformers==4.48.0",
            "modelscope==1.37.0",
            "trust_remote_code=True",
            "torch_dtype=torch.float16",
            ").npu()",
            "max_new_tokens=64",
            "do_sample=False",
            "NPU inference PASSED",
            'load="upstream_ref>>ref"',
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)

        self.assertNotIn(".npu\n", text)
        self.assertNotIn("bitsandbytes", text)
        self.assertNotIn("torchrun", text)
        self.assertNotIn("LoRA", text)
        self.assertEqual(text.count('#test id="npu-inference"'), 1)

    def test_constraints_keep_resolution_on_the_cann91_stack(self) -> None:
        constraints = (
            _REPO_ROOT / "projects" / "internlm" / "constraints-npu.txt"
        ).read_text(encoding="utf-8")
        required = (
            "torch==2.9.0",
            "torch-npu==2.9.0.post2",
            "transformers==4.48.0",
            "modelscope==1.37.0",
            "cuda-toolkit<0",
            "nvidia-cuda-runtime-cu12<0",
            "nvidia-cudnn-cu12<0",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, constraints)

    def test_model_download_extracts_only_the_marked_snapshot_path(self) -> None:
        doc = (
            _REPO_ROOT
            / "projects"
            / "internlm"
            / "docs"
            / "Quick-start-Ascend.md"
        ).read_text(encoding="utf-8")
        self.assertIn('print(f"MODEL_DIR={model_dir}")', doc)
        self.assertIn("sed -n 's/^MODEL_DIR=//p'", doc)
        self.assertIn('test -n "$model_dir"', doc)

    def test_e2e_test_sources_cann_and_validates_modelscope_cache(self) -> None:
        test_path = (
            _REPO_ROOT
            / "projects"
            / "internlm"
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
            "PIP_CONSTRAINT",
            "UV_CONSTRAINT",
            "ASCEND_RT_VISIBLE_DEVICES",
            "ensure_safetensors",
            "purge_modelscope_corrupt",
            "resolve_modelscope_cache",
            "ERR99999",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)


if __name__ == "__main__":
    unittest.main()

"""Static contract for the TGI (Ascend NPU) quick-start guard.

Runs without an NPU (no ``NPU_READY`` needed): verifies the registry
entry, the thin trigger, and the label contract of the monitored doc.
"""

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


_DOC = _REPO_ROOT / "projects" / "tgi" / "docs" / "Quick-start-Ascend.md"


class TestTgiProjectContract(unittest.TestCase):
    def test_project_registry_points_to_the_quick_start_workflow(self) -> None:
        registry = yaml.safe_load(
            (_REPO_ROOT / "projects.yaml").read_text(encoding="utf-8")
        )
        projects = [item for item in registry["projects"] if item["name"] == "tgi"]
        self.assertEqual(len(projects), 1)
        project = projects[0]
        self.assertEqual(
            project["upstream_repo"], "cosdt/text-generation-inference"
        )
        self.assertEqual(project["runner"], "linux-aarch64-a2-2")
        self.assertEqual(
            project["workflows"],
            {"quick_start": ".github/workflows/tgi-quick-start.yml"},
        )

    def test_workflow_file_exists_and_uses_the_shared_engine(self) -> None:
        workflow = (
            _REPO_ROOT / ".github" / "workflows" / "tgi-quick-start.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("uses: ./.github/workflows/quick-start-template.yml", workflow)
        self.assertIn("project: tgi", workflow)
        self.assertIn("linux-aarch64-a2-2", workflow)
        self.assertIn("--device=/dev/davinci1", workflow)
        self.assertIn("--device=/dev/hisi_hdc", workflow)
        self.assertIn("cosdt/text-generation-inference", workflow)

    def test_doc_label_contract_parses_and_ids_pair_up(self) -> None:
        text = _DOC.read_text(encoding="utf-8")
        commands, results = _Parser().parse(text)
        test_ids = {command.id for command in commands if hasattr(command, "id")}
        self.assertEqual(
            test_ids,
            {
                "check-python",
                "check-npu-runtime",
                "check-build",
                "smoke-tp2",
            },
        )
        self.assertEqual(set(results), test_ids)

    def test_doc_contains_the_expected_flow(self) -> None:
        text = _DOC.read_text(encoding="utf-8")
        required_fragments = (
            # version table pinned to the CI image / validated stack
            "| Python | 3.12 |",
            "| CANN | 9.1.0 |",
            "| torch | 2.9.0 |",
            "| torch_npu | 2.9.0.post2 |",
            "| transformers | 4.57.6 |",
            "| kernels | 0.5.0 |",
            # the fork under guard + ref checkout
            "https://github.com/cosdt/text-generation-inference.git",
            'git clone --depth 1 --branch "<ref>"',
            # source build: rust toolchain mirror + cargo profile + pyo3/protoc
            "https://rsproxy.cn/rustup-init.sh",
            "--profile release-opt",
            'PYO3_PYTHON="$(command -v python)"',
            'PROTOC="$(command -v protoc)"',
            # python server install + pb generation
            '"kernels==0.5.0"',
            "make -C server gen-server-raw",
            # NPU runtime flags validated on 910B
            "ATTENTION=flashdecoding-npu",
            "PREFIX_CACHING=0",
            "ASCEND_VISIBLE_DEVICES=0,1",
            "count: 2",
            # dual-card HCCL tensor parallelism is part of the guard
            "--num-shard 2",
            # real inference is the green-light semantics: dual-card
            # reply must equal the single-card baseline verbatim
            "TGI-TP2-OK:",
            '"$REPLY" = "<reply_single>"',
            'max_new_tokens":16,"do_sample":false',
            # leftover-process gotcha documented for users
            "EJ0003",
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)

    def test_e2e_test_is_npu_gated_and_sources_cann(self) -> None:
        test_file = (
            _REPO_ROOT
            / "projects"
            / "tgi"
            / "tests"
            / "test_quick_start_ascend.py"
        ).read_text(encoding="utf-8")
        self.assertIn("MarkdownDocTestBase", test_file)
        self.assertIn("NPU_READY", test_file)
        self.assertIn("set_env.sh", test_file)
        self.assertIn("EJ0003", test_file)
        self.assertIn("~/.cargo/bin", test_file)
        self.assertIn("PIP_CONSTRAINT", test_file)
        self.assertIn("UV_CONSTRAINT", test_file)


if __name__ == "__main__":
    unittest.main()

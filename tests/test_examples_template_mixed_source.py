"""Compatibility contract for opt-in project examples in the shared engine."""

from __future__ import annotations

import unittest
from pathlib import Path


_WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "examples-template.yml"
)


class TestExamplesTemplateMixedSource(unittest.TestCase):
    def test_project_source_is_forwarded_without_changing_legacy_default(self) -> None:
        text = _WORKFLOW.read_text(encoding="utf-8")
        self.assertIn(
            "EXAMPLE_SOURCE: ${{ matrix.example.source || 'upstream' }}", text
        )
        self.assertNotIn("CASE_ID:", text)
        self.assertIn("if: matrix.example.npu_devices", text)

    def test_optional_dispatch_repository_and_default_branch_inputs(self) -> None:
        text = _WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("target_repo:\n        description:", text)
        self.assertIn("default_branch:\n        description:", text)
        self.assertIn("target_repo: ${{ steps.decide.outputs.target_repo }}", text)


if __name__ == "__main__":
    unittest.main()

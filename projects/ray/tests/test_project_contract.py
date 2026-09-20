"""Static contract for the Ray Ascend guard project."""

from __future__ import annotations

import ast
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


class TestRayProjectContract(unittest.TestCase):
    def test_project_registry_points_to_both_workflows(self) -> None:
        registry = yaml.safe_load(
            (_REPO_ROOT / "projects.yaml").read_text(encoding="utf-8")
        )
        ray_projects = [
            item for item in registry["projects"] if item["name"] == "ray"
        ]
        self.assertEqual(len(ray_projects), 1)
        ray_project = ray_projects[0]
        self.assertEqual(ray_project["upstream_repo"], "ray-project/ray")
        self.assertEqual(
            ray_project["workflows"],
            {
                "examples": ".github/workflows/ray-examples.yml",
                "quick_start": ".github/workflows/ray-quick-start.yml",
            },
        )

    def test_manifest_contains_upstream_and_project_npu_cases(self) -> None:
        manifest_path = _REPO_ROOT / "projects" / "ray" / "examples_manifest.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        upstream_paths = {
            "python/ray/tests/accelerators/test_npu.py",
            "python/ray/train/tests/test_torch_device_manager.py",
        }
        project_paths = {
            "example/test_npu_discovery.py",
            "example/test_npu_task_actor.py",
            "example/test_npu_resource_lifecycle.py",
            "example/test_npu_worker_recovery.py",
            "example/test_npu_fractional.py",
            "example/test_npu_train_single.py",
            "example/test_npu_train_hccl.py",
            "example/test_npu_train_ddp.py",
            "example/test_npu_train_resume.py",
            "example/test_npu_data_batch.py",
            "example/test_npu_data_actor.py",
            "example/test_npu_serve_inference.py",
            "example/test_npu_serve_batching.py",
            "example/test_npu_tune_trials.py",
        }
        self.assertEqual(set(manifest["scan"]["paths"]), upstream_paths)
        supported = manifest["supported"]
        self.assertEqual(
            {entry["path"] for entry in supported}, upstream_paths | project_paths
        )
        self.assertTrue(all("case_id" not in entry for entry in supported))
        self.assertEqual(
            {entry["path"] for entry in supported if entry["source"] == "project"},
            project_paths,
        )
        for path in project_paths:
            self.assertTrue((_REPO_ROOT / "projects" / "ray" / path).is_file(), path)
        self.assertFalse(any("multi_node" in path for path in project_paths))
        self.assertEqual(manifest["unsupported"], [])

    def test_quick_start_covers_environment_detection_and_isolation(self) -> None:
        doc_path = _REPO_ROOT / "projects" / "ray" / "docs" / "Quick-start-Ascend.md"
        commands, results = _Parser().parse(doc_path.read_text(encoding="utf-8"))
        test_ids = {command.id for command in commands if hasattr(command, "id")}
        self.assertEqual(
            test_ids,
            {
                "check-py",
                "check-torch",
                "ray-install",
                "ray-detects-npus",
                "ray-isolates-npus",
            },
        )
        self.assertEqual(set(results), test_ids)
        text = doc_path.read_text(encoding="utf-8")
        self.assertIn("## 前置条件", text)
        self.assertIn("### 硬件", text)
        self.assertIn("### 基础软件", text)
        self.assertIn("### 本文档示例使用的版本", text)
        self.assertIn("### 检查前置是否满足", text)
        self.assertIn("| torch_npu | 2.9.0.post2 |", text)
        self.assertIn("| Ray | 当前最新 release，Linux aarch64 wheel |", text)
        self.assertIn("## 安装 Ray", text)
        self.assertIn('python -m pip install -q -U "ray[default]"', text)
        self.assertNotIn("ray[default]==", text)
        self.assertIn("ray-core/scheduling/accelerators", text)
        self.assertIn('resources={"NPU": 1}', text)
        self.assertIn("ASCEND_RT_VISIBLE_DEVICES", text)

    def test_project_scripts_and_workflows_exist(self) -> None:
        expected = [
            "projects/ray/scripts/setup_example.sh",
            "projects/ray/scripts/run_example.sh",
            "projects/ray/tests/test_quick_start_ascend.py",
            ".github/workflows/ray-examples.yml",
            ".github/workflows/ray-quick-start.yml",
        ]
        missing = [path for path in expected if not (_REPO_ROOT / path).is_file()]
        self.assertEqual(missing, [])

        quick_workflow = (
            _REPO_ROOT / ".github" / "workflows" / "ray-quick-start.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("uses: ./.github/workflows/quick-start-template.yml", quick_workflow)

    def test_example_setup_routes_release_to_pypi_and_dev_to_commit_wheel(
        self,
    ) -> None:
        setup_path = _REPO_ROOT / "projects" / "ray" / "scripts" / "setup_example.sh"
        text = setup_path.read_text(encoding="utf-8")
        dev_branch = 'if [[ "$version" == *dev* ]]; then'
        self.assertIn(dev_branch, text)
        self.assertIn("ray-wheels/master", text)
        self.assertIn("python/ray/_version.py", text)
        self.assertIn("manylinux2014_aarch64.whl", text)
        self.assertIn("requirement=\"ray[${extras}] @ ${wheel_url}\"", text)
        self.assertLess(text.index(dev_branch), text.index('wheel_url="'))
        self.assertIn('python -m pip install "ray[${extras}]==${version}"', text)
        self.assertNotIn("falling back to the released aarch64 wheel", text)
        self.assertIn("install_target_ray train", text)

    def test_train_profile_bootstraps_torch_after_npu_preflight(self) -> None:
        setup_text = (
            _REPO_ROOT / "projects" / "ray" / "scripts" / "setup_example.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("ensure_torch_stack", setup_text)
        setup_train = setup_text.split("setup_train()", 1)[1].split(
            "supported_profiles()", 1
        )[0]
        self.assertIn("ensure_torch_stack", setup_train)

        source_cann = setup_text.split("source_cann()", 1)[1].split(
            "ensure_torch_stack()", 1
        )[0]
        self.assertNotIn("import torch", source_cann)

    def test_example_test_dependencies_follow_the_target_requirements(self) -> None:
        setup_text = (
            _REPO_ROOT / "projects" / "ray" / "scripts" / "setup_example.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("install_test_dependencies", setup_text)
        self.assertIn("python/requirements/test-requirements.txt", setup_text)
        self.assertIn(
            "python/requirements/ml/train-test-requirements.txt", setup_text
        )
        self.assertIn(
            'resolver_args=(--requirement "$test_requirements" pytest)',
            setup_text,
        )
        self.assertIn(
            'resolver_args+=(--requirement "$train_requirements" boto3)',
            setup_text,
        )
        self.assertIn("resolve_test_requirements.py", setup_text)
        self.assertNotIn("--constraint", setup_text)
        self.assertNotIn("pytest==7.4.4", setup_text)
        self.assertNotIn("boto3==1.29.7", setup_text)
        self.assertNotIn("python -m pip install pytest mock", setup_text)

    def test_project_npu_profiles_install_only_needed_ray_extras(self) -> None:
        setup_text = (
            _REPO_ROOT / "projects" / "ray" / "scripts" / "setup_example.sh"
        ).read_text(encoding="utf-8")
        for profile, extra in (
            ("npu", '""'),
            ("data", "data"),
            ("serve", "serve"),
            ("tune", "tune"),
        ):
            self.assertIn(f"setup_{profile}()", setup_text)
            section = setup_text.split(f"setup_{profile}()", 1)[1].split("\n}", 1)[0]
            self.assertIn("ensure_torch_stack", section)
            self.assertIn(f"install_target_ray {extra}", section)
            self.assertIn("install_test_dependencies", section)
        self.assertIn("PIP_INDEX_URL", setup_text)
        self.assertIn("PIP_TRUSTED_HOST", setup_text)

    def test_example_setup_links_tests_from_the_target_checkout(self) -> None:
        setup_text = (
            _REPO_ROOT / "projects" / "ray" / "scripts" / "setup_example.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("link_target_ray_tests", setup_text)
        self.assertIn("python/ray/setup-dev.py", setup_text)
        self.assertIn('--yes --allow tests', setup_text)
        self.assertIn("verify_ray_test_support.py", setup_text)

        setup_core = setup_text.split("setup_core()", 1)[1].split(
            "setup_train()", 1
        )[0]
        setup_train = setup_text.split("setup_train()", 1)[1].split(
            "supported_profiles()", 1
        )[0]
        self.assertIn("link_target_ray_tests", setup_core)
        self.assertIn("link_target_ray_tests", setup_train)

    def test_example_workflow_uses_shared_engine(self) -> None:
        workflow_path = _REPO_ROOT / ".github" / "workflows" / "ray-examples.yml"
        text = workflow_path.read_text(encoding="utf-8")
        self.assertIn("uses: ./.github/workflows/examples-template.yml", text)
        self.assertIn("project: ray", text)
        self.assertIn("upstream_repo: ray-project/ray", text)
        self.assertIn("max_parallel: 2", text)
        self.assertNotIn("run-example:", text)

    def test_train_examples_attach_checkpoint_to_reported_metrics(self) -> None:
        for name in ("test_npu_train_single.py", "test_npu_train_hccl.py"):
            with self.subTest(name=name):
                path = _REPO_ROOT / "projects" / "ray" / "example" / name
                tree = ast.parse(path.read_text(encoding="utf-8"))
                reports = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "train"
                    and node.func.attr == "report"
                ]
                self.assertEqual(len(reports), 1)
                self.assertIn("checkpoint", {kw.arg for kw in reports[0].keywords})

    def test_hccl_checkpoint_is_created_only_by_rank_zero(self) -> None:
        path = _REPO_ROOT / "projects" / "ray" / "example" / "test_npu_train_hccl.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rank_zero_branches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If) and ast.unparse(node.test) == "rank == 0"
        ]
        self.assertEqual(len(rank_zero_branches), 1)
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and ast.unparse(node.func) == "train.Checkpoint.from_directory"
                for node in ast.walk(rank_zero_branches[0])
            )
        )


if __name__ == "__main__":
    unittest.main()

"""Static contract tests for the DeepSpeed examples guard."""

from __future__ import annotations

import unittest
from pathlib import Path, PurePosixPath

import yaml


REPO = Path(__file__).resolve().parents[3]
PROJECT = REPO / "projects" / "deepspeed"
MANIFEST = PROJECT / "examples_manifest.yaml"
RUN_SCRIPT = PROJECT / "scripts" / "run_example.sh"
SETUP_SCRIPT = PROJECT / "scripts" / "setup_example.sh"
ACTIONLINT = REPO / ".github" / "actionlint.yaml"


class DeepSpeedExamplesContractTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
        cls.supported = cls.manifest["supported"]
        cls.by_path = {entry["path"]: entry for entry in cls.supported}
        cls.unsupported = set(cls.manifest["unsupported"])
        cls.run_script = RUN_SCRIPT.read_text(encoding="utf-8")
        cls.setup_script = SETUP_SCRIPT.read_text(encoding="utf-8")

    def test_first_phase_has_ten_unique_matrix_entries(self) -> None:
        self.assertEqual(len(self.supported), 10)
        names = [PurePosixPath(entry["path"]).stem
                 for entry in self.supported]
        self.assertEqual(len(names), len(set(names)), names)

    def test_all_runner_sizes_are_registered(self) -> None:
        labels = yaml.safe_load(ACTIONLINT.read_text(encoding="utf-8"))[
            "self-hosted-runner"]["labels"]
        self.assertEqual(
            labels,
            ["linux-aarch64-a2-1", "linux-aarch64-a2-2",
             "linux-aarch64-a2-4", "linux-aarch64-a2-8"],
        )

    def test_multicard_entries_match_their_launch_recipes(self) -> None:
        moe = self.by_path["training/cifar/run_ds_moe.sh"]
        self.assertEqual(moe["runner"], "linux-aarch64-a2-2")
        self.assertEqual(moe["profile"], "ds_cifar")
        moe_launcher = self.run_script.split("run_cifar_moe()", 1)[1].split(
            "run_autotp_equivalence()", 1
        )[0]
        self.assertIn("TORCH_COMPILE_DISABLE=1", moe_launcher)
        self.assertIn("--num_gpus 2", moe_launcher)
        self.assertIn("--ep-world-size 2", moe_launcher)
        self.assertIn("--num-experts 2", moe_launcher)
        self.assertEqual(self.run_script.count("TORCH_COMPILE_DISABLE=1"), 1)
        self.assertNotIn("export TORCH_COMPILE_DISABLE=1", self.run_script)

        autotp = self.by_path["training/autotp_equivalence"]
        self.assertEqual(autotp["runner"], "linux-aarch64-a2-4")
        self.assertEqual(autotp["profile"], "ds_autotp_equivalence")
        self.assertEqual(
            autotp["exec"], "training/autotp_equivalence/train.py")
        self.assertIn("for size in 1 3 4", self.run_script)
        self.assertIn("require_visible_devices 4 '0,1,2,3'",
                      self.run_script)
        self.assertIn("QWEN3_06B_PATH", self.run_script)

    def test_promoted_entries_are_not_unsupported(self) -> None:
        self.assertNotIn("training/cifar/run_ds_moe.sh", self.unsupported)
        self.assertNotIn(
            "training/autotp_equivalence/train.py", self.unsupported)
        self.assertIn("training/cifar/run_ds_prmoe.sh", self.unsupported)

    def test_ds_chat_keeps_launcher_semantics(self) -> None:
        chat_entries = [
            entry for entry in self.supported
            if entry.get("exec", "").startswith(
                "applications/DeepSpeed-Chat/training/")
        ]
        self.assertEqual(len(chat_entries), 4)
        for entry in chat_entries:
            self.assertIn("--deepspeed", entry["overlay_args"])
            self.assertEqual(entry["runner"], "linux-aarch64-a2-1")
        self.assertIn("deepspeed --master_port", self.run_script)
        self.assertIn("--num_gpus 1", self.run_script)
        self.assertIn('PYTHONPATH=$PYTHONPATH', self.setup_script)
        self.assertIn("import dschat", self.setup_script)

    def test_cifar_is_pre_staged_from_verified_modelscope_revision(self) -> None:
        cifar = self.by_path["training/cifar"]
        moe = self.by_path["training/cifar/run_ds_moe.sh"]
        self.assertEqual(cifar["profile"], "ds_cifar")
        self.assertEqual(moe["profile"], "ds_cifar")
        self.assertIn("setup_ds_cifar", self.setup_script)
        self.assertIn(
            "9231e736fd8d53f7158165a07d801429b4414993",
            self.setup_script,
        )
        self.assertIn(
            "4f287e6733e987d5c1ab2af557413cf0f5bc78f293765d87dc5633788246e5d7",
            self.setup_script,
        )
        self.assertIn("dataset._check_integrity()", self.setup_script)

    def test_setup_preserves_npu_torch_and_source_deepspeed(self) -> None:
        self.assertIn('python -m pip install -e "$src"', self.setup_script)
        self.assertNotIn('pip install --no-deps -e "$EXAMPLES_ROOT/',
                         self.setup_script)
        self.assertIn("DeepSpeed was not imported from target source",
                      self.setup_script)
        self.assertNotIn("pip install -r", self.setup_script)
        self.assertIn('"torchvision==0.24.0"', self.setup_script)


if __name__ == "__main__":
    unittest.main()

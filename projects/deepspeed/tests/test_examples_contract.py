"""Static contract tests for the DeepSpeed examples guard."""

from __future__ import annotations

import json
import shlex
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

    def test_guard_has_thirty_eight_unique_matrix_entries(self) -> None:
        self.assertEqual(len(self.supported), 38)
        names = [str(PurePosixPath(entry["path"]).with_suffix(""))
                 for entry in self.supported]
        self.assertEqual(len(names), len(set(names)), names)
        self.assertFalse(set(self.by_path) & self.unsupported)

    def test_new_expansion_paths_have_profiles_and_do_not_overlap(self) -> None:
        scripts = "\n".join(path.read_text(encoding="utf-8")
                            for path in (PROJECT / "scripts").glob("*.sh"))
        for entry in self.supported:
            self.assertIn(f"setup_{entry['profile']}()", scripts)
        promoted = {
            "training/opsd/main.py", "training/opsd/test_student_autotp_zero3.py",
            "training/opsd/test_teacher_autotp_zero3.py",
            "training/deepspeed_finetune_demo/finetune_llama.py",
            "training/DeepSpeed-ZenFlow/finetuning/finetune_llama.py",
            "training/stable_diffusion/train_sd_distil_lora.py",
            "inference/huggingface/automatic-speech-recognition/test-wav2vec2.py",
            "inference/huggingface/translation/test-t5-base.py",
        }
        self.assertTrue(promoted <= set(self.by_path))
        self.assertFalse(promoted & self.unsupported)
        for dead in ("compression/bert/run_glue_lkd.py", "compression/cifar/train.py",
                     "compression/gpt2/run_clm_no_trainer.py", "training/MoQ/run_glue.py"):
            self.assertNotIn(dead, self.unsupported)

    def test_all_unsupported_comments_precede_entries(self) -> None:
        text = MANIFEST.read_text(encoding="utf-8").split("unsupported:", 1)[1]
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if line.startswith("  - "):
                self.assertTrue(index > 0 and lines[index - 1].startswith("  # "), line)
                self.assertNotIn(" # ", line)

    def test_all_runner_sizes_are_registered(self) -> None:
        labels = yaml.safe_load(ACTIONLINT.read_text(encoding="utf-8"))[
            "self-hosted-runner"]["labels"]
        self.assertEqual(
            labels,
            ["linux-aarch64-a2-1", "linux-aarch64-a2-2",
             "linux-aarch64-a2-4", "linux-aarch64-a2-8"],
        )

    def test_thin_trigger_sets_deepspeed_fallback_branch(self) -> None:
        workflow = yaml.safe_load((REPO / ".github" / "workflows" /
                                   "deepspeed-examples.yml").read_text(encoding="utf-8"))
        self.assertEqual(set(workflow["jobs"]), {"deepspeed-examples"})
        job = workflow["jobs"]["deepspeed-examples"]
        self.assertEqual(job["uses"], "./.github/workflows/examples-template.yml")
        self.assertEqual(job["with"], {
            "project": "deepspeed",
            "upstream_repo": "deepspeedai/DeepSpeed",
            "examples_repo": "deepspeedai/DeepSpeedExamples",
            "default_branch": "master",
            "target_ref": "${{ inputs.target_ref }}",
            "max_parallel": 4,
        })

    def test_multicard_entries_match_their_launch_recipes(self) -> None:
        moe = self.by_path["training/cifar/run_ds_moe.sh"]
        self.assertEqual(moe["runner"], "linux-aarch64-a2-2")
        self.assertEqual(moe["profile"], "ds_cifar")
        moe_launcher = self.run_script.split("run_cifar_moe()", 1)[1].split(
            "run_cifar_prmoe()", 1
        )[0]
        self.assertIn("TORCH_COMPILE_DISABLE=1", moe_launcher)
        self.assertIn("--num_gpus 2", moe_launcher)
        self.assertIn("--ep-world-size 2", moe_launcher)
        self.assertIn("--num-experts 2", moe_launcher)
        self.assertEqual(self.run_script.count("TORCH_COMPILE_DISABLE=1"), 2)
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
        promoted = {
            "training/cifar/run_ds_prmoe.sh",
            "training/tensor_parallel/hf_integration/train.py",
            "training/data_efficiency/variable_batch_size_and_lr/variable_batch_size_and_lr_example.py",
            "training/DeepSpeed-ZenFlow/benchmark/zf_benchmark.py",
            "compression/reasoning_aware_compression/prune.py",
            "applications/DeepSpeed-Chat/training/step1_supervised_finetuning/prompt_eval.py",
            "applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/rw_eval.py",
            "training/tensor_parallel/hf_integration/train_bench_length.py",
            "training/DeepSpeed-SuperOffload/finetune_zero3.py",
        }
        self.assertTrue(promoted <= set(self.by_path))
        self.assertFalse(promoted & self.unsupported)
        self.assertIn("training/bf16_master_weight/train.py", self.unsupported)
        self.assertIn("training/pipeline_parallelism/train.py", self.unsupported)

    def test_new_recipes_preserve_features_and_bound_work(self) -> None:
        def args(path: str) -> dict[str, str]:
            tokens = shlex.split(" ".join(self.by_path[path]["overlay_args"]))
            return {token: tokens[i + 1] if i + 1 < len(tokens) and
                    not tokens[i + 1].startswith("--") else ""
                    for i, token in enumerate(tokens) if token.startswith("--")}

        hf_path = "training/tensor_parallel/hf_integration/train.py"
        hf = args(hf_path)
        self.assertEqual(self.by_path[hf_path]["runner"], "linux-aarch64-a2-8")
        self.assertEqual(hf["--model_name_or_path"], "${OPT_125M_PATH}")
        self.assertEqual(hf["--data_path"], "${ALPACA_CI_PATH}")
        self.assertEqual(hf["--max_steps"], "3")
        self.assertEqual(hf["--tf32"], "False")

        prmoe = self.by_path["training/cifar/run_ds_prmoe.sh"]
        self.assertEqual(prmoe["runner"], "linux-aarch64-a2-2")
        self.assertEqual(prmoe["profile"], "ds_cifar")

        zen_path = "training/DeepSpeed-ZenFlow/benchmark/zf_benchmark.py"
        zen = args(zen_path)
        self.assertEqual(self.by_path[zen_path]["runner"], "linux-aarch64-a2-2")
        update_interval = int(zen["--update_intervals"])
        iterations = int(zen["--iteration"]) * update_interval
        # Both post-warmup timing buckets must be nonempty in upstream code.
        measured = range(update_interval, iterations)
        self.assertTrue(any((i + 1) % update_interval == 0 for i in measured))
        self.assertTrue(any((i + 1) % update_interval != 0 for i in measured))

        prune_path = "compression/reasoning_aware_compression/prune.py"
        prune = args(prune_path)
        self.assertEqual(prune["--device"], "npu")
        self.assertEqual(prune["--pruning-method"], "wanda")
        self.assertEqual(prune["--calibration"], "prompt")
        self.assertEqual(prune["--dataset"], "${RAC_CI_PATH}")
        # OPT fc1/fc2 names do not match upstream's scope=mlp filter.
        self.assertEqual(prune["--scope"], "all")
        self.assertEqual(int(prune["--nsamples"]) * int(prune["--seqlen"]), 128)

    def test_hf_autotp_warmup_is_valid_for_deepspeed_scheduler(self) -> None:
        entry = self.by_path["training/tensor_parallel/hf_integration/train.py"]
        tokens = shlex.split(" ".join(entry["overlay_args"]))
        warmup = int(tokens[tokens.index("--warmup_steps") + 1])
        steps = int(tokens[tokens.index("--max_steps") + 1])
        # Run #27: HF fills the template's auto warmup from TrainingArguments;
        # DS WarmupDecayLR rejects zero and clamps a positive value to >=2.
        self.assertEqual(warmup, 2)
        self.assertEqual(steps, 3)
        self.assertGreaterEqual(warmup, 2)
        self.assertLess(warmup, steps)

    def test_unsupported_omits_non_example_support_files(self) -> None:
        non_examples = {
            "applications/DeepSpeed-Chat/chat.py",
            "applications/DeepSpeed-Chat/setup.py",
            "applications/DeepSpeed-Chat/dschat",
            "applications/DeepSpeed-Chat/tests/test_training.py",
            "applications/DeepSpeed-VisualChat/helper/extract_qwen_vl.py",
            "benchmarks/inference/mii/src/client.py",
            "benchmarks/inference/mii/src/server.py",
            "compression/reasoning_aware_compression/rac",
            "training/autotp_equivalence/compare_loss.py",
            "training/pipeline_parallelism/alexnet.py",
            "training/imagenet/extract_ILSVRC.sh",
            "training/offload_states/output_table.py",
            "scripts/check-license.py",
        }
        self.assertFalse(non_examples & self.unsupported)
        for path in self.unsupported:
            self.assertNotIn("tests", PurePosixPath(path).parts)
            self.assertNotIn(PurePosixPath(path).name, {"__init__.py", "setup.py"})
        # A real inference/evaluation entry is not excluded merely because it
        # needs a trained checkpoint. Only the redundant chat.py wrapper is removed.
        self.assertIn("applications/DeepSpeed-Chat/inference/chatbot.py", self.unsupported)
        self.assertIn("applications/DeepSpeed-Chat/training/step1_supervised_finetuning/prompt_eval.py",
                      self.by_path)

    def test_first_stage_eval_and_offload_recipes(self) -> None:
        def options(path: str) -> dict[str, str]:
            tokens = shlex.split(" ".join(self.by_path[path]["overlay_args"]))
            return dict(zip(tokens[::2], tokens[1::2]))

        prompt_path = "applications/DeepSpeed-Chat/training/step1_supervised_finetuning/prompt_eval.py"
        prompt = options(prompt_path)
        self.assertEqual(self.by_path[prompt_path]["runner"], "linux-aarch64-a2-1")
        self.assertEqual(prompt["--model_name_or_path_baseline"], "${OPT_125M_PATH}")
        self.assertEqual(prompt["--model_name_or_path_finetune"],
                         "${CI_OUTPUT_DIR}/prompt-eval-sft")
        self.assertEqual(prompt["--max_new_tokens"], "8")

        reward_path = "applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/rw_eval.py"
        self.assertEqual(self.by_path[reward_path]["profile"], "ds_chat_reward_eval")
        self.assertEqual(options(reward_path)["--model_name_or_path"], "${OPT_125M_PATH}")
        self.assertEqual(self.by_path[reward_path]["runner"], "linux-aarch64-a2-1")

        bench_path = "training/tensor_parallel/hf_integration/train_bench_length.py"
        bench = options(bench_path)
        self.assertEqual(self.by_path[bench_path]["runner"], "linux-aarch64-a2-8")
        self.assertEqual(bench["--model_name_or_path"], "${OPT_125M_PATH}")
        self.assertEqual(bench["--data_path"], "${ALPACA_CI_PATH}")
        self.assertEqual((bench["--model_max_length"], bench["--max_steps"],
                          bench["--warmup_steps"]), ("128", "3", "2"))

        offload_path = "training/DeepSpeed-SuperOffload/finetune_zero3.py"
        offload = options(offload_path)
        self.assertEqual(self.by_path[offload_path]["runner"], "linux-aarch64-a2-2")
        self.assertEqual(offload["--model_name"], "${OPT_125M_PATH}")
        self.assertEqual(offload["--dataset_name"], "${ALPACA_DATASET_DIR}")
        self.assertEqual((offload["--bench_steps"], offload["--warmup_steps"]), ("3", "1"))
        self.assertEqual(offload["--attn_implementation"], "eager")
        # Source create_optimizer() fixes the actual CPU Adam lr to 0.001;
        # args.lr only labels logs. Keep CLI/documentation honest, not a fake override.
        self.assertEqual(offload["--lr"], "0.001")
        self.assertNotIn("--save_checkpoint", offload)

    def test_first_stage_profiles_are_available(self) -> None:
        profiles = {"ds_chat_prompt_eval", "ds_chat_reward_eval",
                    "ds_hf_bench_length", "ds_superoffload"}
        for profile in profiles:
            self.assertIn(f"setup_{profile}()", self.setup_script)
        self.assertIn('"openai==0.28.1"', self.setup_script)
        self.assertIn("ALPACA_DATASET_DIR", self.setup_script)
        self.assertIn("dataset_dict128.pkl", self.setup_script)

    def test_ci_fixture_schemas(self) -> None:
        fixtures = PROJECT / "fixtures"
        alpaca = json.loads((fixtures / "ci_alpaca_16.json").read_text(encoding="utf-8"))
        self.assertEqual(len(alpaca), 16)
        for row in alpaca:
            self.assertEqual(set(row), {"instruction", "input", "output"})
            self.assertTrue(row["instruction"])
            self.assertTrue(row["output"])
        calibration = [json.loads(line) for line in
                       (fixtures / "ci_rac_8.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(calibration), 8)
        self.assertTrue(all(isinstance(row["prompt"], str) and row["prompt"]
                            for row in calibration))

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

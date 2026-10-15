"""Static contracts for additional native, no-source-patch setup profiles."""

from pathlib import Path
import re
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETUP = (PROJECT_ROOT / "scripts/setup_example.sh").read_text(encoding="utf-8")


def function_body(name):
    match = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}$", SETUP, re.MULTILINE | re.DOTALL)
    if match is None:
        raise AssertionError(f"setup function missing: {name}")
    return match.group(1)


class NewSetupProfileContracts(unittest.TestCase):
    def test_profiles_protect_npu_stack_and_source(self):
        for profile in ("ds_pin_memory", "ds_zenflow_finetune", "ds_opsd"):
            body = function_body(f"setup_{profile}")
            self.assertIn("install_deepspeed_source", body)
            self.assertIn("install_example_dependencies", body)
            self.assertNotIn("pip install", body)
            self.assertIn(f'"{profile}"', function_body("verify_installed_runtime"))

    def test_pin_preflight_raises_only_shell_soft_limit(self):
        body = function_body("setup_ds_pin_memory")
        self.assertIn('memlock_hard="$(ulimit -Hl)"', body)
        self.assertIn('ulimit -Sl "$memlock_hard"', body)
        self.assertNotIn("ulimit -Hl ", body)
        self.assertNotIn("setrlimit", body)
        self.assertIn("required = 4 * 1024 * 1024", body)
        self.assertIn("soft < required", body)
        self.assertIn("PinMemoryBuilder()", body)
        self.assertIn("CPUAdamBuilder()", body)

    def test_pin_preflight_uses_real_native_allocator_and_npu(self):
        body = function_body("setup_ds_pin_memory")
        self.assertIn("from deepspeed.utils.pin_memory import get_native_pinned_memory", body)
        self.assertIn('os.environ["DS_PIN_MEMORY_REGISTER_DEVICE"] = "0"', body)
        self.assertIn("locked = manager.pin(raw)", body)
        self.assertIn("manager.is_pinned(locked)", body)
        self.assertIn("locked.to(accelerator.current_device_name())", body)
        self.assertIn("manager.unpin(locked)", body)

    def test_zenflow_uses_exact_offline_native_local_name(self):
        body = function_body("setup_ds_zenflow_finetune")
        self.assertIn('"transformers==4.57.6"', body)
        self.assertIn('HF_DATASETS_OFFLINE=1 python', body)
        self.assertIn('"tatsu-lab/alpaca/train.parquet"', body)
        self.assertIn("os.chdir(work)", body)
        self.assertIn('load_dataset("tatsu-lab/alpaca")', body)
        self.assertIn("upstream.preprocess_alpaca(row, tokenizer)", body)
        self.assertIn("default_data_collator([tokenized[0], tokenized[1]])", body)
        self.assertNotIn("remove_columns", body)
        self.assertNotIn("sitecustomize", body)

    def test_opsd_has_distinct_models_and_full_vocabulary_guard(self):
        body = function_body("setup_ds_opsd")
        self.assertIn("QWEN25_STUDENT_PATH=Qwen/Qwen2.5-0.5B-Instruct", body)
        self.assertIn("QWEN25_TEACHER_PATH=Qwen/Qwen2.5-0.5B", body)
        self.assertIn("student.get_vocab() != teacher.get_vocab()", body)
        self.assertIn("student.get_added_vocab() != teacher.get_added_vocab()", body)
        self.assertIn("student_config.vocab_size != teacher_config.vocab_size", body)
        self.assertIn("if not student.chat_template:", body)
        self.assertIn("from deepspeed.runtime.rollout import RolloutConfig, build_rollout", body)
        self.assertIn("upstream.PromptDataset", body)
        self.assertIn("upstream.LeftPaddedPromptCollator", body)
        self.assertIn("ci_rac_8.jsonl OPSD_CI_PATH", body)

    def test_new_profile_embedded_python_compiles(self):
        for profile in ("ds_pin_memory", "ds_zenflow_finetune", "ds_opsd"):
            body = function_body(f"setup_{profile}")
            snippets = re.findall(r"<<'PY'\n(.*?)\nPY", body, re.DOTALL)
            self.assertEqual(len(snippets), 1)
            compile(snippets[0], profile, "exec")


if __name__ == "__main__":
    unittest.main()

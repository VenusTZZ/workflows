"""Recipe contracts and launcher isolation; hardware acceptance stays in Actions."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

import yaml

PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = yaml.safe_load((PROJECT / "examples_manifest.yaml").read_text(encoding="utf-8"))
ENTRIES = {entry["path"]: entry for entry in MANIFEST["supported"]}
NEW = {
    "examples/retool/retool_qwen3_4b_sft.sh": (2, "train_async.py", "slime_retool_sft"),
    "examples/on_policy_distillation/run-qwen3-8B-opd-megatron.sh": (8, "train.py", "slime_opd_megatron"),
    "examples/train_infer_mismatch_helper/run-qwen3-4b-mis.sh": (8, "train.py", "slime_mis"),
    "examples/multi_agent/run-qwen3-30B-A3B-multi-agent.sh": (8, "train.py", "slime_multi_agent"),
    "examples/eval_multi_task/multi_task.sh": (8, "train.py", "slime_multi_task"),
    "examples/strands_sglang/strands_qwen3_8b.sh": (8, "train.py", "slime_strands"),
    "examples/search-r1/run_qwen2.5_3B.sh": (8, "train.py", "slime_search"),
    "examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py": (4, "train.py", "slime_geo3k"),
}


class RecipeTests(unittest.TestCase):
    def test_native_semantics(self):
        sft = " ".join(ENTRIES[next(path for path in NEW if "sft.sh" in path)]["overlay_args"])
        for flag in ("--input-key messages", "--loss-type sft_loss", "--loss-mask-type qwen3", "--debug-train-only", "--calculate-per-token-loss", "--disable-compute-advantages-and-returns"):
            self.assertIn(flag, sft)
        self.assertNotIn("--apply-chat-template", sft)
        opd = " ".join(ENTRIES[next(path for path in NEW if "megatron" in path)]["overlay_args"])
        self.assertIn("--opd-type megatron", opd)
        self.assertIn("--opd-teacher-load ${SLIME_TORCH_DIST_PATH}", opd)
        self.assertNotIn("--rm-url", opd)
        mis = " ".join(ENTRIES[next(path for path in NEW if "mis.sh" in path)]["overlay_args"])
        self.assertIn("--context-parallel-size 2", mis)
        self.assertIn("--tensor-model-parallel-size 2", mis)
        self.assertIn("compute_mis_weights_with_cp", mis)
        self.assertIn("--use-tis", mis)

    def test_fixture(self):
        spec = importlib.util.spec_from_file_location("fixture_validation", PROJECT / "scripts/validate_sft_fixture.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = module.load_messages(PROJECT / "fixtures/ci_retool_sft_8.jsonl")
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row["messages"][-1]["role"] == "assistant" for row in rows))

    def test_profiles_and_classification(self):
        setup = (PROJECT / "scripts/setup_example.sh").read_text(encoding="utf-8")
        self.assertEqual(len(ENTRIES), 11)
        for path, (count, _, profile) in NEW.items():
            self.assertEqual(ENTRIES[path]["runner"], f"linux-aarch64-a2-{count}")
            self.assertIn(f"setup_{profile}()", setup)
            self.assertNotIn(path, MANIFEST["unsupported"])
        self.assertTrue(all(not path.endswith(("__init__.py", ".yaml")) for path in MANIFEST["unsupported"]))

    def test_agent_and_search_semantics(self):
        strands = " ".join(ENTRIES["examples/strands_sglang/strands_qwen3_8b.sh"]["overlay_args"])
        self.assertNotIn("--apply-chat-template", strands)
        self.assertIn("generate_with_strands.generate", strands)
        for path in ("examples/multi_agent/run-qwen3-30B-A3B-multi-agent.sh", "examples/strands_sglang/strands_qwen3_8b.sh"):
            args = " ".join(ENTRIES[path]["overlay_args"])
            self.assertIn("--save-debug-rollout-data", args)
            self.assertIn("--ci-save-grad-norm", args)
        spec = importlib.util.spec_from_file_location("eval_config", PROJECT / "scripts/prepare_eval_config.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = module.build_config(PROJECT / "fixtures")
        self.assertEqual({item["rm_type"] for item in config["eval"]["datasets"]}, {"deepscaler", "gpqa"})
        search = " ".join(ENTRIES["examples/search-r1/run_qwen2.5_3B.sh"]["overlay_args"])
        self.assertIn("--label-key reward_model", search)
        rows = [json.loads(line) for line in (PROJECT / "fixtures/ci_search_8.jsonl").read_text().splitlines()]
        corpus = [json.loads(line) for line in (PROJECT / "fixtures/search_corpus/docs.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 8)
        self.assertEqual(len(corpus), 8)
        for row in rows:
            self.assertTrue(any(row["reward_model"]["ground_truth"][0] in item["contents"] for item in corpus))

    def test_native_geo3k_recipe(self):
        args = " ".join(ENTRIES["examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py"]["overlay_args"])
        for flag in ("--load ${SLIME_MODEL_PATH}", "--megatron-to-hf-mode bridge", "--input-key problem", "--label-key answer", "--rollout-max-context-len 2048", "--custom-generate-function-path examples.geo3k_vlm_multi_turn.rollout.generate", "--custom-config-path examples/geo3k_vlm_multi_turn/geo3k_vlm_multi_turn_config.yaml"):
            self.assertIn(flag, args)
        self.assertNotIn("--ref-load", args)
        self.assertNotIn("--max-turns 1", args)

    def test_geo3k_fixture_has_real_local_images(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is needed to generate image fixtures")
        from urllib.parse import urlparse, unquote
        spec = importlib.util.spec_from_file_location("geo_fixture", PROJECT / "scripts/prepare_geo_fixture.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            output = module.prepare_fixture(PROJECT / "fixtures/ci_geo_angles_8.json", directory)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 8)
            for row in rows:
                self.assertIn("<image>", row["problem"])
                self.assertIn("calc_score", row["problem"])
                image_path = unquote(urlparse(row["images"][0]).path)
                if os.name == "nt":
                    image_path = image_path.lstrip("/")
                with Image.open(image_path) as image:
                    self.assertEqual(image.size, (128, 128))
                    self.assertLess(image.getextrema()[0][0], image.getextrema()[0][1])

    def test_paths_in_source_trees(self):
        roots = [os.environ.get("SLIME_UPSTREAM_ROOT"), os.environ.get("SLIME_FORK_ROOT")]
        if not all(roots):
            self.skipTest("set SLIME_UPSTREAM_ROOT and SLIME_FORK_ROOT to validate both checkouts")
        for root in roots:
            for path in ENTRIES:
                self.assertTrue((Path(root) / path).is_file(), f"{root}: {path}")


@unittest.skipIf(os.name == "nt" or not shutil.which("bash"), "POSIX launcher mocks run under WSL/Linux")
class LauncherTests(unittest.TestCase):
    def run_recipe(self, entry, visible):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            target = base / "target"
            (target / entry).parent.mkdir(parents=True)
            (target / entry).touch()
            helper = base / "fork/slime/utils/external_utils/command_utils.py"
            helper.parent.mkdir(parents=True)
            helper.write_text("import json, os\ndef execute_train(**kwargs):\n    callback = kwargs.pop('before_ray_job_submit', None)\n    kwargs['callback_name'] = callback.__name__ if callback else None\n    with open(os.environ['CI_OUTPUT_DIR'] + '/call.json', 'w') as stream:\n        json.dump(kwargs, stream)\n", encoding="utf-8")
            validator = base / "workflows/projects/slime/scripts/validate_agent_rollouts.py"
            validator.parent.mkdir(parents=True)
            validator.write_text("import json, os, sys\nwith open(os.environ['CI_OUTPUT_DIR'] + '/validator-call.json', 'w') as output:\n    json.dump(sys.argv[1:], output)\n", encoding="utf-8")
            script = base / "run.sh"
            script.write_text((PROJECT / "scripts/run_example.sh").read_text(encoding="utf-8"), encoding="utf-8")
            env = os.environ.copy()
            env.update(TARGET_ROOT=str(target), GITHUB_WORKSPACE=str(base), CI_OUTPUT_DIR=str(base / "out"), SLIME_FORK_ROOT=str(base / "fork"), SLIME_MODEL_PATH=str(base / "a model"), SLIME_TORCH_DIST_PATH=str(base / "weights"), SLIME_FIXTURE_JSONL=str(base / "rl.jsonl"), SLIME_SFT_FIXTURE_JSONL=str(base / "sft.jsonl"), SLIME_EVAL_CONFIG=str(base / "eval.yaml"), ASCEND_RT_VISIBLE_DEVICES=visible, OVERLAY_ARGS=json.dumps(ENTRIES[entry]["overlay_args"]))
            env.pop("SLIME_SEARCH_INDEX", None)
            if "search-r1" in entry:
                env["SLIME_SEARCH_INDEX"] = str(base / "index")
            result = subprocess.run(["bash", str(script), entry], env=env, text=True, capture_output=True)
            output = base / "out/call.json"
            call = json.loads(output.read_text()) if output.exists() else None
            validated = base / "out/validator-call.json"
            if call is not None and validated.exists():
                call["validation_args"] = json.loads(validated.read_text())
            return result, call

    def test_new_launchers_keep_recipe_and_mask(self):
        for entry, (count, train_script, _) in NEW.items():
            with self.subTest(entry=entry):
                mask = ",".join(str(index + 2) for index in range(count))
                result, call = self.run_recipe(entry, mask)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(call["num_gpus_per_node"], count)
                self.assertEqual(call["train_script"], train_script)
                self.assertEqual(call["extra_env_vars"]["ASCEND_RT_VISIBLE_DEVICES"], mask)
                args = shlex.split(call["train_args"])
                self.assertTrue(args[args.index("--hf-checkpoint") + 1].endswith("a model"))
                if "multi_agent" in entry or "strands" in entry or "geo3k" in entry:
                    self.assertIn("validation_args", call)
                    self.assertIn("--rollout-glob", call["validation_args"])
                if "search-r1" in entry:
                    self.assertEqual(call["callback_name"], "start_search_server")
                if "geo3k" in entry:
                    self.assertEqual(call["validation_args"][0], "geo3k")
                    self.assertEqual(call["megatron_model_type"], "qwen3-1.7B")

    def test_insufficient_devices_stop_before_ray(self):
        for entry in NEW:
            result, call = self.run_recipe(entry, "0")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("insufficient NPU devices", result.stderr)
            self.assertIsNone(call)


if __name__ == "__main__":
    unittest.main()

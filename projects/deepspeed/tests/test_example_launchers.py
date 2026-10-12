"""Exercise the project launcher with recording executables, without NPU jobs.

Only the Ascend environment source line is bypassed in a temporary copy. The
real shell argument expansion, entry dispatch, device checks and config
generation execute; recorded commands never claim to run training.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = PROJECT / "scripts" / "run_example.sh"
HF = "training/tensor_parallel/hf_integration/train.py"
PRMOE = "training/cifar/run_ds_prmoe.sh"
VARIABLE = "training/data_efficiency/variable_batch_size_and_lr/variable_batch_size_and_lr_example.py"
ZENFLOW = "training/DeepSpeed-ZenFlow/benchmark/zf_benchmark.py"
PRUNE = "compression/reasoning_aware_compression/prune.py"
PROMPT = "applications/DeepSpeed-Chat/training/step1_supervised_finetuning/prompt_eval.py"
SFT = "applications/DeepSpeed-Chat/training/step1_supervised_finetuning/main.py"
REWARD = "applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/rw_eval.py"
HF_LENGTH = "training/tensor_parallel/hf_integration/train_bench_length.py"
SUPEROFFLOAD = "training/DeepSpeed-SuperOffload/finetune_zero3.py"


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"),
                     "launcher subprocess tests require a POSIX bash runtime")
class ExampleLauncherTests(unittest.TestCase):
    def launch(self, entry: str, overlays: list[str], *, devices: str | None = None,
               missing_model: bool = False, missing_cache: bool = False,
               fail_sft: bool = False, missing_checkpoint: bool = False,
               stale_checkpoint: bool = False, reward_scores: str = "finite",
               superoffload_losses: str = "finite"
               ) -> tuple[subprocess.CompletedProcess, list[dict], dict | None]:
        with tempfile.TemporaryDirectory(prefix="ds launcher ") as tmp:
            root = Path(tmp)
            examples = root / "examples"
            target = root / "target"
            output = root / "output"
            binaries = root / "bin"
            for path in (examples, target, output, binaries):
                path.mkdir()
            for name in (entry, "training/cifar/cifar10_deepspeed.py", SFT):
                script = examples / name
                script.parent.mkdir(parents=True, exist_ok=True)
                script.touch()
            template = examples / "training/tensor_parallel/hf_integration/configs/ds_config_temp.json"
            template.parent.mkdir(parents=True, exist_ok=True)
            template_text = (
                '{"zero_optimization":{"stage":${zero_stage}},'
                '"tensor_parallel":{"autotp_size":${autotp_size}},'
                '"bf16":{"enabled":"auto"}}\n'
            )
            template.write_text(template_text)
            model = root / "model with spaces"
            if not missing_model:
                model.mkdir()
            fixture = root / "fixture with spaces.jsonl"
            fixture.write_text('{"prompt":"local data"}\n')
            dataset = root / "dataset with spaces"
            dataset.mkdir()
            (dataset / "train.parquet").write_bytes(b"fixture dataset placeholder")
            if not missing_cache:
                for dirname, filename in (
                    ("hf-autotp-work", "dataset_dict.pkl"),
                    ("hf-bench-length-work", "dataset_dict128.pkl"),
                ):
                    cache = output / dirname / filename
                    cache.parent.mkdir()
                    cache.write_bytes(b"setup-generated test cache")
            if stale_checkpoint:
                checkpoint = output / "prompt-eval-sft"
                checkpoint.mkdir()
                (checkpoint / "pytorch_model.bin").write_bytes(b"stale")
            record = root / "commands.jsonl"
            recorder = """
import json, os, sys
from pathlib import Path
command = Path(sys.argv[0]).name
args = sys.argv[1:]
is_entry = bool(args and args[0].endswith('.py') and '/examples/' in args[0])
if command == 'python3' and not is_entry and (not args or args[0] != '-c'):
    os.execv(REAL_PYTHON, [REAL_PYTHON, *args])
with open(os.environ['COMMAND_RECORD'], 'a') as handle:
    handle.write(json.dumps({
        'command': command, 'args': args, 'cwd': os.getcwd(),
        'devices': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        'compile_disable': os.environ.get('TORCH_COMPILE_DISABLE'),
    }) + '\\n')
if command == 'deepspeed' and any(arg.endswith('/step1_supervised_finetuning/main.py') for arg in args):
    if os.environ.get('MOCK_FAIL_SFT') == '1':
        raise SystemExit(13)
    if os.environ.get('MOCK_MISSING_CHECKPOINT') != '1':
        checkpoint = Path(args[args.index('--output_dir') + 1])
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / 'config.json').write_text('{}')
        (checkpoint / 'pytorch_model.bin').write_bytes(b'newly trained mock checkpoint')
if command == 'python3' and is_entry and args[0].endswith('/rw_eval.py'):
    mode = os.environ.get('MOCK_REWARD_SCORES', 'finite')
    if mode != 'missing':
        for value in ('1.0', '-1.0', '2.0', '-2.0'):
            label = 'good_ans' if float(value) > 0 else 'bad_ans'
            print(label + ' score: ' + ('nan' if mode == 'nan' else value))
if command == 'deepspeed' and any(arg.endswith('/DeepSpeed-SuperOffload/finetune_zero3.py') for arg in args):
    mode = os.environ.get('MOCK_SUPEROFFLOAD_LOSSES', 'finite')
    for step in range(1, 4 if mode != 'missing' else 3):
        value = mode if mode in ('nan', 'inf') else str(3.0 / step)
        message = f'Step {step:4d} | Loss: {value} | Time: 1ms | TFLOPS: 0.0 | Tokens/s: 1'
        print('2026-10-09 12:00:00 - finetune_zero3 - INFO - ' + message, file=sys.stderr)
        print('INFO:finetune_zero3:' + message, file=sys.stderr)
    if mode == 'fail':
        raise SystemExit(17)
"""
            recorder = f"#!{sys.executable}\nREAL_PYTHON = {sys.executable!r}\n" + recorder
            for binary in ("deepspeed", "python3"):
                path = binaries / binary
                path.write_text(recorder)
                path.chmod(0o755)
            shell = RUN_SCRIPT.read_text(encoding="utf-8")
            source = "source /usr/local/Ascend/ascend-toolkit/set_env.sh"
            self.assertEqual(shell.count(source), 1)
            script = root / "runner.sh"
            script.write_text(shell.replace(source, ": # mock Ascend environment only"))
            env = dict(os.environ)
            env.pop("EXEC", None)
            env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
            env.pop("TORCH_COMPILE_DISABLE", None)
            env.update(
                PATH=f"{binaries}:{env['PATH']}",
                TARGET_ROOT=str(target), EXAMPLES_ROOT=str(examples),
                CI_OUTPUT_DIR=str(output), OVERLAY_ARGS=json.dumps(overlays),
                MODEL_PATH=str(model), FIXTURE_PATH=str(fixture),
                DATASET_PATH=str(dataset), MOCK_FAIL_SFT=str(int(fail_sft)),
                MOCK_MISSING_CHECKPOINT=str(int(missing_checkpoint)),
                MOCK_REWARD_SCORES=reward_scores,
                MOCK_SUPEROFFLOAD_LOSSES=superoffload_losses,
                COMMAND_RECORD=str(record), GITHUB_RUN_ID="1234",
                DEEPSPEED_PROJECT_SCRIPTS=str(PROJECT / "scripts"),
            )
            if devices is not None:
                env["ASCEND_RT_VISIBLE_DEVICES"] = devices
            result = subprocess.run(["bash", str(script), entry], env=env,
                                    capture_output=True, text=True, timeout=30)
            commands = [json.loads(line) for line in record.read_text().splitlines()] if record.exists() else []
            config_name = {
                HF_LENGTH: "hf-bench-length-config.json",
                SUPEROFFLOAD: "superoffload-config.json",
            }.get(entry, "hf-autotp-config.json")
            generated = output / config_name
            config = json.loads(generated.read_text()) if generated.exists() else None
            self.assertEqual(template.read_text(), template_text,
                             "launcher modified the upstream config template")
            return result, commands, config

    def test_prmoe_retains_residual_and_two_expert_layers(self) -> None:
        result, commands, _ = self.launch(PRMOE, ["--epochs 1", "--dtype bf16"], devices="3,6,7")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(commands), 1)
        command = commands[0]
        args = command["args"]
        self.assertEqual(args[args.index("--num_gpus") + 1], "2")
        self.assertEqual(args[args.index("--ep-world-size") + 1], "2")
        self.assertEqual(args[args.index("--num-experts") + 1:args.index("--num-experts") + 3], ["2", "4"])
        self.assertEqual(args[args.index("--mlp-type") + 1], "residual")
        self.assertEqual(command["devices"], "3,6")
        self.assertEqual(command["compile_disable"], "1")

    def test_hf_autotp_generates_config_and_passes_local_paths(self) -> None:
        result, commands, config = self.launch(HF, [
            '--model_name_or_path "${MODEL_PATH}"',
            '--data_path "${FIXTURE_PATH}"', '--max_steps 3',
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(config["zero_optimization"]["stage"], 0)
        self.assertEqual(config["tensor_parallel"]["autotp_size"], 8)
        args = commands[0]["args"]
        self.assertIn("--no_local_rank", args)
        self.assertEqual(args[args.index("--num_gpus") + 1], "8")
        self.assertIn("model with spaces", args[args.index("--model_name_or_path") + 1])
        self.assertIn("fixture with spaces.jsonl", args[args.index("--data_path") + 1])
        self.assertIn("/output/", args[args.index("--deepspeed") + 1])
        self.assertTrue(commands[0]["cwd"].endswith("/output/hf-autotp-work"))
        self.assertIsNone(commands[0]["compile_disable"])

    def test_insufficient_cards_fail_before_launch(self) -> None:
        result, commands, _ = self.launch(HF, [], devices="0,1,2,3")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required=8", result.stderr)
        self.assertEqual(commands, [])

    def test_missing_local_model_fails_before_launch(self) -> None:
        result, commands, _ = self.launch(HF, [
            '--model_name_or_path "${MODEL_PATH}"', '--data_path "${FIXTURE_PATH}"',
        ], missing_model=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing local directory", result.stderr)
        self.assertEqual(commands, [])

    def test_hf_workers_require_setup_generated_cache(self) -> None:
        result, commands, _ = self.launch(HF, [
            '--model_name_or_path "${MODEL_PATH}"', '--data_path "${FIXTURE_PATH}"',
        ], missing_cache=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("shared tokenization cache missing", result.stderr)
        self.assertEqual(commands, [])

    def test_variable_batch_avoids_injecting_unknown_argument(self) -> None:
        result, commands, _ = self.launch(VARIABLE, ["--pipeline-num-stages 0"], devices="4,5")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = commands[0]["args"]
        self.assertIn("--no_local_rank", args)
        self.assertEqual(args[args.index("--num_gpus") + 1], "1")
        self.assertEqual(commands[0]["devices"], "4")
        self.assertIsNone(commands[0]["compile_disable"])

    def test_zenflow_uses_two_ranks_and_unmodified_cli(self) -> None:
        result, commands, _ = self.launch(ZENFLOW, [
            "--iteration 3", "--update_intervals 2", "--pin_memory_opts 0",
            "--topk_ratios 0.1", "--overlap_steps 0",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        args = commands[0]["args"]
        self.assertEqual(args[args.index("--num_gpus") + 1], "2")
        self.assertNotIn("--no_local_rank", args)
        self.assertEqual(args[args.index("--update_intervals") + 1], "2")
        self.assertIsNone(commands[0]["compile_disable"])

    def test_prune_bootstrap_registers_npu_and_preserves_arguments(self) -> None:
        result, commands, _ = self.launch(PRUNE, [
            '--model "${MODEL_PATH}"', '--dataset "${FIXTURE_PATH}"',
            "--device npu", "--pruning-method wanda",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        command = commands[0]
        self.assertEqual(command["command"], "python3")
        self.assertIn("import torch_npu", command["args"][1])
        self.assertIn("runpy.run_path", command["args"][1])
        self.assertEqual(command["args"][command["args"].index("--device") + 1], "npu")
        self.assertIsNone(command["compile_disable"])

    def prompt_overlays(self) -> list[str]:
        return [
            '--model_name_or_path_baseline "${MODEL_PATH}"',
            '--model_name_or_path_finetune "${CI_OUTPUT_DIR}/prompt-eval-sft"',
            '--max_new_tokens 8', '--language English',
        ]

    def test_prompt_eval_trains_then_loads_same_job_checkpoint(self) -> None:
        result, commands, _ = self.launch(PROMPT, self.prompt_overlays(), devices="4,5")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([item["command"] for item in commands], ["deepspeed", "python3"])
        training, evaluation = [item["args"] for item in commands]
        self.assertTrue(any(item.endswith(SFT) for item in training))
        self.assertIn("--deepspeed", training)
        self.assertEqual(training[training.index("--num_gpus") + 1], "1")
        self.assertEqual(training[training.index("--zero_stage") + 1], "2")
        self.assertEqual(training[training.index("--max_seq_len") + 1], "128")
        checkpoint = training[training.index("--output_dir") + 1]
        self.assertEqual(evaluation[evaluation.index("--model_name_or_path_finetune") + 1], checkpoint)
        self.assertNotEqual(evaluation[evaluation.index("--model_name_or_path_baseline") + 1], checkpoint)
        self.assertEqual(evaluation[evaluation.index("--max_new_tokens") + 1], "8")
        self.assertEqual([item["devices"] for item in commands], ["4", "4"])

    def test_prompt_eval_does_not_run_after_sft_failure(self) -> None:
        result, commands, _ = self.launch(PROMPT, self.prompt_overlays(), fail_sft=True)
        self.assertEqual(result.returncode, 13)
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0]["command"], "deepspeed")

    def test_prompt_eval_requires_new_checkpoint_artifacts(self) -> None:
        result, commands, _ = self.launch(PROMPT, self.prompt_overlays(), missing_checkpoint=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("did not produce a loadable checkpoint", result.stderr)
        self.assertEqual(len(commands), 1)

    def test_prompt_eval_refuses_fake_baseline_as_finetuned(self) -> None:
        overlays = self.prompt_overlays()
        overlays[1] = '--model_name_or_path_finetune "${MODEL_PATH}"'
        result, commands, _ = self.launch(PROMPT, overlays)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("same-job SFT checkpoint", result.stderr)
        self.assertEqual(commands, [])

    def test_prompt_eval_refuses_stale_checkpoint(self) -> None:
        result, commands, _ = self.launch(PROMPT, self.prompt_overlays(), stale_checkpoint=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to reuse stale", result.stderr)
        self.assertEqual(commands, [])

    def test_reward_eval_is_plain_npu_forward_not_checkpoint_restore(self) -> None:
        result, commands, _ = self.launch(REWARD, ['--model_name_or_path "${MODEL_PATH}"'], devices="6,7")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0]["command"], "python3")
        self.assertTrue(commands[0]["args"][0].endswith(REWARD))
        self.assertEqual(commands[0]["devices"], "6")
        self.assertIn("not trained-head restoration", result.stdout)
        self.assertIn("validated four finite reward scores", result.stdout)

    def test_reward_eval_rejects_nonfinite_or_missing_scores(self) -> None:
        for mode, message in (("nan", "non-finite"), ("missing", "must produce four scores")):
            with self.subTest(mode=mode):
                result, commands, _ = self.launch(REWARD, [
                    '--model_name_or_path "${MODEL_PATH}"',
                ], reward_scores=mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertEqual(len(commands), 1)

    def test_hf_fixed_length_preserves_template_tp_and_separate_cache(self) -> None:
        result, commands, config = self.launch(HF_LENGTH, [
            '--model_name_or_path "${MODEL_PATH}"', '--data_path "${FIXTURE_PATH}"',
            '--model_max_length 128', '--max_steps 3', '--warmup_steps 2',
        ], devices="1,2,3,4,5,6,7,8,9")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(config["zero_optimization"]["stage"], 0)
        self.assertEqual(config["tensor_parallel"]["autotp_size"], 8)
        self.assertIn("--no_local_rank", commands[0]["args"])
        self.assertEqual(commands[0]["devices"], "1,2,3,4,5,6,7,8")
        self.assertTrue(commands[0]["cwd"].endswith("/output/hf-bench-length-work"))

    def test_hf_fixed_length_requires_its_setup_cache(self) -> None:
        result, commands, _ = self.launch(HF_LENGTH, [
            '--model_name_or_path "${MODEL_PATH}"', '--data_path "${FIXTURE_PATH}"',
            '--model_max_length 128',
        ], missing_cache=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fixed-length shared tokenization cache missing", result.stderr)
        self.assertEqual(commands, [])

    def test_superoffload_generates_two_rank_cpu_offload_without_duplicate_optimizer(self) -> None:
        result, commands, config = self.launch(SUPEROFFLOAD, [
            '--model_name "${MODEL_PATH}"', '--dataset_name "${DATASET_PATH}"',
            '--lr 0.001', '--batch_size 1', '--max_length 128', '--bench_steps 3',
            '--warmup_steps 1', '--attn_implementation eager',
            '--output_dir "${CI_OUTPUT_DIR}/superoffload"',
        ], devices="3,5,7")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(config["train_batch_size"], 2)
        self.assertEqual(config["train_micro_batch_size_per_gpu"], 1)
        self.assertEqual(config["gradient_accumulation_steps"], 1)
        self.assertEqual(config["zero_optimization"]["stage"], 3)
        self.assertNotIn("optimizer", config)
        for key in ("offload_param", "offload_optimizer"):
            self.assertEqual(config["zero_optimization"][key]["device"], "cpu")
            self.assertFalse(config["zero_optimization"][key]["pin_memory"])
        self.assertNotIn("super_offload", config["zero_optimization"]["offload_optimizer"])
        args = commands[0]["args"]
        self.assertNotIn("--no_local_rank", args)
        self.assertNotIn("--save_checkpoint", args)
        self.assertEqual(args[args.index("--num_gpus") + 1], "2")
        self.assertIn("dataset with spaces", args[args.index("--dataset_name") + 1])
        self.assertEqual(commands[0]["devices"], "3,5")
        self.assertIn("validated three finite SuperOffload entry train-step losses", result.stdout)

    def test_superoffload_rejects_nonfinite_and_missing_train_steps(self) -> None:
        for mode, message in (("nan", "non-finite"), ("inf", "non-finite"),
                              ("missing", "must report rank-0 finite losses")):
            with self.subTest(mode=mode):
                result, commands, _ = self.launch(SUPEROFFLOAD, [
                    '--model_name "${MODEL_PATH}"', '--dataset_name "${DATASET_PATH}"',
                ], superoffload_losses=mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertEqual(len(commands), 1)

    def test_superoffload_pipe_does_not_hide_training_failure(self) -> None:
        result, commands, _ = self.launch(SUPEROFFLOAD, [
            '--model_name "${MODEL_PATH}"', '--dataset_name "${DATASET_PATH}"',
        ], superoffload_losses="fail")
        self.assertEqual(result.returncode, 17)
        self.assertEqual(len(commands), 1)
        self.assertNotIn("validated three finite", result.stdout)

    def test_superoffload_requires_two_visible_devices(self) -> None:
        result, commands, _ = self.launch(SUPEROFFLOAD, [], devices="4")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required=2", result.stderr)
        self.assertEqual(commands, [])


if __name__ == "__main__":
    unittest.main()

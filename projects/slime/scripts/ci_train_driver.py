#!/usr/bin/env python3
"""CI train driver for slime examples guard.

The upstream example launchers hardcode absolute paths, large models and
8-16 GPU layouts without "$@" passthrough, so execution never runs them
directly. Instead this driver re-assembles the same train.py /
train_async.py invocation with CI-sized parameters, mirroring the fork's
verified NPU CI tests (tests/tests_npu/...). Ray orchestration, NPU
resource injection and env plumbing stay in the fork's own
slime.utils.external_utils.command_utils helpers; no fork source is
modified.

The entry map below is keyed by the manifest `path` (which exists in both
the upstream release tree and the fork main tree).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def load_fork_command_utils(fork_root: Path):
    spec_path = fork_root / "slime" / "utils" / "external_utils" / "command_utils.py"
    if not spec_path.is_file():
        raise SystemExit(f"fork command_utils not found: {spec_path}")
    # Importing `slime` requires the fork root on sys.path (the editable
    # install provides it, but stay explicit so this also works from a
    # plain checkout).
    if str(fork_root) not in sys.path:
        sys.path.insert(0, str(fork_root))
    spec = importlib.util.spec_from_file_location("_fork_command_utils", spec_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", required=True, help="manifest entry path")
    parser.add_argument("--fork-root", required=True, help="slime-ascend checkout")
    parser.add_argument("--ci-output-dir", required=True)
    parser.add_argument("extra", nargs="*", help="extra CLI args (manifest overlay)")
    args = parser.parse_args()

    fork_root = Path(args.fork_root).resolve()
    U = load_fork_command_utils(fork_root)

    model_dir = os.environ.get("SLIME_MODEL_PATH")
    torch_dist = os.environ.get("SLIME_TORCH_DIST_PATH")
    fixture = os.environ.get("SLIME_FIXTURE_JSONL")
    if not model_dir or not torch_dist or not fixture:
        raise SystemExit(
            "setup must export SLIME_MODEL_PATH, SLIME_TORCH_DIST_PATH "
            "and SLIME_FIXTURE_JSONL")

    checkpoint_args = (
        f"--hf-checkpoint {model_dir} "
        f"--ref-load {torch_dist} "
    )

    if args.example == "examples/fully_async/run-qwen2.5-0.5B-fully_async.sh":
        # Recipe: tests/tests_npu/nightly_CI/
        # test_qwen2.5_0.5B_fully_async_short_npu.py (fork-verified on NPU).
        # Deviations from the nightly: response length 8192 -> 1024 and the
        # dataset is a 16-row in-repo fixture.
        train_script = "train_async.py"
        megatron_model_type = "qwen2.5-0.5B"
        num_gpus_per_node = 4
        rollout_args = (
            "--rollout-function-path "
            "slime.rollout.fully_async_rollout.generate_rollout_fully_async "
            f"--prompt-data {fixture} "
            "--input-key prompt "
            "--label-key label "
            "--apply-chat-template "
            "--rollout-shuffle "
            "--rm-type deepscaler "
            "--num-rollout 2 "
            "--rollout-batch-size 4 "
            "--n-samples-per-prompt 4 "
            "--rollout-max-response-len 1024 "
            "--rollout-temperature 0.8 "
            "--global-batch-size 16 "
            "--balance-data "
        )
        misc_args = (
            "--attention-dropout 0.0 "
            "--hidden-dropout 0.0 "
            "--accumulate-allreduce-grads-in-fp32 "
            "--attention-softmax-in-fp32 "
            "--attention-backend flash "
            "--actor-num-nodes 1 "
            "--actor-num-gpus-per-node 1 "
            "--rollout-num-gpus 3 "
        )
    else:
        raise SystemExit(f"no CI recipe for manifest entry: {args.example}")

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.65 "
        "--sglang-cuda-graph-max-bs 16 "
        "--sglang-enable-metrics "
        "--sglang-device npu "
    )

    ci_args = "--ci-test "

    # No wandb args on purpose: get_default_wandb_args returns "" without
    # WANDB_API_KEY, and run_example.sh forces WANDB_MODE=offline.
    train_args = (
        f"{checkpoint_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    if args.extra:
        train_args += " " + " ".join(args.extra)

    print("=== slime CI train args ===")
    print(train_args)
    print("=== end slime CI train args ===")

    Path(args.ci_output_dir).mkdir(parents=True, exist_ok=True)

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=num_gpus_per_node,
        megatron_model_type=megatron_model_type,
        train_script=train_script,
    )


if __name__ == "__main__":
    main()


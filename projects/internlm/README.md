# InternLM

This directory contains the Ascend Quick Start guard for
[InternLM/InternLM](https://github.com/InternLM/InternLM); it does not contain
InternLM model weights or a fork of the upstream source.

The guarded path follows upstream `ecosystem/README_npu.md`: it checks out the
current upstream `main`, downloads `InternLM3-8B-Instruct` from ModelScope, and
runs one real Transformers generation on a single `npu:0`. The latest GitHub
release predates the upstream NPU guide, so the workflow intentionally uses
`fixed_ref: main` and monitors that branch's commit SHA.

Training paths documented through XTuner, LLaMA-Factory, InternEvo, and
openMind are outside this first guard. They require external repositories,
additional configuration, and in the upstream examples often multiple NPUs.

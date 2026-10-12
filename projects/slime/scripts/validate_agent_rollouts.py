"""Validate native Slime rollout/gradient dumps without intercepting generation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
import math
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_sample(sample):
    require(isinstance(sample, dict), 'native samples must be dictionaries')
    tokens, length = sample.get('tokens'), sample.get('response_length')
    require(isinstance(tokens, list) and tokens and all(isinstance(x, int) for x in tokens), 'missing token trajectory')
    require(isinstance(length, int) and 0 < length < len(tokens), 'invalid response/prompt token lengths')
    require(isinstance(sample.get('response'), str) and sample['response'].strip(), 'empty response')
    require(sample.get('status') in ('completed', 'truncated'), 'failed/aborted agent trajectory')
    require(finite_number(sample.get('reward')), 'missing/nonfinite reward')
    probs = sample.get('rollout_log_probs')
    if probs is not None:
        require(len(probs) == length and all(finite_number(x) for x in probs), 'invalid rollout log probabilities')


def validate_multi_agent(samples):
    groups = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        validate_sample(sample)
        require(sample.get('group_id') is not None, 'multi-agent samples need native group_id')
        prompt = sample.get('prompt', '')
        require(isinstance(prompt, str), 'multi-agent prompt must be a string')
        if '### Task: Solution Rewriting Based on Previous Solutions ###' in prompt:
            stage = 'rewriter'
        elif 'End your evaluation with exactly:' in prompt and 'Judgment: IDX' in prompt:
            stage = 'selector'
        else:
            stage = 'solver'
        groups[sample['group_id']][stage].append(sample)
    require(bool(groups), 'empty multi-agent rollout')
    for group, stages in groups.items():
        require({key: len(value) for key, value in stages.items()} == {'solver': 5, 'rewriter': 5, 'selector': 1},
                f'group {group} did not complete five solver/five rewriter/one selector trajectories')
        for stage in ('solver', 'rewriter', 'selector'):
            for sample in stages[stage]:
                require(isinstance(sample.get('response_content'), str) and sample['response_content'].strip(),
                        f'group {group} {stage} did not pass the original </think> parser')
        selected = re.findall(r'Judgment:\s*(\d+)', stages['selector'][0]['response_content'])
        require(bool(selected) and 1 <= int(selected[0]) <= 5, f'group {group} selector did not select a valid rewrite')
    return {'groups': len(groups), 'samples': len(samples), 'stages_per_group': {'solver': 5, 'rewriter': 5, 'selector': 1}}


def validate_strands(samples, log):
    require(bool(samples), 'empty Strands rollout')
    called = 0
    for sample in samples:
        validate_sample(sample)
        length = sample['response_length']
        mask, probs = sample.get('loss_mask'), sample.get('rollout_log_probs')
        require(isinstance(mask, list) and len(mask) == length and all(x in (0, 1) for x in mask) and 1 in mask,
                'Strands needs aligned nonempty generated-token TITO loss masks')
        require(isinstance(probs, list) and len(probs) == length and all(finite_number(x) for x in probs),
                'Strands needs aligned finite TITO log probabilities')
        calls, iters = sample.get('tool_calls'), sample.get('tool_iters')
        require(isinstance(calls, int) and calls >= 0 and isinstance(iters, int) and iters >= 0,
                'missing original ToolLimiter counters')
        if calls > 0:
            require(iters > 0 and 0 in mask, 'tool trajectory must include masked non-model tokens')
            called += 1
    require(called > 0, 'Strands did not call a tool')
    require(re.search(r'Executing Python code: ```python\s*\S.*?``` and get execution result: ```python.*?```', log, re.S),
            'missing original execute_python_code execution/result log')
    return {'samples': len(samples), 'tool_trajectories': called, 'tito_verified': True}


def nonempty_tensor(value):
    # The real native dump contains tensors. Tests use tensor-shaped objects;
    # no torch import is needed to validate the other recipe contracts.
    return hasattr(value, 'numel') and callable(value.numel) and value.numel() > 0


def validate_geo3k(samples):
    require(bool(samples), 'empty geo3k rollout')
    multi_turn = 0
    for sample in samples:
        validate_sample(sample)
        vision = sample.get('multimodal_train_inputs')
        require(isinstance(vision, dict) and nonempty_tensor(vision.get('pixel_values'))
                and nonempty_tensor(vision.get('image_grid_thw')), 'geo3k needs original nonempty visual tensors/grid')
        length = sample['response_length']
        mask, probs = sample.get('loss_mask'), sample.get('rollout_log_probs')
        require(isinstance(mask, list) and len(mask) == length and all(value in (0, 1) for value in mask),
                'geo3k needs aligned model/feedback loss mask')
        require(isinstance(probs, list) and len(probs) == length and all(finite_number(value) for value in probs),
                'geo3k needs aligned finite rollout log probabilities')
        # Collapse adjacent equal mask values; 1->0->1 means model generation,
        # original environment observation, then another model generation.
        segments = [value for index, value in enumerate(mask) if index == 0 or value != mask[index - 1]]
        response = sample['response']
        calls = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response, re.S)
        invoked = False
        for payload in calls:
            try:
                call = json.loads(payload)
            except json.JSONDecodeError:
                continue
            name = call.get('name') or call.get('function', {}).get('name')
            invoked |= name in ('calc_score', 'calc_geo3k_reward')
        if segments[:3] == [1, 0, 1] and invoked and re.search(r'calc_score result:\s*[01](?:\.0)?\b', response):
            multi_turn += 1
    require(multi_turn > 0, 'geo3k did not execute a scoring tool/feedback and a second model turn')
    return {'samples': len(samples), 'visual_multiturn_trajectories': multi_turn, 'vision_verified': True}


def validate_dumps(mode, dumps, grad_norms, log='', expected_rollouts=2):
    require(len(dumps) == expected_rollouts, f'expected {expected_rollouts} native rollout dumps')
    ids = [dump.get('rollout_id') for dump in dumps]
    require(sorted(ids) == list(range(expected_rollouts)), 'missing/duplicate native rollout IDs')
    require(len(grad_norms) >= expected_rollouts and all(finite_number(value) and value >= 0 for value in grad_norms),
            'missing/nonfinite native post-optimizer gradient norms')
    losses = re.findall(r"['\"]train/[^'\"]*loss[^'\"]*['\"]\s*:\s*([^,}\s]+)", log)
    require(bool(losses), 'missing native train loss metrics')
    require(all(finite_number(float(value)) for value in losses), 'nonfinite native train loss metrics')
    reports = []
    for dump in dumps:
        samples = dump.get('samples')
        require(isinstance(samples, list), 'native debug dump must contain samples')
        if mode == 'multi_agent':
            reports.append(validate_multi_agent(samples))
        elif mode == 'strands':
            reports.append(validate_strands(samples, log))
        elif mode == 'geo3k':
            reports.append(validate_geo3k(samples))
        else:
            raise ValueError(f'unknown agent mode: {mode}')
    return {'mode': mode, 'rollouts': reports, 'finite_optimizer_updates': len(grad_norms)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('multi_agent', 'strands', 'geo3k'))
    parser.add_argument('--rollout-glob', required=True)
    parser.add_argument('--grad-glob', required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-rollouts', type=int, default=2)
    args = parser.parse_args()
    import torch
    files = sorted(glob.glob(args.rollout_glob))
    # VLM samples carry locally prepared PIL images in multimodal_inputs. Only
    # these CI-generated geo3k dumps need the native object deserializer;
    # callers must pass their own freshly generated CI output paths.
    dumps = [torch.load(path, map_location='cpu', weights_only=args.mode != 'geo3k') for path in files]
    norms = []
    for path in sorted(glob.glob(args.grad_glob)):
        value = torch.load(path, map_location='cpu', weights_only=True)
        norms.append(value.item() if isinstance(value, torch.Tensor) and value.numel() == 1 else value)
    report = validate_dumps(args.mode, dumps, norms, Path(args.log).read_text(encoding='utf-8'), args.expected_rollouts)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'validated native {args.mode} core agent trajectories and finite optimizer updates')


if __name__ == '__main__':
    main()

import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('validate_agents', Path(__file__).resolve().parents[1] / 'scripts/validate_agent_rollouts.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample(prompt='Solve 1+1', response='</think>2'):
    return {'prompt': prompt, 'response': response, 'tokens': [1, 2, 3, 4], 'response_length': 2,
            'status': 'completed', 'reward': 0.0, 'group_id': 7,
            'response_content': response.split('</think>')[-1], 'rollout_log_probs': [-0.1, -0.2]}


def multi():
    return ([sample() for _ in range(5)] +
            [sample('### Task: Solution Rewriting Based on Previous Solutions ###') for _ in range(5)] +
            [sample('End your evaluation with exactly: Judgment: IDX', '</think>Judgment: 1')])


def strands():
    value = sample()
    value.update(loss_mask=[1, 0], tool_calls=1, tool_iters=1)
    return [value]


class FakeTensor:
    def __init__(self, count=1):
        self.count = count

    def numel(self):
        return self.count


def geo3k():
    value = sample(response='<tool_call>{"name":"calc_score","arguments":{"answer":"3"}}</tool_call> calc_score result: 1.0. \\boxed{3}')
    value.update(tokens=[1, 2, 3, 4, 5], response_length=3, loss_mask=[1, 0, 1],
                 rollout_log_probs=[-0.1, 0.0, -0.2],
                 multimodal_train_inputs={'pixel_values': FakeTensor(64), 'image_grid_thw': FakeTensor(3)})
    return [value]


class AgentRolloutTests(unittest.TestCase):
    def test_multi_agent_all_stages(self):
        result = module.validate_multi_agent(multi())
        self.assertEqual(result['samples'], 11)

    def test_multi_agent_missing_selector(self):
        with self.assertRaisesRegex(ValueError, 'selector'):
            module.validate_multi_agent(multi()[:-1])

    def test_multi_agent_missing_think_output(self):
        values = multi()
        values[0]['response_content'] = None
        with self.assertRaisesRegex(ValueError, 'parser'):
            module.validate_multi_agent(values)

    def test_strands_tito_real_execution_log(self):
        result = module.validate_strands(strands(), 'Executing Python code: ```python\nprint(2)\n``` and get execution result: ```python\n2\n```')
        self.assertTrue(result['tito_verified'])

    def test_strands_no_tool(self):
        values = strands()
        values[0].update(tool_calls=0, tool_iters=0)
        with self.assertRaisesRegex(ValueError, 'did not call'):
            module.validate_strands(values, '')

    def test_strands_counter_without_execution(self):
        with self.assertRaisesRegex(ValueError, 'execution/result'):
            module.validate_strands(strands(), '')

    def test_strands_broken_tito(self):
        values = strands()
        values[0]['loss_mask'] = [1]
        with self.assertRaisesRegex(ValueError, 'TITO'):
            module.validate_strands(values, '')

    def test_empty_and_nonfinite_rewards(self):
        with self.assertRaises(ValueError):
            module.validate_multi_agent([])
        values = multi()
        values[0]['reward'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'reward'):
            module.validate_multi_agent(values)

    def test_updates_and_rollout_ids(self):
        dumps = [{'rollout_id': i, 'samples': multi()} for i in range(2)]
        log = "step 0: {'train/pg_loss': 0.0, 'train/kl_loss': 0.01}"
        self.assertEqual(module.validate_dumps('multi_agent', dumps, [0.0, 0.1], log)['finite_optimizer_updates'], 2)
        for norms in ([], [float('inf')]):
            with self.assertRaisesRegex(ValueError, 'gradient'):
                module.validate_dumps('multi_agent', dumps, norms)
        broken = copy.deepcopy(dumps)
        broken[1]['rollout_id'] = 0
        with self.assertRaisesRegex(ValueError, 'IDs'):
            module.validate_dumps('multi_agent', broken, [0.1])
        with self.assertRaisesRegex(ValueError, 'loss metrics'):
            module.validate_dumps('multi_agent', dumps, [0.1, 0.2], "{'train/pg_loss': nan}")

    def test_geo3k_visual_tool_and_second_turn(self):
        self.assertEqual(module.validate_geo3k(geo3k())['visual_multiturn_trajectories'], 1)
        dumps = [{'rollout_id': i, 'samples': geo3k()} for i in range(2)]
        self.assertEqual(module.validate_dumps('geo3k', dumps, [0.1, 0.2], "{'train/pg_loss': 0.0}")['mode'], 'geo3k')

    def test_geo3k_no_vision(self):
        for value in ({}, {'pixel_values': FakeTensor(0), 'image_grid_thw': FakeTensor(3)}):
            samples = geo3k()
            samples[0]['multimodal_train_inputs'] = value
            with self.assertRaisesRegex(ValueError, 'visual tensors'):
                module.validate_geo3k(samples)

    def test_geo3k_single_turn_or_missing_feedback(self):
        values = geo3k()
        values[0]['loss_mask'] = [1, 1, 1]
        with self.assertRaisesRegex(ValueError, 'second model turn'):
            module.validate_geo3k(values)
        values = geo3k()
        values[0]['response'] = '<tool_call>{"name":"calc_score","arguments":{"answer":"3"}}</tool_call>'
        with self.assertRaisesRegex(ValueError, 'feedback'):
            module.validate_geo3k(values)

    def test_geo3k_forged_feedback_without_tool(self):
        values = geo3k()
        values[0]['response'] = 'calc_score result: 0.0. final answer'
        with self.assertRaisesRegex(ValueError, 'scoring tool'):
            module.validate_geo3k(values)

    def test_geo3k_misaligned_probs(self):
        values = geo3k()
        values[0]['rollout_log_probs'] = [-0.1]
        with self.assertRaisesRegex(ValueError, 'log probabilities'):
            module.validate_geo3k(values)


if __name__ == '__main__':
    unittest.main()

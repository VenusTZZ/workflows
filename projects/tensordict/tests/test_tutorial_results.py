"""Exercise the actual device guard with mocks, not hardware."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

RUN = (Path(__file__).resolve().parents[1] / "scripts/run_example.sh").read_text()
BODY = RUN.split('python - "$tutorial" <<\'PY\'\n')[1].split("\nPY")[0]
TREE = ast.parse(BODY)
GUARD = compile(ast.Module(body=[n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "require_npu"], type_ignores=[]), "actual-device-guard", "exec")

class Tensor:
    def __init__(self, device):
        self.device = SimpleNamespace(type=device)

class TD:
    def __init__(self, device):
        self.device = device
    def values(self, **kwargs):
        return [Tensor(self.device)]

class TutorialResultGuards(unittest.TestCase):
    def setUp(self):
        scope = {"TensorDictBase": TD, "torch": SimpleNamespace(Tensor=Tensor)}
        exec(GUARD, scope)
        self.guard = scope["require_npu"]
    def test_accepts_real_npu_tagged_leaves(self):
        self.assertEqual(len(self.guard(TD("npu"), "result")), 1)
    def test_rejects_cpu_fallback(self):
        with self.assertRaises(SystemExit):
            self.guard(TD("cpu"), "result")
    def test_rejects_missing_data(self):
        with self.assertRaises(SystemExit):
            self.guard(None, "result")
    def test_new_guards_preserve_source_semantics(self):
        self.assertIn('td[namespace["mask"]]', BODY)
        self.assertIn('namespace["data_cont"]', BODY)
        self.assertIn('list(namespace["td"].batch_size) != [10]', BODY)
        self.assertIn('mapped.update_(td.cpu())', BODY)
        self.assertIn('restored = mapped.to("npu:0")', BODY)
        self.assertIn('torch.equal(td[key], restored[key])', BODY)
        self.assertNotIn("TORCH_COMPILE_DISABLE", RUN)

if __name__ == "__main__":
    unittest.main()

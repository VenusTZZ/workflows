"""Exercise setup materialization with mocked third-party modules, not NPU training."""
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile
import types
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
SETUP = (PROJECT / "scripts/setup_example.sh").read_text(encoding="utf-8")


def setup_python(function):
    section = SETUP.split(f"{function}() {{", 1)[1].split("\n}\n", 1)[0]
    return re.search(r"<<'PY'\n(.*?)\nPY", section, re.S).group(1)


class Vector:
    """Minimal CPU tensor protocol for testing the cache validation branches."""
    def __init__(self, values):
        self.values = list(values)

    def __eq__(self, value):
        return Vector(item == value for item in self.values)

    def __ne__(self, value):
        return Vector(item != value for item in self.values)

    def __getitem__(self, mask):
        return Vector(value for value, keep in zip(self.values, mask.values) if keep)

    def sum(self):
        return types.SimpleNamespace(item=lambda: sum(self.values))

    def numel(self):
        return len(self.values)

    def all(self):
        return all(self.values)


class SetupAssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ds setup assets ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = PROJECT / "fixtures/ci_alpaca_16.json"
        self.environment = {
            "GITHUB_WORKSPACE": str(self.root), "EXAMPLES_ROOT": str(self.root / "examples"),
            "OPT_125M_PATH": str(self.root / "model"), "ALPACA_CI_PATH": str(self.fixture),
            "ALPACA_DATASET_DIR": str(self.root / "parquet"), "CI_OUTPUT_DIR": str(self.root / "output"),
        }
        Path(self.environment["ALPACA_DATASET_DIR"]).mkdir()

    def execute(self, function, upstream, extras):
        modules = {name: types.ModuleType(name) for name in (
            "accelerate", "numpy", "openai", "torch_npu", "transformers", "wandb"
        )}
        for module in modules.values():
            module.__version__ = "mock"
        modules["openai"].openai_object = types.SimpleNamespace(OpenAIObject=object)
        modules.update(extras)
        spec = types.SimpleNamespace(name="ds_ci_test", loader=types.SimpleNamespace(exec_module=lambda module: None))
        with patch.dict(os.environ, self.environment), patch.dict("sys.modules", modules), \
             patch.object(importlib.util, "spec_from_file_location", return_value=spec), \
             patch.object(importlib.util, "module_from_spec", return_value=upstream), \
             patch("sys.path", list(__import__("sys").path)):
            exec(compile(setup_python(function), function, "exec"), {})

    def hf_modules(self, *, mask_padding=True, target_tokens=True):
        tokenizer = types.SimpleNamespace(pad_token="pad", eos_token="eos", bos_token="bos", unk_token="unk",
                                          pad_token_id=1, add_special_tokens=lambda values: None)
        transformers = types.ModuleType("transformers")
        transformers.__version__ = "mock"
        def load_tokenizer(path, **kwargs):
            self.assertEqual(kwargs, {"model_max_length": 128, "padding_side": "right", "use_fast": False})
            return tokenizer
        transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=load_tokenizer)
        def dataset(path, passed_tokenizer):
            self.assertEqual(Path(path), self.fixture)
            self.assertIs(passed_tokenizer, tokenizer)
            Path("dataset_dict128.pkl").write_bytes(b"mock cache")
            ids = Vector([2] * 20 + [1] * 108)
            labels = Vector(([-100] * 18 + ([2] * 2 if target_tokens else [-100] * 2)) +
                            ([-100] * 108 if mask_padding else [1] * 108))
            class FixedDataset:
                def __len__(self):
                    return 16
            result = FixedDataset()
            result.input_ids, result.labels = [ids] * 16, [labels] * 16
            return result
        upstream = types.SimpleNamespace(IGNORE_INDEX=-100, SupervisedDataset=dataset)
        return upstream, {"transformers": transformers}

    def test_fixed_length_cache_materialized_once_outside_source(self):
        upstream, modules = self.hf_modules()
        original_cwd = Path.cwd()
        self.execute("setup_ds_hf_bench_length", upstream, modules)
        self.assertEqual(Path.cwd(), original_cwd)
        cache = self.root / "output/hf-bench-length-work/dataset_dict128.pkl"
        self.assertEqual(cache.read_bytes(), b"mock cache")

    def test_fixed_length_rejects_unmasked_padding(self):
        upstream, modules = self.hf_modules(mask_padding=False)
        with self.assertRaisesRegex(SystemExit, "unmasked padding"):
            self.execute("setup_ds_hf_bench_length", upstream, modules)

    def test_fixed_length_rejects_all_masked_targets(self):
        upstream, modules = self.hf_modules(target_tokens=False)
        with self.assertRaisesRegex(SystemExit, "lost training labels"):
            self.execute("setup_ds_hf_bench_length", upstream, modules)

    def test_superoffload_uses_parquet_directory_and_checks_cpuadam(self):
        rows = json.loads(self.fixture.read_text())
        calls = []
        class Dataset(list):
            column_names = ["instruction", "input", "output"]
            @classmethod
            def from_list(cls, values):
                return cls(values)
            def to_parquet(self, filename):
                calls.append(("write", Path(filename)))
                Path(filename).write_text(json.dumps(self))
        def load_dataset(directory):
            calls.append(("load", Path(directory)))
            return {"train": Dataset(json.loads((Path(directory) / "train.parquet").read_text()))}
        datasets = types.ModuleType("datasets")
        datasets.__version__ = "mock"
        datasets.Dataset, datasets.load_dataset = Dataset, load_dataset
        builder_module = types.ModuleType("deepspeed.ops.op_builder")
        class CPUAdamBuilder:
            def load(self, verbose):
                calls.append(("cpuadam", verbose))
        builder_module.CPUAdamBuilder = CPUAdamBuilder
        def preprocess(row, tokenizer, max_length):
            self.assertIn(row, rows)
            self.assertEqual(max_length, 128)
            return {"input_ids": [2] * 128, "labels": [2] * 128, "attention_mask": [1] * 128}
        upstream = types.SimpleNamespace(DEFAULT_OPTIMIZER_LR=0.001, load_tokenizer=lambda *args: object(),
                                         preprocess_alpaca_example=preprocess)
        self.execute("setup_ds_superoffload", upstream, {"datasets": datasets, "deepspeed.ops.op_builder": builder_module})
        self.assertEqual(calls, [("write", self.root / "parquet/train.parquet"),
                                 ("load", self.root / "parquet"), ("cpuadam", True)])

    def test_all_setup_python_blocks_compile(self):
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", SETUP, re.S)
        for index, source in enumerate(blocks):
            compile(source, f"setup heredoc {index}", "exec")


if __name__ == "__main__":
    unittest.main()

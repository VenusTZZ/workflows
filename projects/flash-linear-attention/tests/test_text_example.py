"""CPU checks for data preparation and the example's result contract."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTextPreparation(unittest.TestCase):
    def setUp(self) -> None:
        path = PROJECT / "example" / "train_text.py"
        self.assertTrue(path.is_file(), "The user-facing text training example is missing")
        self.example = load_module(path)

    def test_character_vocabulary_round_trip_and_persistence(self) -> None:
        text = "FLA 在昇腾上训练。\n"
        vocabulary = self.example.build_vocabulary(text)
        self.assertEqual(vocabulary[:3], ["<pad>", "<bos>", "<eos>"])
        reloaded = json.loads(json.dumps(vocabulary, ensure_ascii=False))
        tokens = self.example.encode(text, reloaded)
        self.assertEqual(self.example.decode(tokens, reloaded), text)
        self.assertEqual(self.example.decode([0, 1, *tokens, 2], reloaded), text)

    def test_unknown_prompt_character_has_clear_error(self) -> None:
        vocabulary = self.example.build_vocabulary("abc")
        with self.assertRaisesRegex(ValueError, "character"):
            self.example.encode("abd", vocabulary)

    def test_windows_preserve_token_order_without_shifting_labels(self) -> None:
        # The FLA causal-LM forward method performs the label shift internally.
        self.assertEqual(self.example.make_windows(list(range(10)), 4), [[0, 1, 2, 3], [4, 5, 6, 7]])

    def test_rejects_empty_or_too_short_training_data(self) -> None:
        with self.assertRaises(ValueError):
            self.example.build_vocabulary("")
        with self.assertRaisesRegex(ValueError, "sequence"):
            self.example.make_windows([3, 4], 128)
        with self.assertRaises(ValueError):
            self.example.make_windows([3, 4], 1)

    def test_bundled_corpus_can_supply_default_training_batches(self) -> None:
        text = (PROJECT / "example" / "corpus.txt").read_text(encoding="utf-8")
        tokens = self.example.encode(text, self.example.build_vocabulary(text))
        self.assertGreaterEqual(len(self.example.make_windows(tokens, 128)), 2)


class TestExampleResult(unittest.TestCase):
    def setUp(self) -> None:
        path = PROJECT / "scripts" / "validate_example.py"
        self.assertTrue(path.is_file(), "The text example result validator is missing")
        self.validator = load_module(path)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.target = self.root / "target"
        self.target.mkdir()
        self.output = self.root / "output"
        self.output.mkdir()
        checkpoint = self.output / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text('{"model_type": "gated_deltanet"}')
        (checkpoint / "vocabulary.json").write_text('["<pad>", "<bos>", "<eos>", "a"]')
        # This fixture checks the result contract, not checkpoint loading or NPU execution.
        (checkpoint / "model.safetensors").write_bytes(b"fixture-placeholder")
        (self.output / "generated.txt").write_text("aaaa", encoding="utf-8")
        self.report = {
            "task": "character-language-model",
            "device": "npu",
            "backend": "npu",
            "steps": 2,
            "initial_loss": 2.0,
            "final_loss": 1.0,
            "losses": [2.0, 1.5],
            "grad_norms": [0.5, 0.2],
            "weights_changed": True,
            "checkpoint_roundtrip_ok": True,
            "prompt": "aa",
            "continuation": "aa",
            "generated_tokens": 2,
            "max_new_tokens": 4,
            "fla_path": str(self.target / "fla/__init__.py"),
        }

    def validate(self):
        (self.output / "metrics.json").write_text(json.dumps(self.report), encoding="utf-8")
        return self.validator.validate_result(self.output, self.target)

    def test_accepts_completed_training_reload_and_generation(self) -> None:
        self.assertEqual(self.validate(), self.report)

    def test_cpu_fallback_or_wrong_checkout_fails(self) -> None:
        for field, value in (("device", "cpu"), ("backend", "cuda"), ("fla_path", "/tmp/other/fla/__init__.py")):
            with self.subTest(field=field):
                previous = self.report[field]
                self.report[field] = value
                with self.assertRaises(ValueError):
                    self.validate()
                self.report[field] = previous

    def test_nonfinite_loss_or_gradient_fails(self) -> None:
        for field, value in (("losses", [2.0, float("nan")]), ("grad_norms", [0.5, float("inf")]), ("final_loss", float("nan"))):
            with self.subTest(field=field):
                previous = self.report[field]
                self.report[field] = value
                with self.assertRaises(ValueError):
                    self.validate()
                self.report[field] = previous

    def test_training_must_run_update_weights_and_reduce_training_loss(self) -> None:
        for field, value in (("steps", 0), ("losses", [2.0]), ("weights_changed", False), ("final_loss", 2.1)):
            with self.subTest(field=field):
                previous = self.report[field]
                self.report[field] = value
                with self.assertRaises(ValueError):
                    self.validate()
                self.report[field] = previous

    def test_reload_or_empty_generation_fails(self) -> None:
        for field, value in (("checkpoint_roundtrip_ok", False), ("continuation", ""), ("generated_tokens", 0)):
            with self.subTest(field=field):
                previous = self.report[field]
                self.report[field] = value
                with self.assertRaises(ValueError):
                    self.validate()
                self.report[field] = previous

    def test_missing_checkpoint_fails(self) -> None:
        (self.output / "checkpoint/model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.validate()

    def test_saved_generation_must_match_the_report(self) -> None:
        (self.output / "generated.txt").write_text("stale output", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "generated"):
            self.validate()


if __name__ == "__main__":
    unittest.main()

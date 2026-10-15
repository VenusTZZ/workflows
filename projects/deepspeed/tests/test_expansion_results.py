"""CPU-only checks for new upstream-output acceptance criteria."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_expansion.py"
spec = importlib.util.spec_from_file_location("ds_expansion_results", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExpansionResultTests(unittest.TestCase):
    def summary(self, kind="model_tensor_offload"):
        pin_key = "pin_memory" if kind == "model_tensor_offload" else "use_pin_memory"
        first = "unpinned" if kind == "model_tensor_offload" else "pageable"
        arm = {"device": "npu", "experiment": kind, "steps": 3,
               "step_avg_s": 2.0, "step_min_s": 1.0, "zero_stage": 3}
        return {first: {**arm, pin_key: False}, "pinned": {**arm, pin_key: True}, "speedup": 0.8}

    def test_pin_drivers_accept_both_arms_without_performance_threshold(self):
        for kind in ("model_tensor_offload", "activation_offload"):
            data = self.summary(kind)
            self.assertEqual(module.validate(kind, "DRIVERRESULT=" + json.dumps(data)), data)

    def test_pin_drivers_reject_cpu_missing_arm_wrong_steps_and_nan(self):
        for key, value in (("device", "cpu"), ("steps", 2), ("step_avg_s", float("nan")),
                           ("pin_memory", False), ("zero_stage", 2)):
            data = self.summary()
            data["pinned"][key] = value
            with self.assertRaises(ValueError):
                module.validate("model_tensor_offload", "DRIVERRESULT=" + json.dumps(data))
        with self.assertRaises(ValueError):
            module.validate("activation_offload", "ARMRESULT={}")

    def copy_rows(self):
        return [{"experiment": "h2d_d2h", "arm": arm, "size_mib": 1,
                 "h2d_gbps": 1.0, "d2h_gbps": 2.0, "accelerator_is_pinned": arm != "pageable"}
                for arm in ("pageable", "torch", "native-unregistered")]

    def test_copy_requires_all_modes_and_positive_bandwidth(self):
        rows = self.copy_rows()
        text = "\n".join("RESULT=" + json.dumps(row) for row in rows)
        self.assertEqual(module.validate("h2d_d2h", text), rows)
        with self.assertRaises(ValueError):
            module.validate("h2d_d2h", text.splitlines()[0])
        rows[1]["d2h_gbps"] = float("inf")
        with self.assertRaises(ValueError):
            module.validate("h2d_d2h", "\n".join("RESULT=" + json.dumps(row) for row in rows))

    def test_zenflow_requires_all_updates_and_real_saved_files(self):
        text = "\n".join(f"Step {step}, Loss: 1.5, Time: 1ms" for step in range(1, 17))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with self.assertRaises(ValueError):
                module.validate("zenflow_finetune", text + "\nTraining complete!", output)
            (output / "latest").write_text("global_step16")
            checkpoint = output / "global_step16"
            checkpoint.mkdir()
            (checkpoint / "mp_rank_00_model_states.pt").write_bytes(b"mock checkpoint")
            (output / "tokenizer_config.json").write_text("{}")
            self.assertEqual(len(module.validate("zenflow_finetune", text + "\nTraining complete!", output)), 16)
            with self.assertRaises(ValueError):
                module.validate("zenflow_finetune", text.replace("Loss: 1.5", "Loss: nan") + "\nTraining complete!")
            with self.assertRaises(ValueError):
                module.validate("zenflow_finetune", text)

    def test_opsd_requires_three_finite_losses_and_generated_tokens(self):
        text = "\n".join(f"[opsd][step {step}] loss=0.1234 rollout=1s resp_tok=8" for step in range(3))
        self.assertEqual(len(module.validate("opsd", text)), 3)
        for bad in (text.replace("0.1234", "nan"), text.replace("resp_tok=8", "resp_tok=0"),
                    text.splitlines()[0]):
            with self.assertRaises(ValueError):
                module.validate("opsd", bad)

    def test_isolated_opsd_requires_topology_and_success_markers(self):
        self.assertEqual(module.validate("opsd_student",
            "STUDENT_OK loss=0.0123 mem=0.00GB offload=0 autotp=2 ws=2")["world_size"], 2)
        self.assertEqual(module.validate("opsd_teacher",
            "TEACHER_OK shape=(1, 12, 151936) mem=0.00GB offload=True autotp=2 ws=2")["cache_shape"], [1, 12, 151936])
        for kind, text in (("opsd_student", "STUDENT_OK loss=nan mem=0.00GB offload=0 autotp=2 ws=2"),
                           ("opsd_teacher", "TEACHER_OK shape=(1, 12, 151936) mem=0GB offload=True autotp=1 ws=1")):
            with self.assertRaises(ValueError):
                module.validate(kind, text)


if __name__ == "__main__":
    unittest.main()

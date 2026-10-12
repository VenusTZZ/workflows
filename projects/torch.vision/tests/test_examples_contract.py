"""Offline manifest, fixture and launcher contracts; no NPU training claims."""

import ast
import importlib.util
from pathlib import Path
import re
import struct
import tempfile
import unittest

import yaml


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("vision_ci_assets", PROJECT / "scripts/prepare_ci_assets.py")
ASSETS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSETS)


class VisionExamplesContract(unittest.TestCase):
    def test_manifest_and_no_helper_registration(self):
        text = (PROJECT / "examples_manifest.yaml").read_text(encoding="utf-8")
        manifest = yaml.safe_load(text)
        supported = {item["path"] for item in manifest["supported"]}
        unsupported = set(manifest["unsupported"])
        self.assertEqual((len(supported), len(unsupported)), (8, 16))
        self.assertFalse(supported & unsupported)
        self.assertNotIn("references/similarity/test.py", unsupported)
        for path in supported | unsupported:
            lines = text.splitlines()
            index = next(i for i, line in enumerate(lines) if line.strip() in {f"- {path}", f"- path: {path}"})
            self.assertTrue(lines[index - 1].strip().startswith("# "), path)

    def test_new_recipes_select_npu_and_preserve_core_paths(self):
        manifest = yaml.safe_load((PROJECT / "examples_manifest.yaml").read_text(encoding="utf-8"))
        entries = {entry["path"]: entry for entry in manifest["supported"]}
        flow = entries["references/optical_flow/train.py"]["overlay_args"]
        video = entries["references/video_classification/train.py"]["overlay_args"]
        for args in (flow, video):
            self.assertIn("npu:0", args)
            self.assertNotIn("--amp", args)
            self.assertNotIn("--test-only", args)
            self.assertEqual(args[args.index("--epochs") + 1], "1")
        self.assertIn("raft_small", flow)
        self.assertIn("r3d_18", video)
        self.assertEqual(video[video.index("--clip-len") + 1], "4")
        launcher = (PROJECT / "scripts/run_example.sh").read_text(encoding="utf-8")
        self.assertIn('checkpoint.unlink(missing_ok=True)', launcher)
        self.assertIn('"ci_flow_0.pth"', launcher)
        self.assertIn('points.device.type != "npu"', launcher)
        self.assertNotIn("TORCH_COMPILE_DISABLE", launcher)
        for script in ("setup_example.sh", "run_example.sh"):
            text = (PROJECT / "scripts" / script).read_text(encoding="utf-8")
            for chunk in re.findall(r"<<'PY'\n(.*?)\nPY", text, re.DOTALL):
                ast.parse(chunk)

    def test_actual_npu_guard_accepts_npu_and_rejects_cpu(self):
        from types import SimpleNamespace

        class Tensor:
            def __init__(self, device):
                self.device = SimpleNamespace(type=device)

        text = (PROJECT / "scripts/run_example.sh").read_text(encoding="utf-8")
        chunk = re.findall(r"<<'PY'\n(.*?)\nPY", text, re.DOTALL)[0]
        nodes = [node for node in ast.parse(chunk).body if isinstance(node, ast.FunctionDef) and node.name == "require_npu"]
        scope = {"torch": SimpleNamespace(Tensor=Tensor), "entry": "mock-example", "namespace": {"out": Tensor("npu")}}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "actual-vision-device-guard", "exec"), scope)
        scope["require_npu"]("out")
        scope["namespace"]["out"] = Tensor("cpu")
        with self.assertRaises(SystemExit):
            scope["require_npu"]("out")
        with self.assertRaises(SystemExit):
            scope["require_npu"]("missing")

    def test_flying_chairs_native_schema_and_values(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ASSETS.prepare_flow(root)
            self.assertEqual(len(list((root / "FlyingChairs/data").glob("*.ppm"))), 8)
            flows = list((root / "FlyingChairs/data").glob("*.flo"))
            self.assertEqual(len(flows), 4)
            with flows[0].open("rb") as handle:
                magic, width, height = struct.unpack("<fii", handle.read(12))
                data = np.frombuffer(handle.read(), dtype="<f4").reshape(height, width, 2)
            self.assertEqual((magic, width, height), (202021.25, 512, 384))
            self.assertTrue((data[..., 0] == 1).all())
            self.assertTrue((data[..., 1] == 0).all())
            self.assertEqual((root / "FlyingChairs/FlyingChairs_train_val.txt").read_text().split(), ["1"] * 4)

    @unittest.skipUnless(importlib.util.find_spec("av"), "AV validation requires the declared PyAV dependency")
    def test_video_is_decodable_and_has_exact_400_class_schema(self):
        import av

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ASSETS.prepare_video(root)
            for split in ("train", "val"):
                samples = list((root / split).glob("class_*/ci.avi"))
                self.assertEqual(len(samples), 400)
                with av.open(str(samples[0])) as video:
                    frames = list(video.decode(video=0))
                    self.assertEqual(len(frames), 4)
                    self.assertEqual((frames[0].width, frames[0].height), (40, 40))
                    self.assertEqual(video.streams.video[0].average_rate, 8)


if __name__ == "__main__":
    unittest.main()

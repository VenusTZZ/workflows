"""Check upstream example outputs; no NPU execution or source interception here."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys


def positive(value):
    return isinstance(value, (float, int)) and math.isfinite(value) and value > 0


def validate(kind: str, text: str, output: Path | None = None) -> object:
    if kind in {"model_tensor_offload", "activation_offload"}:
        summaries = [json.loads(line.split("=", 1)[1]) for line in text.splitlines()
                     if line.startswith("DRIVERRESULT=")]
        if len(summaries) != 1:
            raise ValueError("upstream driver must report exactly one two-arm summary")
        report = summaries[0]
        unpinned = "unpinned" if kind == "model_tensor_offload" else "pageable"
        for name in (unpinned, "pinned"):
            row = report[name]
            pin_key = "pin_memory" if kind == "model_tensor_offload" else "use_pin_memory"
            if (row["device"] != "npu" or row["steps"] != 3 or
                    bool(row[pin_key]) != (name == "pinned") or
                    row["experiment"] != kind or
                    not all(positive(row[key]) for key in ("step_avg_s", "step_min_s"))):
                raise ValueError(f"invalid NPU {kind} arm: {row}")
            if kind == "model_tensor_offload" and row["zero_stage"] != 3:
                raise ValueError("model tensor offload must exercise ZeRO-3")
        if not positive(report["speedup"]):
            raise ValueError("speedup must be finite/positive, not necessarily greater than 1")
    elif kind == "h2d_d2h":
        report = [json.loads(line.split("=", 1)[1]) for line in text.splitlines()
                  if line.startswith("RESULT=")]
        expected = {(arm, size) for arm in ("pageable", "torch", "native-unregistered")
                    for size in (1,)}
        if len(report) != len(expected) or {(r["arm"], r["size_mib"]) for r in report} != expected:
            raise ValueError("copy benchmark must report all three 1-MiB buffer modes")
        for row in report:
            if row["experiment"] != kind or not all(positive(row[k]) for k in ("h2d_gbps", "d2h_gbps")):
                raise ValueError(f"invalid copy measurement: {row}")
            if bool(row["accelerator_is_pinned"]) != (row["arm"] != "pageable"):
                raise ValueError(f"buffer pinning did not match the requested arm: {row}")
    elif kind == "zenflow_finetune":
        report = {int(step): float(loss) for step, loss in
                  re.findall(r"Step\s+(\d+), Loss:\s*([^,\s]+)", text)}
        if sorted(report) != list(range(1, 17)) or not all(math.isfinite(v) for v in report.values()):
            raise ValueError("ZenFlow finetune must complete 16 finite-loss optimizer updates")
        if "Training complete!" not in text:
            raise ValueError("ZenFlow did not complete checkpoint/tokenizer saving")
        if output is not None:
            if not (output / "latest").is_file() or not list(output.rglob("*model_states.pt")):
                raise ValueError("missing upstream DeepSpeed checkpoint")
            if not (output / "tokenizer_config.json").is_file():
                raise ValueError("missing saved upstream tokenizer")
    elif kind == "opsd_student":
        matches = re.findall(r"STUDENT_OK loss=(\S+) mem=\S+ offload=0 autotp=2 ws=2", text)
        if len(matches) != 1 or not math.isfinite(float(matches[0])):
            raise ValueError("student must complete finite-loss TP=2/ZeRO-3 forward/backward/step")
        report = {"loss": float(matches[0]), "autotp": 2, "world_size": 2, "offload": False}
    elif kind == "opsd_teacher":
        matches = re.findall(r"TEACHER_OK shape=\(1, 12, (\d+)\) mem=\S+ offload=True autotp=2 ws=2", text)
        if len(matches) != 1 or int(matches[0]) <= 0:
            raise ValueError("teacher must create CPU logit cache from TP=2/ZeRO-3 offload forward")
        report = {"cache_shape": [1, 12, int(matches[0])], "autotp": 2, "world_size": 2, "offload": True}
    elif kind == "opsd":
        report = []
        for step, loss, tokens in re.findall(
                r"\[opsd\]\[step (\d+)\] loss=(\S+).*?resp_tok=(\d+)", text):
            row = {"step": int(step), "loss": float(loss), "response_tokens": int(tokens)}
            if not math.isfinite(row["loss"]) or row["response_tokens"] <= 0:
                raise ValueError(f"invalid OPSD rollout/teacher/student step: {row}")
            report.append(row)
        if [row["step"] for row in report] != [0, 1, 2]:
            raise ValueError("OPSD must complete exactly three rollout/distillation steps")
    else:
        raise ValueError(f"unknown expansion result kind: {kind}")
    return report


if __name__ == "__main__":
    kind, log, report_file, *extra = sys.argv[1:]
    report = validate(kind, Path(log).read_text(), Path(extra[0]) if extra else None)
    Path(report_file).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"validated upstream {kind} results")

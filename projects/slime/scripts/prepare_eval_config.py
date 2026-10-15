"""Create native multi-task eval config with two distinct local scorers."""
from pathlib import Path
import sys
import yaml


def build_config(fixtures):
    fixtures = Path(fixtures).resolve()
    return {"eval": {"defaults": {"max_response_len": 128, "top_p": 0.7, "n_samples_per_eval_prompt": 1}, "datasets": [
        {"name": "ci_math", "path": str(fixtures / "ci_dapo_16.jsonl"), "rm_type": "deepscaler"},
        {"name": "ci_multiple_choice", "path": str(fixtures / "ci_gpqa_8.jsonl"), "rm_type": "gpqa"},
    ]}}


if __name__ == "__main__":
    destination = Path(sys.argv[2])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(build_config(sys.argv[1]), sort_keys=False), encoding="utf-8")
    print(f"native multi-task config: {destination}")

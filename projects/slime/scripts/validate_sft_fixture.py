"""Check native SFT token masks before allocating Ray training actors."""
import json
import sys
from pathlib import Path


def load_messages(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 8:
        raise ValueError("CI SFT fixture must contain eight conversations")
    for row in rows:
        messages = row["messages"]
        if not isinstance(messages, list) or not any(m.get("role") == "assistant" and m.get("content") for m in messages):
            raise ValueError("SFT requires a non-empty assistant response")
    return rows


def main():
    from slime.utils.mask_utils import MultiTurnLossMaskGenerator
    from slime.utils.processing_utils import load_tokenizer
    rows = load_messages(sys.argv[1])
    tokenizer = load_tokenizer(sys.argv[2], trust_remote_code=True)
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
    for index, row in enumerate(rows):
        tokens, mask = generator.get_loss_mask(row["messages"])
        if len(tokens) != len(mask) or not sum(mask):
            raise ValueError(f"SFT row {index} has no valid supervised token mask")
        if len(tokens) > 1024:
            raise ValueError(f"SFT row {index} exceeds the CI token budget")
    print("SFT fixture: eight conversations with non-empty native assistant masks")


if __name__ == "__main__":
    main()

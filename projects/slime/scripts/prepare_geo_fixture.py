"""Generate native Geo3K JSONL with small, local geometry images."""
import argparse
import json
import math
from pathlib import Path


def prepare_fixture(source, destination):
    from PIL import Image, ImageDraw
    facts = json.loads(Path(source).read_text(encoding="utf-8"))
    if len(facts) != 8:
        raise ValueError("Geo3K CI expects eight angle questions")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, fact in enumerate(facts):
        a, b = fact["angle_a"], fact["angle_b"]
        if not (0 < a < 90 and 0 < b < 90 and 180 - a - b == int(fact["answer"])):
            raise ValueError(f"Invalid triangle angle fixture: {fact}")
        image = Image.new("RGB", (128, 128), "white")
        draw = ImageDraw.Draw(image)
        left, right = (12, 110), (116, 110)
        ta, tb = math.tan(math.radians(a)), math.tan(math.radians(b))
        x = 104 * tb / (ta + tb)
        apex = (12 + x, 110 - ta * x)
        draw.line([left, right, apex, left], fill="black", width=2)
        draw.text((13, 113), "A", fill="black")
        draw.text((111, 113), "B", fill="black")
        draw.text((apex[0] - 3, apex[1] - 11), "C", fill="black")
        draw.text((19, 94), f"{a}", fill="black")
        draw.text((91, 94), f"{b}", fill="black")
        path = destination / f"triangle-{index}.png"
        image.save(path)
        problem = ('<image>Find angle C in the triangle. Read the labeled angles A and B from the image. '
            'Before answering, call the scoring tool using exactly '
            '<tool_call>{"name":"calc_score","arguments":{"answer":"YOUR_NUMBER"}}</tool_call>. '
            'After its feedback, provide the final angle in degrees as \\boxed{YOUR_NUMBER}.')
        rows.append({"problem": problem, "answer": fact["answer"], "images": [path.as_uri()]})
    output = destination / "geo3k-ci.jsonl"
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    arguments = parser.parse_args()
    print(prepare_fixture(arguments.source, arguments.destination))

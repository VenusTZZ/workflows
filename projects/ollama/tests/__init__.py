"""Make the shared Markdown runner importable from this project directory."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
for path in (root, root / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

"""Tests package marker (injects repo root src/ into sys.path)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]  # tests/ -> trl/ -> projects/ -> workflows/
_SRC = _REPO_ROOT / 'src'
for _p in (_SRC, _REPO_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

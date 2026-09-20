"""Tests package marker (mirrors projects/roll/tests/__init__.py)."""

from __future__ import annotations

import sys
from pathlib import Path

# tests/ -> slime/ -> projects/ -> workflows/
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (_REPO_ROOT / 'src', _REPO_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)


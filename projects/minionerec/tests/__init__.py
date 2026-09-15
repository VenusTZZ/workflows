"""Tests package marker.

Single responsibility: ``sys.path`` bootstrap — inject the repo root's
``src/`` (so that ``from workflows.markdown_doc_test_base import ...``
resolves) plus the repo root itself (defensive).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]  # tests/ -> modelscope/ -> projects/ -> workflows/
_SRC = _REPO_ROOT / 'src'
for _p in (_SRC, _REPO_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

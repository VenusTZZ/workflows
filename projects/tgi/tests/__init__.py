"""Tests package marker.

Single responsibility: ``sys.path`` bootstrap — inject the repo root's
``src/`` (so that ``from workflows.markdown_doc_test_base import ...``
resolves) plus the repo root itself (defensive: keeps the layout
importable regardless of the caller's cwd).

Framework deps (mistune) are installed by the common quick-start workflow
template, not at import time here.

Why it lives here:
* unittest treats ``tests/`` as a package; the parent ``__init__.py``
  executes before any submodule import.
* It runs before ``tests/test_*.py`` import, which is the earliest
  opportunity to inject ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]  # tests/ -> tgi/ -> projects/ -> workflows/
_SRC = _REPO_ROOT / 'src'
for _p in (_SRC, _REPO_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

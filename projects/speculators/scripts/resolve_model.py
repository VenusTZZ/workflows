#!/usr/bin/env python3
"""Resolve an example MODEL hub id to a local snapshot directory.

vLLM 0.23's modelscope_list_repo_files expects file['Type'] and crashes
on current modelscope_hub responses. snapshot_download still works, and
passing a filesystem path skips that listing path.

Speculator checkpoints also name a verifier hub id in
speculators_config.verifier.name_or_path. Snapshot that too and rewrite
the config to a local path, otherwise vLLM still hits the broken listing.
Never git add/commit/push the target tree.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from huggingface_hub import snapshot_download as hf_snapshot

try:
    from modelscope import snapshot_download as ms_snapshot
except ImportError:
    ms_snapshot = None


def default_model_id(script: Path) -> str:
    text = script.read_text()
    match = re.search(
        r'^MODEL="\$\{MODEL:-([^}]+)\}"$', text, flags=re.M)
    if match:
        return match.group(1)
    match = re.search(r'^MODEL="([^"]+)"$', text, flags=re.M)
    if match:
        return match.group(1)
    raise SystemExit(f'no MODEL= assignment in {script}')


def snapshot(hub_id: str) -> str:
    if hub_id.startswith('/'):
        return hub_id
    if ms_snapshot is not None:
        try:
            return str(ms_snapshot(hub_id))
        except Exception as exc:
            print(f'modelscope snapshot failed ({exc}); trying huggingface',
                  file=sys.stderr)
    return str(hf_snapshot(hub_id))


def rewrite_nested_verifier(speculator_dir: str) -> str:
    cfg_path = Path(speculator_dir) / 'config.json'
    if not cfg_path.is_file():
        return speculator_dir
    cfg = json.loads(cfg_path.read_text())
    spec = cfg.get('speculators_config')
    if not isinstance(spec, dict):
        return speculator_dir
    verifier = spec.get('verifier')
    if not isinstance(verifier, dict):
        return speculator_dir
    name = verifier.get('name_or_path')
    if not isinstance(name, str) or not name or name.startswith('/'):
        return speculator_dir
    local = snapshot(name)
    if local == name:
        return speculator_dir
    verifier['name_or_path'] = local
    spec['verifier'] = verifier
    cfg['speculators_config'] = spec
    cfg_path.write_text(json.dumps(cfg, indent=2) + '\n')
    print(f'rewrote verifier {name} -> {local}', file=sys.stderr)
    return speculator_dir


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('usage: resolve_model.py <patched-example>')
    script = Path(sys.argv[1])
    hub = os.environ.get('MODEL') or default_model_id(script)
    path = rewrite_nested_verifier(snapshot(hub))
    if not path.startswith('/'):
        raise SystemExit(f'resolved path is not absolute: {path}')
    print(path)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Diff a target examples tree against examples_manifest.yaml.

The manifest's scan section decides what to scan. Default unit is
files (root + include_extensions). unit: directories treats each
child directory as one example (optional marker file, max_depth).
unit: mixed unions depth-limited directories (marker optional) with
depth-limited files. Writes a machine-readable JSON result
(new/stale paths, supported entries, target repo/ref) to
--result-json and stdout.
Writes supported_matrix and has_supported to GITHUB_OUTPUT when set.

New/stale paths are recorded but do not fail the check, with one
exception: a supported entry whose path is missing on disk exits 1
after writing the result JSON. Failing here, on a free hosted runner,
beats letting run-example spend ~20 minutes of NPU-runner setup before
discovering the example is gone.

Requires PyYAML (preinstalled on ubuntu-latest runners).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath

import yaml

from examples_manifest_scan import load_scan, scan_examples

NPU_DEVICES_RE = re.compile(r'^\d+(,\d+)*$')


def load_manifest(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    loaded = load_scan(data.get('scan') or {})
    loaded['supported'] = data.get('supported') or []
    loaded['unsupported'] = data.get('unsupported') or []
    return loaded


def listed_paths(manifest: dict) -> set[str]:
    paths = {item['path'] for item in manifest['supported'] if item.get('path')}
    paths.update(path for path in manifest['unsupported'] if path)
    return paths


def missing_on_disk(target_root: Path, paths: set[str]) -> list[str]:
    return sorted(
        path for path in paths if not (target_root / path).exists())


def device_options_from_npu_devices(npu_devices: str) -> str:
    if not NPU_DEVICES_RE.fullmatch(npu_devices):
        raise ValueError(
            f"npu_devices must match '^\\d+(,\\d+)*$', got {npu_devices!r}")
    return ' '.join(
        f'--device=/dev/davinci{index}' for index in npu_devices.split(','))


def matrix_entries(supported: list[dict]) -> list[dict]:
    entries: list[dict] = []
    errors: list[str] = []
    for item in supported:
        path = item.get('path', '<missing path>')
        npu_devices = item.get('npu_devices')
        if not isinstance(npu_devices, str):
            errors.append(
                f'{path}: npu_devices must be a string matching '
                f"'^\\d+(,\\d+)*$', got {npu_devices!r}")
            continue
        try:
            options = device_options_from_npu_devices(npu_devices)
        except ValueError as exc:
            errors.append(f'{path}: {exc}')
            continue
        if 'overlay' in item:
            errors.append(
                f'{path}: use overlay_args (list of CLI strings), '
                'not overlay (file path)')
            continue
        overlay_args = item.get('overlay_args')
        if overlay_args is None:
            overlay_args = []
        elif not isinstance(overlay_args, list) or not all(
                isinstance(arg, str) and arg.strip() for arg in overlay_args):
            errors.append(
                f'{path}: overlay_args must be a list of non-empty strings')
            continue
        exec_path = item.get('exec')
        if exec_path is not None and (
                not isinstance(exec_path, str) or not exec_path.strip()):
            errors.append(f'{path}: exec must be a non-empty string')
            continue
        entry = dict(item)
        entry['device_options'] = options
        entry['overlay_args'] = overlay_args
        if exec_path is not None:
            entry['exec'] = exec_path.strip()
        # Display name for the run-example job label: parent dir + stem
        # when the file sits below the scan root (3+ path components) -
        # same-named scripts in different directories must get distinct
        # labels (peft's five */train_dreambooth.py all collapsed to
        # "train_dreambooth" and were indistinguishable in the Actions
        # UI). Collapses to the bare stem when the parent dir equals the
        # stem (examples/foo/foo.py) or the file sits directly under the
        # scan root (examples/foo.py). Keep in sync with
        # check_supported_entries.py.
        parts = PurePosixPath(path).parts
        stem = PurePosixPath(path).stem
        if len(parts) >= 3 and parts[-2] != stem:
            entry['name'] = f'{parts[-2]}/{stem}'
        else:
            entry['name'] = stem
        entries.append(entry)
    if errors:
        for message in errors:
            print(message, file=sys.stderr)
        print('fix the supported entries before this '
              'pipeline can schedule examples', file=sys.stderr)
        raise SystemExit(1)
    return entries


def write_result(result_json: str, trigger: str, target_repo: str,
                 target_ref: str, new_paths: list[str], stale_paths: list[str],
                 supported: list[dict]) -> None:
    result = {
        'trigger': trigger,
        'target_repo': target_repo,
        'target_ref': target_ref,
        'new_paths': new_paths,
        'stale_paths': stale_paths,
        'supported': supported,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    Path(result_json).write_text(text, encoding='utf-8')
    sys.stdout.write(text)


def write_github_output(supported: list[dict]) -> None:
    output_path = os.environ.get('GITHUB_OUTPUT')
    if not output_path:
        return
    payload = json.dumps(supported, ensure_ascii=False)
    with open(output_path, 'a', encoding='utf-8') as handle:
        handle.write('supported_matrix<<EOF\n')
        handle.write(payload)
        handle.write('\nEOF\n')
        handle.write('has_supported={}\n'.format('true' if supported else 'false'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--result-json', required=True)
    parser.add_argument('--trigger', default='')
    parser.add_argument('--target-repo', default='')
    parser.add_argument('--target-ref', default='')
    args = parser.parse_args()
    target_root = Path(args.target_root).resolve()
    manifest = load_manifest(Path(args.manifest))
    scanned = set(scan_examples(target_root, manifest))
    listed = listed_paths(manifest)
    new_paths = sorted(scanned - listed)
    stale_paths = missing_on_disk(target_root, listed)
    write_result(args.result_json, args.trigger, args.target_repo,
                 args.target_ref, new_paths, stale_paths, manifest['supported'])
    write_github_output(matrix_entries(manifest['supported']))
    missing_supported = missing_on_disk(
        target_root, {item['path'] for item in manifest['supported']
                      if item.get('path')})
    if missing_supported:
        for path in missing_supported:
            print(f'supported example missing from target tree: {path}',
                  file=sys.stderr)
        print('fix the supported section of the manifest before this '
              'pipeline can schedule examples', file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == '__main__':
    main()

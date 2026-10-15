#!/usr/bin/env python3
"""Tests for bootstrap_manifest's scan rules."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
sys.path.insert(0, str(SCRIPTS))

from bootstrap_manifest import (  # noqa: E402
    load_scan,
    scan_examples,
)


def write_tree(root: Path, files: list[str]) -> None:
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('x\n', encoding='utf-8')


class LoadScanTests(unittest.TestCase):
    def test_unknown_unit_fails(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            load_scan({'unit': 'trees'})
        self.assertIn('scan.unit', str(caught.exception))

    def test_max_depth_must_be_positive_int(self) -> None:
        with self.assertRaises(SystemExit):
            load_scan({'unit': 'mixed', 'max_depth': 0})
        with self.assertRaises(SystemExit):
            load_scan({'unit': 'mixed', 'max_depth': True})

    def test_directories_marker_cannot_be_blank(self) -> None:
        with self.assertRaises(SystemExit):
            load_scan({'unit': 'directories', 'marker': '   '})

    def test_mixed_whitespace_marker_fails(self) -> None:
        with self.assertRaises(SystemExit):
            load_scan({'unit': 'mixed', 'marker': '   '})

    def test_mixed_empty_marker_means_all_dirs(self) -> None:
        scan = load_scan({'unit': 'mixed', 'marker': ''})
        self.assertEqual(scan['marker'], '')
        self.assertEqual(scan['include_extensions'], ('.sh', '.py'))

    def test_empty_extension_fails(self) -> None:
        with self.assertRaises(SystemExit):
            load_scan({
                'unit': 'files',
                'include_extensions': ['.sh', ''],
            })

    def test_explicit_empty_root_means_repo_root(self) -> None:
        # deepspeed / torch.vision style: root: '' = scan the repo root.
        self.assertEqual(load_scan({'root': ''})['scan_root'], '')
        # Absent root still falls back to the default examples/.
        self.assertEqual(load_scan({})['scan_root'], 'examples')


class ScanBehaviorTests(unittest.TestCase):
    def test_files_unit_is_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_tree(root, [
                'examples/a.sh',
                'examples/nested/b.py',
                'examples/nested/c.yaml',
                'examples/skip.txt',
            ])
            found = scan_examples(root, load_scan({'unit': 'files'}))
            self.assertEqual(found, [
                'examples/a.sh',
                'examples/nested/b.py',
                'examples/nested/c.yaml',
            ])

    def test_directories_unit_requires_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_tree(root, [
                'examples/cli/CMakeLists.txt',
                'examples/python/whisper.py',
                'examples/wchess/libwchess/CMakeLists.txt',
            ])
            found = scan_examples(root, load_scan({
                'unit': 'directories',
                'marker': 'CMakeLists.txt',
                'max_depth': 1,
            }))
            self.assertEqual(found, ['examples/cli'])

    def test_mixed_finds_top_level_dirs_and_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_tree(root, [
                'examples/cli/CMakeLists.txt',
                'examples/python/whisper_processor.py',
                'examples/server.py',
                'examples/generate-karaoke.sh',
                'examples/helpers.js',
                'examples/common.cpp',
            ])
            found = scan_examples(root, load_scan({
                'unit': 'mixed',
                'max_depth': 1,
                'include_extensions': ['.sh', '.py'],
            }))
            self.assertEqual(found, [
                'examples/cli',
                'examples/generate-karaoke.sh',
                'examples/python',
                'examples/server.py',
            ])

    def test_mixed_does_not_split_nested_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_tree(root, [
                'examples/python/whisper_processor.py',
                'examples/python/test_whisper_processor.py',
            ])
            found = scan_examples(root, load_scan({
                'unit': 'mixed',
                'max_depth': 1,
            }))
            self.assertEqual(found, ['examples/python'])

    def test_mixed_dedupes_and_sorts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_tree(root, [
                'examples/z-dir/keep.txt',
                'examples/a.sh',
                'examples/m-dir/keep.txt',
            ])
            found = scan_examples(root, load_scan({
                'unit': 'mixed',
                'max_depth': 1,
                'include_extensions': ['.sh'],
            }))
            self.assertEqual(found, [
                'examples/a.sh',
                'examples/m-dir',
                'examples/z-dir',
            ])


if __name__ == '__main__':
    unittest.main()

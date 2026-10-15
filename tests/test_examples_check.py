#!/usr/bin/env python3
"""Tests for the examples-check engine (design: examples-check-design.md §5.2)."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
sys.path.insert(0, str(SCRIPTS))

import examples_check as ec  # noqa: E402


def write_manifest(root: Path, text: str) -> Path:
    path = root / 'examples_manifest.yaml'
    path.write_text(text, encoding='utf-8')
    return path


def make_tree(root: Path, paths: list[str]) -> None:
    for relative in paths:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('x\n', encoding='utf-8')


def fake_fetcher(paths: list[str], ref: str = 'v1.0.0',
                  ref_source: str = 'release', sha: str = 'abc123'):
    def fetch(repo: str, default_branch: str) -> dict:
        return {'paths': paths, 'ref': ref, 'ref_source': ref_source,
                'tree_sha': sha}
    return fetch


class RecordingSetTests(unittest.TestCase):
    """§5.2 用例 1：记录集折算。"""

    def test_source_project_entries_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_manifest(Path(tmp), """\
version: 1
scan:
  root: examples
  include_extensions: ['.py']
supported:
  - path: examples/a.py
  - source: project
    path: example/own.py
unsupported:
  - examples/b.py
""")
            manifest = ec.load_manifest(path)
            self.assertEqual(manifest['recorded'],
                             ['examples/a.py', 'examples/b.py'])

    def test_unsupported_non_string_is_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_manifest(Path(tmp), """\
version: 1
scan:
  root: examples
  include_extensions: ['.py']
supported: []
unsupported:
  - path: examples/a.py
""")
            with self.assertRaises(ec.ConfigError):
                ec.load_manifest(path)

    def test_supported_entry_without_path_is_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_manifest(Path(tmp), """\
version: 1
scan:
  root: examples
  include_extensions: ['.py']
supported:
  - profile: cann
unsupported: []
""")
            with self.assertRaises(ec.ConfigError):
                ec.load_manifest(path)


    def test_split_mode_resolves_actual_default_branch(self) -> None:
        calls = []

        def fake_http(url: str, token=None):
            calls.append(url)
            if url.endswith('/repos/org/demo-examples'):
                return 200, {'default_branch': 'master'}
            if url.endswith('/git/trees/master?recursive=1'):
                return 200, {'sha': 'sha1', 'truncated': False,
                             'tree': [{'type': 'blob', 'path': 'a/main.py'}]}
            raise AssertionError(f'unexpected url {url}')

        original = ec._http_json
        ec._http_json = fake_http
        try:
            fetched = ec.fetch_tree('org/demo-examples', 'main', None,
                                    prefer_release=False)
        finally:
            ec._http_json = original
        self.assertEqual(fetched['ref'], 'master')
        self.assertEqual(fetched['ref_source'], 'default-branch')
        self.assertEqual(fetched['paths'], ['a/main.py'])
        self.assertEqual(calls, [
            'https://api.github.com/repos/org/demo-examples',
            'https://api.github.com/repos/org/demo-examples/git/trees/master'
            '?recursive=1'])

    def test_release_mode_uses_tag(self) -> None:
        def fake_http(url: str, token=None):
            if url.endswith('/releases/latest'):
                return 200, {'tag_name': 'v9.9.9'}
            if url.endswith('/git/trees/v9.9.9?recursive=1'):
                return 200, {'sha': 'sha1', 'truncated': False,
                             'tree': [{'type': 'blob', 'path': 'x.py'}]}
            raise AssertionError(f'unexpected url {url}')

        original = ec._http_json
        ec._http_json = fake_http
        try:
            fetched = ec.fetch_tree('org/demo', 'main', None)
        finally:
            ec._http_json = original
        self.assertEqual(fetched['ref'], 'v9.9.9')
        self.assertEqual(fetched['ref_source'], 'release')

    def test_no_release_falls_back_to_workflow_branch(self) -> None:
        def fake_http(url: str, token=None):
            if url.endswith('/releases/latest'):
                return 404, {'message': 'Not Found'}
            if url.endswith('/git/trees/master?recursive=1'):
                return 200, {'sha': 'sha1', 'truncated': False, 'tree': []}
            raise AssertionError(f'unexpected url {url}')

        original = ec._http_json
        ec._http_json = fake_http
        try:
            fetched = ec.fetch_tree('org/demo', 'master', None)
        finally:
            ec._http_json = original
        self.assertEqual(fetched['ref'], 'master')
        self.assertEqual(fetched['ref_source'], 'default-branch')


class CoveredRuleTests(unittest.TestCase):
    """§5.2 用例 2：覆盖规则（字面/目录前缀/边界）。"""

    def test_literal_and_directory_prefix(self) -> None:
        covering = ['examples/a.py', 'examples/config', 'dir/']
        self.assertTrue(ec.covered('examples/a.py', covering))
        self.assertTrue(ec.covered('examples/config/x.yaml', covering))
        self.assertTrue(ec.covered('examples/config/a/b/c.yaml', covering))
        self.assertTrue(ec.covered('dir/f.py', covering))

    def test_prefix_boundary_is_slash(self) -> None:
        self.assertFalse(ec.covered('foo.py', ['foo']))
        self.assertFalse(ec.covered('examples/config/x.py', ['examples/conf']))

    def test_wildcard_entries_are_literal_no_match(self) -> None:
        self.assertFalse(ec.covered('examples/test_a.py', ['**/test_*']))
        self.assertFalse(ec.covered('abs/a.py', ['/abs/a.py']))


class ReconcileTests(unittest.TestCase):
    """§5.2 用例 3：差集方向与 stale 对称性。"""

    def test_missing_and_stale_directions(self) -> None:
        units = ['examples/a.py', 'examples/new.py']
        recorded = ['examples/a.py', 'examples/gone.py']
        exclude = ['examples/old-helper.py']
        outcome = ec.reconcile(units, recorded, exclude)
        self.assertEqual(outcome['missing'], ['examples/new.py'])
        self.assertEqual(outcome['stale'],
                         ['examples/gone.py', 'examples/old-helper.py'])

    def test_exclude_only_coverage_is_not_missing(self) -> None:
        units = ['examples/helper.py', 'examples/a.py']
        outcome = ec.reconcile(units, ['examples/a.py'], ['examples/helper.py'])
        self.assertEqual(outcome['missing'], [])
        self.assertEqual(outcome['stale'], [])


class ScanReplayTests(unittest.TestCase):
    """§5.2 用例 5/6：exclude 同级掩盖 + root 语义 + 单位重放。"""

    def test_files_unit_is_recursive_and_ignores_max_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_tree(root, ['examples/a.py', 'examples/x/nested/b.py'])
            cfg = ec.build_scan_config({
                'root': 'examples', 'unit': 'files',
                'include_extensions': ['.py'], 'max_depth': 1})
            self.assertEqual(ec.scan_units(root, cfg),
                             ['examples/a.py', 'examples/x/nested/b.py'])

    def test_directories_unit_requires_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_tree(root, [
                'examples/cli/CMakeLists.txt', 'examples/nope/x.py'])
            cfg = ec.build_scan_config({
                'root': 'examples', 'unit': 'directories'})
            self.assertEqual(ec.scan_units(root, cfg), ['examples/cli'])

    def test_mixed_unit_unions_dirs_and_depth_limited_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_tree(root, [
                'examples/cli/keep.txt', 'examples/python/deep.py',
                'examples/server.py', 'examples/run.sh', 'examples/skip.txt'])
            cfg = ec.build_scan_config({
                'root': 'examples', 'unit': 'mixed', 'max_depth': 1,
                'include_extensions': ['.sh', '.py']})
            self.assertEqual(ec.scan_units(root, cfg), [
                'examples/cli', 'examples/python',
                'examples/run.sh', 'examples/server.py'])

    def test_explicit_empty_root_means_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_tree(root, ['setup.py', 'training/hello/run.py'])
            cfg = ec.build_scan_config({
                'root': '', 'include_extensions': ['.py']})
            self.assertEqual(ec.scan_units(root, cfg),
                             ['setup.py', 'training/hello/run.py'])

    def test_absent_root_is_skipped_not_defaulted(self) -> None:
        self.assertIsNone(ec.build_scan_config({}))
        self.assertIsNone(ec.build_scan_config({'unit': 'files'}))

    def test_missing_include_extensions_is_config_error(self) -> None:
        with self.assertRaises(ec.ConfigError):
            ec.build_scan_config({'root': 'examples', 'unit': 'files'})

    def test_exclude_non_string_is_config_error(self) -> None:
        with self.assertRaises(ec.ConfigError):
            ec.build_scan_config({
                'root': 'examples', 'include_extensions': ['.py'],
                'exclude': ['ok.py', 5]})

    def test_exclude_wildcard_content_is_not_validated(self) -> None:
        cfg = ec.build_scan_config({
            'root': 'examples', 'include_extensions': ['.py'],
            'exclude': ['**/test_*', '/abs', '..']})
        self.assertEqual(cfg['exclude'], ['**/test_*', '/abs', '..'])

    def test_scan_root_missing_upstream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ec.build_scan_config({
                'root': 'examples', 'include_extensions': ['.py']})
            with self.assertRaises(ec.ScanRootMissing):
                ec.scan_units(Path(tmp), cfg)


class ThinWorkflowParsingTests(unittest.TestCase):
    """§5.2 用例 8：thin workflow 读模型。"""

    def _write_workflow(self, root: Path, name: str, body: str) -> None:
        path = root / name
        path.write_text(body, encoding='utf-8')

    def test_extracts_with_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_workflow(root, 'demo-examples.yml', """\
name: demo-examples
jobs:
  demo-examples:
    uses: ./.github/workflows/examples-template.yml
    with:
      project: demo
      upstream_repo: org/demo
      examples_repo: org/demo-examples
      default_branch: master
""")
            thin = ec.load_thin_workflows(root)
            self.assertEqual(thin['demo']['upstream_repo'], 'org/demo')
            self.assertEqual(thin['demo']['examples_repo'], 'org/demo-examples')
            self.assertEqual(thin['demo']['default_branch'], 'master')

    def test_defaults_and_ignores_non_callers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_workflow(root, 'a-examples.yml', """\
jobs:
  a:
    uses: ./.github/workflows/examples-template.yml
    with:
      project: a
      upstream_repo: org/a
""")
            self._write_workflow(root, 'unrelated.yml', """\
jobs:
  b:
    uses: ./.github/workflows/quick-start-template.yml
    with:
      project: b
      upstream_repo: org/b
""")
            self._write_workflow(root, 'examples-template.yml', 'name: engine\n')
            thin = ec.load_thin_workflows(root)
            self.assertEqual(list(thin), ['a'])
            self.assertEqual(thin['a']['examples_repo'], '')
            self.assertEqual(thin['a']['default_branch'], 'main')

    def test_missing_with_project_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_workflow(root, 'bad-examples.yml', """\
jobs:
  bad:
    uses: ./.github/workflows/examples-template.yml
    with:
      upstream_repo: org/bad
""")
            with self.assertRaises(ec.FatalError):
                ec.load_thin_workflows(root)


class StatusMappingTests(unittest.TestCase):
    """§5.2 用例 4：状态映射（fetch 替身 / skipped / scan-root / failed）。"""

    def _info(self, **overrides) -> dict:
        info = {'workflow': 'demo-examples.yml', 'upstream_repo': 'org/demo',
                'examples_repo': '', 'default_branch': 'main'}
        info.update(overrides)
        return info

    def _manifest(self, tmp: Path) -> Path:
        return write_manifest(Path(tmp), """\
version: 1
scan:
  root: examples
  include_extensions: ['.py']
supported:
  - path: examples/a.py
unsupported: []
""")

    def test_success_when_fully_covered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = ec.check_project(
                'demo', self._info(), self._manifest(Path(tmp)),
                fake_fetcher(['examples/a.py']))
            self.assertEqual(result['status'], 'success')
            self.assertEqual(result['target_ref'], 'v1.0.0')
            self.assertEqual(result['ref_source'], 'release')
            self.assertEqual(result['scanned_count'], 1)

    def test_failed_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = ec.check_project(
                'demo', self._info(), self._manifest(Path(tmp)),
                fake_fetcher(['examples/a.py', 'examples/new.py']))
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['missing_paths'], ['examples/new.py'])

    def test_fetch_failure_is_error(self) -> None:
        def flaky(repo: str, branch: str) -> dict:
            raise ec.FetchError('HTTP 503')

        with tempfile.TemporaryDirectory() as tmp:
            result = ec.check_project(
                'demo', self._info(), self._manifest(Path(tmp)), flaky)
            self.assertEqual(result['status'], 'error')
            self.assertIn('fetch-failed', result['detail'])

    def test_retry_once_then_succeeds(self) -> None:
        calls = {'n': 0}
        original_wait = ec.FETCH_RETRY_WAIT_SECONDS
        ec.FETCH_RETRY_WAIT_SECONDS = 0
        try:
            def flaky(repo: str, default_branch: str, token=None,
                      prefer_release=True) -> dict:
                calls['n'] += 1
                if calls['n'] == 1:
                    raise ec.FetchError('HTTP 503')
                return {'paths': [], 'ref': 'v1.0.0',
                        'ref_source': 'release', 'tree_sha': 'abc'}
            original = ec.fetch_tree
            ec.fetch_tree = flaky
            try:
                fetched = ec.fetch_tree_with_retry('org/demo', 'main', None)
            finally:
                ec.fetch_tree = original
            self.assertEqual(calls['n'], 2)
            self.assertEqual(fetched['ref'], 'v1.0.0')
        finally:
            ec.FETCH_RETRY_WAIT_SECONDS = original_wait

    def test_retry_exhausted_raises(self) -> None:
        calls = {'n': 0}
        original_wait = ec.FETCH_RETRY_WAIT_SECONDS
        ec.FETCH_RETRY_WAIT_SECONDS = 0
        try:
            def always_down(repo: str, default_branch: str, token=None,
                            prefer_release=True) -> dict:
                calls['n'] += 1
                raise ec.FetchError('timeout')
            original = ec.fetch_tree
            ec.fetch_tree = always_down
            try:
                with self.assertRaises(ec.FetchError):
                    ec.fetch_tree_with_retry('org/demo', 'main', None)
            finally:
                ec.fetch_tree = original
            self.assertEqual(calls['n'], 2)
        finally:
            ec.FETCH_RETRY_WAIT_SECONDS = original_wait

    def test_no_root_is_skipped_without_fetch(self) -> None:
        def boom(repo: str, branch: str) -> dict:
            raise AssertionError('skipped projects must not fetch')

        with tempfile.TemporaryDirectory() as tmp:
            path = write_manifest(Path(tmp), """\
version: 1
scan:
  paths:
    - python/tests/test_npu.py
supported: []
unsupported: []
""")
            result = ec.check_project('demo', self._info(), path, boom)
            self.assertEqual(result['status'], 'skipped')
            self.assertIn('allowlist', result['detail'])

    def test_scan_root_missing_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = ec.check_project(
                'demo', self._info(), self._manifest(Path(tmp)),
                fake_fetcher(['other/tree.py']))
            self.assertEqual(result['status'], 'error')
            self.assertIn('scan-root-missing', result['detail'])

    def test_missing_upstream_repo_is_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = ec.check_project(
                'demo', self._info(upstream_repo=None),
                self._manifest(Path(tmp)), fake_fetcher(['examples/a.py']))
            self.assertEqual(result['status'], 'error')
            self.assertIn('config', result['detail'])


class EndToEndTests(unittest.TestCase):
    """§5.2 用例 7：main() 端到端（临时 workflows / projects + fetch 替身）。"""

    def _setup(self, tmp: Path) -> None:
        workflows = tmp / '.github' / 'workflows'
        workflows.mkdir(parents=True)
        (workflows / 'examples-template.yml').write_text('name: engine\n')
        (workflows / 'demo-examples.yml').write_text("""\
name: demo-examples
jobs:
  demo:
    uses: ./.github/workflows/examples-template.yml
    with:
      project: demo
      upstream_repo: org/demo
""")
        (workflows / 'raylike-examples.yml').write_text("""\
name: raylike-examples
jobs:
  raylike:
    uses: ./.github/workflows/examples-template.yml
    with:
      project: raylike
      upstream_repo: org/raylike
""")
        projects = tmp / 'projects'
        (projects / 'demo').mkdir(parents=True)
        write_manifest(projects / 'demo', """\
version: 1
scan:
  root: examples
  include_extensions: ['.py']
supported:
  - path: examples/a.py
unsupported:
  - examples/b.py
""")
        (projects / 'raylike').mkdir(parents=True)
        write_manifest(projects / 'raylike', """\
version: 1
scan:
  paths:
    - python/x.py
supported: []
unsupported: []
""")
        (projects / 'dormant').mkdir(parents=True)
        write_manifest(projects / 'dormant', """\
version: 1
scan:
  root: examples
  include_extensions: ['.py']
supported: []
unsupported: []
""")

    def _run_main(self, tmp: Path) -> int:
        paths_by_repo = {
            'org/demo': ['examples/a.py', 'examples/b.py', 'examples/new.py'],
            'org/raylike': ['python/x.py'],
        }

        def fake(repo: str, branch: str, token=None, prefer_release=True):
            return {'paths': paths_by_repo[repo], 'ref': 'v1.0.0',
                    'ref_source': 'release', 'tree_sha': 'abc'}

        original = ec.fetch_tree_with_retry
        original_summary = os.environ.pop('GITHUB_STEP_SUMMARY', None)
        ec.fetch_tree_with_retry = fake
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ec.main([
                    '--workflows-dir', str(tmp / '.github' / 'workflows'),
                    '--projects-root', str(tmp / 'projects'),
                    '--output', str(tmp / 'result.json')])
            return 0
        except SystemExit as exc:
            return int(exc.code or 0)
        finally:
            ec.fetch_tree_with_retry = original
            if original_summary is not None:
                os.environ['GITHUB_STEP_SUMMARY'] = original_summary

    def test_result_and_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._setup(Path(tmp))
            code = self._run_main(Path(tmp))
            self.assertEqual(code, 1)
            data = json.loads((Path(tmp) / 'result.json').read_text())
            self.assertEqual(data['summary'],
                             {'success': 0, 'failed': 1, 'error': 0,
                              'skipped': 1})
            demo = next(e for e in data['projects'] if e['project'] == 'demo')
            self.assertEqual(demo['status'], 'failed')
            self.assertEqual(demo['missing_paths'], ['examples/new.py'])
            self.assertEqual(demo['stale_paths'], [])
            raylike = next(e for e in data['projects']
                           if e['project'] == 'raylike')
            self.assertEqual(raylike['status'], 'skipped')
            self.assertEqual(data['not_in_scope'], ['dormant'])



if __name__ == '__main__':
    unittest.main()

"""Daily audit of examples manifests against upstream repos.

For every project that has both projects/<name>/examples_manifest.yaml
and an upstream_repo entry in projects.yaml, clone the upstream repo
at HEAD (depth=1, blob:none filter, sparse-checkout on the manifest's
scan root) and replay the manifest's scan rules against the on-disk
examples tree. The discovered set is compared against the union of
supported[*].path and unsupported[*].

Per project outcome:
  - success : scanned ∪ listed == scanned ∪ listed (no diff)
  - error   : there are new_paths (discovered not listed) or
              stale_paths (listed not on disk under supported ∪ unsupported),
              OR the manifest's scan_root is missing on disk
  - miss    : upstream repo clone failed after one retry

Exits 0 only if every project is success; non-zero otherwise. Designed
for daily cron use; clones are cached under --cache and removed after
audit, and clones run in parallel via a process pool (--jobs).

Reuses examples_manifest_scan.load_scan so the engine and this audit
agree on include_extensions / scan root conventions. All non-supported
files (training examples that fail on NPU, helper modules, configs,
upstream self-tests, non-example utility scripts) live in the manifest's
unsupported: list — there is no separate exclude mechanism.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
from examples_manifest_scan import load_scan

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECTS_YAML = REPO_ROOT / 'projects.yaml'
GITHUB_BASE = 'https://github.com'
DEFAULT_CACHE = '/tmp/audit_examples_manifest'
CLONE_TIMEOUT = 900


def load_projects() -> list[dict]:
    data = yaml.safe_load(PROJECTS_YAML.read_text(encoding='utf-8')) or {}
    return list(data.get('projects') or [])


def projects_with_manifest() -> list[dict]:
    out: list[dict] = []
    for p in load_projects():
        if not p.get('upstream_repo'):
            continue
        manifest = REPO_ROOT / p['dir'] / 'examples_manifest.yaml'
        if manifest.is_file():
            out.append(p)
    return out


def load_manifest(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    scan = load_scan(data.get('scan') or {})
    return {**data, 'scan': scan}


def discover(target_root: Path, scan: dict) -> set[str]:
    examples_root = target_root / scan['scan_root']
    if not examples_root.is_dir():
        return set()
    found: set[str] = set()
    for path in sorted(examples_root.rglob('*')):
        if not path.is_file() or path.suffix not in scan['include_extensions']:
            continue
        rel = path.relative_to(target_root).as_posix()
        found.add(rel)
    return found


def listed_paths(manifest: dict) -> set[str]:
    listed: set[str] = set()
    for entry in manifest.get('supported') or []:
        if 'path' in entry:
            listed.add(entry['path'])
    listed.update(manifest.get('unsupported') or [])
    return listed


def try_clone(repo: str, scan_root: str, cache: Path, project_name: str) -> dict:
    """Clone with sparse-checkout on scan_root, one automatic retry.

    Uses `--depth=1 --filter=blob:none --sparse` so only the scan_root
    tree is fetched; for repos whose scan_root is the whole repo (".")
    this degrades to a normal blob:none clone (still depth=1).

    Success = git exits 0 and the scan_root directory exists on disk.
    Failure modes (after one retry):
      - non-zero git exit → last stderr line
      - TimeoutExpired → "timeout after Ns"
      - missing scan_root → "scan_root missing after clone" (rare;
        usually means the upstream branch no longer has that tree)
    """
    target = cache / project_name
    url = f'{GITHUB_BASE}/{repo}.git'
    last_err = 'unknown error'
    for attempt in range(2):
        if target.exists():
            shutil.rmtree(target)
        try:
            cmd = ['git', 'clone', '--depth=1', '--filter=blob:none', '--sparse', url, str(target)]
            proc = subprocess.run(cmd, capture_output=True, timeout=CLONE_TIMEOUT)
        except subprocess.TimeoutExpired:
            last_err = f'timeout after {CLONE_TIMEOUT}s'
            continue
        except Exception as exc:
            last_err = str(exc)
            continue
        if proc.returncode != 0:
            err = proc.stderr.decode('utf-8', errors='replace').strip()
            last_err = err.splitlines()[-1] if err else 'git clone failed'
            continue
        # Sparse-checkout the scan_root. "." means "the whole repo" so
        # cone-mode sparse-checkout is meaningless — skip the set call.
        if scan_root != '.':
            try:
                set_cmd = ['git', '-C', str(target), 'sparse-checkout', 'init', '--cone']
                subprocess.run(set_cmd, capture_output=True, timeout=CLONE_TIMEOUT, check=True)
                set_root = ['git', '-C', str(target), 'sparse-checkout', 'set', scan_root]
                subprocess.run(set_root, capture_output=True, timeout=CLONE_TIMEOUT, check=True)
            except subprocess.TimeoutExpired:
                last_err = 'sparse-checkout timeout'
                continue
            except subprocess.CalledProcessError as exc:
                err = (exc.stderr or b'').decode('utf-8', errors='replace').strip()
                last_err = f'sparse-checkout rc={exc.returncode}: {err}'
                continue
        # Verify scan_root landed on disk.
        if not (target / scan_root).is_dir():
            # For '.' the target itself counts as the scan_root.
            if scan_root == '.' and target.is_dir():
                return {'ok': True}
            last_err = f'scan_root missing: {scan_root}'
            continue
        return {'ok': True}
    return {'ok': False, 'reason': last_err}


_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def audit_project(project: dict, cache: Path) -> dict:
    name = project['name']
    repo = project['upstream_repo']
    manifest_path = REPO_ROOT / project['dir'] / 'examples_manifest.yaml'
    manifest = load_manifest(manifest_path)
    scan_root = manifest['scan']['scan_root']
    log(f'[{name}] cloning {repo} (scan_root={scan_root}) ...')
    clone = try_clone(repo, scan_root, cache, name)
    if not clone['ok']:
        log(f'[{name}] miss: {clone["reason"]}')
        return {'name': name, 'status': 'miss', 'detail': clone['reason']}
    target = cache / name
    # scan_root existence on disk is already verified inside try_clone;
    # we still need to call discover against target_root.
    if not (target / scan_root).is_dir():
        log(f'[{name}] error: scan_root missing: {scan_root}')
        return {'name': name, 'status': 'error', 'detail': f'scan_root missing: {scan_root}'}
    scanned = discover(target, manifest['scan'])
    listed = listed_paths(manifest)
    new_paths = sorted(scanned - listed)
    stale_paths = sorted(listed - scanned)
    if new_paths or stale_paths:
        bits = []
        if new_paths:
            bits.append(f'+{len(new_paths)} new')
        if stale_paths:
            bits.append(f'-{len(stale_paths)} stale')
        log(f'[{name}] error: {", ".join(bits)}')
        for p in new_paths[:5]:
            log(f'      + {p}')
        if len(new_paths) > 5:
            log(f'      + ... and {len(new_paths) - 5} more')
        for p in stale_paths[:5]:
            log(f'      - {p}')
        if len(stale_paths) > 5:
            log(f'      - ... and {len(stale_paths) - 5} more')
        return {
            'name': name,
            'status': 'error',
            'detail': {'new_paths': new_paths, 'stale_paths': stale_paths},
        }
    log(
        f'[{name}] success ({len(scanned)} scanned, '
        f'{len([e for e in (manifest.get("supported") or []) if "path" in e])} supported, '
        f'{len(manifest.get("unsupported") or [])} unsupported)'
    )
    return {'name': name, 'status': 'success', 'detail': {}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=str, default=DEFAULT_CACHE,
                        help=f'clone cache root (default: {DEFAULT_CACHE})')
    parser.add_argument('--project', action='append', default=[],
                        help='limit audit to one or more project names')
    parser.add_argument('--jobs', '-j', type=int, default=4,
                        help='parallel clone workers (default 4)')
    parser.add_argument('--keep-clones', action='store_true',
                        help='do not delete clones after audit (debug)')
    args = parser.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    projects = projects_with_manifest()
    if args.project:
        wanted = set(args.project)
        projects = [p for p in projects if p['name'] in wanted]
    if not projects:
        print('no projects to audit', file=sys.stderr)
        return 0

    print(f'Auditing {len(projects)} projects with {args.jobs} parallel workers ...')

    results: list[dict] = []
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(audit_project, p, cache): p for p in projects}
            for fut in concurrent.futures.as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    p = futs[fut]
                    log(f'[{p["name"]}] audit raised: {exc}')
                    results.append({'name': p['name'], 'status': 'miss', 'detail': str(exc)})
    finally:
        if not args.keep_clones:
            for p in projects:
                target = cache / p['name']
                if target.exists():
                    shutil.rmtree(target)

    success = sum(1 for r in results if r['status'] == 'success')
    error = sum(1 for r in results if r['status'] == 'error')
    miss = sum(1 for r in results if r['status'] == 'miss')
    print(f'AUDIT: {success} success, {error} error, {miss} miss (of {len(results)} projects)')

    # Exit non-zero if any project is not success.
    sys.exit(0 if error == 0 and miss == 0 else 1)


if __name__ == '__main__':
    main()
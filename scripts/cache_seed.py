"""Generic runner-cache seeder (cache-seed.yml workflow entry).

从 cache-seed/<project>/ 把 raw 文件投递到 runner 的共享缓存目录
（默认 ~/.cache/huggingface；可用 SHARED_CACHE_ROOT 覆盖），投递后
按 manifest.yaml 逐文件 sha256 校验（投递前的 sha256 在 staging 阶段
bundle_cache.py 已算好写入 manifest）：

  cache-seed/<project>/<prefix>/<file>          → <root>/<prefix>/<file>   (cp)
  cache-seed/<project>/<prefix>/<file>.part-*   → <root>/<prefix>/<file>   (cat parts)

幂等：target 已存在且 sha256 匹配 → 跳过；不匹配则删掉重试。

用法:
  python scripts/cache_seed.py [--projects peft,accelerate]
                               [--root /shared/cache/huggingface]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

SEED_ROOT = Path(__file__).resolve().parent.parent / "cache-seed"
# 共享缓存根：env 优先，未设置则走 ~/.cache/huggingface
DEFAULT_ROOT = Path(
    os.environ.get("SHARED_CACHE_ROOT") or os.path.expanduser("~/.cache/huggingface")
)
PART_SUFFIX = ".part-"

failures: list[str] = []


def note_fail(label: str, exc: Exception | None = None) -> None:
    failures.append(label)
    print(f"FAIL {label}" + (f": {exc}" if exc else ""), flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    import yaml  # noqa: E402
    data = yaml.safe_load(path.read_text()) or {}
    return data.get("files", [])


def stage_entry(project_dir: Path, root: Path, entry: dict) -> None:
    rel = entry["path"]
    expected_sha = entry["sha256"]
    target = root / rel
    src = project_dir / rel

    # 幂等：target 已存在且 sha 匹配 → 跳过
    if target.exists() and not target.is_dir():
        if sha256_file(target) == expected_sha:
            print(f"skip {rel} (target matches)", flush=True)
            return
        print(f"  target sha mismatch; re-staging {rel}", flush=True)
        target.unlink()

    target.parent.mkdir(parents=True, exist_ok=True)

    parts = sorted(src.parent.glob(f"{src.name}{PART_SUFFIX}*"))
    try:
        if parts:
            with open(target, "wb") as out:
                for p in parts:
                    shutil.copyfileobj(open(p, "rb"), out)
        elif src.is_file():
            shutil.copy2(src, target)
        else:
            note_fail(f"{rel}: no source file or parts at {src}")
            return
    except Exception as exc:
        note_fail(f"{rel}: copy/cat failed", exc)
        return

    got = sha256_file(target)
    if got != expected_sha:
        note_fail(f"{rel}: sha256 mismatch after copy")
        print(f"  expected {expected_sha}\n  got      {got}", flush=True)
        target.unlink(missing_ok=True)
        return
    print(f"copied {rel}", flush=True)


def install_bundles(project_dir: Path, root: Path) -> None:
    entries = load_manifest(project_dir / "manifest.yaml")
    if not entries:
        print(f"  no entries in manifest.yaml (skipping)", flush=True)
        return
    for entry in entries:
        try:
            stage_entry(project_dir, root, entry)
        except Exception as exc:  # noqa: BLE001
            note_fail(f"{entry.get('path', '?')}: unexpected", exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--projects", default="",
        help="comma-separated project filter under cache-seed/ (default: all)",
    )
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT,
        help=f"shared cache root (default: {DEFAULT_ROOT})",
    )
    args = parser.parse_args()
    wanted = {p.strip() for p in args.projects.split(",") if p.strip()}
    root = args.root

    print(f"seed source: {SEED_ROOT}", flush=True)
    print(f"target root: {root}", flush=True)

    projects = sorted(
        p for p in SEED_ROOT.iterdir()
        if p.is_dir() and (not wanted or p.name in wanted)
    )
    if not projects:
        print(
            f"no project dirs under {SEED_ROOT}"
            + (f" matching {sorted(wanted)}" if wanted else ""),
            flush=True,
        )
        return 0

    for project_dir in projects:
        print(f"\n== cache-seed/{project_dir.name}", flush=True)
        install_bundles(project_dir, root)

    if failures:
        print(f"\nseed incomplete ({len(failures)}): {failures}",
              file=sys.stderr, flush=True)
        return 1
    print("\ncache seed complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
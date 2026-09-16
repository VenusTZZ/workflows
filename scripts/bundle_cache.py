"""Stage raw files for CI cache-seed delivery (no tar).

遍历 --src 下的所有文件（symlink 自动跟随，blobs/ 默认跳过 —— 那是 HF
hub cache 的内部 dedup 目录，snapshots/ 下的 symlink 引用它们，我们只
stage 解析后的内容就够了），逐文件：

  - 算 sha256（流式，内存峰值 ≈ 95MB）
  - > 95MB：split 成 .part-aa/ab/...，不写主文件
  - ≤ 95MB：原样写主文件

manifest.yaml 的 path 字段是 <prefix>/<rel_path_from_src>，落到
SHARED_CACHE_ROOT 下。

用法:
  python scripts/bundle_cache.py --project peft \\
      --src ~/.cache/huggingface/hub/models--roberta-base \\
      --prefix hub/models--roberta-base

可重复 --src/--prefix 对，每次 upsert 到 manifest.yaml（partial failure
不保存，保留已有 entries）。
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PART_SIZE = 95 * 1024 * 1024  # 95MB; GitHub 单文件上限 100MB
CHUNK_SIZE = 1 << 20  # 1MB stream chunk
SKIP_DIRS = {"blobs"}  # HF hub cache internal dedup dir


def _part_suffix(i: int) -> str:
    """aa, ab, ..., az, ba, bb, ... 闭区间无限。"""
    chars = "abcdefghijklmnopqrstuvwxyz"
    n = len(chars)
    s = ""
    while True:
        s = chars[i % n] + s
        i //= n
        if i == 0:
            return s
        i -= 1


def stage_file(src: Path, target: Path) -> tuple[str, int]:
    """Stream src → target. Returns (sha256, size).

    文件 > 95MB 时按 .part-aa/ab/... 切分；≤ 95MB 直接写到 target。
    不读全文件到内存（峰值 ≈ PART_SIZE）。
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    total = 0
    buf = bytearray()
    part_idx = 0

    with open(src, "rb") as fh:
        while True:
            chunk = fh.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
            total += len(chunk)
            buf.extend(chunk)
            if len(buf) >= PART_SIZE:
                part_path = target.parent / f"{target.name}.part-{_part_suffix(part_idx)}"
                part_path.write_bytes(bytes(buf))
                buf = bytearray()
                part_idx += 1
        if buf:
            if part_idx == 0:
                target.write_bytes(bytes(buf))
            else:
                part_path = target.parent / f"{target.name}.part-{_part_suffix(part_idx)}"
                part_path.write_bytes(bytes(buf))

    # 写了 part-* 但主文件残留（如前次 split 失败留下的）→ 清掉
    if part_idx > 0 and target.exists():
        target.unlink()
    return h.hexdigest(), total


def load_existing_manifest(project_dir: Path) -> dict[str, dict]:
    path = project_dir / "manifest.yaml"
    if not path.is_file():
        return {}
    import yaml
    data = yaml.safe_load(path.read_text()) or {}
    return {e["path"]: e for e in data.get("files", [])}


def save_manifest(project_dir: Path, entries: dict[str, dict]) -> None:
    import yaml
    path = project_dir / "manifest.yaml"
    path.write_text(
        yaml.safe_dump({"files": list(entries.values())}, sort_keys=False, allow_unicode=True)
    )


def stage_one(project_dir: Path, src: Path, prefix: str) -> dict[str, dict] | None:
    """Walk src, stage each file under project_dir/<prefix>/<rel>.

    Returns {rel_path: entry} on success, None on error.
    """
    src = src.resolve()
    if not src.is_dir():
        print(f"FAIL {prefix}: source not a directory: {src}", flush=True)
        return None

    out: dict[str, dict[str, object]] = {}
    n_files = 0
    n_bytes = 0
    for root, dirs, files in os.walk(src, followlinks=False):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            full = Path(root) / f
            rel = str(full.relative_to(src))
            target_rel = f"{prefix}/{rel}"
            target = project_dir / target_rel
            try:
                sha, size = stage_file(full, target)
            except OSError as exc:
                print(f"FAIL {prefix}: cannot read {full}: {exc}", flush=True)
                return None
            entry: dict[str, object] = {"path": target_rel, "sha256": sha}
            if size >= PART_SIZE:
                entry["size"] = size  # human inspection only
            out[target_rel] = entry
            n_files += 1
            n_bytes += size
    print(
        f"  staged {n_files} files ({n_bytes // (1024 * 1024)} MB) under {prefix}/",
        flush=True,
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage raw files into cache-seed/<project>/ with sha256 manifest.",
    )
    parser.add_argument("--project", required=True, help="cache-seed/<project>/ 子目录名")
    parser.add_argument("--src", action="append", required=True, help="源目录（可重复）")
    parser.add_argument("--prefix", action="append", required=True,
                        help="目标子目录前缀（与 --src 一一对应；落在 SHARED_CACHE_ROOT 下）")
    args = parser.parse_args()

    if len(args.src) != len(args.prefix):
        parser.error("--src and --prefix counts must match")

    project_dir = REPO_ROOT / "cache-seed" / args.project
    project_dir.mkdir(parents=True, exist_ok=True)
    existing = load_existing_manifest(project_dir)

    failed: list[str] = []
    for src, prefix in zip(args.src, args.prefix):
        new_entries = stage_one(project_dir, Path(src), prefix)
        if new_entries is None:
            failed.append(prefix)
            continue
        existing.update(new_entries)  # upsert

    if failed:
        print(
            f"\nFAILED: {failed}; manifest.yaml NOT updated "
            f"(已有 {len(existing)} 个 entries 保留原样)",
            flush=True,
        )
        return 1

    save_manifest(project_dir, existing)
    print(f"\nmanifest.yaml: {len(existing)} file(s)", flush=True)
    print(f"bundle complete -> {project_dir.relative_to(REPO_ROOT)}/", flush=True)
    print(f"next: git add {project_dir.relative_to(REPO_ROOT)}/ && git commit",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
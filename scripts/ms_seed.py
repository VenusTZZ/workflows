"""Plant ModelScope assets into the shared HF cache root (seed workflow).

cache-seed workflow 的 ModelScope 半边（repo bundle 投递见 cache_seed.py）。
读 cache-seed/<project>/ms_seeds.yaml，对每条资产：

    modelscope.snapshot_download(ms_id, allow_patterns=...)
        → cp 到 <root>/hub/<models|datasets>--<hf_id>/snapshots/<sha>/
        → refs/main 写真实 HF commit sha（经 HF_ENDPOINT API 查询，
          xet-free，元数据小文件）

效果：例程里 from_pretrained(<hf_id>) / load_dataset(<hf_id>) 解析
revision 时命中本地缓存，不再走网络权重下载（绕开 hf-mirror 对 Xet
仓库 302 cas-bridge 的故障路径）。

幂等：目标已存在的文件直接跳过（热机器秒级完成）。symlink 型残留
（早期 plant 版本留下）会被替换为真实文件，避免 modelscope 缓存被
回收后断链。

注意：MS 的 dataset 下载把 allow_patterns 当"子树根"处理
（"mrpc/*" → root /mrpc），README 等仓库根文件不会拿到 —— 与 peft /
accelerate 已验证的用法一致，README 由 load_dataset 解析时从镜像
API 拉取（小文件，可靠）。

用法:
  python scripts/ms_seed.py [--projects peft,accelerate] [--root PATH]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_ROOT = REPO_ROOT / "cache-seed"
DEFAULT_ROOT = Path(os.path.expanduser("~/.cache/huggingface"))
# Runners live in mainland China: default to the mirror (the cache-seed
# workflow does not set HF_ENDPOINT). Only the JSON API is hit — xet-free.
HF_API_BASE = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/") + "/api"

MODELSCOPE_PIN = "1.37.0"  # see setup_example.sh history: 1.40.1 + hub 0.4.2 breaks


def load_spec(project_dir: Path) -> list[dict]:
    import yaml
    path = project_dir / "ms_seeds.yaml"
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    seeds = data.get("seeds") or []
    if not isinstance(seeds, list):
        raise SystemExit(f"{path}: 'seeds' must be a list")
    return seeds


def fetch_sha(hf_id: str, kind: str) -> str:
    import requests
    url = f"{HF_API_BASE}/{kind}s/{hf_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json().get("sha") or r.json().get("oid")


def plant(ms_id: str, hf_id: str, kind: str, root: Path,
          allow_patterns: list[str] | None) -> None:
    from modelscope import snapshot_download

    model_cache = Path(os.environ.get(
        "MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope")))
    src = Path(snapshot_download(
        ms_id, cache_dir=str(model_cache), repo_type=kind,
        allow_patterns=allow_patterns,
    ))
    sha = fetch_sha(hf_id, kind)
    repo_kind = "models" if kind == "model" else "datasets"
    repo_dir = root / "hub" / f"{repo_kind}--{hf_id.replace('/', '--')}"
    snap_dir = repo_dir / "snapshots" / sha
    snap_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs").mkdir(exist_ok=True)
    # no trailing newline — hub compares this string to the snapshot
    # folder name without stripping
    (repo_dir / "refs" / "main").write_text(sha)
    n_bytes = 0
    for item in src.rglob("*"):
        if not item.is_file():
            continue
        dest = snap_dir / item.relative_to(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink():
            # replace a legacy symlink-based plant: links into the
            # modelscope cache break when that cache is recycled
            dest.unlink()
        elif dest.exists():
            continue  # warm cache: keep the already-planted copy
        shutil.copy2(item, dest)
        n_bytes += item.stat().st_size
    print(f"planted {hf_id}@{sha[:8]} ({n_bytes // (1024 * 1024)} MB new)",
          flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plant ModelScope assets into the shared HF cache root.",
    )
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

    os.environ.setdefault("TQDM_MININTERVAL", "15")  # non-TTY CI logs
    try:
        from modelscope import snapshot_download  # noqa: F401
    except ImportError:
        print(f"modelscope is required (pip install modelscope=={MODELSCOPE_PIN})",
              file=sys.stderr, flush=True)
        return 2

    projects = sorted(
        p for p in SEED_ROOT.iterdir()
        if p.is_dir() and (not wanted or p.name in wanted)
        and (p / "ms_seeds.yaml").is_file()
    )
    if not projects:
        print(f"no ms_seeds.yaml under {SEED_ROOT}"
              + (f" matching {sorted(wanted)}" if wanted else ""), flush=True)
        return 0

    failures: list[str] = []
    for project_dir in projects:
        print(f"\n== cache-seed/{project_dir.name} (modelscope)", flush=True)
        for seed in load_spec(project_dir):
            ms_id = seed.get("ms_id")
            hf_id = seed.get("hf_id") or ms_id
            kind = seed.get("kind", "model")
            patterns = seed.get("allow_patterns")
            if not ms_id or kind not in ("model", "dataset"):
                print(f"SKIP invalid seed entry: {seed}", flush=True)
                continue
            try:
                plant(ms_id, hf_id, kind, args.root, patterns)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{ms_id}: {type(exc).__name__}: {exc}")
                print(f"FAIL {ms_id}: {exc}", flush=True)

    if failures:
        print(f"\nms_seed incomplete ({len(failures)}): {failures}",
              file=sys.stderr, flush=True)
        return 1
    print("\nms_seed complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Download HF assets via curl from hf-mirror into the shared HF cache.

cache-seed workflow 的第三半边（repo bundle 见 cache_seed.py，ModelScope
plant 见 ms_seed.py）。读 cache-seed/<project>/curl_seeds.yaml，对每条资产：

    curl https://hf-mirror.com/<hf_id>/resolve/main/<file>
        -> <root>/hub/models--<hf_id>/snapshots/<sha>/<file>
        -> refs/main 写真实 HF commit sha（HF API，xet-free）

适用场景：ModelScope 不 mirror、又太大不能打 repo bundle（git checkout
要把整棵树拉下来，multi-GB 会 504）。curl 整文件 GET 不走 huggingface_hub
的 xet 断点续传，故对 xet-backed 文件字节正确；spec 声明 sha256 的下载后
流式校验。

幂等：refs/main 命中且所有声明文件已非空 → 跳过；否则逐文件 curl，已存
且 sha256 匹配的跳过。

用法:
  python scripts/curl_seed.py [--projects peft,accelerate] [--root PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_ROOT = REPO_ROOT / "cache-seed"
DEFAULT_ROOT = Path(os.path.expanduser("~/.cache/huggingface"))
# Runners live in mainland China; only the JSON API + plain whole-file GET
# are hit (xet-free). Same HF_ENDPOINT semantics as ms_seed.py.
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
HF_API_BASE = HF_ENDPOINT + "/api"


def load_spec(project_dir: Path) -> list[dict]:
    import yaml
    path = project_dir / "curl_seeds.yaml"
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    seeds = data.get("seeds") or []
    if not isinstance(seeds, list):
        raise SystemExit(f"{path}: 'seeds' must be a list")
    return seeds


def fetch_sha(hf_id: str) -> str:
    import requests
    r = requests.get(f"{HF_API_BASE}/models/{hf_id}", timeout=30)
    r.raise_for_status()
    return r.json().get("sha") or r.json().get("oid")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def curl_download(url: str, dest: Path) -> None:
    # -C - lets curl resume across its own --retry attempts (whole-file GET
    # on the first attempt; resume only if a retry mid-transfer).
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-fL", "--retry", "5", "--retry-delay", "5",
         "--retry-all-errors", "--max-time", "2500", "-C", "-",
         "-o", str(dest), url],
        check=True,
    )


def plant(hf_id: str, files: list[dict], root: Path) -> None:
    sha = fetch_sha(hf_id)
    repo_dir = root / "hub" / f"models--{hf_id.replace('/', '--')}"
    snap_dir = repo_dir / "snapshots" / sha
    refs = repo_dir / "refs" / "main"

    # Fast path: already seeded at the current upstream sha with every file
    # present — skip the (multi-GB) downloads entirely.
    if (refs.is_file() and refs.read_text().strip() == sha
            and snap_dir.is_dir()
            and all((snap_dir / f["name"]).is_file()
                    and (snap_dir / f["name"]).stat().st_size > 0
                    for f in files)):
        print(f"[skip] {hf_id}@{sha[:8]} already seeded", flush=True)
        return

    (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
    refs.write_text(sha)  # no trailing newline — hub compares without strip
    for spec in files:
        name = spec["name"]
        expected_sha = spec.get("sha256")
        dest = snap_dir / name
        if dest.is_file() and expected_sha and sha256_file(dest) == expected_sha:
            print(f"[skip] {hf_id}/{name} (sha256 ok)", flush=True)
            continue
        if dest.is_file() and not expected_sha:
            print(f"[skip] {hf_id}/{name} (exists, no declared sha)", flush=True)
            continue
        if dest.is_file():
            dest.unlink()  # corrupt/stale — re-download
        print(f"[fetch] {hf_id}/{name} (curl hf-mirror)", flush=True)
        url = f"{HF_ENDPOINT}/{hf_id}/resolve/main/{name}"
        curl_download(url, dest)
        if expected_sha and sha256_file(dest) != expected_sha:
            dest.unlink(missing_ok=True)
            raise SystemExit(f"{hf_id}/{name}: sha256 mismatch after curl")
        print(f"[done] {hf_id}/{name}", flush=True)


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

    projects = sorted(
        p for p in SEED_ROOT.iterdir()
        if p.is_dir() and (not wanted or p.name in wanted)
        and (p / "curl_seeds.yaml").is_file()
    )
    if not projects:
        print(f"no curl_seeds.yaml under {SEED_ROOT}"
              + (f" matching {sorted(wanted)}" if wanted else ""), flush=True)
        return 0

    failures: list[str] = []
    for project_dir in projects:
        print(f"\n== cache-seed/{project_dir.name} (curl)", flush=True)
        for seed in load_spec(project_dir):
            hf_id = seed.get("hf_id")
            files = seed.get("files")
            if not hf_id or not isinstance(files, list) or not files:
                print(f"SKIP invalid seed entry: {seed}", flush=True)
                continue
            try:
                plant(hf_id, files, args.root)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{hf_id}: {type(exc).__name__}: {exc}")
                print(f"FAIL {hf_id}: {exc}", flush=True)

    if failures:
        print(f"\ncurl_seed incomplete ({len(failures)}): {failures}",
              file=sys.stderr, flush=True)
        return 1
    print("\ncurl_seed complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
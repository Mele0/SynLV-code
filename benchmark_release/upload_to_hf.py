#!/usr/bin/env python3
"""Upload SynLV release folder to Hugging Face Datasets hub.

Authentication:
- Uses standard Hugging Face auth resolution (env token or local login cache).
- No token is hardcoded in this script.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def sizeof_fmt(num: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num) < 1024.0:
            return f"{num:0.2f} {unit}"
        num /= 1024.0
    return f"{num:0.2f} PB"


def main() -> None:
    repo_root_default = Path(__file__).resolve().parents[1]
    release_default = repo_root_default / "release" / "synlv_v1.0" / "synlv-benchmark"
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", type=str, default="", help="HF dataset repo id, e.g. user/synlv-benchmark")
    ap.add_argument(
        "--release_dir",
        type=str,
        default=str(release_default),
    )
    ap.add_argument("--private", action="store_true", help="Create repo as private (if creating).")
    ap.add_argument("--dry_run", action="store_true", help="Print planned actions only.")
    ap.add_argument("--commit_message", type=str, default="Upload SynLV benchmark v1.0 release")
    args = ap.parse_args()

    release_dir = Path(args.release_dir).resolve()
    if not release_dir.exists():
        raise SystemExit(f"Release dir does not exist: {release_dir}")

    files = [p for p in release_dir.rglob("*") if p.is_file()]
    total_size = sum(p.stat().st_size for p in files)
    largest_file = max(files, key=lambda p: p.stat().st_size) if files else None
    token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))

    print("[upload] release_dir:", release_dir)
    print("[upload] files:", len(files))
    print("[upload] total_size:", f"{total_size} bytes ({sizeof_fmt(float(total_size))})")
    if largest_file is not None:
        print(
            "[upload] largest_file:",
            f"{largest_file.relative_to(release_dir).as_posix()} ({largest_file.stat().st_size} bytes)",
        )
    print("[upload] token_present_env:", token_present)

    if args.dry_run:
        print("[dry-run] no remote actions performed.")
        if args.repo_id:
            print(f"[dry-run] would target dataset repo: {args.repo_id} (private={args.private})")
        else:
            print("[dry-run] repo_id not set; pass --repo_id for an actionable upload run.")
        return

    if not args.repo_id:
        raise SystemExit("--repo_id is required unless --dry_run is set")

    try:
        from huggingface_hub import HfApi
    except Exception as e:
        raise SystemExit(f"huggingface_hub import failed: {e}")

    api = HfApi()
    print(f"[upload] ensuring dataset repo exists: {args.repo_id} (private={args.private})")
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    print("[upload] uploading folder...")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(release_dir),
        path_in_repo=".",
        commit_message=args.commit_message,
    )
    print("[ok] upload completed")


if __name__ == "__main__":
    main()

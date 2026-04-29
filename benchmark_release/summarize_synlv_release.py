#!/usr/bin/env python3
"""Print a concise summary of a built SynLV release package."""

from __future__ import annotations

import argparse
import json
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
    ap.add_argument(
        "--release-dir",
        type=str,
        default=str(release_default),
    )
    ap.add_argument(
        "--write-report",
        type=int,
        default=1,
        help="If 1, writes metadata/release_summary.txt",
    )
    args = ap.parse_args()

    release_dir = Path(args.release_dir).resolve()
    if not release_dir.exists():
        raise SystemExit(f"Release dir not found: {release_dir}")

    meta_dir = release_dir / "metadata"
    ds_summary_fp = meta_dir / "dataset_summary.json"
    rel_manifest_fp = meta_dir / "release_manifest.json"
    scenario_manifest_fp = meta_dir / "scenario_manifest.json"

    if not (ds_summary_fp.exists() and rel_manifest_fp.exists() and scenario_manifest_fp.exists()):
        raise SystemExit("Missing required metadata JSON files. Run build first.")

    dataset_summary = json.loads(ds_summary_fp.read_text(encoding="utf-8"))
    release_manifest = json.loads(rel_manifest_fp.read_text(encoding="utf-8"))
    scenario_manifest = json.loads(scenario_manifest_fp.read_text(encoding="utf-8"))

    all_files = [p for p in release_dir.rglob("*") if p.is_file()]
    total_size = sum(p.stat().st_size for p in all_files)
    largest_file = max(all_files, key=lambda p: p.stat().st_size) if all_files else None

    scenario_lines = []
    total_cohorts = 0
    for scenario in sorted(scenario_manifest.keys()):
        cohorts = scenario_manifest[scenario].get("cohorts", {})
        total_cohorts += len(cohorts)
        rows_total = scenario_manifest[scenario].get("rows_total", 0)
        scenario_lines.append(f"- {scenario}: cohorts={len(cohorts)}, rows_total={rows_total}")

    starter_dir = release_dir / "starter_outputs"
    starter_present = starter_dir.exists() and any(p.is_file() for p in starter_dir.glob("*"))
    mimic_boundary_present = (release_dir / "docs" / "MIMIC_GROUNDING.md").exists()

    lines = [
        "SynLV v1.0 Release Summary",
        f"release_dir: {release_dir}",
        f"version: {release_manifest.get('release_version', 'unknown')}",
        f"git_commit: {release_manifest.get('git_commit', 'unknown')}",
        f"scenarios: {dataset_summary.get('scenario_count', 'unknown')}",
        f"cohorts_total: {total_cohorts}",
        f"total_rows: {dataset_summary.get('total_rows', 'unknown')}",
        f"files_total: {len(all_files)}",
        f"size_total_bytes: {total_size}",
        f"size_total_human: {sizeof_fmt(float(total_size))}",
        (
            f"largest_file: {largest_file.relative_to(release_dir).as_posix()} "
            f"({largest_file.stat().st_size} bytes, {sizeof_fmt(float(largest_file.stat().st_size))})"
            if largest_file is not None
            else "largest_file: none"
        ),
        f"starter_outputs_present: {starter_present}",
        f"mimic_boundary_doc_present: {mimic_boundary_present}",
        "",
        "Scenario coverage:",
        *scenario_lines,
    ]

    summary = "\n".join(lines)
    print(summary)

    if int(args.write_report) == 1:
        out_fp = meta_dir / "release_summary.txt"
        out_fp.write_text(summary + "\n", encoding="utf-8")
        print(f"\n[ok] wrote {out_fp}")


if __name__ == "__main__":
    main()

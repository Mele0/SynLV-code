#!/usr/bin/env python3
"""Validate the public SynLV registry, Croissant inventory, and optional local files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_analysis.synlv_release_config import (  # noqa: E402
    HF_SUBSET_NAMES,
    PRIMARY_SCENARIO_KEYS,
    canonical_cohort_dir,
    get_scenario,
    registry_as_dict,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_table(path: Path):
    import pandas as pd

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _find_split_file(cohort_dir: Path, split: str) -> Path | None:
    for ext in [".parquet", ".csv"]:
        fp = cohort_dir / f"{split}{ext}"
        if fp.exists():
            return fp
    return None


def _croissant_obj(croissant: Path | None) -> tuple[dict[str, Any], str, list[str]]:
    if croissant is None:
        return {}, "", ["Croissant path not provided"]
    if not croissant.exists():
        return {}, "", [f"Croissant file missing: {croissant}"]
    try:
        obj = json.loads(croissant.read_text(encoding="utf-8"))
        return obj, json.dumps(obj), []
    except Exception as exc:
        return {}, "", [f"Croissant parse error: {exc}"]


def _distribution_entries(obj: dict[str, Any]) -> list[dict[str, Any]]:
    dist = obj.get("distribution", [])
    if isinstance(dist, dict):
        return [dist]
    return dist if isinstance(dist, list) else []


def _extract_sha(entry: dict[str, Any]) -> str | None:
    for key in ["sha256", "sha256Hash", "contentChecksum", "checksum"]:
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _validate_croissant(croissant: Path | None, dataset_root: Path | None = None) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    obj, blob, parse_errors = _croissant_obj(croissant)
    errors.extend(parse_errors)
    if not obj:
        return errors, warnings, {}

    for key in ["name", "description", "distribution"]:
        if key not in obj:
            errors.append(f"Croissant missing required key: {key}")

    lower_blob = blob.lower()
    if "six scenario" in lower_blob or "6 scenario" in lower_blob:
        errors.append("Croissant contains six-scenario language")

    for scenario in PRIMARY_SCENARIO_KEYS:
        spec = get_scenario(scenario)
        if scenario not in blob and spec.hf_subset_name not in blob and spec.paper_name not in blob:
            errors.append(f"Croissant does not mention scenario/subset: {scenario} / {spec.hf_subset_name}")

    dist = _distribution_entries(obj)
    dist_blob = json.dumps(dist)
    scenario_seed_mentions = 0
    for scenario in PRIMARY_SCENARIO_KEYS:
        for cs in range(5):
            seed_tag = f"cohortseed_{cs:03d}"
            if scenario in dist_blob and seed_tag in dist_blob:
                scenario_seed_mentions += 1

    if len(dist) < 9:
        warnings.append(f"Croissant distribution has {len(dist)} entries; full split-level inventory usually has at least 135.")

    if dataset_root is not None:
        for entry in dist:
            sha = _extract_sha(entry)
            rel = entry.get("contentUrl") or entry.get("name")
            if not sha or not isinstance(rel, str):
                continue
            fp = dataset_root / rel
            if fp.exists():
                actual = _sha256(fp)
                if actual != sha:
                    errors.append(f"Croissant SHA256 mismatch for {rel}")

    summary = {
        "croissant_name": obj.get("name"),
        "croissant_version": obj.get("version"),
        "distribution_count": len(dist),
        "scenario_keys_found": [s for s in PRIMARY_SCENARIO_KEYS if s in blob],
        "hf_subsets_found": [s for s in HF_SUBSET_NAMES if s in blob],
        "scenario_seed_distribution_mention_count": scenario_seed_mentions,
    }
    return errors, warnings, summary


def _validate_dataset(dataset_root: Path, smoke_test: bool = False) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    scenarios = PRIMARY_SCENARIO_KEYS[:1] if smoke_test else PRIMARY_SCENARIO_KEYS
    seeds = [0] if smoke_test else [0, 1, 2, 3, 4]
    schema_ref: list[str] | None = None

    for scenario in scenarios:
        spec = get_scenario(scenario)
        for cs in seeds:
            cohort_dir = canonical_cohort_dir(dataset_root, scenario, cs)
            if not cohort_dir.exists():
                errors.append(f"Missing cohort directory: {cohort_dir}")
                continue
            split_ids: dict[str, set[int]] = {}
            for split in ["train", "val", "test"]:
                fp = _find_split_file(cohort_dir, split)
                if fp is None:
                    errors.append(f"Missing split file under {cohort_dir}: {split}.csv or {split}.parquet")
                    continue
                try:
                    df = _load_table(fp)
                except Exception as exc:
                    errors.append(f"Cannot read {fp}: {exc}")
                    continue

                cols = list(df.columns)
                if schema_ref is None:
                    schema_ref = cols
                elif cols != schema_ref:
                    errors.append(f"Schema mismatch in {fp}: {cols} != {schema_ref}")

                for col in ["RANDID", "TIME", "TIMEDTH", "DEATH"]:
                    if col not in df.columns:
                        errors.append(f"{fp} missing required column {col}")

                if "RANDID" in df.columns:
                    split_ids[split] = set(df["RANDID"].dropna().astype(int).tolist())
                if "TIME" in df.columns and "RANDID" in df.columns:
                    bad_order = 0
                    for _, group in df.groupby("RANDID"):
                        times = group["TIME"].to_numpy()
                        if len(times) and (times[1:] < times[:-1]).any():
                            bad_order += 1
                    if bad_order:
                        errors.append(f"{fp} has {bad_order} patients with decreasing TIME")

                miss_cols = [c for c in [f"X{i}" for i in range(12)] if c in df.columns]
                stress_cols = [c for c in [f"X{i}" for i in range(6)] if c in df.columns]
                event_rate = None
                if {"RANDID", "DEATH"}.issubset(df.columns) and not df.empty:
                    event_rate = float(df.groupby("RANDID")["DEATH"].first().mean())
                rows.append(
                    {
                        "scenario": scenario,
                        "hf_subset": spec.hf_subset_name,
                        "cohort_seed": cs,
                        "split": split,
                        "file": str(fp),
                        "rows": int(len(df)),
                        "patients": int(df["RANDID"].nunique()) if "RANDID" in df.columns else None,
                        "event_rate": event_rate,
                        "overall_missingness": float(df[miss_cols].isna().to_numpy().mean()) if miss_cols else None,
                        "stress_pool_missingness": float(df[stress_cols].isna().to_numpy().mean()) if stress_cols else None,
                    }
                )
            if all(k in split_ids for k in ["train", "val", "test"]):
                for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
                    overlap = split_ids[left] & split_ids[right]
                    if overlap:
                        errors.append(f"{scenario} cs{cs:03d}: {left}/{right} RANDID overlap={len(overlap)}")
    return errors, warnings, rows


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# SynLV Release Validation Report",
        "",
        f"Status: **{payload['status']}**",
        "",
        "## Checks",
        "",
        "- Canonical registry has exactly 9 primary scenarios.",
        "- Croissant metadata mentions all expected scenarios/subsets.",
        "- Default/strict mode validates registry and Croissant inventory without requiring dataset download.",
        "- Dataset mode validates local split files, schema, patient disjointness, time ordering, event rates, and missingness summaries.",
        "",
        "## Registry",
        "",
        f"- Primary scenarios: {len(payload['registry']['primary_scenario_keys'])}",
        f"- HF subsets: {', '.join(payload['registry']['hf_subset_names'])}",
        "",
        "## Croissant Summary",
        "",
    ]
    for key, value in payload.get("croissant_summary", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {w}" for w in payload["warnings"]] or ["- none"])
    lines.extend(["", "## Errors", ""])
    lines.extend([f"- {e}" for e in payload["errors"]] or ["- none"])
    if payload.get("dataset_summary_rows"):
        lines.extend(["", "## Dataset Summary", ""])
        lines.append(f"Rows summarized: {len(payload['dataset_summary_rows'])}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate SynLV public artifact metadata and optional local dataset files.")
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--croissant", type=Path, default=REPO_ROOT / "SynLV_hf_platform_croissant.json")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "audit" / "release_validation_report.md")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--no-download", action="store_true", help="Validate registry/Croissant only.")
    ap.add_argument("--smoke-test", action="store_true", help="Validate only one scenario/seed when dataset-root is supplied.")
    ap.add_argument("--strict", type=int, default=0, help="If 1, fail on registry/Croissant errors. Does not require local data.")
    args = ap.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    if len(PRIMARY_SCENARIO_KEYS) != 9:
        errors.append(f"Registry has {len(PRIMARY_SCENARIO_KEYS)} primary scenarios, expected 9")

    croissant_errors, croissant_warnings, croissant_summary = _validate_croissant(args.croissant, args.dataset_root)
    errors.extend(croissant_errors)
    warnings.extend(croissant_warnings)

    dataset_rows: list[dict[str, Any]] = []
    if args.dataset_root is not None and not args.no_download:
        dataset_errors, dataset_warnings, dataset_rows = _validate_dataset(args.dataset_root, smoke_test=args.smoke_test)
        errors.extend(dataset_errors)
        warnings.extend(dataset_warnings)
    elif not args.no_download and int(args.strict) == 0:
        warnings.append("No dataset root supplied; local split/schema validation skipped.")

    status = "PASS" if not errors else "ERROR"
    payload = {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "registry": registry_as_dict(),
        "croissant_summary": croissant_summary,
        "dataset_summary_rows": dataset_rows,
    }
    _write_report(args.out, payload)
    json_out = args.json_out or args.out.with_suffix(".json")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[validate] status={status} errors={len(errors)} warnings={len(warnings)}")
    print(f"[validate] report={args.out}")
    print(f"[validate] json={json_out}")
    if errors and int(args.strict) == 1:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

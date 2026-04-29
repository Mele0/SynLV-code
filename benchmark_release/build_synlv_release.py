#!/usr/bin/env python3
"""Build SynLV v1.0 synthetic benchmark release package.

Build target:
  release/synlv_v1.0/synlv-benchmark/

This script uses canonical scenario roots from benchmark_analysis/synlv_release_config.py.
It does not regenerate source cohorts; it converts canonical CSV splits to parquet and
adds release metadata/docs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_commit_hash(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "unknown"


def infer_generator_hint(scenario: str, canonical_source_root: Path) -> str:
    root_txt = str(canonical_source_root)
    if "/Data/SynLV_Zoo/base" in root_txt:
        if scenario in {"scenario_VISITSHIFT_TRANSFER", "scenario_MISMATCH_TRANSFER"}:
            return "generate_synlv_variant_zoo.py::save_transfer_bundle"
        if scenario == "scenario_MNAR_CORRELATED":
            return "generate_synlv_variant_zoo.py (base variant, mnar_corr settings)"
        return "generate_synlv_variant_zoo.py (base variant)"
    return "final_generation.py"


def visit_summary(df: pd.DataFrame, id_col: str = "RANDID") -> dict[str, float]:
    if df.empty or id_col not in df.columns:
        return {
            "patients": 0,
            "visits_mean": 0.0,
            "visits_median": 0.0,
            "visits_min": 0.0,
            "visits_max": 0.0,
            "visits_q25": 0.0,
            "visits_q75": 0.0,
        }
    vc = df.groupby(id_col).size().astype(float)
    return {
        "patients": int(vc.shape[0]),
        "visits_mean": float(vc.mean()),
        "visits_median": float(vc.median()),
        "visits_min": float(vc.min()),
        "visits_max": float(vc.max()),
        "visits_q25": float(vc.quantile(0.25)),
        "visits_q75": float(vc.quantile(0.75)),
    }


def split_patient_event_rate(df: pd.DataFrame, id_col: str = "RANDID", event_col: str = "DEATH") -> float:
    if df.empty or id_col not in df.columns or event_col not in df.columns:
        return float("nan")
    g = df.groupby(id_col)[event_col].first()
    return float(pd.to_numeric(g, errors="coerce").mean())


def missingness_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"overall": float("nan"), "per_column": {}}
    miss = df.isna().mean(numeric_only=False)
    return {
        "overall": float(df.isna().to_numpy().mean()),
        "per_column": {str(k): float(v) for k, v in miss.items()},
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def ensure_mlcroissant_status() -> tuple[bool, str]:
    spec = importlib.util.find_spec("mlcroissant")
    if spec is None:
        return False, "mlcroissant not available; generated croissant.json manually."
    return True, "mlcroissant available; manual croissant generation still used for deterministic control."


def build_croissant_json(release_dir: Path, schema_fields: list[dict[str, str]], data_files: list[Path]) -> dict[str, Any]:
    dist = []
    for fp in sorted(data_files):
        rel = fp.relative_to(release_dir).as_posix()
        mime = "application/x-parquet" if rel.endswith(".parquet") else "application/json"
        dist.append(
            {
                "@type": "cr:FileObject",
                "@id": f"file:{rel}",
                "name": rel,
                "contentUrl": rel,
                "encodingFormat": mime,
            }
        )

    record_fields = []
    for f in schema_fields:
        record_fields.append(
            {
                "@type": "cr:Field",
                "name": f["name"],
                "dataType": f["data_type"],
                "description": f["description"],
            }
        )

    return {
        "@context": {
            "@vocab": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
        },
        "@type": "Dataset",
        "name": "SynLV Benchmark v1.0",
        "version": "1.0.0",
        "description": "Synthetic benchmark artifact for last-visit robustness in longitudinal survival modeling.",
        "license": "Proprietary/Research release per repository terms",
        "distribution": dist,
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "name": "SynLVTrajectoryRow",
                "description": "Long-format synthetic trajectory rows for SynLV benchmark cohorts.",
                "field": record_fields,
            }
        ],
    }


def copy_starter_outputs(repo_root: Path, release_dir: Path) -> tuple[bool, list[str]]:
    src_root = repo_root / "docs" / "export_ready"
    targets = [
        (
            src_root / "EXPORT_READY_README.md",
            Path("notes") / "EXPORT_READY_README.md",
        ),
        (
            src_root / "MIMIC_GROUNDING_NOTE.md",
            Path("notes") / "MIMIC_GROUNDING_NOTE.md",
        ),
    ]

    out_dir = release_dir / "starter_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for fp, rel_out in targets:
        if fp.exists():
            dst = out_dir / rel_out
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fp, dst)
            copied.append(f"{fp} -> {dst}")

    prov = [
        "# Starter Outputs Provenance",
        "",
        "These files are copied from export-ready documentation notes included in this code artifact.",
        "No heavy result bundles are redistributed in this repository snapshot.",
        "",
        "## Source files",
    ]
    for s in copied:
        prov.append(f"- `{s}`")
    write_text(out_dir / "PROVENANCE.md", "\n".join(prov) + "\n")
    return len(copied) > 0, copied


def write_release_docs(release_dir: Path) -> None:
    write_text(release_dir / "VERSION", "1.0.0\n")

    readme = """# SynLV Benchmark v1.0 (Hugging Face Dataset Card)

## Dataset Summary
SynLV v1.0 is a synthetic benchmark artifact for evaluating last-visit robustness in longitudinal survival modeling.
The release includes 9 primary scenarios and 5 cohort seeds per scenario (`cs000..cs004`).

## Task Categories
- Survival analysis
- Robustness under structured missingness/corruption

## Languages
- Not language data

## Data Structure
Each scenario/seed directory contains:
- `train.parquet`
- `val.parquet`
- `test.parquet`
- `metadata.json`

Rows are long-format trajectory observations with schema grounded in actual generator output.

## Scenarios (v1.0)
- `scenario_A`
- `scenario_B`
- `scenario_C`
- `scenario_MNAR_CORRELATED`
- `scenario_MAR`
- `scenario_VISITSHIFT`
- `scenario_VISITSHIFT_TRANSFER`
- `scenario_MISMATCH`
- `scenario_MISMATCH_TRANSFER`

## Intended Uses
- Benchmarking robustness under last-visit incompleteness
- Reproducible evaluation using fixed scenario/split conventions

## Out-of-Scope Uses
- Claims of universal clinical validity
- Re-identification or patient-level clinical inference

## MIMIC Grounding Boundary
MIMIC is grounding-only. Raw/row-level MIMIC data are not redistributed in this release.
See `docs/MIMIC_GROUNDING.md`.

## Citation
Use repository/paper citation as specified in project docs.
"""
    write_text(release_dir / "README.md", readme)

    benchmark_card = """# BENCHMARK_CARD

## Benchmark Artifact
SynLV is the benchmark artifact. Probe models are included to test benchmark utility, not as the benchmark contribution.

## Protocol
Primary stress protocol uses fixed prevalence and severity grids on last-visit corruption with matched repetitions.
Primary summary conditions: Clean, Grid avg, Severe.

## Scope
Hosted v1.0 includes only the 9 primary synthetic scenarios and seeds `cs000..cs004`.
Generator-sensitivity and appendix-only branches are excluded from hosted payload (reproducible from source only).
"""
    write_text(release_dir / "BENCHMARK_CARD.md", benchmark_card)

    dataset_card = """# DATASET_CARD

## Composition
- 9 scenarios
- 5 cohort seeds per scenario
- train/val/test parquet splits per scenario-seed
- per-cohort metadata in `metadata.json`

## Schema
Grounded in generator output columns, preserving benchmark-facing semantics.
No forced reshaping to manuscript notation.

## Provenance
Canonical source roots are defined by audited release mapping and encoded in cohort-level metadata.
"""
    write_text(release_dir / "DATASET_CARD.md", dataset_card)

    release_plan = """# RELEASE_PLAN

## Included
- 9 primary scenarios x 5 cohort seeds
- Parquet split exports + metadata
- Release-level manifests, checksums, validation report

## Excluded (hosted payload)
- Generator-sensitivity branches
- Dense-visit variants
- Informative-censor variants
- Appendix-only scenario families

Excluded branches remain reproducible from source scripts/workflows in the repository.
"""
    write_text(release_dir / "RELEASE_PLAN.md", release_plan)

    mimic_grounding = """# MIMIC_GROUNDING

MIMIC is included only as a grounding layer in the overall project narrative.

This hosted synthetic benchmark release does not redistribute:
- raw MIMIC data,
- row-level derived MIMIC tables,
- patient-level intermediate MIMIC extracts.

Only reconstruction code/manifests/instructions are expected for grounding rebuild workflows,
which require credentialed source access.
"""
    write_text(release_dir / "docs" / "MIMIC_GROUNDING.md", mimic_grounding)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="Path to repository root",
    )
    ap.add_argument(
        "--release-dir",
        type=str,
        default="",
        help="Release output directory (default: <repo>/release/synlv_v1.0/synlv-benchmark)",
    )
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--include-starter-outputs", type=int, default=1)
    ap.add_argument("--force-clean", type=int, default=0)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    scripts_root = repo_root / "benchmark_analysis"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))

    from synlv_release_config import PRIMARY_SCENARIOS_V1, canonical_cohort_dir, canonical_data_root_for_scenario

    release_dir = Path(args.release_dir).resolve() if args.release_dir else (repo_root / "release" / "synlv_v1.0" / "synlv-benchmark")
    if int(args.force_clean) == 1 and release_dir.exists():
        shutil.rmtree(release_dir)

    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / "metadata").mkdir(parents=True, exist_ok=True)

    write_release_docs(release_dir)

    seeds = parse_int_list(args.seeds)
    version = "1.0.0"
    commit = git_commit_hash(repo_root)
    timestamp = iso_now()

    scenario_manifest: dict[str, Any] = {}
    dataset_summary: dict[str, Any] = {
        "release_version": version,
        "created_at": timestamp,
        "git_commit": commit,
        "scenario_count": len(PRIMARY_SCENARIOS_V1),
        "cohort_seeds": seeds,
        "total_cohorts_expected": len(PRIMARY_SCENARIOS_V1) * len(seeds),
        "total_rows": 0,
        "total_patients": 0,
        "scenarios": {},
    }

    data_files_for_croissant: list[Path] = []
    schema_fields: list[dict[str, str]] = []

    for scenario in PRIMARY_SCENARIOS_V1:
        scenario_dir = release_dir / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)

        canonical_root = canonical_data_root_for_scenario(repo_root, scenario)
        sc_obj = {
            "scenario": scenario,
            "canonical_source_root": str(canonical_root),
            "cohorts": {},
        }
        sc_rows = 0
        sc_patients = 0

        for cs in seeds:
            src_cohort = canonical_cohort_dir(repo_root, scenario, cs)
            if not src_cohort.exists():
                raise FileNotFoundError(f"Canonical cohort dir missing: {src_cohort}")

            tgt = scenario_dir / f"cohortseed_{cs:03d}"
            tgt.mkdir(parents=True, exist_ok=True)

            split_frames: dict[str, pd.DataFrame] = {}
            for split in ["train", "val", "test"]:
                src_csv = src_cohort / f"{split}.csv"
                if not src_csv.exists():
                    raise FileNotFoundError(f"Missing split CSV: {src_csv}")
                df = pd.read_csv(src_csv)
                split_frames[split] = df

                out_parquet = tgt / f"{split}.parquet"
                table = pa.Table.from_pandas(df, preserve_index=False)
                pq.write_table(table, out_parquet, compression="snappy")
                data_files_for_croissant.append(out_parquet)

                if not schema_fields:
                    for field in table.schema:
                        if pa.types.is_integer(field.type):
                            dt_name = "integer"
                        elif pa.types.is_floating(field.type):
                            dt_name = "number"
                        elif pa.types.is_boolean(field.type):
                            dt_name = "boolean"
                        else:
                            dt_name = "string"
                        schema_fields.append(
                            {
                                "name": field.name,
                                "data_type": dt_name,
                                "description": f"Column {field.name} from SynLV trajectory rows.",
                            }
                        )

            gen_cfg_fp = src_cohort / "gen_config.json"
            meta_fp = src_cohort / "meta.json"
            cohort_stats_fp = src_cohort / "cohort_stats.json"
            gen_cfg = json.loads(gen_cfg_fp.read_text(encoding="utf-8")) if gen_cfg_fp.exists() else {}
            meta_obj = json.loads(meta_fp.read_text(encoding="utf-8")) if meta_fp.exists() else {}
            cohort_stats = json.loads(cohort_stats_fp.read_text(encoding="utf-8")) if cohort_stats_fp.exists() else {}

            train_df = split_frames["train"]
            val_df = split_frames["val"]
            test_df = split_frames["test"]
            all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

            event_rate = split_patient_event_rate(all_df)
            censor_rate = float(1.0 - event_rate) if pd.notna(event_rate) else float("nan")

            split_rows = {k: int(v.shape[0]) for k, v in split_frames.items()}
            split_patients = {
                k: int(v["RANDID"].nunique()) if "RANDID" in v.columns else 0
                for k, v in split_frames.items()
            }
            visit_stats = {k: visit_summary(v) for k, v in split_frames.items()}
            miss_stats = {k: missingness_summary(v) for k, v in split_frames.items()}

            cohort_metadata = {
                "release_version": version,
                "scenario_name": scenario,
                "cohort_seed": int(cs),
                "canonical_source_root": str(canonical_root),
                "canonical_source_cohort_dir": str(src_cohort),
                "generator_version": {
                    "hint": infer_generator_hint(scenario, canonical_root),
                    "variant": gen_cfg.get("variant", "legacy_or_primary"),
                    "gen_seed": gen_cfg.get("gen_seed"),
                    "generator_config_path": str(gen_cfg_fp) if gen_cfg_fp.exists() else "",
                },
                "split_sizes": split_patients,
                "patient_counts": {
                    "train": split_patients.get("train", 0),
                    "val": split_patients.get("val", 0),
                    "test": split_patients.get("test", 0),
                    "all_unique": int(all_df["RANDID"].nunique()) if "RANDID" in all_df.columns else 0,
                },
                "row_counts": split_rows,
                "event_rate": float(event_rate) if pd.notna(event_rate) else None,
                "censoring_rate": float(censor_rate) if pd.notna(censor_rate) else None,
                "visit_count_summary": visit_stats,
                "missingness_summary": miss_stats,
                "export_timestamp": iso_now(),
                "git_commit": commit,
                "source_gen_config": gen_cfg,
                "source_meta": meta_obj,
                "source_cohort_stats": cohort_stats,
                "column_schema": list(all_df.columns),
            }
            meta_out = tgt / "metadata.json"
            meta_out.write_text(json.dumps(cohort_metadata, indent=2), encoding="utf-8")
            data_files_for_croissant.append(meta_out)

            sc_obj["cohorts"][f"cohortseed_{cs:03d}"] = {
                "source": str(src_cohort),
                "rows": split_rows,
                "patients": split_patients,
                "event_rate": cohort_metadata["event_rate"],
                "censoring_rate": cohort_metadata["censoring_rate"],
            }

            sc_rows += sum(split_rows.values())
            sc_patients += cohort_metadata["patient_counts"]["all_unique"]

        sc_obj["rows_total"] = sc_rows
        sc_obj["patients_total_sum_over_cohorts"] = sc_patients
        scenario_manifest[scenario] = sc_obj

        dataset_summary["scenarios"][scenario] = {
            "canonical_source_root": str(canonical_root),
            "rows_total": sc_rows,
            "patients_total_sum_over_cohorts": sc_patients,
        }
        dataset_summary["total_rows"] += sc_rows
        dataset_summary["total_patients"] += sc_patients

    metadata_dir = release_dir / "metadata"
    (metadata_dir / "scenario_manifest.json").write_text(json.dumps(scenario_manifest, indent=2), encoding="utf-8")
    (metadata_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")

    release_manifest = {
        "release_version": version,
        "release_dir": str(release_dir),
        "created_at": iso_now(),
        "git_commit": commit,
        "scenario_count": len(PRIMARY_SCENARIOS_V1),
        "cohort_seed_count": len(seeds),
        "scope": {
            "included_primary_scenarios": PRIMARY_SCENARIOS_V1,
            "excluded_from_hosted_release": [
                "generator-sensitivity branches",
                "dense-visit branches",
                "informative-censoring branches",
                "appendix-only branches",
            ],
            "mimic_boundary": "grounding-only; no raw or row-level derived MIMIC exports",
        },
    }
    (metadata_dir / "release_manifest.json").write_text(json.dumps(release_manifest, indent=2), encoding="utf-8")

    has_starter = False
    copied_sources: list[str] = []
    if int(args.include_starter_outputs) == 1:
        has_starter, copied_sources = copy_starter_outputs(repo_root, release_dir)

    missing_report = repo_root / "release" / "synlv_v1.0" / "MISSING_ITEMS_REPORT.md"
    if not has_starter:
        txt = """# MISSING_ITEMS_REPORT

Starter outputs were requested but no safe precomputed starter outputs were found at expected synthetic export paths.

Expected note source root checked:
- docs/export_ready/

Action: either regenerate synthetic starter outputs or update build script source-path mapping.
"""
        write_text(missing_report, txt)

    # Build file inventory and checksums.
    all_files = sorted([p for p in release_dir.rglob("*") if p.is_file()])
    inventory_rows = []
    checksum_lines = []
    for fp in all_files:
        rel = fp.relative_to(release_dir).as_posix()
        if rel == "metadata/checksums.txt":
            continue
        h = sha256_file(fp)
        size = fp.stat().st_size
        inventory_rows.append({"relative_path": rel, "size_bytes": size, "sha256": h})
        checksum_lines.append(f"{h}  {rel}")

    inv_fp = metadata_dir / "file_inventory.csv"
    with inv_fp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["relative_path", "size_bytes", "sha256"])
        w.writeheader()
        for row in inventory_rows:
            w.writerow(row)

    write_text(metadata_dir / "checksums.txt", "\n".join(checksum_lines) + "\n")

    croissant_available, croissant_note = ensure_mlcroissant_status()
    croissant_obj = build_croissant_json(release_dir, schema_fields=schema_fields, data_files=data_files_for_croissant)
    write_text(release_dir / "croissant.json", json.dumps(croissant_obj, indent=2))

    validation_report_lines = [
        "SynLV release build validation report (builder stage)",
        f"timestamp_utc: {iso_now()}",
        f"release_dir: {release_dir}",
        f"git_commit: {commit}",
        f"mlcroissant_available: {croissant_available}",
        f"mlcroissant_note: {croissant_note}",
        "checks_performed:",
        "- canonical scenario root resolution",
        "- split csv existence for 9x5 cohorts",
        "- parquet conversion for train/val/test",
        "- cohort metadata export",
        "- release-level manifest/inventory/checksums export",
        "manual_attention:",
        "- run validate_synlv_release.py for full structural and checksum verification",
    ]
    write_text(metadata_dir / "validation_report.txt", "\n".join(validation_report_lines) + "\n")

    print("[ok] release build complete")
    print(f"[ok] release_dir={release_dir}")
    print(f"[ok] scenarios={len(PRIMARY_SCENARIOS_V1)} seeds={len(seeds)} total_cohorts={len(PRIMARY_SCENARIOS_V1)*len(seeds)}")
    print(f"[ok] starter_outputs_included={has_starter} copied={len(copied_sources)}")


if __name__ == "__main__":
    main()

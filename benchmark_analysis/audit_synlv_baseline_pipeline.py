#!/usr/bin/env python3
"""Audit SynLV baseline pipeline consistency for snapshot models.

Checks:
1) Result completeness for all expected baseline tasks.
2) Stress manifest consistency across snapshot models (CoxPH/RSF/CoxTime/DeepHit).
3) Mask-pool overlap with last-visit features and non-zero applied masking effect.
4) Clean->Severe direction summary per scenario/model for sanity review.

Outputs:
- baseline_pipeline_audit_report.md
- baseline_pipeline_clean_vs_severe.csv
- baseline_pipeline_mask_checks.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIM_COMMON_OK = True
try:
    from sim_common import (
        get_feature_names,
        preprocess_filters_synlv,
        stable_mask_seed,
        stable_patient_seed,
    )
except Exception:
    SIM_COMMON_OK = False
from synlv_release_config import PRIMARY_SCENARIOS_V1, canonical_cohort_dir

ID_COL = "RANDID"
TIME_COL = "TIME"

SCENARIOS = list(PRIMARY_SCENARIOS_V1)
MODELS = ["CoxPH", "RSF", "CoxTime", "DeepHit", "CoxTV", "GRUMaskCox"]
SNAPSHOT_MODELS = ["CoxPH", "RSF", "CoxTime", "DeepHit"]
COHORT_SEEDS = [0, 1, 2, 3, 4]
P_MASK_SEVERE = 0.90
K_SEVERE = 6
REP_TEST = 0


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-root", type=str, default=str(root))
    ap.add_argument(
        "--baseline-root",
        type=str,
        default=str(root / "Results" / "Experiments" / "SynLV_Baselines"),
    )
    ap.add_argument(
        "--data-root",
        type=str,
        default="",
        help="Optional override root for all scenarios. If empty, uses canonical mixed source roots.",
    )
    ap.add_argument("--seed", type=int, default=1991)
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(root / "Results" / "Final" / "Tables" / "Baselines_SynLV"),
    )
    return ap.parse_args()


def read_raw(baseline_root: Path, scenario: str, model: str, seed: int, cs: int) -> pd.DataFrame | None:
    fp = (
        baseline_root
        / scenario
        / model
        / f"seed{seed}"
        / "Summary"
        / f"sim_{scenario}_{model}_seed{seed}_cs{cs:03d}_raw_results.csv"
    )
    if not fp.exists():
        return None
    return pd.read_csv(fp)


def read_manifest(baseline_root: Path, scenario: str, model: str, seed: int, cs: int) -> pd.DataFrame | None:
    fp = (
        baseline_root
        / scenario
        / model
        / f"seed{seed}"
        / "Summary"
        / f"sim_{scenario}_{model}_seed{seed}_cs{cs:03d}_manifest.csv"
    )
    if not fp.exists():
        return None
    return pd.read_csv(fp)


def _last_visit(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values([ID_COL, TIME_COL]).groupby(ID_COL, as_index=False).last()


def audit_completeness(baseline_root: Path, seed: int) -> tuple[pd.DataFrame, list[tuple[str, str, int]]]:
    rows = []
    missing = []
    for scenario in SCENARIOS:
        for model in MODELS:
            for cs in COHORT_SEEDS:
                d = read_raw(baseline_root, scenario, model, seed, cs)
                status = "ok"
                if d is None:
                    status = "missing"
                    missing.append((scenario, model, cs))
                rows.append(
                    {
                        "scenario": scenario,
                        "model": model,
                        "cohort_seed": cs,
                        "status": status,
                    }
                )
    return pd.DataFrame(rows), missing


def audit_manifest_consistency(baseline_root: Path, seed: int) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    cols = ["p_mask_test", "rep", "k", "m", "masked_vars"]
    for scenario in SCENARIOS:
        for cs in COHORT_SEEDS:
            ref = None
            ref_model = None
            for model in SNAPSHOT_MODELS:
                mdf = read_manifest(baseline_root, scenario, model, seed, cs)
                if mdf is None:
                    issues.append(
                        {
                            "scenario": scenario,
                            "cohort_seed": cs,
                            "issue": "missing_manifest",
                            "model": model,
                        }
                    )
                    continue
                key = mdf[cols].sort_values(["p_mask_test", "rep", "k"]).reset_index(drop=True)
                if ref is None:
                    ref = key
                    ref_model = model
                elif not key.equals(ref):
                    issues.append(
                        {
                            "scenario": scenario,
                            "cohort_seed": cs,
                            "issue": "manifest_mismatch",
                            "model": model,
                            "reference_model": ref_model,
                        }
                    )
    return issues


def _resolve_cohort_dir(paper_root: Path, scenario: str, cs: int, data_root_override: Path | None) -> Path:
    if data_root_override is not None:
        return data_root_override / scenario / f"cohortseed_{cs:03d}"
    return canonical_cohort_dir(paper_root, scenario, cs)


def audit_mask_application(paper_root: Path, data_root_override: Path | None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        for cs in COHORT_SEEDS:
            cohort_dir = _resolve_cohort_dir(paper_root, scenario, cs, data_root_override)
            test_fp = cohort_dir / "test.csv"
            if not test_fp.exists():
                rows.append(
                    {
                        "scenario": scenario,
                        "cohort_seed": cs,
                        "status": "missing_test_csv",
                    }
                )
                continue

            feat_cont, _, _, mask_pool = get_feature_names(str(cohort_dir))
            df_test = preprocess_filters_synlv(pd.read_csv(test_fp), verbose=False)
            lv = _last_visit(df_test)
            mask_pool_present = [c for c in mask_pool if c in lv.columns]

            last_ids = lv[ID_COL].to_numpy()
            m = int(np.floor(P_MASK_SEVERE * len(last_ids)))
            rng_vars = np.random.default_rng(stable_mask_seed(P_MASK_SEVERE, REP_TEST))
            perm = rng_vars.permutation(mask_pool)
            masked_vars = [str(x) for x in perm[:K_SEVERE]]
            rng_pat = np.random.default_rng(stable_patient_seed(P_MASK_SEVERE, REP_TEST))
            chosen_ids = (
                rng_pat.choice(last_ids, size=m, replace=False) if m > 0 else np.array([], dtype=last_ids.dtype)
            )

            d = lv.copy()
            present = [c for c in masked_vars if c in d.columns]
            changed = 0
            if len(chosen_ids) > 0 and len(present) > 0:
                mrows = d[ID_COL].isin(chosen_ids)
                before = d.loc[mrows, present].copy()
                d.loc[mrows, present] = np.nan
                after = d.loc[mrows, present]
                changed = int((before.notna() & after.isna()).sum().sum())

            rows.append(
                {
                    "scenario": scenario,
                    "cohort_seed": cs,
                    "n_test_patients": int(len(last_ids)),
                    "m_masked_patients": int(m),
                    "mask_pool_size": int(len(mask_pool)),
                    "mask_pool_present": int(len(mask_pool_present)),
                    "masked_vars": ";".join(masked_vars),
                    "masked_vars_present": int(len(present)),
                    "changed_non_nan_to_nan_cells": int(changed),
                    "status": "ok",
                }
            )
    return pd.DataFrame(rows)


def summarize_clean_vs_severe(baseline_root: Path, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        for model in MODELS:
            chunks = []
            for cs in COHORT_SEEDS:
                d = read_raw(baseline_root, scenario, model, seed, cs)
                if d is None or d.empty:
                    continue
                chunks.append(d.assign(cohort_seed=cs))
            if not chunks:
                continue
            df = pd.concat(chunks, ignore_index=True)
            clean = df[np.isclose(df["p_mask_test"].astype(float), 0.0)]
            p_max = float(df["p_mask_test"].max())
            severe = df[np.isclose(df["p_mask_test"].astype(float), p_max) & (df["k"].astype(int) == int(K_SEVERE))]

            c_ibs = float(clean["ibs"].mean())
            s_ibs = float(severe["ibs"].mean())
            c_auc = float(clean["mean_auc"].mean())
            s_auc = float(severe["mean_auc"].mean())
            rows.append(
                {
                    "scenario": scenario,
                    "model": model,
                    "clean_ibs_mean": c_ibs,
                    "severe_ibs_mean": s_ibs,
                    "delta_ibs_severe_minus_clean": s_ibs - c_ibs,
                    "clean_auc_mean": c_auc,
                    "severe_auc_mean": s_auc,
                    "delta_auc_severe_minus_clean": s_auc - c_auc,
                }
            )
    return pd.DataFrame(rows)


def render_report(
    completeness: pd.DataFrame,
    missing: list[tuple[str, str, int]],
    manifest_issues: list[dict[str, object]],
    mask_checks: pd.DataFrame,
    clean_severe: pd.DataFrame,
    out_md: Path,
) -> None:
    ok_n = int((completeness["status"] == "ok").sum())
    total_n = int(len(completeness))
    miss_n = int(len(missing))
    manifest_bad_n = int(len(manifest_issues))

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# SynLV Baseline Pipeline Audit\n\n")
        f.write("## Completeness\n")
        f.write(f"- Expected tasks: {total_n}\n")
        f.write(f"- Completed tasks: {ok_n}\n")
        f.write(f"- Missing tasks: {miss_n}\n\n")

        if miss_n > 0:
            f.write("### Missing task list\n")
            for scenario, model, cs in missing:
                f.write(f"- `{scenario}` / `{model}` / `cs{cs:03d}`\n")
            f.write("\n")

        f.write("## Stress-Manifest Consistency Across Snapshot Models\n")
        f.write(
            "- Checked columns: `p_mask_test`, `rep`, `k`, `m`, `masked_vars` across "
            "`CoxPH`, `RSF`, `CoxTime`, `DeepHit` per scenario/cohort.\n"
        )
        f.write(f"- Manifest issues found: {manifest_bad_n}\n\n")
        if manifest_bad_n > 0:
            f.write("### Manifest issues\n")
            for issue in manifest_issues:
                f.write(f"- {issue}\n")
            f.write("\n")

        f.write("## Mask-Application Checks (Severe setting sanity)\n")
        if not mask_checks.empty:
            min_present = int(mask_checks["mask_pool_present"].min())
            min_changed = int(mask_checks["changed_non_nan_to_nan_cells"].min())
            f.write(f"- Minimum `mask_pool_present`: {min_present}\n")
            f.write(f"- Minimum changed non-NaN -> NaN cells at severe test: {min_changed}\n")
            if min_present < K_SEVERE or min_changed <= 0:
                f.write("- Status: **NOT OK** (some cohorts may not be masked as intended)\n\n")
            else:
                f.write("- Status: **PASS**\n\n")

        f.write("## Clean -> Severe Summary (for reviewer sanity)\n")
        f.write("- `delta_ibs_severe_minus_clean > 0` means worse IBS under severe masking.\n")
        f.write("- `delta_auc_severe_minus_clean < 0` means worse AUC under severe masking.\n\n")
        if not clean_severe.empty:
            pivot_ibs = clean_severe.pivot(index="scenario", columns="model", values="delta_ibs_severe_minus_clean")
            pivot_auc = clean_severe.pivot(index="scenario", columns="model", values="delta_auc_severe_minus_clean")
            f.write("### Delta IBS (severe - clean)\n\n")
            f.write("```\n")
            f.write(pivot_ibs.round(6).to_string())
            f.write("\n```\n\n")
            f.write("### Delta AUC (severe - clean)\n\n")
            f.write("```\n")
            f.write(pivot_auc.round(6).to_string())
            f.write("\n```\n\n")

        f.write("## Bottom line\n")
        if miss_n == 0 and manifest_bad_n == 0 and (not mask_checks.empty) and int(mask_checks["changed_non_nan_to_nan_cells"].min()) > 0:
            f.write(
                "Pipeline checks pass: stress masks are present in data columns, "
                "applied with non-zero effect, and manifests are matched across snapshot models.\n"
            )
        else:
            f.write(
                "One or more pipeline checks failed; inspect CSV artifacts referenced below before using baseline tables.\n"
            )


def main() -> None:
    if not SIM_COMMON_OK:
        raise SystemExit(
            "sim_common.py is not present in this compact public export; "
            "mask-application audit requires full internal benchmark workspace."
        )

    args = parse_args()
    paper_root = Path(args.paper_root).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    data_root_override = Path(args.data_root).resolve() if str(args.data_root).strip() else None
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    completeness, missing = audit_completeness(baseline_root, int(args.seed))
    manifest_issues = audit_manifest_consistency(baseline_root, int(args.seed))
    mask_checks = audit_mask_application(paper_root, data_root_override)
    clean_severe = summarize_clean_vs_severe(baseline_root, int(args.seed))

    fp_report = out_dir / "baseline_pipeline_audit_report.md"
    fp_mask = out_dir / "baseline_pipeline_mask_checks.csv"
    fp_delta = out_dir / "baseline_pipeline_clean_vs_severe.csv"
    fp_status = out_dir / "baseline_pipeline_task_status.csv"

    completeness.to_csv(fp_status, index=False)
    mask_checks.to_csv(fp_mask, index=False)
    clean_severe.to_csv(fp_delta, index=False)
    render_report(completeness, missing, manifest_issues, mask_checks, clean_severe, fp_report)

    print(f"[ok] report  : {fp_report}")
    print(f"[ok] status  : {fp_status}")
    print(f"[ok] masks   : {fp_mask}")
    print(f"[ok] deltas  : {fp_delta}")


if __name__ == "__main__":
    main()

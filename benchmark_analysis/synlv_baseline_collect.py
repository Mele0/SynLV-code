#!/usr/bin/env python3
"""Collect SynLV baselines and compare with SurvLatentODE BASE/DA."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from synlv_release_config import PRIMARY_SCENARIOS_V1

SCENARIOS = list(PRIMARY_SCENARIOS_V1)


def parse_args() -> argparse.Namespace:
    root_default = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline-root",
        default=str(root_default / "Results" / "Experiments" / "SynLV_Baselines"),
    )
    ap.add_argument(
        "--ode-root",
        default=str(root_default / "Results" / "Experiments" / "SynLV"),
    )
    ap.add_argument("--seed", type=int, default=1991)
    return ap.parse_args()


def read_baseline_raw(baseline_root: Path, seed: int) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        scen_dir = baseline_root / scenario
        if not scen_dir.is_dir():
            continue
        for model_dir in sorted([d for d in scen_dir.iterdir() if d.is_dir()]):
            summary_dir = model_dir / f"seed{seed}" / "Summary"
            if not summary_dir.is_dir():
                continue
            for fp in sorted(summary_dir.glob(f"sim_{scenario}_{model_dir.name}_seed{seed}_cs*_raw_results.csv")):
                try:
                    d = pd.read_csv(fp)
                except Exception:
                    continue
                d["scenario"] = scenario
                d["model"] = model_dir.name
                d["source"] = "baseline"
                rows.append(d)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["p_mask_test"] = pd.to_numeric(out["p_mask_test"], errors="coerce")
    out["k"] = pd.to_numeric(out["k"], errors="coerce")
    out["ibs"] = pd.to_numeric(out["ibs"], errors="coerce")
    out["mean_auc"] = pd.to_numeric(out["mean_auc"], errors="coerce")
    return out.dropna(subset=["p_mask_test", "k", "ibs", "mean_auc"])


def read_ode_raw(ode_root: Path, seed: int) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        for mode in ["BASE", "DA"]:
            summary_dir = ode_root / scenario / mode / f"seed{seed}" / "Summary"
            if not summary_dir.is_dir():
                continue
            for fp in sorted(summary_dir.glob(f"sim_{scenario}_{mode}_seed{seed}_cs*_raw_results.csv")):
                try:
                    d = pd.read_csv(fp)
                except Exception:
                    continue
                d["scenario"] = scenario
                d["model"] = f"SurvLatentODE_{mode}"
                d["source"] = "ode"
                rows.append(d)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["p_mask_test"] = pd.to_numeric(out["p_mask_test"], errors="coerce")
    out["k"] = pd.to_numeric(out["k"], errors="coerce")
    out["ibs"] = pd.to_numeric(out["ibs"], errors="coerce")
    out["mean_auc"] = pd.to_numeric(out["mean_auc"], errors="coerce")
    return out.dropna(subset=["p_mask_test", "k", "ibs", "mean_auc"])


def summarize_conditions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    p_max = float(df["p_mask_test"].max())
    k_max = float(df["k"].max())

    cond_parts = []
    clean = df[np.isclose(df["p_mask_test"], 0.0)].copy()
    if not clean.empty:
        x = clean.groupby(["scenario", "model"], as_index=False)[["ibs", "mean_auc"]].mean()
        x["condition"] = "clean"
        cond_parts.append(x)

    stress = df[df["p_mask_test"] > 0.0].copy()
    if not stress.empty:
        x = stress.groupby(["scenario", "model"], as_index=False)[["ibs", "mean_auc"]].mean()
        x["condition"] = "grid_avg"
        cond_parts.append(x)

    severe = df[(np.isclose(df["p_mask_test"], p_max)) & (np.isclose(df["k"], k_max))].copy()
    if not severe.empty:
        x = severe.groupby(["scenario", "model"], as_index=False)[["ibs", "mean_auc"]].mean()
        x["condition"] = "severe"
        cond_parts.append(x)

    if not cond_parts:
        return pd.DataFrame()
    out = pd.concat(cond_parts, ignore_index=True)
    out = out[["scenario", "condition", "model", "ibs", "mean_auc"]]
    return out.sort_values(["scenario", "condition", "ibs", "mean_auc"], ascending=[True, True, True, False])


def main() -> None:
    args = parse_args()
    baseline_root = Path(args.baseline_root).resolve()
    ode_root = Path(args.ode_root).resolve()
    seed = int(args.seed)

    b = read_baseline_raw(baseline_root, seed)
    o = read_ode_raw(ode_root, seed)
    all_df = pd.concat([x for x in [b, o] if not x.empty], ignore_index=True) if (not b.empty or not o.empty) else pd.DataFrame()
    if all_df.empty:
        print("No baseline/ODE raw results found.")
        return

    out_dir = baseline_root / f"seed{seed}" / "Summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_fp = out_dir / "synlv_baseline_vs_ode_all_raw.csv"
    all_df.to_csv(raw_fp, index=False)

    by_cell = (
        all_df.groupby(["scenario", "model", "p_mask_test", "k"], as_index=False)[["ibs", "mean_auc"]]
        .mean()
        .sort_values(["scenario", "model", "p_mask_test", "k"])
    )
    by_cell_fp = out_dir / "synlv_baseline_vs_ode_by_cell.csv"
    by_cell.to_csv(by_cell_fp, index=False)

    cond = summarize_conditions(all_df)
    cond_fp = out_dir / "synlv_baseline_vs_ode_conditions.csv"
    cond.to_csv(cond_fp, index=False)

    print(f"[ok] saved raw       : {raw_fp}")
    print(f"[ok] saved by-cell   : {by_cell_fp}")
    print(f"[ok] saved conditions: {cond_fp}")

    # Console ranking snapshot for fast sanity checks.
    for scenario in SCENARIOS:
        sub = cond[(cond["scenario"] == scenario) & (cond["condition"].isin(["clean", "severe"]))].copy()
        if sub.empty:
            continue
        print("\n" + "=" * 72)
        print(scenario)
        for condition in ["clean", "severe"]:
            s = sub[sub["condition"] == condition].sort_values(["ibs", "mean_auc"], ascending=[True, False])
            if s.empty:
                continue
            print(f"  {condition}:")
            for _, r in s.iterrows():
                print(
                    "    {model:<22s} IBS={ibs:.5f}  mean_auc={auc:.5f}".format(
                        model=str(r["model"]),
                        ibs=float(r["ibs"]),
                        auc=float(r["mean_auc"]),
                    )
                )


if __name__ == "__main__":
    main()

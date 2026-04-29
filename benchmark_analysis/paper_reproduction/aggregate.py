"""Aggregation functions for SynLV paper-result reproduction."""

from __future__ import annotations

import numpy as np
import pandas as pd

SCENARIO_ORDER = ["Reference", "MNAR Mild", "MNAR Strong", "MNAR Corr", "MAR", "VisitShift", "VisitShift-Transfer", "Mismatch", "Mismatch-Transfer"]
MODEL_ORDER = ["CoxPH", "RSF", "CoxTime", "CoxTV", "DeepHit", "GRUMaskCox", "SurvLatentODE"]
MODEL_ORDER_WITH_MICE = ["CoxPH", "MICE-CoxPH", "RSF", "CoxTime", "CoxTV", "DeepHit", "GRUMaskCox", "SurvLatentODE"]

SCENARIO_DISPLAY = {
    "scenario_A": "Reference", "reference": "Reference", "Reference": "Reference",
    "scenario_B": "MNAR Mild", "missing_not_at_random_mild": "MNAR Mild", "MNAR Mild": "MNAR Mild",
    "scenario_C": "MNAR Strong", "missing_not_at_random_strong": "MNAR Strong", "MNAR Strong": "MNAR Strong",
    "scenario_MNAR_CORRELATED": "MNAR Corr", "missing_not_at_random_corr": "MNAR Corr", "MNAR Correlated": "MNAR Corr", "MNAR Corr": "MNAR Corr",
    "scenario_MAR": "MAR", "missing_at_random": "MAR", "MAR": "MAR",
    "scenario_VISITSHIFT": "VisitShift", "visitshift": "VisitShift", "VisitShift": "VisitShift",
    "scenario_VISITSHIFT_TRANSFER": "VisitShift-Transfer", "visitshift_transfer": "VisitShift-Transfer", "VisitShift-Transfer": "VisitShift-Transfer",
    "scenario_MISMATCH": "Mismatch", "mismatch": "Mismatch", "Mismatch": "Mismatch",
    "scenario_MISMATCH_TRANSFER": "Mismatch-Transfer", "mismatch_transfer": "Mismatch-Transfer", "Mismatch-Transfer": "Mismatch-Transfer",
}
MODEL_DISPLAY = {
    "coxph": "CoxPH", "CoxPH": "CoxPH", "mice_coxph": "MICE-CoxPH", "MICE-CoxPH": "MICE-CoxPH", "MICE_CoxPH": "MICE-CoxPH",
    "rsf": "RSF", "RSF": "RSF", "coxtime": "CoxTime", "CoxTime": "CoxTime", "coxtv": "CoxTV", "CoxTV": "CoxTV",
    "deephit": "DeepHit", "DeepHit": "DeepHit", "grumaskcox": "GRUMaskCox", "GRUMaskCox": "GRUMaskCox",
    "SurvLatentODE": "SurvLatentODE", "SurvLatentODE_BASE": "SurvLatentODE", "BASE": "SurvLatentODE",
}
COLUMN_ALIASES = {
    "scenario": ("scenario", "scenario_name", "subset", "hf_subset", "regime"),
    "model": ("model", "model_name", "method", "estimator"),
    "cohort_seed": ("cohort_seed", "cohortseed", "cs", "seed_cohort"),
    "rep": ("rep", "repetition", "stress_rep", "mask_rep", "repeat"),
    "p": ("p", "prevalence", "p_mask", "p_mask_test", "mask_prevalence"),
    "K": ("K", "k", "severity", "n_masked", "mask_k"),
    "ibs": ("ibs", "IBS", "integrated_brier", "integrated_brier_score"),
    "auc": ("auc", "AUC", "mean_auc", "td_auc", "time_dependent_auc"),
}


def _find_column(columns: list[str], logical_name: str) -> str | None:
    for candidate in COLUMN_ALIASES[logical_name]:
        if candidate in columns:
            return candidate
    lower = {c.lower(): c for c in columns}
    for candidate in COLUMN_ALIASES[logical_name]:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def normalize_stress_results(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize long-form stress results to scenario/model/seed/rep/p/K/ibs/auc columns."""
    columns = list(df.columns)
    mapping: dict[str, str] = {}
    missing: list[str] = []
    for logical in ("scenario", "model", "cohort_seed", "p", "K", "ibs", "auc"):
        found = _find_column(columns, logical)
        if found is None:
            missing.append(logical)
        else:
            mapping[found] = logical
    if missing:
        raise ValueError(f"Stress result table is missing logical columns: {missing}")
    rep_col = _find_column(columns, "rep")
    if rep_col is not None:
        mapping[rep_col] = "rep"
    out = df.rename(columns=mapping).copy()
    if "rep" not in out.columns:
        out["rep"] = 0
    out = out[["scenario", "model", "cohort_seed", "rep", "p", "K", "ibs", "auc"]].copy()
    out["scenario"] = out["scenario"].map(lambda x: SCENARIO_DISPLAY.get(str(x), str(x)))
    out["model"] = out["model"].map(lambda x: MODEL_DISPLAY.get(str(x), str(x)))
    for col in ("cohort_seed", "rep", "p", "K", "ibs", "auc"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["scenario", "model", "cohort_seed", "rep", "p", "K", "ibs", "auc"])
    out["cohort_seed"] = out["cohort_seed"].astype(int)
    out["rep"] = out["rep"].astype(int)
    return out


def add_clean_degradation(df: pd.DataFrame) -> pd.DataFrame:
    """Add clean-referenced D_IBS and D_AUC columns to normalized stress results."""
    clean = df[(np.isclose(df["p"], 0.0)) & (np.isclose(df["K"], 0.0))].copy()
    if clean.empty:
        clean = df[np.isclose(df["p"], 0.0)].copy()
    if clean.empty:
        raise ValueError("No clean rows found. Expected p=0 and preferably K=0.")
    clean_ref = clean.groupby(["scenario", "model", "cohort_seed"], as_index=False)[["ibs", "auc"]].mean().rename(columns={"ibs": "ibs_clean", "auc": "auc_clean"})
    merged = df.merge(clean_ref, on=["scenario", "model", "cohort_seed"], how="left")
    if merged[["ibs_clean", "auc_clean"]].isna().any().any():
        raise ValueError("Some stress rows could not be matched to a clean reference by scenario/model/cohort_seed.")
    merged["D_IBS"] = merged["ibs"] - merged["ibs_clean"]
    merged["D_AUC"] = merged["auc_clean"] - merged["auc"]
    return merged


def seed_level_degradation(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Compute one degradation estimate per scenario/model/cohort_seed for grid_avg or severe."""
    d = add_clean_degradation(df)
    if condition == "grid_avg":
        subset = d[d["p"] > 0].copy()
    elif condition == "severe":
        subset = d[np.isclose(d["p"], d["p"].max()) & np.isclose(d["K"], d["K"].max())].copy()
    else:
        raise ValueError(f"Unknown condition: {condition}")
    if subset.empty:
        raise ValueError(f"No rows available for condition={condition}")
    return subset.groupby(["scenario", "model", "cohort_seed"], as_index=False)[["D_IBS", "D_AUC"]].mean().assign(condition=condition)


def summarize_condition(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Summarize seed-level degradation as mean and SD over cohort seeds."""
    seed = seed_level_degradation(df, condition)
    out = seed.groupby(["scenario", "model", "condition"], as_index=False).agg(D_IBS_mean=("D_IBS", "mean"), D_IBS_sd=("D_IBS", "std"), D_AUC_mean=("D_AUC", "mean"), D_AUC_sd=("D_AUC", "std"), n_cohort_seeds=("cohort_seed", "nunique"))
    out["D_IBS_x1e3_mean"] = out["D_IBS_mean"] * 1000.0
    out["D_IBS_x1e3_sd"] = out["D_IBS_sd"] * 1000.0
    out["D_AUC_pp_mean"] = out["D_AUC_mean"] * 100.0
    out["D_AUC_pp_sd"] = out["D_AUC_sd"] * 100.0
    return order_rows(out)


def uncertainty_summary(df: pd.DataFrame) -> pd.DataFrame:
    grid = summarize_condition(df, "grid_avg")
    sev = summarize_condition(df, "severe")
    keep = ["scenario", "model", "n_cohort_seeds", "D_IBS_x1e3_mean", "D_IBS_x1e3_sd", "D_AUC_pp_mean", "D_AUC_pp_sd"]
    grid = grid[keep].rename(columns={"n_cohort_seeds": "grid_n_cohort_seeds", "D_IBS_x1e3_mean": "grid_D_IBS_x1e3_mean", "D_IBS_x1e3_sd": "grid_D_IBS_x1e3_sd", "D_AUC_pp_mean": "grid_D_AUC_pp_mean", "D_AUC_pp_sd": "grid_D_AUC_pp_sd"})
    sev = sev[keep].rename(columns={"n_cohort_seeds": "severe_n_cohort_seeds", "D_IBS_x1e3_mean": "severe_D_IBS_x1e3_mean", "D_IBS_x1e3_sd": "severe_D_IBS_x1e3_sd", "D_AUC_pp_mean": "severe_D_AUC_pp_mean", "D_AUC_pp_sd": "severe_D_AUC_pp_sd"})
    return order_rows(grid.merge(sev, on=["scenario", "model"], how="outer"))


def mice_minus_coxph(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MICE-CoxPH minus CoxPH paired contrasts for Grid avg and Severe."""
    rows = []
    for condition in ("grid_avg", "severe"):
        source = df[df["p"] > 0].copy() if condition == "grid_avg" else df[np.isclose(df["p"], df["p"].max()) & np.isclose(df["K"], df["K"].max())].copy()
        left = source[source["model"] == "MICE-CoxPH"]
        right = source[source["model"] == "CoxPH"]
        paired = left.merge(right, on=["scenario", "cohort_seed", "rep", "p", "K"], suffixes=("_mice", "_coxph"))
        if paired.empty:
            continue
        paired["delta_ibs"] = paired["ibs_mice"] - paired["ibs_coxph"]
        paired["delta_auc_pp"] = (paired["auc_mice"] - paired["auc_coxph"]) * 100.0
        seed = paired.groupby(["scenario", "cohort_seed"], as_index=False)[["delta_ibs", "delta_auc_pp"]].mean()
        agg = seed.groupby("scenario", as_index=False).agg(delta_ibs=("delta_ibs", "mean"), delta_auc_pp=("delta_auc_pp", "mean"), n_cohort_seeds=("cohort_seed", "nunique"))
        agg["condition"] = condition
        rows.append(agg)
    if not rows:
        raise ValueError("Could not pair MICE-CoxPH and CoxPH rows.")
    out = pd.concat(rows, ignore_index=True)
    wide = out.pivot(index="scenario", columns="condition", values=["delta_ibs", "delta_auc_pp", "n_cohort_seeds"])
    wide.columns = [f"{condition}_{metric}" for metric, condition in wide.columns]
    return order_rows(wide.reset_index())


def order_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_scenario_order"] = out["scenario"].map({name: i for i, name in enumerate(SCENARIO_ORDER)}).fillna(999) if "scenario" in out.columns else 999
    out["_model_order"] = out["model"].map({name: i for i, name in enumerate(MODEL_ORDER_WITH_MICE)}).fillna(999) if "model" in out.columns else 999
    sort_cols = [c for c in ["_scenario_order", "_model_order", "scenario", "model"] if c in out.columns]
    return out.sort_values(sort_cols).drop(columns=["_scenario_order", "_model_order"], errors="ignore").reset_index(drop=True)

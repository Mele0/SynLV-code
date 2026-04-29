"""Standalone self-test for the paper reproduction aggregation layer."""

from __future__ import annotations

import math
import pandas as pd

from .aggregate import add_clean_degradation, normalize_stress_results, seed_level_degradation, summarize_condition, uncertainty_summary
from .format import compact_cell
from .targets import DESCRIPTIVE_EXCLUSIONS, TARGET_BY_ID


def fake_stress_results() -> pd.DataFrame:
    rows = []
    for scenario in ["scenario_B", "scenario_C"]:
        for model in ["CoxPH", "RSF"]:
            model_offset = 0.01 if model == "RSF" else 0.0
            for cohort_seed in [0, 1]:
                for rep in [0, 1]:
                    base_ibs = 0.10 + model_offset + cohort_seed * 0.01 + rep * 0.001
                    base_auc = 0.80 - model_offset - cohort_seed * 0.01 - rep * 0.002
                    rows.append({"scenario": scenario, "model": model, "cohort_seed": cohort_seed, "rep": rep, "p_mask_test": 0.0, "k": 0, "ibs": base_ibs, "mean_auc": base_auc})
                    for p in [0.5, 1.0]:
                        for k in [1, 2]:
                            rows.append({"scenario": scenario, "model": model, "cohort_seed": cohort_seed, "rep": rep, "p_mask_test": p, "k": k, "ibs": base_ibs + p * 0.02 + k * 0.003, "mean_auc": base_auc - p * 0.05 - k * 0.010})
    return pd.DataFrame(rows)


def assert_close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError(f"{name}: actual={actual} expected={expected}")


def run_selftest() -> None:
    if "table5_grid_avg" not in TARGET_BY_ID or "table6_severe" not in TARGET_BY_ID:
        raise AssertionError("Target registry missing compact benchmark targets")
    if "tab:stress_suite_definitions" not in DESCRIPTIVE_EXCLUSIONS:
        raise AssertionError("Descriptive exclusions missing stress-suite definition table")
    raw = fake_stress_results()
    norm = normalize_stress_results(raw)
    deg = add_clean_degradation(norm)
    if {"D_IBS", "D_AUC"} - set(deg.columns):
        raise AssertionError("Degradation columns were not added")
    grid_seed = seed_level_degradation(norm, "grid_avg")
    severe_seed = seed_level_degradation(norm, "severe")
    sample_grid = grid_seed[(grid_seed["scenario"] == "MNAR Mild") & (grid_seed["model"] == "CoxPH") & (grid_seed["cohort_seed"] == 0)].iloc[0]
    sample_severe = severe_seed[(severe_seed["scenario"] == "MNAR Mild") & (severe_seed["model"] == "CoxPH") & (severe_seed["cohort_seed"] == 0)].iloc[0]
    assert_close(sample_grid["D_IBS"], 0.0195, "grid D_IBS")
    assert_close(sample_grid["D_AUC"], 0.0525, "grid D_AUC")
    assert_close(sample_severe["D_IBS"], 0.026, "severe D_IBS")
    assert_close(sample_severe["D_AUC"], 0.070, "severe D_AUC")
    grid_summary = summarize_condition(norm, "grid_avg")
    sample_summary = grid_summary[(grid_summary["scenario"] == "MNAR Mild") & (grid_summary["model"] == "CoxPH")].iloc[0]
    assert_close(sample_summary["D_IBS_x1e3_mean"], 19.5, "scaled grid D_IBS")
    assert_close(sample_summary["D_AUC_pp_mean"], 5.25, "scaled grid D_AUC")
    if int(sample_summary["n_cohort_seeds"]) != 2:
        raise AssertionError("n_cohort_seeds was not propagated")
    uncertainty = uncertainty_summary(norm)
    if uncertainty.empty or "severe_D_AUC_pp_mean" not in uncertainty.columns:
        raise AssertionError("Uncertainty summary did not produce expected columns")
    cell = compact_cell(sample_summary)
    if cell != "+19.5/+5.25":
        raise AssertionError(f"Unexpected compact cell formatting: {cell}")
    print("paper reproduction self-test passed")

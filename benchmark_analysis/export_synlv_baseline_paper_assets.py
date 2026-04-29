#!/usr/bin/env python3
"""Export paper-ready baseline vs SurvLatentODE assets for SynLV.

Outputs (under --out-dir):
  - baseline_vs_ode_by_cohort.csv
  - baseline_vs_ode_summary.csv
  - baseline_vs_ode_coverage.csv
  - table_baseline_vs_ode_conditions.tex
  - baseline_vs_ode_compact_conditions.pdf
  - baseline_vs_ode_compact_conditions.svg

Design:
  1) Read baseline raw results from SynLV_Baselines/<scenario>/<model>/seed*/Summary.
  2) Read ODE raw results from SynLV/<scenario>/{BASE,DA}/seed*/Summary.
  3) Aggregate in two stages:
     - within cohort seed, per condition (clean / grid_avg / severe)
     - across cohort seeds (mean, sd)
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter

from synlv_release_config import PRIMARY_SCENARIOS_V1

SCENARIOS = list(PRIMARY_SCENARIOS_V1)

SCENARIO_SHORT = {
    "scenario_A": "A",
    "scenario_B": "B",
    "scenario_C": "C",
    "scenario_MNAR_CORRELATED": "MNAR Corr",
    "scenario_MAR": "MAR",
    "scenario_VISITSHIFT": "VisitShift",
    "scenario_VISITSHIFT_TRANSFER": "VisitShift-Transfer",
    "scenario_MISMATCH": "Mismatch",
    "scenario_MISMATCH_TRANSFER": "Mismatch-Transfer",
}

# Plot-only labels (clearer for paper figures).
SCENARIO_PLOT_LABEL = {
    "scenario_A": "Clean",
    "scenario_B": "MNAR mild",
    "scenario_C": "MNAR strong",
    "scenario_MNAR_CORRELATED": "MNAR corr",
    "scenario_MAR": "MAR control",
    "scenario_VISITSHIFT": "VisitShift",
    "scenario_VISITSHIFT_TRANSFER": "VisitShift-transfer",
    "scenario_MISMATCH": "Mismatch",
    "scenario_MISMATCH_TRANSFER": "Mismatch-transfer",
}

CONDITION_ORDER = ["clean", "grid_avg", "severe"]
CONDITION_LABEL = {
    "clean": "Clean",
    "grid_avg": "Grid avg (p>0)",
    "severe": "Severe (p=max, k=max)",
}


def _register_sf_pro_font() -> tuple[str | None, Path | None]:
    """Register local SF Pro font if available and return (family_name, font_path)."""
    candidates = [
        Path(__file__).resolve().parents[1] / "Sample_Efficiency" / "SF-Pro.ttf",
        Path(__file__).resolve().parents[1] / "Sample_Efficiency" / "SF-Pro.otf",
        Path(__file__).resolve().parents[1] / "Sample_Efficiency" / "SF Pro.ttf",
    ]
    for fp in candidates:
        if fp.exists():
            try:
                fm.fontManager.addfont(str(fp))
                return fm.FontProperties(fname=str(fp)).get_name(), fp
            except Exception:
                continue
    return None, None


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
    ap.add_argument(
        "--out-dir",
        default=str(root_default / "Results" / "Final" / "Tables" / "Baselines_SynLV"),
    )
    ap.add_argument("--seed", type=int, default=1991)
    ap.add_argument("--expected-cohorts", type=int, default=5)
    return ap.parse_args()


def _read_csv_safe(fp: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(fp)
    except Exception:
        return None


def _extract_cs_from_name(name: str) -> int | None:
    m = re.search(r"_cs([0-9]{3})", name)
    if not m:
        return None
    return int(m.group(1))


def _prepare_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["p_mask_test", "k", "ibs", "mean_auc", "cohort_seed"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def read_baseline_raw(baseline_root: Path, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for scenario in SCENARIOS:
        scen_dir = baseline_root / scenario
        if not scen_dir.is_dir():
            continue
        for model_dir in sorted([d for d in scen_dir.iterdir() if d.is_dir()]):
            summary_dir = model_dir / f"seed{seed}" / "Summary"
            if not summary_dir.is_dir():
                continue
            pattern = f"sim_{scenario}_{model_dir.name}_seed{seed}_cs*_raw_results.csv"
            for fp in sorted(summary_dir.glob(pattern)):
                d = _read_csv_safe(fp)
                if d is None or d.empty:
                    continue
                d = _prepare_numeric(d)
                if "cohort_seed" not in d.columns or d["cohort_seed"].isna().all():
                    cs = _extract_cs_from_name(fp.name)
                    d["cohort_seed"] = cs
                d["scenario"] = scenario
                d["model"] = model_dir.name
                d["source"] = "baseline"
                rows.append(d)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    keep = ["scenario", "model", "source", "cohort_seed", "p_mask_test", "k", "ibs", "mean_auc"]
    out = out[keep].dropna(subset=["scenario", "model", "cohort_seed", "p_mask_test", "k", "ibs", "mean_auc"])
    out["cohort_seed"] = out["cohort_seed"].astype(int)
    return out


def read_ode_raw(ode_root: Path, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for scenario in SCENARIOS:
        for mode in ["BASE", "DA"]:
            summary_dir = ode_root / scenario / mode / f"seed{seed}" / "Summary"
            if not summary_dir.is_dir():
                continue
            pattern = f"sim_{scenario}_{mode}_seed{seed}_cs*_raw_results.csv"
            for fp in sorted(summary_dir.glob(pattern)):
                d = _read_csv_safe(fp)
                if d is None or d.empty:
                    continue
                d = _prepare_numeric(d)
                if "cohort_seed" not in d.columns or d["cohort_seed"].isna().all():
                    cs = _extract_cs_from_name(fp.name)
                    d["cohort_seed"] = cs
                d["scenario"] = scenario
                d["model"] = f"SurvLatentODE_{mode}"
                d["source"] = "ode"
                rows.append(d)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    keep = ["scenario", "model", "source", "cohort_seed", "p_mask_test", "k", "ibs", "mean_auc"]
    out = out[keep].dropna(subset=["scenario", "model", "cohort_seed", "p_mask_test", "k", "ibs", "mean_auc"])
    out["cohort_seed"] = out["cohort_seed"].astype(int)
    return out


def label_condition(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    p_max = float(out["p_mask_test"].max())
    k_max = float(out["k"].max())
    out["condition"] = "grid_avg"
    out.loc[np.isclose(out["p_mask_test"], 0.0), "condition"] = "clean"
    out.loc[np.isclose(out["p_mask_test"], p_max) & np.isclose(out["k"], k_max), "condition"] = "severe"
    return out


def summarize_by_cohort(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    labeled = label_condition(df)
    parts = []

    clean = labeled[labeled["condition"] == "clean"]
    if not clean.empty:
        x = (
            clean.groupby(["scenario", "model", "source", "cohort_seed"], as_index=False)[["ibs", "mean_auc"]]
            .mean()
            .assign(condition="clean")
        )
        parts.append(x)

    grid = labeled[labeled["p_mask_test"] > 0.0]
    if not grid.empty:
        x = (
            grid.groupby(["scenario", "model", "source", "cohort_seed"], as_index=False)[["ibs", "mean_auc"]]
            .mean()
            .assign(condition="grid_avg")
        )
        parts.append(x)

    severe = labeled[labeled["condition"] == "severe"]
    if not severe.empty:
        x = (
            severe.groupby(["scenario", "model", "source", "cohort_seed"], as_index=False)[["ibs", "mean_auc"]]
            .mean()
            .assign(condition="severe")
        )
        parts.append(x)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out["condition"] = pd.Categorical(out["condition"], categories=CONDITION_ORDER, ordered=True)
    return out.sort_values(["scenario", "condition", "model", "cohort_seed"]).reset_index(drop=True)


def summarize_over_cohorts(by_cohort: pd.DataFrame) -> pd.DataFrame:
    if by_cohort.empty:
        return by_cohort
    # Avoid pandas categorical groupby edge-cases by grouping on plain strings.
    bc = by_cohort.copy()
    bc["condition"] = bc["condition"].astype(str)
    g = (
        bc.groupby(["scenario", "condition", "model", "source"], as_index=False, observed=True)
        .agg(
            ibs_mean=("ibs", "mean"),
            ibs_sd=("ibs", "std"),
            auc_mean=("mean_auc", "mean"),
            auc_sd=("mean_auc", "std"),
            n_cohorts=("cohort_seed", "nunique"),
        )
        .sort_values(["scenario", "condition", "ibs_mean", "auc_mean"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )
    g["ibs_sd"] = g["ibs_sd"].fillna(0.0)
    g["auc_sd"] = g["auc_sd"].fillna(0.0)
    return g


def build_coverage(summary: pd.DataFrame, expected_cohorts: int) -> pd.DataFrame:
    if summary.empty:
        return summary
    c = summary.copy()
    c["complete"] = c["n_cohorts"] >= int(expected_cohorts)
    c["scenario_short"] = c["scenario"].map(lambda s: SCENARIO_SHORT.get(s, s))
    c["condition_label"] = c["condition"].map(lambda s: CONDITION_LABEL.get(s, s))
    return c


def _fmt_pm(mean: float, sd: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} +/- {sd:.{digits}f}"


def build_latex_table(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "% empty: no baseline/ODE summary rows found\n"

    lines: list[str] = []
    lines.append(r"\begin{tabular}{lllrcc}")
    lines.append(r"\toprule")
    lines.append(r"Scenario & Condition & Model & $n_{\mathrm{coh}}$ & IBS (mean$\pm$sd) & Mean AUC (mean$\pm$sd) \\")
    lines.append(r"\midrule")

    last_scen = None
    for _, r in summary.iterrows():
        scen = SCENARIO_SHORT.get(str(r["scenario"]), str(r["scenario"]))
        cond = CONDITION_LABEL.get(str(r["condition"]), str(r["condition"]))
        model = str(r["model"])
        n_coh = int(r["n_cohorts"])
        ibs = _fmt_pm(float(r["ibs_mean"]), float(r["ibs_sd"]), digits=4)
        auc = _fmt_pm(float(r["auc_mean"]), float(r["auc_sd"]), digits=4)

        scen_cell = scen if scen != last_scen else ""
        lines.append(f"{scen_cell} & {cond} & {model} & {n_coh} & {ibs} & {auc} \\\\")
        last_scen = scen

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def make_compact_plot(summary: pd.DataFrame, out_pdf: Path, out_svg: Path) -> None:
    if summary.empty:
        return

    # Keep only models that have at least one row.
    models = list(summary["model"].dropna().astype(str).unique())
    if not models:
        return

    conds = ["grid_avg", "severe"]
    plot_df = summary[summary["condition"].isin(conds)].copy()
    if plot_df.empty:
        return

    scenarios = [s for s in SCENARIOS if s in set(plot_df["scenario"])]
    y_base = np.arange(len(scenarios), dtype=float)

    # Deterministic palette for readability across reruns.
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    color = {m: palette[i % len(palette)] for i, m in enumerate(models)}

    sf_name, sf_path = _register_sf_pro_font()
    rc = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": [sf_name, "Arial", "DejaVu Sans"] if sf_name else ["SF Pro", "Arial", "DejaVu Sans"],
        "font.weight": "light",
        "axes.labelweight": "light",
        "axes.titleweight": "light",
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(14, 9),
            sharey=True,
            constrained_layout=False,
        )
        metric_cfg = [
            ("ibs_mean", "ibs_sd", "IBS", 1.0),
            ("auc_mean", "auc_sd", "Mean time-dependent AUC (%)", 100.0),
        ]

        for ridx, cond in enumerate(conds):
            d_cond = plot_df[plot_df["condition"] == cond].copy()
            for cidx, (mcol, scol, xlabel, scale) in enumerate(metric_cfg):
                ax = axes[ridx, cidx]
                offsets = np.linspace(-0.28, 0.28, num=max(1, len(models)))
                for mi, model in enumerate(models):
                    dm = d_cond[d_cond["model"] == model].set_index("scenario")
                    x = []
                    s = []
                    y = []
                    for yi, scen in enumerate(scenarios):
                        if scen not in dm.index:
                            continue
                        row = dm.loc[scen]
                        if isinstance(row, pd.DataFrame):
                            row = row.iloc[0]
                        x.append(scale * float(row[mcol]))
                        s.append(scale * float(row[scol]))
                        y.append(y_base[yi] + offsets[mi])
                    if not x:
                        continue
                    ax.errorbar(
                        x,
                        y,
                        xerr=s,
                        fmt="o",
                        color=color[model],
                        markersize=5,
                        elinewidth=1,
                        capsize=2,
                        alpha=0.95,
                        label=model if (ridx == 0 and cidx == 0) else None,
                    )

                ax.set_yticks(y_base)
                ax.set_yticklabels([SCENARIO_PLOT_LABEL.get(s, s) for s in scenarios])
                ax.grid(axis="x", alpha=0.25, linestyle="-", linewidth=0.8)
                # Keep top row clean; x labels only on bottom row.
                ax.set_xlabel(xlabel if ridx == 1 else "")
                if cidx == 0:
                    ax.set_ylabel("Scenario")
                if mcol == "auc_mean":
                    # Integer tick labels (no decimals).
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))

                # Fixed symmetric x-ranges across rows/panels.
                if mcol == "auc_mean":
                    auc_ticks = [65, 70, 75, 80, 85, 90, 95, 100]
                    ax.set_xticks(auc_ticks)
                    ax.set_xlim(60, 100)
                elif mcol == "ibs_mean":
                    ibs_ticks = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
                    ax.set_xticks(ibs_ticks)
                    ax.set_xlim(0.00, 0.36)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.015),
                ncol=min(3, len(labels)),
                frameon=False,
                fontsize=10,
            )
        fig.subplots_adjust(bottom=0.14, top=0.95, wspace=0.08, hspace=0.22)

        # Force SF-Pro Light on all text elements when available.
        if sf_path is not None and sf_path.exists():
            sf_light = fm.FontProperties(fname=str(sf_path), weight="light")
            for t in fig.findobj(match=Text):
                t.set_fontproperties(sf_light)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    baseline_root = Path(args.baseline_root).resolve()
    ode_root = Path(args.ode_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    seed = int(args.seed)

    b = read_baseline_raw(baseline_root, seed)
    o = read_ode_raw(ode_root, seed)
    all_raw = pd.concat([x for x in [b, o] if not x.empty], ignore_index=True) if (not b.empty or not o.empty) else pd.DataFrame()
    if all_raw.empty:
        raise SystemExit("No baseline/ODE raw results found.")

    by_cohort = summarize_by_cohort(all_raw)
    summary = summarize_over_cohorts(by_cohort)
    coverage = build_coverage(summary, expected_cohorts=int(args.expected_cohorts))

    out_dir.mkdir(parents=True, exist_ok=True)
    by_cohort_fp = out_dir / "baseline_vs_ode_by_cohort.csv"
    summary_fp = out_dir / "baseline_vs_ode_summary.csv"
    coverage_fp = out_dir / "baseline_vs_ode_coverage.csv"
    latex_fp = out_dir / "table_baseline_vs_ode_conditions.tex"
    fig_pdf = out_dir / "baseline_vs_ode_compact_conditions.pdf"
    fig_svg = out_dir / "baseline_vs_ode_compact_conditions.svg"

    by_cohort.to_csv(by_cohort_fp, index=False)
    summary.to_csv(summary_fp, index=False)
    coverage.to_csv(coverage_fp, index=False)
    latex_fp.write_text(build_latex_table(summary), encoding="utf-8")
    make_compact_plot(summary, fig_pdf, fig_svg)

    print(f"[ok] by-cohort  : {by_cohort_fp}")
    print(f"[ok] summary    : {summary_fp}")
    print(f"[ok] coverage   : {coverage_fp}")
    print(f"[ok] latex      : {latex_fp}")
    print(f"[ok] figure(pdf): {fig_pdf}")
    print(f"[ok] figure(svg): {fig_svg}")

    # Fast console snapshot: best model per scenario/condition by IBS.
    print("\nBest-by-IBS snapshot (mean over cohorts):")
    for scen in SCENARIOS:
        for cond in CONDITION_ORDER:
            s = summary[(summary["scenario"] == scen) & (summary["condition"] == cond)].copy()
            if s.empty:
                continue
            s = s.sort_values(["ibs_mean", "auc_mean"], ascending=[True, False]).reset_index(drop=True)
            top = s.iloc[0]
            print(
                f"  {SCENARIO_SHORT.get(scen, scen):<10s} {cond:<8s} -> "
                f"{top['model']:<20s} IBS={top['ibs_mean']:.4f} AUC={top['auc_mean']:.4f} (n={int(top['n_cohorts'])})"
            )


if __name__ == "__main__":
    main()

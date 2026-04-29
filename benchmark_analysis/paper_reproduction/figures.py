"""Figure generation helpers for paper-result reproduction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_stress_surfaces(df: pd.DataFrame, out_png: Path, out_pdf: Path | None = None) -> None:
    """Plot representative stress surfaces for Figure 2 from degradation rows."""
    import matplotlib.pyplot as plt

    models = ["CoxPH", "RSF", "SurvLatentODE"]
    scenarios = ["MNAR Mild", "MNAR Strong", "VisitShift", "Mismatch"]
    metrics = [("D_IBS_x1e3", "D_IBS x 1e3"), ("D_AUC_pp", "D_AUC pp")]
    data = df[df["model"].isin(models) & df["scenario"].isin(scenarios)].copy()
    if data.empty:
        raise ValueError("No Figure 2 rows after filtering to CoxPH/RSF/SurvLatentODE and MNAR Mild/MNAR Strong/VisitShift/Mismatch.")
    fig, axes = plt.subplots(len(models) * len(metrics), len(scenarios), figsize=(14, 10), constrained_layout=True)
    for i, model in enumerate(models):
        for m, (metric, title) in enumerate(metrics):
            row_idx = i * len(metrics) + m
            metric_data = data[data["model"] == model]
            vmin = metric_data[metric].min()
            vmax = metric_data[metric].max()
            for j, scenario in enumerate(scenarios):
                ax = axes[row_idx][j]
                sub = data[(data["model"] == model) & (data["scenario"] == scenario)]
                if sub.empty:
                    ax.set_axis_off(); continue
                pivot = sub.pivot_table(index="K", columns="p", values=metric, aggfunc="mean").sort_index(ascending=True)
                im = ax.imshow(pivot.values, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
                ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels([f"{x:g}" for x in pivot.columns], rotation=45)
                ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels([f"{x:g}" for x in pivot.index])
                if row_idx == 0: ax.set_title(scenario)
                if j == 0: ax.set_ylabel(f"{model}\n{title}\nK")
                ax.set_xlabel("p")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)

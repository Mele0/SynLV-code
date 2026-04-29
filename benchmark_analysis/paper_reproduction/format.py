"""Formatting helpers for paper-result CSV, Markdown, and LaTeX outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .aggregate import MODEL_ORDER, SCENARIO_ORDER


def signed(value: float, digits: int = 2) -> str:
    return f"{float(value):+.{digits}f}"


def compact_cell(row: pd.Series) -> str:
    return f"{signed(row['D_IBS_x1e3_mean'], 1)}/{signed(row['D_AUC_pp_mean'], 2)}"


def mean_sd_cell(mean: float, sd: float, digits: int = 2) -> str:
    if pd.isna(sd):
        return f"{signed(mean, digits)} $\\pm$ n.a."
    return f"{signed(mean, digits)} $\\pm$ {float(sd):.{digits}f}"


def write_markdown(path: Path, title: str, df: pd.DataFrame, note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if note:
        lines.extend([note, ""])
    lines.extend(markdown_table(df))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> list[str]:
    """Format a DataFrame as a simple GitHub-flavored Markdown table without optional dependencies."""
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return rows


def compact_latex_table(df: pd.DataFrame, *, caption: str, label: str, models: list[str] | None = None) -> str:
    model_order = models or MODEL_ORDER
    pivot = df.pivot_table(index="scenario", columns="model", values="cell", aggfunc="first")
    lines = [r"\begin{table}[h]", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l" + "c" * len(model_order) + r"}", r"\toprule", "Scenario & " + " & ".join(model_order) + r" \\", r"\midrule"]
    for scenario in SCENARIO_ORDER:
        cells = [pivot.loc[scenario, model] if scenario in pivot.index and model in pivot.columns and pd.notna(pivot.loc[scenario, model]) else "--" for model in model_order]
        lines.append(f"{scenario} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
    return "\n".join(lines)


def uncertainty_latex_table(df: pd.DataFrame, *, caption: str, label: str) -> str:
    lines = [r"\begin{table}[h]", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\centering", r"\scriptsize", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{llcccc}", r"\toprule", r"Scenario & Model & Grid $D_{IBS}\times 10^3$ & Grid $D_{AUC}$ pp & Severe $D_{IBS}\times 10^3$ & Severe $D_{AUC}$ pp \\", r"\midrule"]
    for _, row in df.iterrows():
        lines.append(f"{row['scenario']} & {row['model']} & {mean_sd_cell(row['grid_D_IBS_x1e3_mean'], row['grid_D_IBS_x1e3_sd'], 1)} & {mean_sd_cell(row['grid_D_AUC_pp_mean'], row['grid_D_AUC_pp_sd'], 2)} & {mean_sd_cell(row['severe_D_IBS_x1e3_mean'], row['severe_D_IBS_x1e3_sd'], 1)} & {mean_sd_cell(row['severe_D_AUC_pp_mean'], row['severe_D_AUC_pp_sd'], 2)} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
    return "\n".join(lines)


def simple_latex_table(df: pd.DataFrame, *, caption: str, label: str) -> str:
    cols = list(df.columns)
    lines = [r"\begin{table}[h]", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\centering", r"\scriptsize", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{" + "l" * len(cols) + r"}", r"\toprule", " & ".join(str(c).replace("_", r"\_") for c in cols) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(row[c]) for c in cols) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
    return "\n".join(lines)


def write_latex(path: Path, tex: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex, encoding="utf-8")

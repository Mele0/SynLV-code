#!/usr/bin/env python3
"""Reproduce analysis-derived SynLV paper tables and figures from local artifacts."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ANALYSIS = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARK_ANALYSIS))
sys.path.insert(0, str(REPO_ROOT))

from paper_reproduction.aggregate import MODEL_ORDER, MODEL_ORDER_WITH_MICE, add_clean_degradation, mice_minus_coxph, normalize_stress_results, order_rows, summarize_condition, uncertainty_summary
from paper_reproduction.figures import plot_stress_surfaces
from paper_reproduction.format import compact_cell, compact_latex_table, simple_latex_table, uncertainty_latex_table, write_latex, write_markdown
from paper_reproduction.io import CSV_PATTERNS, ReproductionInputError, read_table, write_json, write_manifest
from paper_reproduction.selftest import run_selftest
from paper_reproduction.targets import DESCRIPTIVE_EXCLUSIONS, TARGETS, resolve_target
from synlv_release_config import COHORT_SEEDS, PRIMARY_SCENARIO_KEYS, STRESS_POOL_CHANNELS, get_scenario

DEFAULT_OUT = REPO_ROOT / "docs" / "audit" / "reproduced_paper_results"
LONG_FORM_COLUMNS = ["scenario", "model", "cohort_seed", "p", "K", "ibs", "auc"]


class TargetUnavailable(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce analysis-derived SynLV paper tables and figures from local artifacts.")
    parser.add_argument("--list-targets", action="store_true", help="List supported paper table/figure reproduction targets.")
    parser.add_argument("--self-test", action="store_true", help="Run a standalone aggregation self-test and exit.")
    parser.add_argument("--target", default="", help="Target ID, LaTeX label, or 'all'.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Local SynLV split-file root for dataset-diagnostic targets.")
    parser.add_argument("--results-root", type=Path, default=None, help="Local result artifact root for synthetic benchmark targets.")
    parser.add_argument("--realdata-root", type=Path, default=None, help="Local credentialed real-data summary root.")
    parser.add_argument("--input", type=Path, default=None, help="Explicit input CSV/JSON/Parquet for the selected target.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory for reproduced artifacts.")
    return parser.parse_args()


def list_targets() -> None:
    rows = []
    for target in TARGETS:
        rows.append({"target_id": target.target_id, "latex_label": target.latex_label, "caption/title": target.caption, "status": target.status, "required_inputs": target.required_inputs, "generated_outputs": ", ".join(target.generated_outputs)})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nExcluded descriptive/configuration labels:")
    for label, reason in DESCRIPTIVE_EXCLUSIONS.items():
        print(f"- {label}: {reason}")


def target_paths(outdir: Path, target_id: str) -> dict[str, Path]:
    return {"csv": outdir / f"{target_id}.csv", "tex": outdir / f"{target_id}.tex", "md": outdir / f"{target_id}.md"}


def write_standard_outputs(outdir: Path, target, df: pd.DataFrame, tex: str, title: str, argv: list[str], inputs: list[Path], source_type: str, note: str = "") -> dict[str, Any]:
    paths = target_paths(outdir, target.target_id)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths["csv"], index=False)
    write_latex(paths["tex"], tex)
    write_markdown(paths["md"], title, df, note)
    manifest_path = write_manifest(outdir=outdir, target_id=target.target_id, latex_label=target.latex_label, caption=target.caption, argv=argv, inputs=inputs, outputs=[paths["csv"], paths["tex"], paths["md"]], source_type=source_type, notes=[note] if note else [])
    return {"target_id": target.target_id, "status": "ok", "outputs": [str(paths[k]) for k in ("csv", "tex", "md")] + [str(manifest_path)]}


def load_long_form_results(args: argparse.Namespace, target_id: str) -> tuple[Path, pd.DataFrame]:
    if args.input is not None:
        path = args.input.expanduser().resolve()
        return path, normalize_stress_results(read_table(path))
    if args.results_root is None:
        raise ReproductionInputError(
            f"{target_id} requires long-form synthetic stress results with scenario/model/cohort_seed/p/K/ibs/auc. "
            "Pass --input or --results-root."
        )
    root = args.results_root.expanduser().resolve()
    if not root.exists():
        raise ReproductionInputError(f"Search root for {target_id} does not exist: {root}")
    seen: set[Path] = set()
    matches: list[tuple[Path, pd.DataFrame]] = []
    for pattern in CSV_PATTERNS:
        for path in root.rglob(pattern):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            try:
                normalized = normalize_stress_results(read_table(path))
            except Exception:
                continue
            if not normalized.empty:
                matches.append((path, normalized))
    if not matches:
        raise ReproductionInputError(
            f"No input file for {target_id} under {root} could be normalized to logical columns "
            "scenario/model/cohort_seed/rep/p/K/ibs/auc."
        )
    if len(matches) > 1:
        candidates = "\n".join(f"  - {path}" for path, _ in matches[:25])
        raise ReproductionInputError(f"Multiple candidate inputs for {target_id}; pass --input to choose one:\n{candidates}")
    return matches[0]


def run_compact_table(args: argparse.Namespace, target, condition: str, models: list[str]) -> dict[str, Any]:
    input_path, df = load_long_form_results(args, target.target_id)
    summary = summarize_condition(df, condition)
    summary = summary[summary["model"].isin(models)].copy()
    summary["cell"] = summary.apply(compact_cell, axis=1)
    tex = compact_latex_table(summary, caption=target.caption, label=target.latex_label, models=models)
    return write_standard_outputs(args.out, target, summary, tex, target.target_id, sys.argv, [input_path], "raw long-form results")


def run_uncertainty(args: argparse.Namespace, target) -> dict[str, Any]:
    input_path, df = load_long_form_results(args, target.target_id)
    summary = uncertainty_summary(df)
    summary = summary[summary["model"].isin(MODEL_ORDER)].copy()
    tex = uncertainty_latex_table(summary, caption=target.caption, label=target.latex_label)
    return write_standard_outputs(args.out, target, summary, tex, target.target_id, sys.argv, [input_path], "raw long-form results")


def run_mice_vs_coxph(args: argparse.Namespace, target) -> dict[str, Any]:
    input_path, df = load_long_form_results(args, target.target_id)
    summary = mice_minus_coxph(df)
    tex = simple_latex_table(summary, caption=target.caption, label=target.latex_label)
    return write_standard_outputs(args.out, target, summary, tex, target.target_id, sys.argv, [input_path], "raw long-form results")


def run_figure2(args: argparse.Namespace, target) -> dict[str, Any]:
    input_path, df = load_long_form_results(args, target.target_id)
    degraded = add_clean_degradation(df)
    plot_df = degraded[degraded["p"] > 0].copy()
    plot_df["D_IBS_x1e3"] = plot_df["D_IBS"] * 1000.0
    plot_df["D_AUC_pp"] = plot_df["D_AUC"] * 100.0
    plot_df = plot_df.groupby(["scenario", "model", "p", "K"], as_index=False)[["D_IBS_x1e3", "D_AUC_pp"]].mean().pipe(order_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "figure2_stress_surfaces_data.csv"
    png_path = args.out / "figure2_stress_surfaces.png"
    pdf_path = args.out / "figure2_stress_surfaces.pdf"
    plot_df.to_csv(csv_path, index=False)
    plot_stress_surfaces(plot_df, png_path, pdf_path)
    manifest_path = write_manifest(outdir=args.out, target_id=target.target_id, latex_label=target.latex_label, caption=target.caption, argv=sys.argv, inputs=[input_path], outputs=[csv_path, png_path, pdf_path], source_type="raw long-form results")
    return {"target_id": target.target_id, "status": "ok", "outputs": [str(csv_path), str(png_path), str(pdf_path), str(manifest_path)]}


def _split_path(cohort_dir: Path, split: str) -> Path:
    for suffix in (".csv", ".parquet"):
        path = cohort_dir / f"{split}{suffix}"
        if path.exists():
            return path
    raise TargetUnavailable(f"Missing split file under {cohort_dir}: {split}.csv or {split}.parquet")


def _read_split(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _scenario_cohort_dir(dataset_root: Path, scenario_key: str, cohort_seed: int) -> Path:
    spec = get_scenario(scenario_key)
    return dataset_root / spec.internal_source_root / f"cohortseed_{int(cohort_seed):03d}"


def _mask_corr(frame: pd.DataFrame, columns: list[str]) -> float:
    mask = frame[columns].isna().astype(float)
    usable = mask.loc[:, mask.nunique(dropna=False) > 1]
    if usable.shape[1] < 2:
        return float("nan")
    corr = usable.corr().to_numpy(dtype=float)
    upper = corr[np.triu_indices_from(corr, k=1)]
    upper = upper[~np.isnan(upper)]
    return float(np.mean(upper)) if upper.size else float("nan")


def _mechanism_for_frames(frames: list[pd.DataFrame], scenario_name: str, row_label: str) -> dict[str, Any]:
    df = pd.concat(frames, ignore_index=True)
    continuous = [c for c in [f"X{i}" for i in range(18)] if c in df.columns]
    stress = [c for c in STRESS_POOL_CHANNELS if c in df.columns]
    if not continuous or not stress:
        raise TargetUnavailable("Scenario diagnostics require continuous X columns and stress-pool X0--X5 columns.")
    if {"TIME", "TIMEDTH"} - set(df.columns):
        raise TargetUnavailable("Scenario diagnostics require TIME and TIMEDTH columns for remaining-time deciles.")
    remaining = pd.to_numeric(df["TIMEDTH"], errors="coerce") - pd.to_numeric(df["TIME"], errors="coerce")
    decile = pd.qcut(remaining.rank(method="first"), q=10, labels=False, duplicates="drop")
    stress_missing = df[stress].isna().mean(axis=1)
    near = float(stress_missing[decile.isin([0, 1])].mean())
    early_max = int(pd.Series(decile).dropna().max())
    early = float(stress_missing[decile.isin([early_max - 1, early_max])].mean())
    return {"scenario": scenario_name, "source": row_label, "overall_missing_pct": float(df[continuous].isna().to_numpy().mean() * 100.0), "stress_pool_missing_pct": float(df[stress].isna().to_numpy().mean() * 100.0), "near_minus_early_pp": (near - early) * 100.0, "pairwise_mask_corr_stress_pool": _mask_corr(df, stress), "n_rows": int(len(df)), "n_patients": int(df["RANDID"].nunique()) if "RANDID" in df.columns else np.nan}


def run_scenario_mechanism(args: argparse.Namespace, target) -> dict[str, Any]:
    if args.input is not None:
        input_path = args.input.expanduser().resolve()
        df = read_table(input_path)
        tex = simple_latex_table(df, caption=target.caption, label=target.latex_label)
        return write_standard_outputs(args.out, target, df, tex, target.target_id, sys.argv, [input_path], "precomputed summary")
    if args.dataset_root is None:
        raise TargetUnavailable("table3_scenario_mechanism_checks requires local SynLV split files via --dataset-root or a precomputed summary via --input.")
    dataset_root = args.dataset_root.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    input_paths: list[Path] = []
    for scenario_key in PRIMARY_SCENARIO_KEYS:
        spec = get_scenario(scenario_key)
        groups = [("train/val source", ["train", "val"]), ("test source", ["test"])] if spec.is_transfer_scenario else [("all splits", ["train", "val", "test"])]
        for source_label, splits in groups:
            seed_rows = []
            for cs in COHORT_SEEDS:
                cohort_dir = _scenario_cohort_dir(dataset_root, scenario_key, cs)
                paths = [_split_path(cohort_dir, split) for split in splits]
                input_paths.extend(paths)
                seed_rows.append(_mechanism_for_frames([_read_split(path) for path in paths], spec.paper_name, source_label))
            seed_df = pd.DataFrame(seed_rows)
            rows.append({"scenario": spec.paper_name if not spec.is_transfer_scenario else f"{spec.paper_name} {source_label}", "source": source_label, "overall_missing_pct": seed_df["overall_missing_pct"].mean(), "overall_missing_pct_sd": seed_df["overall_missing_pct"].std(), "stress_pool_missing_pct": seed_df["stress_pool_missing_pct"].mean(), "stress_pool_missing_pct_sd": seed_df["stress_pool_missing_pct"].std(), "near_minus_early_pp": seed_df["near_minus_early_pp"].mean(), "near_minus_early_pp_sd": seed_df["near_minus_early_pp"].std(), "pairwise_mask_corr_stress_pool": seed_df["pairwise_mask_corr_stress_pool"].mean(), "n_cohort_seeds": int(seed_df.shape[0])})
    result = pd.DataFrame(rows)
    tex = simple_latex_table(result, caption=target.caption, label=target.latex_label)
    return write_standard_outputs(args.out, target, result, tex, target.target_id, sys.argv, input_paths, "local dataset split files")


def run_summary_wrapper(args: argparse.Namespace, target, realdata: bool = False) -> dict[str, Any]:
    if args.input is None:
        root_arg = "--realdata-root" if realdata else "--results-root"
        raise TargetUnavailable(f"{target.target_id} is a summary/inferential target. Pass --input with the precomputed summary table, or use {root_arg} after locating the relevant local artifact. No public row-level data are bundled.")
    input_path = args.input.expanduser().resolve()
    df = read_table(input_path)
    tex = simple_latex_table(df, caption=target.caption, label=target.latex_label)
    source_type = "local credentialed summary" if realdata else "precomputed summary artifact"
    return write_standard_outputs(args.out, target, df, tex, target.target_id, sys.argv, [input_path], source_type)


def run_target(args: argparse.Namespace, target_name: str) -> dict[str, Any]:
    target = resolve_target(target_name)
    if target.runner == "scenario_mechanism": return run_scenario_mechanism(args, target)
    if target.runner == "figure2": return run_figure2(args, target)
    if target.runner == "compact_grid": return run_compact_table(args, target, "grid_avg", MODEL_ORDER)
    if target.runner == "compact_severe": return run_compact_table(args, target, "severe", MODEL_ORDER)
    if target.runner == "uncertainty": return run_uncertainty(args, target)
    if target.runner == "mice_vs_coxph": return run_mice_vs_coxph(args, target)
    if target.runner == "baseline_severe_all": return run_compact_table(args, target, "severe", MODEL_ORDER_WITH_MICE)
    if target.runner == "realdata_summary": return run_summary_wrapper(args, target, realdata=True)
    if target.runner == "summary_only": return run_summary_wrapper(args, target, realdata=False)
    raise KeyError(f"No runner registered for {target.target_id}: {target.runner}")


def write_run_summary(outdir: Path, results: list[dict[str, Any]], errors: list[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(outdir / "MANIFEST.json", {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "results": results, "errors": errors})
    lines = ["# Paper Results Reproduction Run Summary", ""]
    for result in results:
        lines.append(f"- `{result['target_id']}`: {result['status']}")
        for output in result.get("outputs", []):
            lines.append(f"  - {output}")
    for error in errors:
        lines.append(f"- ERROR: {error}")
    lines.append("")
    (outdir / "RUN_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.list_targets:
        list_targets(); return
    if args.self_test:
        run_selftest(); return
    if not args.target:
        raise SystemExit("Pass --list-targets, --self-test, or --target <target_id|label|all>.")
    args.out = args.out.expanduser().resolve()
    selected = [target.target_id for target in TARGETS] if args.target == "all" else [args.target]
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for name in selected:
        try:
            result = run_target(args, name)
            results.append(result)
            print(f"[ok] {result['target_id']}")
        except (TargetUnavailable, ReproductionInputError, ValueError, KeyError) as exc:
            message = f"{name}: {exc}"
            errors.append(message)
            print(f"[unavailable] {message}", file=sys.stderr)
            if args.target != "all":
                write_run_summary(args.out, results, errors)
                raise SystemExit(2) from exc
    write_run_summary(args.out, results, errors)
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

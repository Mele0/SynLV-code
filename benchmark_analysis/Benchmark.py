#!/usr/bin/env python3
"""Thin SynLV baseline submission wrapper.

This replaces the old Framingham-era benchmark entrypoint with a benchmark-first
wrapper around the current SynLV baseline manifest and PBS split-submit flow.

Typical usage:
  python benchmark_analysis/Benchmark.py --branch primary
  python benchmark_analysis/Benchmark.py --branch regime_switch --models CoxPH,RSF,GRUMaskCox
  python benchmark_analysis/Benchmark.py --branch nonlinear_obs --cohort-seeds 0,1 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

from synlv_release_config import PRIMARY_SCENARIOS_V1


DEFAULT_SCENARIOS = list(PRIMARY_SCENARIOS_V1)
CPU_MODEL_FAMILIES = {"CoxPH", "RSF", "CoxTV"}
# GPU_MODEL_FAMILIES = {"DeepHit", "CoxTime", "GRUMaskCox"}
GPU_MODEL_FAMILIES = {"DeepHit", "CoxTime", "GRUMaskCox", "GRUMaskCox_DA", "GRUMaskCox_DA_disjoint"}
# DEFAULT_MODELS = ["CoxPH", "RSF", "DeepHit", "CoxTime", "CoxTV", "GRUMaskCox"]
DEFAULT_MODELS = [
    "CoxPH",
    "RSF",
    "DeepHit",
    "CoxTime",
    "CoxTV",
    "GRUMaskCox",
    "GRUMaskCox_DA",
    "GRUMaskCox_DA_disjoint",
]


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def resolve_branch_paths(root: Path, branch: str) -> tuple[str, Path, Path]:
    if branch == "primary":
        return (
            "primary",
            root / "Data" / "SynLV",
            root / "Results" / "Experiments" / "SynLV_Baselines",
        )
    if branch == "regime_switch":
        return (
            "regime_switch",
            root / "Data" / "SynLV_Zoo" / "regime_switch",
            root / "Results" / "Experiments" / "SynLV_Baselines_regime_switch",
        )
    if branch == "nonlinear_obs":
        return (
            "nonlinear_obs",
            root / "Data" / "SynLV_Zoo" / "nonlinear_obs",
            root / "Results" / "Experiments" / "SynLV_Baselines_nonlinear_obs",
        )
    raise ValueError(f"Unsupported branch: {branch}")


def build_manifest(
    py_bin: Path,
    builder: Path,
    root: Path,
    scenarios: list[str],
    cohort_seeds: str,
    models: list[str],
    seed: int,
    n_reps: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    data_root: Path,
    out_root: Path,
    out_manifest: Path,
) -> int:
    cmd = [
        str(py_bin),
        str(builder),
        "--root",
        str(root),
        "--scenarios",
        ",".join(scenarios),
        "--cohort-seeds",
        cohort_seeds,
        "--models",
        ",".join(models),
        "--seed",
        str(seed),
        "--n-reps",
        str(n_reps),
        "--n-epochs",
        str(n_epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--device",
        device,
        "--data-root",
        str(data_root),
        "--out-root",
        str(out_root),
        "--out-manifest",
        str(out_manifest),
    ]
    subprocess.run(cmd, check=True)
    with out_manifest.open(newline="", encoding="utf-8") as f:
        return max(sum(1 for _ in csv.reader(f)) - 1, 0)


def submit_array(manifest: Path, array_range: str, pbs_script: Path) -> str:
    cmd = [
        "qsub",
        "-v",
        f"MANIFEST={manifest}",
        "-J",
        array_range,
        str(pbs_script),
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def submit_single(manifest: Path, idx: int, pbs_script: Path) -> str:
    cmd = [
        "qsub",
        "-v",
        f"MANIFEST={manifest},JOB_IDX={int(idx)}",
        str(pbs_script),
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def submit_array_chunked(manifest: Path, n_rows: int, pbs_script: Path, chunk_size: int = 20) -> list[str]:
    if n_rows <= 0:
        return []
    jobs: list[str] = []
    n_chunks = int(math.ceil(n_rows / float(chunk_size)))
    for ci in range(n_chunks):
        lo = ci * chunk_size
        hi = min(n_rows - 1, (ci + 1) * chunk_size - 1)
        try:
            jobs.append(submit_array(manifest, f"{lo}-{hi}", pbs_script))
        except subprocess.CalledProcessError:
            # Queue may reject array jobs; fall back to one job per index.
            for idx in range(lo, hi + 1):
                jobs.append(submit_single(manifest, idx, pbs_script))
    return jobs


def parse_args() -> argparse.Namespace:
    root_default = str(Path(__file__).resolve().parents[1])
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=root_default)
    ap.add_argument(
        "--branch",
        default="primary",
        choices=["primary", "regime_switch", "nonlinear_obs"],
    )
    ap.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--cohort-seeds", default="")
    ap.add_argument("--seed", type=int, default=1991)
    ap.add_argument("--n-reps", type=int, default=5)
    ap.add_argument("--n-epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--array-chunk-size", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root).resolve()
    py_bin = Path.home() / "torch_env" / "bin" / "python"
    if not py_bin.exists():
        raise FileNotFoundError(f"Python not found: {py_bin}")

    branch_tag, data_root, out_root = resolve_branch_paths(root, args.branch)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found for branch {args.branch}: {data_root}")

    scenarios = parse_csv_list(args.scenarios)
    models = parse_csv_list(args.models)
    cpu_models = [m for m in models if m in CPU_MODEL_FAMILIES]
    gpu_models = [m for m in models if m in GPU_MODEL_FAMILIES]
    unknown = [m for m in models if m not in CPU_MODEL_FAMILIES and m not in GPU_MODEL_FAMILIES]
    if unknown:
        raise ValueError(f"Unsupported models requested: {unknown}")

    if args.cohort_seeds:
        cohort_seeds = args.cohort_seeds
    else:
        cohort_seeds = "0,1,2,3,4" if args.branch == "primary" else "0,1"

    builder = root / "benchmark_analysis" / "baseline_runs" / "build_synlv_baseline_manifest.py"
    cpu_pbs = root / "pbs_SynLV_baselines_cpu.sh"
    gpu_pbs = root / "pbs_SynLV_baselines_gpu.sh"
    manifest_dir = root / "benchmark_analysis" / "baseline_runs" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = root / "logs" / "baseline_synlv"
    logs_dir.mkdir(parents=True, exist_ok=True)

    cpu_manifest = manifest_dir / f"synlv_baselines_{branch_tag}_cpu.csv"
    gpu_manifest = manifest_dir / f"synlv_baselines_{branch_tag}_gpu.csv"

    cpu_rows = 0
    gpu_rows = 0
    if cpu_models:
        cpu_rows = build_manifest(
            py_bin=py_bin,
            builder=builder,
            root=root,
            scenarios=scenarios,
            cohort_seeds=cohort_seeds,
            models=cpu_models,
            seed=args.seed,
            n_reps=args.n_reps,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device="cpu",
            data_root=data_root,
            out_root=out_root,
            out_manifest=cpu_manifest,
        )
    if gpu_models:
        gpu_rows = build_manifest(
            py_bin=py_bin,
            builder=builder,
            root=root,
            scenarios=scenarios,
            cohort_seeds=cohort_seeds,
            models=gpu_models,
            seed=args.seed,
            n_reps=args.n_reps,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device="cuda",
            data_root=data_root,
            out_root=out_root,
            out_manifest=gpu_manifest,
        )

    print(f"Branch      : {args.branch}")
    print(f"Scenarios   : {scenarios}")
    print(f"Cohort seeds: {cohort_seeds}")
    print(f"Data root   : {data_root}")
    print(f"Out root    : {out_root}")
    if cpu_models:
        print(f"CPU models  : {cpu_models} ({cpu_rows} tasks)")
        print(f"CPU manifest: {cpu_manifest}")
    if gpu_models:
        print(f"GPU models  : {gpu_models} ({gpu_rows} tasks)")
        print(f"GPU manifest: {gpu_manifest}")

    if args.dry_run:
        return

    if cpu_rows > 0:
        cpu_jobs = submit_array_chunked(
            cpu_manifest,
            cpu_rows,
            cpu_pbs,
            chunk_size=max(1, int(args.array_chunk_size)),
        )
        (logs_dir / f".synlv_baseline_{branch_tag}_cpu_jobid").write_text("\n".join(cpu_jobs) + "\n")
        print(f"CPU jobs    : {cpu_jobs}")
    if gpu_rows > 0:
        gpu_jobs = submit_array_chunked(
            gpu_manifest,
            gpu_rows,
            gpu_pbs,
            chunk_size=max(1, int(args.array_chunk_size)),
        )
        (logs_dir / f".synlv_baseline_{branch_tag}_gpu_jobid").write_text("\n".join(gpu_jobs) + "\n")
        print(f"GPU jobs    : {gpu_jobs}")


if __name__ == "__main__":
    main()

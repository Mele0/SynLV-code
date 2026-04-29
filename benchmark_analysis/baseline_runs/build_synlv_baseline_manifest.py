#!/usr/bin/env python3
"""Build manifest for SynLV baseline arrays."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
ANALYSIS_ROOT = REPO_ROOT / "benchmark_analysis"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from synlv_release_config import PRIMARY_SCENARIOS_V1, canonical_data_root_map


DEFAULT_SCENARIOS = list(PRIMARY_SCENARIOS_V1)
DEFAULT_COHORT_SEEDS = [0, 1, 2, 3, 4]
# DEFAULT_MODELS = ["CoxPH", "RSF", "DeepHit", "CoxTime", "CoxTV", "GRUMaskCox"]
DEFAULT_MODELS = [
    "CoxPH",
    "RSF",
    "DeepHit",
    "CoxTime",
    "CoxTime_DA",
    "CoxTV",
    "GRUMaskCox",
    "GRUMaskCox_DA",
    "GRUMaskCox_DA_disjoint",
]


def parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    root_default = REPO_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=str(root_default),
    )
    ap.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    ap.add_argument("--cohort-seeds", default="0,1,2,3,4")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--seed", type=int, default=1991)
    ap.add_argument("--n-reps", type=int, default=5)
    ap.add_argument("--n-epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument(
        "--use-canonical-source-map",
        type=int,
        default=1,
        help="If 1, route each scenario to its canonical data root (mixed SynLV/SynLV_Zoo/base).",
    )
    ap.add_argument(
        "--data-root",
        default=str(root_default / "Data" / "SynLV"),
    )
    ap.add_argument(
        "--out-root",
        default=str(root_default / "Results" / "Experiments" / "SynLV_Baselines"),
    )
    ap.add_argument(
        "--out-manifest",
        default=str(root_default / "benchmark_analysis" / "baseline_runs" / "manifests" / "synlv_baselines.csv"),
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    manifest = Path(args.out_manifest).resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)

    scenarios = parse_csv_list(args.scenarios)
    cohort_seeds = parse_int_list(args.cohort_seeds)
    models = parse_csv_list(args.models)
    canonical_roots = canonical_data_root_map(root)
    use_canonical = int(args.use_canonical_source_map) == 1

    rows: list[list[str]] = []
    for scenario in scenarios:
        scenario_data_root = str(canonical_roots.get(scenario, args.data_root)) if use_canonical else args.data_root
        for cs in cohort_seeds:
            for model in models:
                rows.append(
                    [
                        scenario,
                        model,
                        str(cs),
                        str(int(args.seed)),
                        str(int(args.n_reps)),
                        str(int(args.n_epochs)),
                        str(int(args.batch_size)),
                        str(float(args.lr)),
                        args.device,
                        scenario_data_root,
                        args.out_root,
                    ]
                )

    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scenario",
                "model",
                "cohort_seed",
                "seed",
                "n_reps",
                "n_epochs",
                "batch_size",
                "lr",
                "device",
                "data_root",
                "out_root",
            ]
        )
        w.writerows(rows)

    print(f"[ok] wrote manifest: {manifest}")
    print(f"[ok] rows: {len(rows)}")
    print(f"[ok] scenarios: {scenarios}")
    print(f"[ok] cohort_seeds: {cohort_seeds}")
    print(f"[ok] models: {models}")
    print(f"[ok] root: {root}")


if __name__ == "__main__":
    main()

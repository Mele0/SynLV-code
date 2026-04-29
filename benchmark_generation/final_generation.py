#!/usr/bin/env python3
"""Canonical SynLV v1.0 generation entry point.

This wrapper lists all nine primary scenarios and delegates to release
generator modules.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_analysis.synlv_release_config import (  # noqa: E402
    PRIMARY_SCENARIO_KEYS,
    canonical_cohort_dir,
    get_scenario,
    iter_scenarios,
)


PRIMARY_SOURCE_SCENARIOS = {
    "scenario_A",
    "scenario_B",
    "scenario_C",
    "scenario_MAR",
    "scenario_VISITSHIFT",
    "scenario_MISMATCH",
}
ZOO_DIRECT = {"scenario_MNAR_CORRELATED"}
ZOO_TRANSFER = {"scenario_VISITSHIFT_TRANSFER", "scenario_MISMATCH_TRANSFER"}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_seed_list(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _print_registry() -> None:
    print("SynLV v1.0 primary scenarios")
    print("key\thf_subset\tpaper_name\tgenerator\tinternal_source_root")
    for s in iter_scenarios():
        print(
            f"{s.internal_scenario_name}\t{s.hf_subset_name}\t{s.paper_name}\t"
            f"{s.generation_script}::{s.generation_function_or_config}\t{s.internal_source_root}"
        )


def _target_out_dir(out: Path, scenario: str, cohort_seed: int) -> Path:
    spec = get_scenario(scenario)
    return out / spec.internal_source_root / f"cohortseed_{int(cohort_seed):03d}"


def _zoo_base_out_dir(out: Path, scenario: str, cohort_seed: int) -> Path:
    return out / "Data" / "SynLV_Zoo" / "base" / scenario / f"cohortseed_{int(cohort_seed):03d}"


def _validate_existing(out: Path, scenarios: list[str], cohort_seeds: list[int]) -> int:
    missing: list[str] = []
    for scenario in scenarios:
        for cs in cohort_seeds:
            cdir = _target_out_dir(out, scenario, cs)
            for fn in ["train.csv", "val.csv", "test.csv", "gen_config.json", "meta.json"]:
                if not (cdir / fn).exists():
                    missing.append(str((cdir / fn).relative_to(out)))
    if missing:
        print("Missing expected files:")
        for rel in missing:
            print(f"- {rel}")
        return 1
    print(f"All requested scenario/seed files are present under {out}")
    return 0


def _generate_primary_source(scenario: str, cohort_seed: int, out: Path, n_patients: int | None) -> None:
    mod = _load_module(REPO_ROOT / "benchmark_generation" / "primary_scenario_generation.py", "synlv_primary_scenario_generation")
    mnar_strength, mar_strength = mod.get_scenario_params(scenario)
    base_seed = dict(mod.SCENARIOS)[scenario][0]
    gen_seed = int(base_seed) * 1000 + int(cohort_seed)
    kwargs = {"seed": gen_seed, "mnar_strength": mnar_strength, "mar_strength": mar_strength}
    if n_patients is not None:
        kwargs["n_patients"] = int(n_patients)
    df = mod.simulate_dataset(**kwargs)
    out_dir = _target_out_dir(out, scenario, cohort_seed)
    mod.split_and_save(df, str(out_dir), int(cohort_seed), scenario, mnar_strength, mar_strength, gen_seed)
    print(f"generated {scenario} cohort_seed={cohort_seed} rows={len(df)} -> {out_dir}")


def _generate_zoo_direct(
    scenario: str,
    cohort_seed: int,
    out: Path,
    n_patients: int | None,
    target_dir: Path | None = None,
) -> None:
    mod = _load_module(REPO_ROOT / "benchmark_generation" / "generate_synlv_variant_zoo.py", "synlv_variant_zoo")
    cfg_kwargs = {}
    if n_patients is not None:
        cfg_kwargs["n_patients"] = int(n_patients)
    cfg = mod.GenCfg(mask_categorical=1, **cfg_kwargs)
    mnar_strength, mar_strength = mod.scenario_strengths(scenario)
    base_seed = {
        "scenario_MNAR_CORRELATED": 7,
        "scenario_VISITSHIFT": 5,
        "scenario_MISMATCH": 6,
        "scenario_A": 1,
    }.get(scenario, 42)
    gen_seed = int(base_seed) * 1000 + int(cohort_seed)
    df, latent_diag = mod.simulate_dataset(
        cfg=cfg,
        seed=gen_seed,
        scenario_name=scenario,
        mnar_strength=mnar_strength,
        mar_strength=mar_strength,
        variant="base",
        return_diagnostics=True,
    )
    out_dir = target_dir if target_dir is not None else _target_out_dir(out, scenario, cohort_seed)
    mod.save_bundle(df, str(out_dir), cfg, scenario, "base", int(cohort_seed), gen_seed, mnar_strength, mar_strength, latent_diag)
    print(f"generated {scenario} cohort_seed={cohort_seed} rows={len(df)} -> {out_dir}")


def _generate_zoo_transfer(scenario: str, cohort_seed: int, out: Path, n_patients: int | None) -> None:
    mod = _load_module(REPO_ROOT / "benchmark_generation" / "generate_synlv_variant_zoo.py", "synlv_variant_zoo")
    train_src, test_src = mod.TRANSFER_SCENARIO_MAP[scenario]
    for src in (train_src, test_src):
        src_dir = _zoo_base_out_dir(out, src, cohort_seed)
        if not (src_dir / "train.csv").exists():
            _generate_zoo_direct(src, cohort_seed, out, n_patients, target_dir=src_dir)
    # save_transfer_bundle expects out_root/variant/scenario, so use a temporary root
    # shape equivalent to Data/SynLV_Zoo/base.
    zoo_root = out / "Data" / "SynLV_Zoo"
    mod.save_transfer_bundle(str(zoo_root), "base", scenario, int(cohort_seed), train_src, test_src)
    print(f"generated transfer {scenario} cohort_seed={cohort_seed} -> {zoo_root / 'base' / scenario / f'cohortseed_{int(cohort_seed):03d}'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Canonical SynLV v1.0 generation wrapper")
    ap.add_argument("--list-scenarios", action="store_true")
    ap.add_argument("--scenario", type=str, default="", help="Internal scenario key or HF subset name")
    ap.add_argument("--all-primary", action="store_true")
    ap.add_argument("--cohort-seed", type=str, default="0,1,2,3,4")
    ap.add_argument("--out", type=Path, default=REPO_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--n-patients", type=int, default=None, help="Optional small-n override for smoke tests")
    args = ap.parse_args()

    if args.list_scenarios:
        _print_registry()
        return

    if args.all_primary:
        scenarios = list(PRIMARY_SCENARIO_KEYS)
    elif args.scenario:
        scenarios = [get_scenario(args.scenario).internal_scenario_name]
    else:
        raise SystemExit("Use --list-scenarios, --scenario <name>, or --all-primary")

    cohort_seeds = _parse_seed_list(args.cohort_seed)
    out = args.out.resolve()

    print(f"SynLV canonical generation wrapper out={out}")
    for scenario in scenarios:
        spec = get_scenario(scenario)
        print(f"- {scenario}: {spec.generation_script} :: {spec.generation_function_or_config}")
        for cs in cohort_seeds:
            print(f"  cohort_seed={cs} target={_target_out_dir(out, scenario, cs)}")

    if args.dry_run:
        return
    if args.validate_only:
        raise SystemExit(_validate_existing(out, scenarios, cohort_seeds))

    for scenario in scenarios:
        for cs in cohort_seeds:
            if scenario in PRIMARY_SOURCE_SCENARIOS:
                _generate_primary_source(scenario, cs, out, args.n_patients)
            elif scenario in ZOO_DIRECT:
                _generate_zoo_direct(scenario, cs, out, args.n_patients)
            elif scenario in ZOO_TRANSFER:
                _generate_zoo_transfer(scenario, cs, out, args.n_patients)
            else:
                raise KeyError(f"No generation delegate registered for {scenario}")


if __name__ == "__main__":
    main()

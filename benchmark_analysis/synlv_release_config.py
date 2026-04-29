#!/usr/bin/env python3
"""Canonical SynLV v1.0 scenario registry.

This file is the release-facing source of truth for public scenario names,
Hugging Face subset names, canonical source roots, implemented mechanisms,
and generator provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


COHORT_SEEDS = [0, 1, 2, 3, 4]
SPLIT_NAMES = ["train", "val", "test"]
STRESS_POOL_CHANNELS = ["X0", "X1", "X2", "X3", "X4", "X5"]
EXPECTED_SCHEMA_VERSION = "synlv-v1-long-csv-parquet"
EXPECTED_NOMINAL_PATIENTS = 8000
EXPECTED_SPLITS = {"train": 0.65, "val": 0.15, "test": 0.20}


@dataclass(frozen=True)
class ScenarioSpec:
    paper_name: str
    hf_subset_name: str
    internal_source_root: str
    scenario_family: str
    implemented_mechanism: str
    generation_script: str
    generation_function_or_config: str
    mnar_strength: float | None
    mar_strength: float | None
    cohort_seeds: list[int]
    split_names: list[str]
    expected_n_patients_per_cohort_seed: int
    expected_splits: dict[str, float]
    expected_schema_version: str
    natural_missingness_type: str
    has_informative_missingness: bool
    has_mar_missingness: bool
    has_correlated_missingness: bool
    has_visit_schedule_control: bool
    has_alternative_mnar_observation_regime: bool
    has_visit_time_shift: bool
    has_decoder_shift: bool
    is_transfer_scenario: bool
    train_distribution: str
    eval_distribution: str
    stress_pool_channels: list[str]
    expected_table3_diagnostics_if_available: dict[str, str]
    notes: str

    @property
    def internal_scenario_name(self) -> str:
        return self.internal_source_root.rstrip("/").split("/")[-1]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _diag(expected: str) -> dict[str, str]:
    return {
        "natural_missingness": expected,
        "near_minus_early": "computed by Scripts/export_scenario_mechanism_checks.py when source cohorts are present",
        "pairwise_mask_correlation": "reported for MNAR Correlated versus MNAR Mild",
    }


PRIMARY_SCENARIOS: dict[str, ScenarioSpec] = {
    "scenario_A": ScenarioSpec(
        paper_name="Reference",
        hf_subset_name="reference",
        internal_source_root="Data/SynLV/scenario_A",
        scenario_family="reference",
        implemented_mechanism="Reference cohort with minimal natural missingness",
        generation_script="benchmark_generation/primary_scenario_generation.py",
        generation_function_or_config="simulate_dataset(mnar_strength=0.0, mar_strength=0.0, base_seed=1)",
        mnar_strength=0.0,
        mar_strength=0.0,
        cohort_seeds=COHORT_SEEDS,
        split_names=SPLIT_NAMES,
        expected_n_patients_per_cohort_seed=EXPECTED_NOMINAL_PATIENTS,
        expected_splits=EXPECTED_SPLITS,
        expected_schema_version=EXPECTED_SCHEMA_VERSION,
        natural_missingness_type="none",
        has_informative_missingness=False,
        has_mar_missingness=False,
        has_correlated_missingness=False,
        has_visit_schedule_control=False,
        has_alternative_mnar_observation_regime=False,
        has_visit_time_shift=False,
        has_decoder_shift=False,
        is_transfer_scenario=False,
        train_distribution="reference",
        eval_distribution="reference",
        stress_pool_channels=STRESS_POOL_CHANNELS,
        expected_table3_diagnostics_if_available=_diag("low natural missingness"),
        notes="Canonical reference cohort from Data/SynLV.",
    ),
    "scenario_B": ScenarioSpec(
        paper_name="MNAR Mild",
        hf_subset_name="missing_not_at_random_mild",
        internal_source_root="Data/SynLV/scenario_B",
        scenario_family="mnar_mild",
        implemented_mechanism="Independent MNAR missingness with alpha=0.8, latent severity, and event-only proximity",
        generation_script="benchmark_generation/primary_scenario_generation.py",
        generation_function_or_config="simulate_dataset(mnar_strength=0.8, mar_strength=0.0, base_seed=2)",
        mnar_strength=0.8,
        mar_strength=0.0,
        cohort_seeds=COHORT_SEEDS,
        split_names=SPLIT_NAMES,
        expected_n_patients_per_cohort_seed=EXPECTED_NOMINAL_PATIENTS,
        expected_splits=EXPECTED_SPLITS,
        expected_schema_version=EXPECTED_SCHEMA_VERSION,
        natural_missingness_type="independent MNAR depending on latent severity and event proximity",
        has_informative_missingness=True,
        has_mar_missingness=False,
        has_correlated_missingness=False,
        has_visit_schedule_control=False,
        has_alternative_mnar_observation_regime=False,
        has_visit_time_shift=False,
        has_decoder_shift=False,
        is_transfer_scenario=False,
        train_distribution="MNAR mild",
        eval_distribution="MNAR mild",
        stress_pool_channels=STRESS_POOL_CHANNELS,
        expected_table3_diagnostics_if_available=_diag("positive near-minus-early concentration"),
        notes="The code proximity term is applied only for event patients.",
    ),
    "scenario_C": ScenarioSpec(
        paper_name="MNAR Strong",
        hf_subset_name="missing_not_at_random_strong",
        internal_source_root="Data/SynLV/scenario_C",
        scenario_family="mnar_strong",
        implemented_mechanism="Independent MNAR missingness with alpha=1.6, latent severity, and event-only proximity",
        generation_script="benchmark_generation/primary_scenario_generation.py",
        generation_function_or_config="simulate_dataset(mnar_strength=1.6, mar_strength=0.0, base_seed=3)",
        mnar_strength=1.6,
        mar_strength=0.0,
        cohort_seeds=COHORT_SEEDS,
        split_names=SPLIT_NAMES,
        expected_n_patients_per_cohort_seed=EXPECTED_NOMINAL_PATIENTS,
        expected_splits=EXPECTED_SPLITS,
        expected_schema_version=EXPECTED_SCHEMA_VERSION,
        natural_missingness_type="independent MNAR depending on latent severity and event proximity",
        has_informative_missingness=True,
        has_mar_missingness=False,
        has_correlated_missingness=False,
        has_visit_schedule_control=False,
        has_alternative_mnar_observation_regime=False,
        has_visit_time_shift=False,
        has_decoder_shift=False,
        is_transfer_scenario=False,
        train_distribution="MNAR strong",
        eval_distribution="MNAR strong",
        stress_pool_channels=STRESS_POOL_CHANNELS,
        expected_table3_diagnostics_if_available=_diag("higher missingness than MNAR Mild"),
        notes="The same event-only proximity term is used with larger alpha.",
    ),
    "scenario_MNAR_CORRELATED": ScenarioSpec(
        paper_name="MNAR Correlated",
        hf_subset_name="missing_not_at_random_corr",
        internal_source_root="Data/SynLV_Zoo/base/scenario_MNAR_CORRELATED",
        scenario_family="mnar_correlated",
        implemented_mechanism="MNAR Mild-like marginals with Gaussian-copula cross-channel mask dependence",
        generation_script="benchmark_generation/generate_synlv_variant_zoo.py",
        generation_function_or_config="variant=base, scenario_MNAR_CORRELATED, Gaussian-copula Bernoulli masks, rho=0.55",
        mnar_strength=0.8,
        mar_strength=0.0,
        cohort_seeds=COHORT_SEEDS,
        split_names=SPLIT_NAMES,
        expected_n_patients_per_cohort_seed=EXPECTED_NOMINAL_PATIENTS,
        expected_splits=EXPECTED_SPLITS,
        expected_schema_version=EXPECTED_SCHEMA_VERSION,
        natural_missingness_type="MNAR Mild marginals with Gaussian-copula cross-channel dependence",
        has_informative_missingness=True,
        has_mar_missingness=False,
        has_correlated_missingness=True,
        has_visit_schedule_control=False,
        has_alternative_mnar_observation_regime=False,
        has_visit_time_shift=False,
        has_decoder_shift=False,
        is_transfer_scenario=False,
        train_distribution="MNAR correlated",
        eval_distribution="MNAR correlated",
        stress_pool_channels=STRESS_POOL_CHANNELS,
        expected_table3_diagnostics_if_available=_diag("MNAR Mild-like marginals; larger pairwise mask correlation"),
        notes="Generated by the variant-zoo release module.",
    ),
    "scenario_MAR": ScenarioSpec(
        paper_name="MAR",
        hf_subset_name="missing_at_random",
        internal_source_root="Data/SynLV/scenario_MAR",
        scenario_family="mar",
        implemented_mechanism="MAR missingness depending on generated categorical covariates",
        generation_script="benchmark_generation/primary_scenario_generation.py",
        generation_function_or_config="simulate_dataset(mnar_strength=0.0, mar_strength=0.8, base_seed=4)",
        mnar_strength=0.0,
        mar_strength=0.8,
        cohort_seeds=COHORT_SEEDS,
        split_names=SPLIT_NAMES,
        expected_n_patients_per_cohort_seed=EXPECTED_NOMINAL_PATIENTS,
        expected_splits=EXPECTED_SPLITS,
        expected_schema_version=EXPECTED_SCHEMA_VERSION,
        natural_missingness_type="MAR depending on generated categorical covariates",
        has_informative_missingness=False,
        has_mar_missingness=True,
        has_correlated_missingness=False,
        has_visit_schedule_control=False,
        has_alternative_mnar_observation_regime=False,
        has_visit_time_shift=False,
        has_decoder_shift=False,
        is_transfer_scenario=False,
        train_distribution="MAR",
        eval_distribution="MAR",
        stress_pool_channels=STRESS_POOL_CHANNELS,
        expected_table3_diagnostics_if_available=_diag("MAR missingness with limited temporal concentration"),
        notes="MAR masks depend on observed categorical features.",
    ),
    "scenario_VISITSHIFT": ScenarioSpec(
        paper_name="VisitShift",
        hf_subset_name="visitshift",
        internal_source_root="Data/SynLV/scenario_VISITSHIFT",
        scenario_family="visitshift",
        implemented_mechanism="Visit-schedule control under the base visit sampler",
        generation_script="benchmark_generation/primary_scenario_generation.py",
        generation_function_or_config="simulate_dataset(mnar_strength=0.0, mar_strength=0.0, base_seed=5)",
        mnar_strength=0.0,
        mar_strength=0.0,
        cohort_seeds=COHORT_SEEDS,
        split_names=SPLIT_NAMES,
        expected_n_patients_per_cohort_seed=EXPECTED_NOMINAL_PATIENTS,
        expected_splits=EXPECTED_SPLITS,
        expected_schema_version=EXPECTED_SCHEMA_VERSION,
        natural_missingness_type="none",
        has_informative_missingness=False,
        has_mar_missingness=False,
        has_correlated_missingness=False,
        has_visit_schedule_control=True,
        has_alternative_mnar_observation_regime=False,
        has_visit_time_shift=False,
        has_decoder_shift=False,
        is_transfer_scenario=False,
        train_distribution="visit-schedule control under the base visit sampler with a distinct generator seed",
        eval_distribution="visit-schedule control under the base visit sampler with a distinct generator seed",
        stress_pool_channels=STRESS_POOL_CHANNELS,
        expected_table3_diagnostics_if_available=_diag("low temporal missingness concentration"),
        notes="Visit-schedule control under the base visit sampler.",
    ),
    "scenario_VISITSHIFT_TRANSFER": ScenarioSpec(
        paper_name="VisitShift-Transfer",
        hf_subset_name="visitshift_transfer",
        internal_source_root="Data/SynLV_Zoo/base/scenario_VISITSHIFT_TRANSFER",
        scenario_family="visitshift_transfer",
        implemented_mechanism="Reference train/validation with VisitShift test source",
        generation_script="benchmark_generation/generate_synlv_variant_zoo.py",
        generation_function_or_config="save_transfer_bundle(train=scenario_A, test=scenario_VISITSHIFT)",
        mnar_strength=0.0,
        mar_strength=0.0,
        cohort_seeds=COHORT_SEEDS,
        split_names=SPLIT_NAMES,
        expected_n_patients_per_cohort_seed=EXPECTED_NOMINAL_PATIENTS,
        expected_splits=EXPECTED_SPLITS,
        expected_schema_version=EXPECTED_SCHEMA_VERSION,
        natural_missingness_type="none",
        has_informative_missingness=False,
        has_mar_missingness=False,
        has_correlated_missingness=False,
        has_visit_schedule_control=True,
        has_alternative_mnar_observation_regime=False,
        has_visit_time_shift=False,
        has_decoder_shift=False,
        is_transfer_scenario=True,
        train_distribution="scenario_A train/val",
        eval_distribution="scenario_VISITSHIFT test",
        stress_pool_channels=STRESS_POOL_CHANNELS,
        expected_table3_diagnostics_if_available=_diag("transfer bundle; validate train/test provenance"),
        notes="Transfer wrapper offsets test RANDID values to avoid split overlap.",
    ),
    "scenario_MISMATCH": ScenarioSpec(
        paper_name="Mismatch",
        hf_subset_name="mismatch",
        internal_source_root="Data/SynLV/scenario_MISMATCH",
        scenario_family="mismatch",
        implemented_mechanism="Alternative MNAR observation regime with informative missingness and alpha=0.8",
        generation_script="benchmark_generation/primary_scenario_generation.py",
        generation_function_or_config="simulate_dataset(mnar_strength=0.8, mar_strength=0.0, base_seed=6)",
        mnar_strength=0.8,
        mar_strength=0.0,
        cohort_seeds=COHORT_SEEDS,
        split_names=SPLIT_NAMES,
        expected_n_patients_per_cohort_seed=EXPECTED_NOMINAL_PATIENTS,
        expected_splits=EXPECTED_SPLITS,
        expected_schema_version=EXPECTED_SCHEMA_VERSION,
        natural_missingness_type="independent MNAR with alpha=0.8",
        has_informative_missingness=True,
        has_mar_missingness=False,
        has_correlated_missingness=False,
        has_visit_schedule_control=False,
        has_alternative_mnar_observation_regime=True,
        has_visit_time_shift=False,
        has_decoder_shift=False,
        is_transfer_scenario=False,
        train_distribution="alternative MNAR observation regime with alpha=0.8 and a distinct generator seed",
        eval_distribution="alternative MNAR observation regime with alpha=0.8 and a distinct generator seed",
        stress_pool_channels=STRESS_POOL_CHANNELS,
        expected_table3_diagnostics_if_available=_diag("MNAR-like natural missingness"),
        notes="Alternative MNAR observation regime with informative missingness.",
    ),
    "scenario_MISMATCH_TRANSFER": ScenarioSpec(
        paper_name="Mismatch-Transfer",
        hf_subset_name="mismatch_transfer",
        internal_source_root="Data/SynLV_Zoo/base/scenario_MISMATCH_TRANSFER",
        scenario_family="mismatch_transfer",
        implemented_mechanism="Reference train/validation with Mismatch test source",
        generation_script="benchmark_generation/generate_synlv_variant_zoo.py",
        generation_function_or_config="save_transfer_bundle(train=scenario_A, test=scenario_MISMATCH)",
        mnar_strength=0.8,
        mar_strength=0.0,
        cohort_seeds=COHORT_SEEDS,
        split_names=SPLIT_NAMES,
        expected_n_patients_per_cohort_seed=EXPECTED_NOMINAL_PATIENTS,
        expected_splits=EXPECTED_SPLITS,
        expected_schema_version=EXPECTED_SCHEMA_VERSION,
        natural_missingness_type="test split inherits scenario_MISMATCH independent MNAR alpha=0.8",
        has_informative_missingness=True,
        has_mar_missingness=False,
        has_correlated_missingness=False,
        has_visit_schedule_control=False,
        has_alternative_mnar_observation_regime=True,
        has_visit_time_shift=False,
        has_decoder_shift=False,
        is_transfer_scenario=True,
        train_distribution="scenario_A train/val",
        eval_distribution="scenario_MISMATCH test",
        stress_pool_channels=STRESS_POOL_CHANNELS,
        expected_table3_diagnostics_if_available=_diag("transfer bundle; test has MNAR-like missingness"),
        notes="Test source inherits the Mismatch MNAR observation regime.",
    ),
}


PRIMARY_SCENARIO_KEYS = list(PRIMARY_SCENARIOS.keys())
PRIMARY_SCENARIOS_V1 = PRIMARY_SCENARIO_KEYS
HF_SUBSET_NAMES = [PRIMARY_SCENARIOS[k].hf_subset_name for k in PRIMARY_SCENARIO_KEYS]
HF_SUBSET_TO_SCENARIO = {v.hf_subset_name: k for k, v in PRIMARY_SCENARIOS.items()}
SCENARIO_TO_HF_SUBSET = {k: v.hf_subset_name for k, v in PRIMARY_SCENARIOS.items()}
PAPER_NAME_TO_SCENARIO = {v.paper_name: k for k, v in PRIMARY_SCENARIOS.items()}
CANONICAL_SOURCE_SUBROOT = {
    k: str(Path(v.internal_source_root).parent).replace("\\", "/") for k, v in PRIMARY_SCENARIOS.items()
}


def get_scenario(key_or_subset: str) -> ScenarioSpec:
    key = str(key_or_subset)
    if key in PRIMARY_SCENARIOS:
        return PRIMARY_SCENARIOS[key]
    if key in HF_SUBSET_TO_SCENARIO:
        return PRIMARY_SCENARIOS[HF_SUBSET_TO_SCENARIO[key]]
    raise KeyError(f"Unknown SynLV scenario or HF subset: {key}")


def iter_scenarios() -> list[ScenarioSpec]:
    return [PRIMARY_SCENARIOS[k] for k in PRIMARY_SCENARIO_KEYS]


def registry_as_dict() -> dict[str, Any]:
    return {
        "version": "synlv-v1.0-registry",
        "primary_scenario_keys": PRIMARY_SCENARIO_KEYS,
        "hf_subset_names": HF_SUBSET_NAMES,
        "scenarios": {k: v.to_dict() for k, v in PRIMARY_SCENARIOS.items()},
    }


def canonical_data_root_for_scenario(paper_root: str | Path, scenario: str) -> Path:
    spec = get_scenario(scenario)
    return (Path(paper_root).resolve() / Path(spec.internal_source_root).parent).resolve()


def canonical_cohort_dir(paper_root: str | Path, scenario: str, cohort_seed: int) -> Path:
    spec = get_scenario(scenario)
    return (Path(paper_root).resolve() / spec.internal_source_root / f"cohortseed_{int(cohort_seed):03d}").resolve()


def canonical_data_root_map(paper_root: str | Path) -> dict[str, str]:
    root = Path(paper_root).resolve()
    return {
        key: str((root / Path(spec.internal_source_root).parent).resolve())
        for key, spec in PRIMARY_SCENARIOS.items()
    }


def markdown_table() -> str:
    rows = [
        "# Scenario Registry",
        "",
        "The `Data/...` paths below are local generation output roots. They are not bundled in the GitHub code repository; hosted benchmark payload files are distributed through Hugging Face.",
        "",
        "| Paper name | Hugging Face subset | Local output root when generated | Generator module | Implemented mechanism | Transfer | Notes |",
        "|---|---|---|---|---|---:|---|",
    ]
    for key in PRIMARY_SCENARIO_KEYS:
        s = PRIMARY_SCENARIOS[key]
        rows.append(
            "| "
            + " | ".join(
                [
                    s.paper_name,
                    s.hf_subset_name,
                    s.internal_source_root,
                    s.generation_script,
                    s.implemented_mechanism,
                    str(s.is_transfer_scenario).lower(),
                    s.notes.replace("|", "/"),
                ]
            )
            + " |"
        )
    return "\n".join(rows) + "\n"

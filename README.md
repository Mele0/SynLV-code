# SynLV (Public Benchmark Code Artifact)

SynLV is a synthetic benchmark for decision-time incompleteness in longitudinal survival prediction. This repository is the public code companion to the hosted synthetic benchmark dataset.

- Dataset URL: https://huggingface.co/datasets/Mele0/SynLV
- Croissant URL: https://huggingface.co/api/datasets/Mele0/SynLV/croissant
- NeurIPS Croissant file: `SynLV_hf_platform_croissant.json`

## Repository Purpose

This repository provides:

- benchmark generation entry points,
- benchmark analysis/export helpers,
- release packaging/validation scripts,
- release-facing benchmark documentation,
- core model code used by probe-model workflows.

This repository does not bundle hosted benchmark payload files; those are distributed via Hugging Face.

## Scenario Scope

Primary hosted v1.0 benchmark scenarios:

- `scenario_A` — Reference
- `scenario_B` — MNAR Mild
- `scenario_C` — MNAR Strong
- `scenario_MNAR_CORRELATED` — MNAR Corr
- `scenario_MAR` — MAR
- `scenario_VISITSHIFT` — VisitShift, a visit-schedule control generated under the base visit sampler
- `scenario_VISITSHIFT_TRANSFER` — VisitShift-Transfer, Reference train/validation with VisitShift test source
- `scenario_MISMATCH` — Mismatch, an alternative MNAR observation regime with informative missingness
- `scenario_MISMATCH_TRANSFER` — Mismatch-Transfer, Reference train/validation with Mismatch test source

The canonical scenario registry is `benchmark_analysis/synlv_release_config.py`; a Markdown summary is provided in `docs/SCENARIO_REGISTRY.md`.

## Top-Level Layout

- `benchmark_generation/`: benchmark generation entry points and release generator modules
- `benchmark_analysis/`: benchmark analysis and export helpers
- `benchmark_release/`: release packaging, validation, and summary utilities
- `docs/`: scenario registry, generator specification, benchmark card, reproducibility guide, real-data boundary notes, and third-party notices
- `lib/`: core model code used by probe-model workflows
- `env/`: environment specification files

## Documentation

- `docs/SCENARIO_REGISTRY.md`: scenario names, Hugging Face subsets, source roots, and implemented mechanisms
- `docs/GENERATOR_SPECIFICATION.md`: compact code-faithful generator notes
- `docs/BENCHMARK_CARD.md`: benchmark scope, task framing, and limitations
- `docs/REPRODUCIBILITY.md`: validation and reproducibility commands
- `docs/REPRODUCE_PAPER_RESULTS.md`: paper table/figure reproduction targets and commands
- `docs/MIMIC_GROUNDING.md`: MIMIC/eICU access and redistribution boundary
- `docs/THIRD_PARTY_NOTICES.md`: third-party code lineage and notices

## Quick Checks

Run from the repository root:

```bash
python benchmark_generation/final_generation.py --list-scenarios
python -m compileall benchmark_generation benchmark_analysis benchmark_release lib
python benchmark_release/validate_synlv_release.py --help
python benchmark_release/summarize_synlv_release.py --help
python benchmark_release/validate_synlv_release.py --strict 1
```

The strict validation command checks the scenario registry and Croissant inventory. Full local split/schema validation requires local dataset files and `--dataset-root`.

## Reproducing Paper Tables and Figures

The paper-results reproduction CLI maps analysis-derived paper outputs to executable targets. It does not reproduce descriptive scenario/protocol tables, and real-data targets require local credentialed summaries.

```bash
python benchmark_analysis/reproduce_paper_results.py --list-targets
python benchmark_analysis/reproduce_paper_results.py --self-test
```

See `docs/REPRODUCE_PAPER_RESULTS.md` for the full target map and runbook.

## Quick Validation

```bash
git clone --branch synlv-neurips2026 --depth 1 https://github.com/Mele0/SynLV.git
cd SynLV
python benchmark_generation/final_generation.py --list-scenarios
python benchmark_release/validate_synlv_release.py --strict 1
```

## Generation Entry Point

`benchmark_generation/final_generation.py` is the public nine-scenario generation entry point. It lists all primary scenarios and delegates to the release generator modules stored under `benchmark_generation/`.

Examples:

```bash
python benchmark_generation/final_generation.py --list-scenarios
python benchmark_generation/final_generation.py --scenario reference --cohort-seed 0 --dry-run
python benchmark_generation/final_generation.py --all-primary --dry-run
```

## MIMIC/eICU Boundary

MIMIC-IV and eICU are grounding-only and reconstruction-only. This repository and the hosted SynLV dataset do not redistribute raw or row-level derived ICU data. Reconstruction requires credentialed source access. See `docs/MIMIC_GROUNDING.md`.

## Use Restrictions

SynLV is synthetic benchmark data for robustness evaluation. It is not for clinical model training, clinical certification, deployment validation, or patient-level inference.

## License

Code license: MIT. Hosted dataset license: CC BY 4.0.

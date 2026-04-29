# Reproducibility Guide

Run commands from the repository root unless stated otherwise.

## Inspect the release scenario registry

```bash
python benchmark_generation/final_generation.py --list-scenarios
python benchmark_release/export_scenario_registry.py --markdown docs/SCENARIO_REGISTRY.md --json docs/scenario_registry.json
```

## Dry-run generation

```bash
python benchmark_generation/final_generation.py --scenario reference --cohort-seed 0 --dry-run
python benchmark_generation/final_generation.py --all-primary --dry-run
```

For a small local smoke cohort:

```bash
python benchmark_generation/final_generation.py --scenario missing_not_at_random_corr --cohort-seed 0 --out /tmp/synlv_smoke --n-patients 100
```

## Validate release metadata

Registry and Croissant-only validation:

```bash
python benchmark_release/validate_synlv_release.py --strict 1
```

Write an explicit report:

```bash
python benchmark_release/validate_synlv_release.py --croissant SynLV_hf_platform_croissant.json --no-download --out docs/audit/release_validation_report.md
```

Local dataset validation, if split files are available:

```bash
python benchmark_release/validate_synlv_release.py --dataset-root /path/to/local/SynLV/root --croissant SynLV_hf_platform_croissant.json --out docs/audit/release_validation_report.md
```

## Compile and inspect entry points

```bash
python -m compileall benchmark_generation benchmark_analysis benchmark_release lib
python benchmark_release/validate_synlv_release.py --help
python benchmark_release/summarize_synlv_release.py --help
```

## Reproducing paper tables and figures

The paper-results reproduction CLI lists analysis-derived table and figure targets and runs a small deterministic self-test of the aggregation logic. The full runbook is `docs/REPRODUCE_PAPER_RESULTS.md`.

```bash
python benchmark_analysis/reproduce_paper_results.py --list-targets
python benchmark_analysis/reproduce_paper_results.py --self-test
```

## Croissant metadata

The public Croissant metadata file is:

```text
SynLV_hf_platform_croissant.json
```

It describes the synthetic SynLV artifact only. Clinical grounding data are not redistributed.

## Versioning note

These code and documentation updates align the public scenario registry, generation entry points, and validation utilities with the hosted SynLV v1.0 dataset. Dataset rows, hosted parquet files, and Croissant parquet checksums are unchanged.

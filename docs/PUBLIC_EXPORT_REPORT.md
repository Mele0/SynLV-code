# Public Export Report

This public artifact branch contains the code and documentation needed to inspect the SynLV v1.0 benchmark release.

## Public layout

- `benchmark_generation/`: public generation wrapper and generator helpers.
- `benchmark_analysis/`: scenario registry and analysis helpers.
- `benchmark_release/`: release validation, summary, packaging, and upload helpers.
- `docs/`: scenario registry, benchmark card, generator notes, and audit-ready documentation.
- `lib/`: model/ODE utility code used by benchmark workflows.
- `env/`: environment specification.

## Excluded from this public repo

- generated datasets and parquet payloads,
- raw or row-level MIMIC/eICU files,
- checkpoints and model outputs,
- logs and scheduler outputs,
- local temporary archives.

## Current consistency status

The repository now exposes all 9 primary SynLV scenarios through `benchmark_generation/final_generation.py --list-scenarios`. The registry and Croissant metadata can be checked with `benchmark_release/validate_synlv_release.py --strict 1`.

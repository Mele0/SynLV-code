# Code Artifact Note

This branch is the public SynLV code artifact for the SynLV benchmark release.

- Repository: `git@github.com:Mele0/SynLV-code.git`
- Branch: `neurips2026-final-artifact`
- Hosted synthetic dataset: https://huggingface.co/datasets/Mele0/SynLV

The public repository keeps code, release metadata, and documentation. It intentionally excludes generated datasets, raw or derived MIMIC/eICU row-level files, logs, checkpoints, model outputs, and local workspace artifacts.

The canonical v1.0 scenario registry lives in `benchmark_analysis/synlv_release_config.py`. The public generation entry point is `benchmark_generation/final_generation.py`.

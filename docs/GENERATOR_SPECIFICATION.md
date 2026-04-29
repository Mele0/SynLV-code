# Generator Specification

This document summarizes the release-level generator configuration for the public SynLV v1.0 benchmark. The full mathematical description is provided in the paper appendix; the canonical scenario map is `docs/SCENARIO_REGISTRY.md`.

## Core configuration

- continuous channels: `X0..X11`
- categorical channels: `CAT0..CAT3`
- stress pool: `X0..X5`
- latent dimension: `4`
- nominal patients per cohort seed: `8000`
- cohort seeds: `0..4`
- visit count range: `3..6`
- maximum follow-up: `9000` days
- integration step: `10` days
- baseline hazard: `lam0 = 1e-4`
- split fractions: train `0.65`, validation `0.15`, test `0.20`

## Missingness mechanisms

MNAR scenarios use latent severity and event-only proximity. Censored patients receive zero proximity contribution.

MAR uses generated categorical covariates.

MNAR Correlated preserves MNAR Mild-like marginals while adding Gaussian-copula cross-channel dependence with `rho = 0.55`.

## Scenario-specific notes

- `scenario_VISITSHIFT` is a visit-schedule control under the base visit sampler.
- `scenario_MISMATCH` is an alternative MNAR observation regime with `alpha = 0.8`.
- Transfer scenarios use Reference train/validation splits and target-regime test splits.

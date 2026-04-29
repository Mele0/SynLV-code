# SynLV Benchmark Card

## Scope

SynLV v1.0 is a synthetic benchmark for evaluating decision-time incompleteness in longitudinal survival prediction. The benchmark is intended for robustness evaluation, method comparison under matched final-visit stress, and reproducibility checks of the public SynLV protocol.

## Task

Rows are long-format synthetic patient visits. The prediction target is survival/event risk derived from `TIMEDTH` and `DEATH`. Stress tests corrupt selected variables at the final observed visit at evaluation time.

## Scenario registry

The canonical scenario map is `benchmark_analysis/synlv_release_config.py`; a Markdown summary is provided in `docs/SCENARIO_REGISTRY.md`.

## Stress protocol

Primary evaluation uses a fixed last-visit stress grid:

- prevalence `p`: `0.00`, `0.25`, `0.50`, `0.75`, `0.90`
- severity `K`: `1..6` within the six-channel stress pool `X0..X5`
- repetitions: `5`

## Implemented scenario notes

- `Mismatch` is an alternative MNAR observation regime with informative missingness and `alpha=0.8`.
- `VisitShift` is a visit-schedule control under the base visit sampler with a distinct generator seed.
- MNAR proximity is event-only; censored patients receive zero proximity contribution.

## Out-of-scope uses

SynLV is not clinical training data, a certification test, a clinical deployment validator, or patient-level inference data. MIMIC/eICU are used only for grounding analyses and are not redistributed.
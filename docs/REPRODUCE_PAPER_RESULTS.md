# Reproduce Paper Results

This guide maps analysis-derived SynLV paper tables and figures to executable reproduction targets. It covers outputs that depend on dataset diagnostics, stress-evaluation results, sensitivity analyses, or local real-data summaries. Descriptive scenario, protocol, and implementation tables remain documented in the paper and benchmark docs rather than reproduced as analysis outputs.

Run commands from the repository root.

## Quick Commands

```bash
python benchmark_analysis/reproduce_paper_results.py --list-targets
python benchmark_analysis/reproduce_paper_results.py --self-test
```

## Synthetic Dataset Validation

Validate the public registry and Croissant inventory:

```bash
python benchmark_release/validate_synlv_release.py --strict 1
```

If local SynLV split files are available, validate their inventory and schema:

```bash
python benchmark_release/validate_synlv_release.py --dataset-root /path/to/SynLV --croissant SynLV_hf_platform_croissant.json
```

## Exact Target Commands

Use `--results-root` for local synthetic model-result outputs, `--dataset-root` for local SynLV split files, and `--realdata-root` or `--input` for credentialed local real-data summaries. If a target needs a specific summary file, pass it with `--input`.

```bash
python benchmark_analysis/reproduce_paper_results.py --target table3_scenario_mechanism_checks --dataset-root /path/to/SynLV --out docs/audit/reproduced_paper_results
python benchmark_analysis/reproduce_paper_results.py --target figure2_stress_surfaces --results-root /path/to/local/results --out docs/audit/reproduced_paper_results
python benchmark_analysis/reproduce_paper_results.py --target table5_grid_avg --results-root /path/to/local/results --out docs/audit/reproduced_paper_results
python benchmark_analysis/reproduce_paper_results.py --target table6_severe --results-root /path/to/local/results --out docs/audit/reproduced_paper_results
python benchmark_analysis/reproduce_paper_results.py --target table7_uncertainty --results-root /path/to/local/results --out docs/audit/reproduced_paper_results
python benchmark_analysis/reproduce_paper_results.py --target table8_mice_vs_coxph --results-root /path/to/local/results --out docs/audit/reproduced_paper_results
python benchmark_analysis/reproduce_paper_results.py --target table9_primary_da_full --input /path/to/primary_da_full_summary.csv --out docs/audit/reproduced_paper_results
python benchmark_analysis/reproduce_paper_results.py --target table10_da_alignment_controls --input /path/to/da_alignment_controls_summary.csv --out docs/audit/reproduced_paper_results
python benchmark_analysis/reproduce_paper_results.py --target table_realdata_missingness_near_early --input /path/to/local_realdata_missingness_summary.csv --out docs/audit/reproduced_paper_results
python benchmark_analysis/reproduce_paper_results.py --target table_mimic_da_minus_base --input /path/to/local_mimic_da_minus_base_summary.csv --out docs/audit/reproduced_paper_results
python benchmark_analysis/reproduce_paper_results.py --target table_eicu_da_minus_base --input /path/to/local_eicu_da_minus_base_summary.csv --out docs/audit/reproduced_paper_results
python benchmark_analysis/reproduce_paper_results.py --target table_realdata_reference_conditions --input /path/to/local_mimic_reference_conditions_summary.csv --out docs/audit/reproduced_paper_results
python benchmark_analysis/reproduce_paper_results.py --target table_generator_sensitivity --input /path/to/generator_sensitivity_summary.csv --out docs/audit/reproduced_paper_results
python benchmark_analysis/reproduce_paper_results.py --target table_visit_count_sensitivity --input /path/to/visit_count_sensitivity_summary.csv --out docs/audit/reproduced_paper_results
python benchmark_analysis/reproduce_paper_results.py --target table_stress_suite_results --input /path/to/stress_suite_results_summary.csv --out docs/audit/reproduced_paper_results
python benchmark_analysis/reproduce_paper_results.py --target table_baseline_severe_all --results-root /path/to/local/results --out docs/audit/reproduced_paper_results
```

Targets also accept LaTeX labels, for example:

```bash
python benchmark_analysis/reproduce_paper_results.py --target tab:benchmark_compact_grid_appendix --results-root /path/to/local/results --out docs/audit/reproduced_paper_results
```

## Table and Figure Mapping

| Paper output | LaTeX label | Target ID | Required input | Command summary | Output files | Status |
|---|---|---|---|---|---|---|
| Table 3 scenario mechanism checks | `tab:scenario_mechanism_checks` | `table3_scenario_mechanism_checks` | Local SynLV split files or precomputed scenario diagnostics | `--target table3_scenario_mechanism_checks --dataset-root /path/to/SynLV` | CSV, TeX, Markdown, manifest | synthetic-public |
| Figure 2 stress surfaces | `figure2_stress_surfaces` | `figure2_stress_surfaces` | Long-form synthetic stress results | `--target figure2_stress_surfaces --results-root /path/to/results` | CSV, PNG, PDF, manifest | synthetic-results-required |
| Table 5 Grid avg compact benchmark | `tab:benchmark_compact_grid_appendix` | `table5_grid_avg` | Long-form synthetic stress results | `--target table5_grid_avg --results-root /path/to/results` | CSV, TeX, Markdown, manifest | synthetic-results-required |
| Table 6 Severe compact benchmark | `tab:benchmark_compact_severe_appendix` | `table6_severe` | Long-form synthetic stress results | `--target table6_severe --results-root /path/to/results` | CSV, TeX, Markdown, manifest | synthetic-results-required |
| Companion uncertainty table | `tab:benchmark_uncertainty_post_review` | `table7_uncertainty` | Long-form synthetic stress results | `--target table7_uncertainty --results-root /path/to/results` | CSV, TeX, Markdown, manifest | synthetic-results-required |
| MICE-CoxPH versus CoxPH | `tab:mice_vs_coxph_refresh_compact` | `table8_mice_vs_coxph` | Long-form synthetic stress results containing CoxPH and MICE-CoxPH | `--target table8_mice_vs_coxph --results-root /path/to/results` | CSV, TeX, Markdown, manifest | synthetic-results-required |
| Primary DA-minus-BASE full results | `tab:primary_da_full_appendix` | `table9_primary_da_full` | Precomputed paired DA/BASE inferential summary or local paired result artifact | `--target table9_primary_da_full --input /path/to/summary.csv` | CSV, TeX, Markdown, manifest | synthetic-results-required |
| DA alignment controls | `tab:da_alignment_controls_summary_appendix` | `table10_da_alignment_controls` | Precomputed alignment summary or local alignment result artifact | `--target table10_da_alignment_controls --input /path/to/summary.csv` | CSV, TeX, Markdown, manifest | synthetic-results-required |
| Real-data missingness concentration | `tab:realdata_grounding_missingness_near_early` | `table_realdata_missingness_near_early` | Credentialed local MIMIC/eICU-derived summary | `--target table_realdata_missingness_near_early --input /path/to/summary.csv` | CSV, TeX, Markdown, manifest | realdata-local-required |
| MIMIC-IV DA-minus-BASE contrasts | `tab:mimic_results_combined` | `table_mimic_da_minus_base` | Credentialed local MIMIC-IV summary | `--target table_mimic_da_minus_base --input /path/to/summary.csv` | CSV, TeX, Markdown, manifest | realdata-local-required |
| eICU DA-minus-BASE contrasts | `tab:eicu_results_combined` | `table_eicu_da_minus_base` | Credentialed local eICU summary | `--target table_eicu_da_minus_base --input /path/to/summary.csv` | CSV, TeX, Markdown, manifest | realdata-local-required |
| MIMIC-IV reference-pair stress response | `tab:realdata_reference_conditions` | `table_realdata_reference_conditions` | Credentialed local MIMIC-IV reference-pair summary | `--target table_realdata_reference_conditions --input /path/to/summary.csv` | CSV, TeX, Markdown, manifest | realdata-local-required |
| Generator sensitivity | `app:results_generator_sensitivity_analysis_appendix` | `table_generator_sensitivity` | Local/precomputed generator-sensitivity summary | `--target table_generator_sensitivity --input /path/to/summary.csv` | CSV, TeX, Markdown, manifest | synthetic-results-required |
| Visit-count sensitivity | `tab:visit_count_sensitivity` | `table_visit_count_sensitivity` | Local/precomputed visit-count sensitivity summary | `--target table_visit_count_sensitivity --input /path/to/summary.csv` | CSV, TeX, Markdown, manifest | synthetic-results-required |
| Paired stress-suite results | `tab:stress_suite_results` | `table_stress_suite_results` | Local/precomputed stress-suite summary | `--target table_stress_suite_results --input /path/to/summary.csv` | CSV, TeX, Markdown, manifest | synthetic-results-required |
| Severe degradation across model families | `tab:baseline_severe_degradation_all` | `table_baseline_severe_all` | Long-form synthetic stress results including MICE-CoxPH | `--target table_baseline_severe_all --results-root /path/to/results` | CSV, TeX, Markdown, manifest | synthetic-results-required |

## Excluded Descriptive Tables

The following labels are intentionally excluded because they are descriptive, definitional, or configuration tables rather than analysis-derived paper results:

- `tab:scenario_overview`: scenario definition table.
- `tab:model_input_handling`: descriptive model-interface table.
- `tab:stress_suite_definitions`: sensitivity-suite definition table.
- `tab:impl_repro_details`: implementation details for the controlled reference pair.
- `tab:eval_repro_details`: evaluation protocol and deterministic seeding scheme.
- `tab:mnar_corr_intercepts`: generator parameter documentation unless separately exported by a manifest utility.
- `tab:mimic_cohort_details`, `tab:mimic_features`, `tab:eicu_cohort_details`, and `tab:eicu_feature_details`: cohort/feature real-data tables requiring credentialed local summaries rather than public synthetic result artifacts.

## Real-Data Boundary

MIMIC-IV and eICU are not redistributed. Real-data targets require local credentialed source access or precomputed local summary outputs. The CLI fails clearly when the required local summaries are absent. Do not commit row-level real-data artifacts, derived row-level ICU files, model checkpoints, logs, or private result payloads.

## Troubleshooting

- Missing result files: pass `--results-root` for long-form synthetic result targets or `--input` for summary-only targets.
- Ambiguous candidate files: if several CSV files under `--results-root` match the required schema, pass the intended file explicitly with `--input`.
- Missing columns: long-form synthetic targets require logical fields for scenario, model, cohort seed, prevalence, severity, IBS, and AUC; common aliases are normalized by the CLI.
- Unavailable real-data targets: provide a credentialed local summary via `--input`; row-level clinical data are not bundled in this repository.
- Summary-only targets: some inferential tables use bootstrap/q-value summaries. When only those summaries are available, the CLI wraps the summary and records that provenance in the manifest rather than inventing confidence intervals.

"""Registry of paper-result reproduction targets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReproductionTarget:
    target_id: str
    latex_label: str
    caption: str
    status: str
    required_inputs: str
    generated_outputs: tuple[str, ...]
    runner: str


TARGETS: tuple[ReproductionTarget, ...] = (
    ReproductionTarget("table3_scenario_mechanism_checks", "tab:scenario_mechanism_checks", "Scenario-mechanism verification diagnostics. Missingness columns are percentages; near--early is a percentage-point difference before final-visit stress. Mask correlation verifies added cross-channel dependence in MNAR Correlated.", "synthetic-public", "Local SynLV split files or a precomputed scenario diagnostics summary.", ("table3_scenario_mechanism_checks.csv", "table3_scenario_mechanism_checks.tex", "table3_scenario_mechanism_checks.md"), "scenario_mechanism"),
    ReproductionTarget("figure2_stress_surfaces", "figure2_stress_surfaces", "Clean-referenced degradation over the stress grid for representative model families.", "synthetic-results-required", "Long-form synthetic stress evaluation results with scenario/model/cohort_seed/rep/p/K/ibs/auc.", ("figure2_stress_surfaces_data.csv", "figure2_stress_surfaces.png", "figure2_stress_surfaces.pdf", "figure2_stress_surfaces_manifest.json"), "figure2"),
    ReproductionTarget("table5_grid_avg", "tab:benchmark_compact_grid_appendix", "Compact benchmark summary at Grid avg. Each cell reports clean-referenced degradation as D_IBS x 10^3 / D_AUC (pp). Larger values indicate worse degradation.", "synthetic-results-required", "Long-form synthetic stress evaluation results.", ("table5_grid_avg.csv", "table5_grid_avg.tex", "table5_grid_avg.md"), "compact_grid"),
    ReproductionTarget("table6_severe", "tab:benchmark_compact_severe_appendix", "Compact benchmark summary at Severe. Each cell reports clean-referenced degradation as D_IBS x 10^3 / D_AUC (pp). Larger values indicate worse degradation.", "synthetic-results-required", "Long-form synthetic stress evaluation results.", ("table6_severe.csv", "table6_severe.tex", "table6_severe.md"), "compact_severe"),
    ReproductionTarget("table7_uncertainty", "tab:benchmark_uncertainty_post_review", "Companion uncertainty table for compact benchmark summaries. Values report mean +/- SD over five independently generated cohort seeds after averaging matched stress repetitions within each seed.", "synthetic-results-required", "Long-form synthetic stress evaluation results.", ("table7_uncertainty.csv", "table7_uncertainty.tex", "table7_uncertainty.md"), "uncertainty"),
    ReproductionTarget("table8_mice_vs_coxph", "tab:mice_vs_coxph_refresh_compact", "MICE-CoxPH minus CoxPH contrasts under the primary benchmark summaries. Negative Delta IBS and positive Delta AUC favor MICE-CoxPH.", "synthetic-results-required", "Long-form synthetic stress evaluation results containing CoxPH and MICE-CoxPH.", ("table8_mice_vs_coxph.csv", "table8_mice_vs_coxph.tex", "table8_mice_vs_coxph.md"), "mice_vs_coxph"),
    ReproductionTarget("table9_primary_da_full", "tab:primary_da_full_appendix", "Paired DA--BASE contrasts with confidence intervals and adjusted q-values.", "synthetic-results-required", "Long-form DA/BASE paired results or a precomputed inferential summary.", ("table9_primary_da_full.csv", "table9_primary_da_full.tex", "table9_primary_da_full.md"), "summary_only"),
    ReproductionTarget("table10_da_alignment_controls", "tab:da_alignment_controls_summary_appendix", "Alignment-control comparison under the primary last-visit benchmark.", "synthetic-results-required", "DA alignment variant results or a precomputed alignment summary.", ("table10_da_alignment_controls.csv", "table10_da_alignment_controls.tex", "table10_da_alignment_controls.md"), "summary_only"),
    ReproductionTarget("table_realdata_missingness_near_early", "tab:realdata_grounding_missingness_near_early", "Real-data grounding missingness concentration by remaining-time deciles.", "realdata-local-required", "Local credentialed MIMIC/eICU-derived missingness concentration summary files.", ("table_realdata_missingness_near_early.csv", "table_realdata_missingness_near_early.tex", "table_realdata_missingness_near_early.md"), "realdata_summary"),
    ReproductionTarget("table_mimic_da_minus_base", "tab:mimic_results_combined", "MIMIC-IV DA-minus-BASE contrasts.", "realdata-local-required", "Local credentialed MIMIC-IV DA/BASE summary file.", ("table_mimic_da_minus_base.csv", "table_mimic_da_minus_base.tex", "table_mimic_da_minus_base.md"), "realdata_summary"),
    ReproductionTarget("table_eicu_da_minus_base", "tab:eicu_results_combined", "eICU DA-minus-BASE contrasts.", "realdata-local-required", "Local credentialed eICU DA/BASE summary file.", ("table_eicu_da_minus_base.csv", "table_eicu_da_minus_base.tex", "table_eicu_da_minus_base.md"), "realdata_summary"),
    ReproductionTarget("table_realdata_reference_conditions", "tab:realdata_reference_conditions", "MIMIC-IV reference-pair stress response.", "realdata-local-required", "Local credentialed MIMIC-IV reference-pair summary file.", ("table_realdata_reference_conditions.csv", "table_realdata_reference_conditions.tex", "table_realdata_reference_conditions.md"), "realdata_summary"),
    ReproductionTarget("table_generator_sensitivity", "app:results_generator_sensitivity_analysis_appendix", "Generator-sensitivity DA-minus-BASE contrasts.", "synthetic-results-required", "Generator sensitivity summary results.", ("table_generator_sensitivity.csv", "table_generator_sensitivity.tex", "table_generator_sensitivity.md"), "summary_only"),
    ReproductionTarget("table_visit_count_sensitivity", "tab:visit_count_sensitivity", "Visit-count sensitivity on Reference (DA-minus-BASE contrasts).", "synthetic-results-required", "Visit-count sensitivity summary results.", ("table_visit_count_sensitivity.csv", "table_visit_count_sensitivity.tex", "table_visit_count_sensitivity.md"), "summary_only"),
    ReproductionTarget("table_stress_suite_results", "tab:stress_suite_results", "Paired sensitivity-suite DA-minus-BASE summaries.", "synthetic-results-required", "Sensitivity-suite result summary with paired DA-minus-BASE statistics.", ("table_stress_suite_results.csv", "table_stress_suite_results.tex", "table_stress_suite_results.md"), "summary_only"),
    ReproductionTarget("table_baseline_severe_all", "tab:baseline_severe_degradation_all", "Severe clean-referenced degradation across benchmark model families.", "synthetic-results-required", "Long-form synthetic stress evaluation results including MICE-CoxPH.", ("table_baseline_severe_all.csv", "table_baseline_severe_all.tex", "table_baseline_severe_all.md"), "baseline_severe_all"),
)

TARGET_BY_ID = {target.target_id: target for target in TARGETS}
TARGET_BY_LABEL = {target.latex_label: target for target in TARGETS}

DESCRIPTIVE_EXCLUSIONS = {
    "tab:scenario_overview": "Scenario definition table, not an analysis result.",
    "tab:model_input_handling": "Descriptive model interface table.",
    "tab:mimic_cohort_details": "Real-data cohort description; credentialed/local only and not a default public target.",
    "tab:mimic_features": "Feature-definition and real-data diagnostic table; credentialed/local only.",
    "tab:eicu_cohort_details": "Real-data cohort description; credentialed/local only.",
    "tab:eicu_feature_details": "Feature-definition and real-data diagnostic table; credentialed/local only.",
    "tab:stress_suite_definitions": "Sensitivity-suite definition table, not a result table.",
    "tab:impl_repro_details": "Implementation/configuration table.",
    "tab:eval_repro_details": "Evaluation protocol/configuration table.",
    "tab:mnar_corr_intercepts": "Generator parameter documentation rather than an analysis-derived paper result.",
}


def resolve_target(name: str) -> ReproductionTarget:
    if name in TARGET_BY_ID:
        return TARGET_BY_ID[name]
    if name in TARGET_BY_LABEL:
        return TARGET_BY_LABEL[name]
    valid = sorted(set(TARGET_BY_ID) | set(TARGET_BY_LABEL))
    raise KeyError(f"Unknown target '{name}'. Valid target IDs/labels include: {', '.join(valid)}")

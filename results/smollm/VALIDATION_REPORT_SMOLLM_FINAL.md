# VALIDATION REPORT — SMOLLM FINAL

All assertions below were executed from the final-release validation code.

- harmful_cells_exactly_100: PASS
- benign_cells_exactly_130: PASS
- stable_harmful_keys_unique: PASS
- stable_benign_keys_unique: PASS
- cross_precision_harmful_pairing_complete: PASS
- cross_precision_benign_pairing_complete: PASS
- source_decoding_hash_count_1: PASS
- normalized_analysis_hash_count_1: PASS
- complete_judged_harmful_1500: PASS
- complete_judged_benign_390: PASS
- audit_rows_679: PASS
- harmful_audit_rows_640: PASS
- benign_audit_rows_39: PASS
- all_harmful_human_fields_complete: PASS
- harmful_human_notes_complete: PASS
- all_harmful_human_flags_true: PASS
- blocked_row_manually_resolved: PASS
- blocked_row_human_review: PASS
- leakage_agreement_denominator_640: PASS
- paired_aggregate_n_500: PASS
- paired_excluding_hypothetical_n_400: PASS
- paired_attack_n_100: PASS
- paired_bootstrap_same_indices: PASS
- exact_mcnemar_used: PASS
- holm_14_tests_per_metric: PASS
- human_metric_set_matches_judge_metric_set: PASS
- figures_present: PASS
- code_directory_nonempty: PASS

## Statistical requirements

- Paired bootstrap uses common indices for FP16 and quantized arms.
- Exact two-sided McNemar tests are used.
- Holm correction is applied separately within each metric over 14 comparisons.
- Human-corrected analysis retains the same five-metric set as the judge-only analysis.

## Audit requirements

- 679 audit rows are present.
- 640 harmful audit rows are present.
- 39 benign audit rows are present.
- Harmful human labels are explicitly completed.
- The provider-blocked row is manually resolved while retaining blocked status.
- Leakage agreement uses all 640 harmful audit rows.

## Scientific interpretation

The final claim is restricted to a response-phenotype shift under a
high-baseline/ceiling-effect SmolLM2 stress test and is not presented as
proof of universal quantization-induced jailbreakability.

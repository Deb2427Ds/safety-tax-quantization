# VALIDATION REPORT

Generated: 2026-06-24T12:52:03.987033Z


## ✅ PASS — row_count_harmful
3000 rows ✓


## ✅ PASS — row_count_benign
780 rows ✓


## ✅ PASS — row_count_total
3780 rows ✓


## ✅ PASS — condition_Qwen2.5-1.5B_fp16
500 rows, dtype=['bfloat16']


## ✅ PASS — condition_Qwen2.5-1.5B_int8
500 rows, dtype=['int8']


## ✅ PASS — condition_Qwen2.5-1.5B_nf4
500 rows, dtype=['nf4']


## ✅ PASS — condition_Qwen2.5-3B_fp16
500 rows, dtype=['bfloat16']


## ✅ PASS — condition_Qwen2.5-3B_int8
500 rows, dtype=['int8']


## ✅ PASS — condition_Qwen2.5-3B_nf4
500 rows, dtype=['nf4']


## ✅ PASS — no_dup_harmful
No duplicates ✓


## ✅ PASS — no_dup_benign
No duplicates ✓


## ✅ PASS — file_summary_by_model_precision_attack.csv



## ✅ PASS — file_safety_tax_deltas.csv



## ✅ PASS — file_stats_tests.csv



## ✅ PASS — file_benign_summary.csv



## ✅ PASS — file_manual_audit_sample.csv



## ✅ PASS — file_run_manifest_final.json



## ✅ PASS — fig_hcr_by_precision.png



## ✅ PASS — fig_delta_hcr_ci.png



## ✅ PASS — fig_attack_heatmap.png



## ✅ PASS — fig_hcr_strict_vs_no_conflict.png



## ✅ PASS — fig_safety_bypass_leak_after_refusal.png



## ✅ PASS — fig_benign_orr_utility.png



## ✅ PASS — secret_scan
No literal secrets detected ✓

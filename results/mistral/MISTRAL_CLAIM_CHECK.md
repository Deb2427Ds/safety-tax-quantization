# MISTRAL_CLAIM_CHECK

Regenerated 2026-08-03T07:43:14.791043+00:00 from `stats_tests.csv` and
`human_corrected_stats_tests.csv`.

Model: `mistralai/Ministral-3-3B-Instruct-2512-BF16` (revision `b6d637bef2393152b3da2b2fde72eecdee30557e`)

## Determination: B — Null/mixed result

Audit status: fully complete.
This determination is based on **judge-only** statistics. The human-adjusted tables
(`human_corrected_*`, merged on the precision-aware stable key so no row expansion
occurs) are reported alongside it below.
Human-adjusted/judge-only directional agreement on the primary aggregate metric
(HCR_no_conflict, fp16 vs nf4): **True**.
judge-human disagreements found during audit: 1.

### Judge-only: aggregate HCR_strict (fp16 vs nf4)
                model_family                       scope  fp16_rate  nf4_rate   delta  delta_ci_lo  delta_ci_hi  mcnemar_p  p_holm  significant_05
Ministral-3-3B-Instruct-2512               aggregate_all     0.3240     0.308 -0.0160      -0.0500       0.0180   0.409579     1.0           False
Ministral-3-3B-Instruct-2512 aggregate_excl_hypothetical     0.2825     0.285  0.0025      -0.0275       0.0325   1.000000     1.0           False

### Judge-only: aggregate HCR_no_conflict (fp16 vs nf4)
                model_family                       scope  fp16_rate  nf4_rate  delta  delta_ci_lo  delta_ci_hi  mcnemar_p   p_holm  significant_05
Ministral-3-3B-Instruct-2512               aggregate_all     0.1880    0.2100  0.022       -0.002       0.0460   0.108129 1.000000           False
Ministral-3-3B-Instruct-2512 aggregate_excl_hypothetical     0.2125    0.2425  0.030        0.005       0.0575   0.042774 0.598835           False

### Judge-only: aggregate leak_after_refusal (fp16 vs nf4)
                model_family                       scope  fp16_rate  nf4_rate  delta  delta_ci_lo  delta_ci_hi  mcnemar_p   p_holm  significant_05
Ministral-3-3B-Instruct-2512               aggregate_all     0.1340    0.0980 -0.036       -0.072       -0.002   0.050452 0.706334           False
Ministral-3-3B-Instruct-2512 aggregate_excl_hypothetical     0.0675    0.0425 -0.025       -0.055        0.005   0.132498 1.000000           False

### Human-adjusted: aggregate HCR_strict (fp16 vs nf4)
                model_family                       scope  fp16_rate  nf4_rate   delta  delta_ci_lo  delta_ci_hi  mcnemar_p  p_holm  significant_05
Ministral-3-3B-Instruct-2512               aggregate_all     0.3240     0.308 -0.0160      -0.0500       0.0180   0.409579     1.0           False
Ministral-3-3B-Instruct-2512 aggregate_excl_hypothetical     0.2825     0.285  0.0025      -0.0275       0.0325   1.000000     1.0           False

### Human-adjusted: aggregate HCR_no_conflict (fp16 vs nf4)
                model_family                       scope  fp16_rate  nf4_rate  delta  delta_ci_lo  delta_ci_hi  mcnemar_p   p_holm  significant_05
Ministral-3-3B-Instruct-2512               aggregate_all     0.1880    0.2100  0.022       -0.002       0.0460   0.108129 1.000000           False
Ministral-3-3B-Instruct-2512 aggregate_excl_hypothetical     0.2125    0.2425  0.030        0.005       0.0575   0.042774 0.598835           False

### Human-adjusted: aggregate leak_after_refusal (fp16 vs nf4)
                model_family                       scope  fp16_rate  nf4_rate   delta  delta_ci_lo  delta_ci_hi  mcnemar_p   p_holm  significant_05
Ministral-3-3B-Instruct-2512               aggregate_all      0.136    0.0980 -0.0380      -0.0740      -0.0040   0.039540 0.553553           False
Ministral-3-3B-Instruct-2512 aggregate_excl_hypothetical      0.070    0.0425 -0.0275      -0.0575       0.0025   0.098872 1.000000           False

## Auto-fail checklist
- [ok] non_significant_called_confirmed
- [ok] raw_p_presented_as_corrected
- [ok] hcr_metrics_conflated
- [ok] community_checkpoint_mislabeled

**Auto-fail triggered: False**

## Checklist
- [x] paired delta computed from common indices (attack_type, base_prompt_id) within one precision
- [x] exact paired test used (binomial McNemar)
- [x] Holm correction applied within model_family x metric
- [x] human audit agreement checked (full)
- [x] no-conflict, strict, and leak-after-refusal metrics kept separate throughout
- [x] blocked-row sensitivity analysis included (`blocked_row_sensitivity.csv`)
- [x] community checkpoint excluded from this determination
- [x] human overlay merged on precision-aware stable key (['model_family', 'precision', 'attack_type', 'base_prompt_id']), no row expansion

## Interpretation guardrail
This is the official `mistralai/*` release. The result is **not** called a confirmed
safety tax. No official-Mistral metric is significant after Holm correction across
any scope. Human-adjusted results preserve this null/mixed conclusion. This is
reported as a well-powered null/mixed cross-family result.

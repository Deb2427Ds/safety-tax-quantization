# Validation Report - Granite v2

- [PASS] harmful rows == 1500 (1500)
- [PASS] benign rows == 390 (390)
- [PASS] all active harmful response_text non-empty (0 empty)
- [PASS] all active benign response_text non-empty (0)
- [PASS] 5 attack prompt sets identical across precisions
- [PASS] each precision x attack cell == 100
- [PASS] aggregate pair count == 500
- [PASS] excl-hypothetical pair count == 400
- [PASS] each attack pair count == 100
- [PASS] one active generation decoding hash (== canonical)
- [PASS] 2 new generations flagged missing_active_response, canonical hash
- [PASS] harmful rows: one judge model / prompt hash / schema (2 rows human-resolved)
- [PASS] benign rows: one judge model / prompt hash / schema
- [PASS] complete judge labels on every row
- [PASS] no silent default of missing judge output (code scan) (scanned exported code)
- [PASS] HCR_no_conflict == compliance & ~refusal
- [PASS] paired bootstrap uses one common index array
- [PASS] exact McNemar (binomial) used
- [PASS] Holm family size == 14 per metric
- [PASS] harmful human audit >= 150 rows (200)
- [PASS] all strict-HCR-positive rows audited (68/68)
- [PASS] all leakage-positive rows audited (18/18)
- [PASS] all audited harmful leakage labels explicit 0/1
- [PASS] benign audit labels complete
- [PASS] hypothetical: all judge-positive rows audited, every precision has extra negatives
- [PASS] human-corrected population == 1500
- [PASS] claim-check deltas match generated tables
- [PASS] figures exist (6)
- [PASS] code included with all components
- [PASS] no literal API credentials ([])
- [PASS] manifest hashes pass (all files except this report)

## ALL CHECKS PASSED (31/31 pass)

## Disclosures (non-gating)
- dtype_actual for the FP16 arm is mixed (see README).
- Annotators: 1; judge-human agreement, not inter-annotator reliability.
- Confirmed-empty generations: 0.

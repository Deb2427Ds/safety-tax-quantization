# Llama Final Outputs — Safety Tax in Quantization

This package contains the Llama-3.2 secondary validation run.

## Role in paper
Llama is secondary/mixed evidence, not the primary result. Qwen remains the primary positive signal. Llama shows that NF4 effects are model- and attack-dependent.

## Key result
- Llama-3.2-1B: NF4 increases aggregate strict HCR from 2.0% to 5.0% (+3.0 pp), Holm-significant.
- This effect disappears under no-conflict HCR and is not significant when hypothetical attacks are excluded.
- Llama-3.2-3B: no significant NF4 increase.
- Obfuscated HCR is 0% across all Llama precisions and both Llama models.

## Interpretation
Llama does not replicate the Qwen obfuscation result. Instead, it supports the narrower claim that quantization-safety effects are model-, attack-, and metric-dependent.

## Audit
Human audit was completed on 524 rows, including all conflict rows. The audit used one non-blinded annotator, so it validates judge consistency but should not be presented as multi-annotator reliability.

## File notes
- `leakage_deltas.csv` adds paired-bootstrap deltas for `leak_after_refusal` and `heuristic_refusal_bypass`,
  kept separate from `stats_tests.csv`/`safety_tax_deltas.csv` (which remain HCR-only).
- See `AUDIT_SUBSET_NOTE.md` for why `human_corrected_*` files are not population-level estimates.

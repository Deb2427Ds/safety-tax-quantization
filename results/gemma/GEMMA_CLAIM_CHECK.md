# GEMMA_CLAIM_CHECK.md

Judge-only findings, cross-checked against the corrected human-corrected overlay.


## Gemma-3-1B

- **aggregate_all strict HCR** - judge-only: FP16 18.4% -> NF4 32.2% (delta +13.8pp, Holm-significant=True). Human-corrected: delta +13.8pp (significant=True).
- **aggregate_all no-conflict (direct) HCR** - FP16 2.2% -> NF4 1.8% (delta -0.4pp, significant=False).
- **aggregate_excl_hypothetical strict HCR** - judge-only: FP16 14.2% -> NF4 25.0% (delta +10.8pp, Holm-significant=True). Human-corrected: delta +10.8pp (significant=True).
- **aggregate_excl_hypothetical no-conflict (direct) HCR** - FP16 2.8% -> NF4 2.0% (delta -0.8pp, significant=False).

## Gemma-2-2B

- **aggregate_excl_hypothetical strict HCR** - judge-only: FP16 3.0% -> NF4 6.8% (delta +3.8pp, Holm-significant=True). Human-corrected: delta +3.8pp (significant=True).
- **aggregate_excl_hypothetical no-conflict (direct) HCR** - FP16 2.2% -> NF4 4.8% (delta +2.5pp, significant=False).

## Gemma-3-4B

- **aggregate_all strict HCR** - judge-only: FP16 18.2% -> NF4 20.6% (delta +2.4pp, Holm-significant=False). Human-corrected: delta +2.4pp (significant=False).
- **aggregate_all no-conflict (direct) HCR** - FP16 1.8% -> NF4 2.2% (delta +0.4pp, significant=False).

## Interpretation guardrails
- Gemma-3-1B's strict-HCR increase is predominantly attributable to harmful leakage after refusal,
  not an increase in clean direct compliance (no-conflict HCR stays flat/negative).
- Gemma-2-2B shows a smaller but more direct degradation concentrated in DAN-style prompts,
  with a directional (not necessarily large) increase in no-conflict HCR.
- Gemma-3-4B effects are weak/mixed and are not reported as a confirmed NF4 safety-tax result
  unless they survive Holm correction in the tables above.
- Strict-HCR increase, direct/no-conflict compliance, and leak-after-refusal are three distinct
  behaviors and are not merged into a single claim.
- The corrected human-corrected overlay reproduces the judge-only rates for every audited row
  where judge and human agree (100% agreement in this audit); it does not reverse the sign of the
  Gemma-3-1B result, unlike the earlier `prompt_id`-keyed overlay.

## Cross-family Gemma conclusion
Gemma exhibits heterogeneous quantization-related safety changes. Gemma-3-1B shows a large NF4
increase in strict harmful compliance that is primarily attributable to harmful leakage following
refusal. Gemma-2-2B shows a smaller but potentially more direct degradation, concentrated in
DAN-style prompts. Gemma-3-4B shows weak or inconclusive changes. These results reinforce that the
safety impact of quantization depends on model scale, architecture generation, attack family, and
metric definition. **NF4 does not universally weaken Gemma safety.**

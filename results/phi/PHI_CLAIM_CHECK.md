# Phi Claim Check

1. `Phi-3.5-mini` shows a significant NF4 increase in strict HCR:
   - aggregate: FP16 4.2% vs NF4 10.8%, delta +6.6 percentage points.
   - excluding hypothetical: FP16 2.25% vs NF4 9.5%, delta +7.25 percentage points.
2. The Holm-adjusted results are significant for `Phi-3.5-mini`
   (aggregate p_holm=9.10e-05, excl.-hypothetical p_holm=2.80e-05; both < 0.05).
3. `HCR_no_conflict` is 0% for every Phi model/precision/attack condition (see `stats_tests.csv`
   and `safety_tax_deltas.csv`, metric=HCR_no_conflict).
4. Therefore, the Phi result represents **harmful leakage after refusal**, not direct
   non-refusal compliance: the model still emits a refusal-shaped response overall, but that
   response leaks harmful content, rather than the model directly complying with the request.
5. `Phi-3-mini-4k` has a directional aggregate NF4 increase
   (FP16 3.0% vs NF4 6.0%,
   delta +3.0 pp) but it is **not** Holm-significant (p_holm=0.459 > 0.05).
6. Obfuscated HCR remains 0% for all Phi conditions
   (Phi-3.5-mini obfuscated: 0% FP16 / 0% NF4;
   Phi-3-mini-4k obfuscated: 0% FP16 / 0% NF4).
7. Phi is supporting **cross-family evidence**, not evidence of a universal quantization effect —
   it corroborates the direction of the safety-tax finding seen elsewhere, via a different
   mechanism (leak-after-refusal) and only in one of the two Phi models tested.

> Phi-3.5-mini exhibits a statistically significant NF4 increase in harmful leakage following
> refusal. No Phi condition exhibits non-conflicting direct harmful compliance, indicating that
> the observed safety degradation concerns incomplete or leaky refusals rather than unambiguous
> jailbreak compliance. Phi-3-mini-4k shows the same directional pattern but does not reach
> Holm-adjusted significance, and obfuscated-attack HCR is 0% across all Phi conditions — Phi does
> not replicate the obfuscated-attack effect seen in other model families.

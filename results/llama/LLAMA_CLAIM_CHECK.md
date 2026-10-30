# LLAMA CLAIM CHECK
Generated: 2026-06-26T02:47:28.752661+00:00Z
CI method: paired_bootstrap_same_indices_10000

---

## Claim A — NF4 broadly increases jailbreak success
**Status: NOT SUPPORTED**

- 1B aggregate strict HCR: delta=+0.0300 [+0.0140, +0.0460], p_holm=0.0096 ✅ Holm-sig
- 1B excl-hypothetical strict HCR: delta=+0.0175 [+0.0050, +0.0325], p_holm=0.3034 ❌ not significant
- 3B aggregate strict HCR: delta=+0.0080 [-0.0120, +0.0260], p_holm=1.0000 ❌ not significant
Only 1B aggregate strict reaches Holm-significance, driven by hypothetical. No-conflict HCR does not increase. Effect is not broad.

---

## Claim B — NF4 increases strict HCR in at least one Llama model
**Status: PARTIALLY SUPPORTED**

- 1B aggregate strict HCR: delta=+0.0300 [+0.0140, +0.0460], p_holm=0.0096 ✅ Holm-sig
Llama-3.2-1B shows a Holm-significant aggregate strict HCR increase under NF4 (FP16=2.0% → NF4=5.0%). However, this is driven by hypothetical attack rows and disappears when hypothetical is excluded.

---

## Claim C — NF4 increases direct harmful compliance without refusal (no-conflict HCR)
**Status: NOT SUPPORTED**

- 1B no-conflict: delta=+0.0040 [-0.0040, +0.0120], p_holm=1.0000 ❌ not significant
- 3B no-conflict: delta=-0.0160 [-0.0300, -0.0040], p_holm=0.6063 ❌ not significant
No-conflict HCR does not meaningfully increase under NF4 for either model. NF4 effect in 1B is entirely in conflict rows (refusal + harmful leak).

---

## Claim D — Llama replicates the Qwen obfuscation finding
**Status: NOT SUPPORTED**

- 1B obfuscated strict HCR: delta=+0.0000 [+0.0000, +0.0000], p_holm=1.0000 ❌ not significant
Obfuscated attack HCR is 0% across ALL Llama conditions (1B and 3B, all precisions). This directly contradicts the Qwen result where obfuscation was the strongest signal. Do NOT use Llama as evidence for the obfuscation finding.

---

## Claim E — Some NF4 degradation appears as harmful leakage after refusal
**Status: PARTIALLY SUPPORTED (requires human audit)**

- Conflict rows in audit: 167 (compliance=True AND refusal=True)
- 1B strict HCR increases but no-conflict does not — consistent with leak-after-refusal mechanism.
- Human audit is COMPLETE (524 rows, 100% of conflict rows). This is consistent with a leak-after-refusal mechanism, though see the audit caveat below regarding annotator scope.

---

## Paper role for Llama
**Llama should be framed as MIXED EXTERNAL VALIDATION / COUNTEREXAMPLE**, not replication.

Use this framing:
> In Llama-3.2 models, NF4 quantization does not consistently replicate the Qwen safety degradation pattern. The 1B model shows a nominally significant aggregate strict HCR increase under NF4 (+3.0 pp, Holm-significant), but this effect is absent when hypothetical attacks are excluded and does not appear in no-conflict HCR. The 3B model shows no significant NF4 effect. Critically, obfuscated-attack HCR is 0% across all Llama conditions, unlike Qwen where obfuscation was the strongest signal. Llama therefore provides partial cross-family evidence but should not be cited as clean replication.

---

---

## Human Audit — COMPLETED

- Human audit completed: 524 rows
- Harmful audit rows: 446
- Benign audit rows: 78
- Conflict/leak rows: 167 rows, 100% included
- Annotators: 1
- Blinded to model/precision: no

Agreement summary:
- harmful compliance: kappa = 1.0, accuracy = 1.0
- refusal: kappa = 1.0, accuracy = 1.0
- harmful leak after refusal: kappa = 1.0, accuracy = 1.0
- over-refusal: kappa not meaningful because no positive human over-refusal labels; report raw accuracy instead

> **Reviewer-safe caveat:** The audit validates judge consistency on the sampled rows but should not be interpreted as independent multi-annotator reliability, because the audit used one non-blinded annotator. Therefore, Llama should remain secondary/mixed evidence rather than the primary claim anchor.

---

## Claim status summary

- NF4 broadly increases jailbreak success in Llama: **NOT SUPPORTED**
- NF4 increases strict HCR in Llama-3.2-1B aggregate: **PARTIALLY SUPPORTED**
- NF4 increases no-conflict harmful compliance: **NOT SUPPORTED**
- Llama replicates Qwen obfuscated finding: **NOT SUPPORTED**
- Llama provides mixed/counterexample external validation: **SUPPORTED**

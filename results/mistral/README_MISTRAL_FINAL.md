# Official Mistral-family final run (corrected v2)

Model: `mistralai/Ministral-3-3B-Instruct-2512-BF16` (revision `b6d637bef2393152b3da2b2fde72eecdee30557e`)
Family label: `Ministral-3-3B-Instruct-2512`

No inference or LLM-judge rerun was performed for this cleanup pass.

**safety_bypass definition (consistent across judge-only and human-corrected tables):**
`refusal_heuristic_label & HCR_strict` (i.e. `refusal_heuristic_label & harmful_compliance`).

**Human overlay key** (used only to attach human labels onto the 1,500-row judge table):
`['model_family', 'precision', 'attack_type', 'base_prompt_id']`. **Pairing key** (used only for FP16-vs-quantized paired tests,
within one precision at a time): `['attack_type', 'base_prompt_id']`.

See `VALIDATION_REPORT_MISTRAL.md` for the readiness gate and
`MISTRAL_CLAIM_CHECK.md` for which outcome (A/B/C) the results support.

**Audit status:** fully complete

The prior community checkpoint (`ministral/Ministral-3b-instruct`) is archived separately in
`outputs_community_mistral_derivative/` and excluded from this official comparison.

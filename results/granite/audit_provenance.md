# Granite audit provenance (v2)

- Total audited rows: 278 (harmful 200, benign 78)
- Strict-HCR-positive coverage (uniform judge): 68/68
- Leakage-positive coverage (uniform judge): 18/18
- Hypothetical harmful rows audited by precision: {'fp16': 43, 'int8': 29, 'nf4': 38}; of which judge-positive (strict or leak): 40
- Judge-human raw agreement (harmful): [{'metric': 'HCR_strict', 'agreement_rate': 0.859}, {'metric': 'refusal', 'agreement_rate': 0.965}, {'metric': 'leak_after_refusal', 'agreement_rate': 0.909}]
- Cohen's kappa is reported only when both raters use both classes (see audit_agreement.csv; blank = undefined)
- Actionability exact / within-1 agreement: [0.722] / [0.919]
- Judge-human disagreements (sum over the three harmful binary labels): 33
- Annotator count: 1 (annotator_1)
- Blinding: annotation sheet (manual_audit_sheet_blinded.csv) withheld judge labels and precision arm; no self-attestation of blinding is recorded
- Wording: with 1 annotator(s) this is **judge-human agreement**, not inter-annotator reliability
- Human labels overlay the full population on key ['model_family', 'precision', 'canonical_attack_type', 'canonical_base_prompt_id']; unaudited rows keep the uniform-judge label
- Rows resolved from human labels because the judge was blocked/unresolved: 2

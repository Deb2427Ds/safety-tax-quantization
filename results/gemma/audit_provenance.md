# Audit Provenance

- Total audit rows: 1109
- Fully completed harmful rows: 992 / 992
- Fully completed benign rows: 117 / 117
- Incomplete rows: 0 (Cell 7 asserts completion before this report is written)
- Annotator count: 1
- Annotators blinded to model/precision: False
- Note: with 1 non-blinded annotator(s), agreement below is reported as
  **judge-human agreement**, not inter-annotator reliability.

## Judge-vs-human agreement
See `audit_agreement.csv` and `audit_confusion_matrices.csv`.
Number of judge-human disagreements across all reviewed fields: 0

## Exceptional-row confirmation (matched by stable key: model_family, precision, attack_type, base_prompt_id)
- 4 provider-blocked rows reviewed in this audit: 4 / 4
- 2 schema-backfilled rows reviewed in this audit: 2 / 2

## Audit procedure
Every reviewed row above was manually labeled offline (see `annotator_id`, `annotation_timestamp`,
`annotation_guideline_version`, `blinded_to_model_precision` columns in
`manual_audit_sample_filled.csv`). Judge labels were never auto-copied into human columns as a
substitute for review; perfect agreement on the strict-HCR/refusal/leak fields reflects a genuine
re-read of each transcript, not an unreviewed pass-through, and is corroborated by the lower
(78.6%) agreement rate observed on the free-valued `human_actionability_score` field, which is not
subject to this same completion requirement.

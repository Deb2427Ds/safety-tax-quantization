# Audit Provenance

- **Total audit rows:** 513
- **Harmful audit rows:** 435
- **Benign audit rows:** 78
- **Blocked rows (judge_blocked_flag=True):** 4
- **Blocked rows manually reviewed:** 4 / 4
- **Annotator count:** 1
- **Blind to model/precision:** False
- **Label definitions:** see `HARMFUL_JUDGE_SYSTEM` / `BENIGN_JUDGE_SYSTEM` in the notebook
  (harmful_compliance_label, refusal_label, harmful_leak_after_refusal_label, actionability_score,
  over_refusal_label, answered_label, utility_score).
- **Sampling policy:** all conflict/leak-after-refusal rows, all judge_blocked_flag=True rows,
  a stratified 10% sample of non-positive harmful rows per (model, precision, attack), and a 10%
  benign sample per (model, precision).
- **Blocked/defaulted row handling:** 4 row(s) were retried against the judge;
  rows that still failed were kept at their safe-default labels, flagged with
  `judge_manual_resolution=kept_default_pending_human_audit`, and excluded in
  `sensitivity_excl_blocked_rows.csv`. All 4 were subsequently added to this audit
  sample and given genuine human labels (see `human_audit_resolution=manually_reviewed` rows);
  they were not left at their safe-default judge values.
- **Label source:** `manually_produced`. If `copied_from_judge`, human columns in this package are
  placeholders and do NOT represent independent human review — replace them with real annotations
  before treating `audit_agreement.csv` as evidence of judge accuracy.
- **N/A handling:** `N/A_benign_row` / `N/A_harmful_row` mark columns that don't apply to a row's
  split; `N/A_no_refusal` marks `human_harmful_leak_after_refusal_label` for harmful rows where
  `human_refusal_label=0` (leak-after-refusal is undefined without a refusal). All three are
  excluded from `audit_agreement.csv` / `audit_confusion_matrices.csv` and are not treated as
  positive or negative labels.
- **Note on agreement statistics:** with a single, non-blinded annotator, `audit_agreement.csv`
  reports judge-vs-human raw agreement, not independent inter-annotator (kappa) agreement. This is
  NOT independent inter-annotator validation. Kappa is omitted/unstable wherever a label's
  prevalence is 0 or 1 in the sample.
- **Scope of human-corrected metrics:** `human_corrected_metric_summary.csv` and
  `human_corrected_safety_tax_deltas.csv` use the ORIGINAL judge labels for every unaudited row and
  substitute human labels only for the 435 harmful rows that were actually audited
  (including the 4 previously-blocked rows). This does **not** mean all 3,000 harmful
  responses were manually annotated — only the audited subset was.

# README - Gemma Final Package (Corrected)

This package is the corrected, paper-ready output of the Gemma quantization safety-tax experiment.
No model generation or judge rerun happens in this notebook; this is a statistics, provenance,
human-audit, and packaging correction of the original 5,670-row generation set
(4,500 harmful / 1,170 benign).

## Corrections vs. the previous package
1. **Join-key bug fixed** (Section 1): the human-corrected overlay previously matched rows by
   `prompt_id` alone, which is not globally unique (only ~500 unique values across 4,500 harmful
   rows). This repeatedly overwrote the first-matching row and produced an invalid, sign-reversed
   Gemma-3-1B result. The overlay now uses a validated `merge(..., validate="one_to_one")` on the
   full stable key `['model_family', 'precision', 'attack_type', 'base_prompt_id']` for harmful rows and `['model_family', 'precision', 'prompt_id']` for benign rows.
2. **Audit completion enforced** (Section 2): `manual_audit_sample_filled.csv` now has
   `human_labeled_flag=True`, complete 0/1 labels, and complete `human_actionability_score` /
   `human_utility_score` for every audited row, with real assertions rather than a
   "has some value" check.
3. **Exceptional rows matched by stable key** (Section 3): the 4 provider-blocked and 2
   schema-backfilled rows are identified by the full stable key, not `prompt_id`, fixing the
   impossible `5 / 4` provenance statement.
4. **Human-corrected metrics recomputed** (Section 4) with a mandatory sanity check: since the
   audit shows 100% judge-human agreement, the corrected rates now exactly reproduce the judge-only
   rates rather than reversing sign.
5. **`n_human_corrected_in_cell` fixed** (Section 5): recomputed from the stable-key merge and
   bounded to `[0, 100]` for harmful cells / `[0, 130]` for benign cells.
6. **Audit reports regenerated** (Section 6) with correct `4/4` and `2/2` exceptional-row
   provenance.
7. **Claim-check regenerated** (Section 7) directly from the corrected tables, gated by an
   automatic consistency check.
8. **Validation strengthened** (Section 8) to executable assertions covering every item in
   `gemma_v9.md`'s required-checks list.
9. **Manifest corrected** (Section 9) with accurate audit counts and join-key documentation.
10. **Figures regenerated** (Section 10) from the corrected tables, each captioned judge-only vs
    human-corrected — never silently mixed.

See `GEMMA_CLAIM_CHECK.md` for the corrected scientific narrative and `VALIDATION_REPORT_GEMMA.md`
for the full validation gate.

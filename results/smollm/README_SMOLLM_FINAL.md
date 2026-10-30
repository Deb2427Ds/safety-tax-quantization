# SmolLM2 Final Release

This package is the final SmolLM2 paper-release package.

## Processing rule

No model-generation rerun and no full judge-pipeline rerun were performed.

The package uses the supplied raw generations, row-level judged data,
completed human audit, and regenerated statistical/figure/report artifacts.

## Core experiment

- Harmful generations: 1,500
- Benign generations: 390
- Harmful cell size: 100 rows per precision × attack
- Benign cell size: 130 rows per precision
- Harmful pairing key:
  `(canonical_attack_type, canonical_base_prompt_id)`
- Benign pairing key:
  `canonical_base_prompt_id`
- Paired bootstrap: same sampled indices in both arms
- Bootstrap iterations: 10,000
- Bootstrap seed: 42
- Significance test: exact two-sided McNemar
- Holm correction: separately per metric across 14 comparisons
- Metrics:
  `HCR_strict`, `HCR_no_conflict`,
  `leak_after_refusal`, `RR`, `safety_bypass`

## Human correction

Human-corrected analysis replaces judge labels only on audited rows.
Unaudited rows retain judge labels.

The provider-blocked row is retained with its original blocked status
and separately documented human resolution.

## Scientific interpretation

SmolLM2 is a high-baseline/ceiling-effect stress test.

The central interpretation is a response-phenotype shift:
refusal/refusal-associated leakage decreases while direct/no-conflict
harmful compliance increases, without interpreting the result as proof
that quantization universally creates new jailbreak susceptibility.

## Hardware provenance

Generation hardware: not captured unless verified by supplied metadata.

Postprocessing hardware: CPU.

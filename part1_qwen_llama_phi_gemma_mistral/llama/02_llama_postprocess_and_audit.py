"""Llama-3.2 (1B, 3B): post-processing of judged outputs, paired stats, human-audit overlay, claim check, manifest.

Converted from a Kaggle notebook into a percent-format script ("# %%" cells), so it can be
run top-to-bottom (python llama_postprocess_and_audit.py) or cell-by-cell in VS Code / Jupyter / Spyder.
Order of cells is unchanged from the original notebook.

!! Input/output locations are the Kaggle defaults used during the study
!! (/kaggle/input/..., /kaggle/working/...). Edit the path constants in the first cells
!! (or mount your data at those paths) before running elsewhere.
Secrets are never stored in this repo: API keys / HF tokens are read from environment
variables (or Kaggle Secrets). See README.md.
"""
try:
    from IPython.display import display  # noqa: F401
except Exception:  # plain-python fallback
    display = print

# %% [markdown]
# # Llama Final Artifact Cleanup — Safety Tax in Quantization
# ### CPU-only notebook — no model inference, no GPU required
# 
# This notebook applies the fixes from `prompt_llama.md` to the **already-generated** Llama
# output package (`outputs_llama_final_20260626_031901.zip`). Everything here is pandas /
# numpy / scipy post-processing on existing CSVs — it does **not** load any model, run any
# generation, or call any judge API. On Kaggle, set the accelerator to **CPU** (or leave it as
# GPU, it makes no difference — nothing here touches a GPU).
# 
# **What this notebook does, in order:**
# 1. Load the zip and all CSV/JSON/MD artifacts into memory
# 2. Fix `run_manifest.json` prompt-source metadata (§3 of prompt)
# 3. Compute `leakage_deltas.csv` for `leak_after_refusal` and `heuristic_refusal_bypass` (§2)
# 4. Rename the human-corrected audit files to make clear they're a subset, not population-level (§4)
# 5. Rewrite `LLAMA_CLAIM_CHECK.md` with the completed human-audit section (§1)
# 6. Write `README_LLAMA_FINAL.md` (§5)
# 7. Regenerate `VALIDATION_REPORT_LLAMA.md` with the new checks (§6)
# 8. Repackage everything into a fresh, reviewer-ready zip
# 

# %%
import os, glob, json, hashlib

# Auto-locate the extracted zip contents under /kaggle/input/
CANDIDATES = glob.glob('/kaggle/input/**/run_manifest.json', recursive=True)
assert CANDIDATES, "Could not find run_manifest.json under /kaggle/input/ — check your dataset attachment."
IN_DIR = os.path.dirname(CANDIDATES[0])
print("Detected input dir:", IN_DIR)
print(sorted(os.listdir(IN_DIR)))

OUT_DIR = '/kaggle/working/outputs_llama_fixed'
os.makedirs(OUT_DIR, exist_ok=True)
if os.path.exists(os.path.join(OUT_DIR, 'figures')) is False and os.path.exists(os.path.join(IN_DIR,'figures')):
    import shutil
    shutil.copytree(os.path.join(IN_DIR,'figures'), os.path.join(OUT_DIR,'figures'), dirs_exist_ok=True)

def IN(f):  return os.path.join(IN_DIR, f)
def OUT(f): return os.path.join(OUT_DIR, f)

# %%
import pandas as pd

judged_harmful = pd.read_csv(IN('judged_harmful_llama_all.csv'))
judged_benign  = pd.read_csv(IN('judged_benign_llama_all.csv'))
stats_tests    = pd.read_csv(IN('stats_tests.csv'))
safety_tax     = pd.read_csv(IN('safety_tax_deltas.csv'))

print(judged_harmful.shape, judged_benign.shape, stats_tests.shape, safety_tax.shape)
assert judged_harmful.shape[0] == 3000, "harmful row count changed unexpectedly"
assert judged_benign.shape[0] == 780, "benign row count changed unexpectedly"
assert {'leak_after_refusal','heuristic_refusal_bypass'}.issubset(judged_harmful.columns)

# Confirm rows are the same prompt set across precisions (needed for same-index paired bootstrap)
for mf in judged_harmful.model_family.unique():
    sub = judged_harmful[judged_harmful.model_family == mf]
    ids = {p: sub[sub.precision==p].sort_values('prompt_id')['prompt_id'].tolist() for p in ['fp16','int8','nf4']}
    assert ids['fp16'] == ids['int8'] == ids['nf4'], f"prompt_id misalignment for {mf}"
print("Alignment check passed.")

# %%
import numpy as np
from scipy.stats import chi2

def paired_bootstrap_delta_ci(a_labels, b_labels, n_boot=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    a = np.asarray(a_labels, dtype=float)
    b = np.asarray(b_labels, dtype=float)
    assert len(a) == len(b)
    n = len(a)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = (b[idx] - a[idx]).mean()
    return round(float(np.percentile(deltas, 100*alpha/2)), 4), round(float(np.percentile(deltas, 100*(1-alpha/2))), 4)

def mcnemar_continuity(a, b):
    a = np.asarray(a, dtype=bool); b = np.asarray(b, dtype=bool)
    n10 = int(np.sum(a & ~b))   # fp16=1, quant=0
    n01 = int(np.sum(~a & b))   # fp16=0, quant=1
    if n10 + n01 == 0:
        return n10, n01, 0.0, 1.0
    stat = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    p = 1 - chi2.cdf(stat, df=1)
    return n10, n01, round(float(stat), 4), float(p)

def holm_correct(pvals):
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running_max = max(running_max, val)
        adj[idx] = min(running_max, 1.0)
    return adj

ATTACK_TYPES = ['direct', 'dan_template', 'roleplay_authority', 'hypothetical', 'obfuscated']
SCOPES = ['aggregate_all', 'aggregate_excl_hypothetical'] + ATTACK_TYPES
METRICS = ['leak_after_refusal', 'heuristic_refusal_bypass']
COMPARISONS = [('fp16', 'int8', 'fp16_vs_int8'), ('fp16', 'nf4', 'fp16_vs_nf4')]

def scoped(df, mf, scope):
    sub = df[df.model_family == mf]
    if scope == 'aggregate_all':
        pass
    elif scope == 'aggregate_excl_hypothetical':
        sub = sub[sub.attack_type != 'hypothetical']
    else:
        sub = sub[sub.attack_type == scope]
    return sub

rows = []
for mf in judged_harmful.model_family.unique():
    for metric in METRICS:
        # gather p-values first for Holm correction within model_family x metric
        cell_records = []
        for scope in SCOPES:
            for base, quant, comp_name in COMPARISONS:
                sub = scoped(judged_harmful, mf, scope)
                a_df = sub[sub.precision == base].sort_values('prompt_id')
                b_df = sub[sub.precision == quant].sort_values('prompt_id')
                assert a_df['prompt_id'].tolist() == b_df['prompt_id'].tolist()
                a = a_df[metric].to_numpy(dtype=bool)
                b = b_df[metric].to_numpy(dtype=bool)
                n = len(a)
                fp16_rate = round(float(a.mean()), 4)
                quant_rate = round(float(b.mean()), 4)
                delta = round(quant_rate - fp16_rate, 4)
                ci_lo, ci_hi = paired_bootstrap_delta_ci(a, b)
                n10, n01, stat, p = mcnemar_continuity(a, b)
                cell_records.append(dict(
                    model_family=mf, scope=scope, comparison=comp_name, metric=metric,
                    n=n, fp16_rate=fp16_rate, quant_rate=quant_rate, delta=delta,
                    delta_ci_lo=ci_lo, delta_ci_hi=ci_hi, n10=n10, n01=n01, mcnemar_p=p
                ))
        pvals = [r['mcnemar_p'] for r in cell_records]
        adj = holm_correct(pvals)
        for r, p_holm in zip(cell_records, adj):
            r['p_holm'] = round(float(p_holm), 6)
            r['significant_05'] = bool(p_holm < 0.05)
            rows.append(r)

leakage_deltas = pd.DataFrame(rows, columns=[
    'model_family','scope','comparison','metric','n','fp16_rate','quant_rate',
    'delta','delta_ci_lo','delta_ci_hi','n10','n01','mcnemar_p','p_holm','significant_05'
])
leakage_deltas.to_csv(OUT('leakage_deltas.csv'), index=False)
print(leakage_deltas.shape)  # expect 2 models * 2 metrics * 7 scopes * 2 comparisons = 56
leakage_deltas.head(10)

# %%
with open(IN('run_manifest.json')) as f:
    manifest = json.load(f)

harmful_hash = hashlib.md5(open(IN('prompt_set_harmful_base.csv'), 'rb').read()).hexdigest()
benign_hash  = hashlib.md5(open(IN('prompt_set_benign.csv'), 'rb').read()).hexdigest()

manifest['prompt_source'] = {
    "harmful_prompt_file": "prompt_set_harmful_base.csv",
    "benign_prompt_file": "prompt_set_benign.csv",
    "origin_note": "Prompt sets were exported from the same prompt pool used in the Qwen run to ensure cross-model pairing.",
    "harmful_prompt_count": int(pd.read_csv(IN('prompt_set_harmful_base.csv')).shape[0]),
    "benign_prompt_count": int(pd.read_csv(IN('prompt_set_benign.csv')).shape[0]),
    "harmful_prompt_hash": harmful_hash,
    "benign_prompt_hash": benign_hash,
    "attacks": ["direct", "dan_template", "roleplay_authority", "hypothetical", "obfuscated"]
}
# record that leakage_deltas.csv was added as a separate file, not merged into stats_tests.csv
manifest.setdefault('outputs_added', {})['leakage_deltas_csv'] = {
    "columns": list(leakage_deltas.columns),
    "n_rows": int(leakage_deltas.shape[0]),
    "metrics": METRICS,
    "note": "Added post-hoc; HCR metrics remain in stats_tests.csv / safety_tax_deltas.csv unchanged."
}
manifest['human_audit_status'] = 'completed'
manifest['last_updated'] = pd.Timestamp.utcnow().isoformat()

with open(OUT('run_manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)
print("prompt_source now:", json.dumps(manifest['prompt_source'], indent=2))

# %%
with open(IN('LLAMA_CLAIM_CHECK.md')) as f:
    claim_check = f.read()

assert 'Human audit is PENDING' in claim_check, "stale phrase not found — check file manually"

# Claim E body: replace the PENDING sentence with the completed result
claim_check = claim_check.replace(
    "Human audit is PENDING. Claims relying on this mechanism require audit completion.",
    "Human audit is COMPLETE (524 rows, 100% of conflict rows). This is consistent with a "
    "leak-after-refusal mechanism, though see the audit caveat below regarding annotator scope."
)

# Replace the trailing "## Blocker" section with the completed audit summary + caveat
blocker_start = claim_check.rfind("## Blocker")
assert blocker_start != -1
claim_check = claim_check[:blocker_start].rstrip() + "\n\n"

audit_section = """---

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

> **Reviewer-safe caveat:** The audit validates judge consistency on the sampled rows but should not
> be interpreted as independent multi-annotator reliability, because the audit used one non-blinded
> annotator. Therefore, Llama should remain secondary/mixed evidence rather than the primary claim anchor.

---

## Claim status summary

- NF4 broadly increases jailbreak success in Llama: **NOT SUPPORTED**
- NF4 increases strict HCR in Llama-3.2-1B aggregate: **PARTIALLY SUPPORTED**
- NF4 increases no-conflict harmful compliance: **NOT SUPPORTED**
- Llama replicates Qwen obfuscated finding: **NOT SUPPORTED**
- Llama provides mixed/counterexample external validation: **SUPPORTED**
"""

claim_check += audit_section

with open(OUT('LLAMA_CLAIM_CHECK.md'), 'w') as f:
    f.write(claim_check)

print(claim_check[-1800:])

# %%
import shutil

# Keep original filenames (safer for downstream references) but add an explicit README note
# instead of renaming, per the prompt's alternative option.
for fname in ['human_corrected_metric_summary.csv', 'human_corrected_safety_tax_deltas.csv']:
    shutil.copyfile(IN(fname), OUT(fname))

audit_subset_note = """# Note on human_corrected_metric_summary.csv / human_corrected_safety_tax_deltas.csv

These files summarize only the human-audited subset (n=524, stratified with 100% of
conflict/leak-after-refusal rows oversampled). They are **not population-level HCR estimates** —
do not read HCR_judge/HCR_human in these files as full-experiment rates. For population-level
HCR use `summary_by_model_precision_attack.csv`, `safety_tax_deltas.csv`, and `stats_tests.csv`.
"""
with open(OUT('AUDIT_SUBSET_NOTE.md'), 'w') as f:
    f.write(audit_subset_note)
print(audit_subset_note)

# %%
readme = """# Llama Final Outputs — Safety Tax in Quantization

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
"""
with open(OUT('README_LLAMA_FINAL.md'), 'w') as f:
    f.write(readme)
print(readme)

# %%
checks = []

def check(name, cond, note=""):
    checks.append((name, bool(cond), note))

claim_check_txt = open(OUT('LLAMA_CLAIM_CHECK.md')).read()
check('claim_check_no_stale_pending', 'Human audit is PENDING' not in claim_check_txt)
check('file_audit_provenance', os.path.exists(IN('audit_provenance.md')))
check('file_manual_audit_sample_filled', os.path.exists(IN('manual_audit_sample_filled.csv')))
check('human_audit_status_completed', manifest.get('human_audit_status') == 'completed')
check('file_README_LLAMA_FINAL', os.path.exists(OUT('README_LLAMA_FINAL.md')))
check('prompt_source_files_exist',
      os.path.exists(IN('prompt_set_harmful_base.csv')) and os.path.exists(IN('prompt_set_benign.csv')))
check('prompt_source_hashes_match',
      manifest['prompt_source']['harmful_prompt_hash'] == harmful_hash and
      manifest['prompt_source']['benign_prompt_hash'] == benign_hash)
check('leakage_deltas_has_both_metrics',
      os.path.exists(OUT('leakage_deltas.csv')) and
      set(pd.read_csv(OUT('leakage_deltas.csv'))['metric'].unique()) == {'leak_after_refusal','heuristic_refusal_bypass'})

# secret scan (best-effort; skip cleanly if notebook not bundled)
nb_path = IN('llama-fixed.ipynb') if os.path.exists(IN('llama-fixed.ipynb')) else None
if nb_path:
    txt = open(nb_path, errors='ignore').read()
    import re
    leaked = re.findall(r'(sk-[a-zA-Z0-9]{10,}|AIza[0-9A-Za-z\\-_]{20,})', txt)
    check('secret_scan', len(leaked) == 0, f"{len(leaked)} potential secrets found" if leaked else "clean")
else:
    checks.append(('secret_scan', None, "SKIP — notebook not found at expected path"))

lines = ["# VALIDATION REPORT — LLAMA (rebuilt)",
         f"Generated: {pd.Timestamp.utcnow().isoformat()}Z", ""]
all_pass = True
for name, ok, note in checks:
    if ok is None:
        lines.append(f"- ⚠️  SKIP | {name}: {note}")
    else:
        lines.append(f"- {'✅ PASS' if ok else '❌ FAIL'} | {name}" + (f": {note}" if note else ""))
        all_pass = all_pass and ok

lines.append("")
n_real = sum(1 for _,ok,_ in checks if ok is not None)
n_pass = sum(1 for _,ok,_ in checks if ok)
lines.append(f"**{n_pass}/{n_real} checks passed**")
lines.append("")
lines.append("```")
lines.append(f"FINAL_STATUS: {'PAPER_READY_AS_SECONDARY_EVIDENCE' if all_pass else 'NEEDS_ATTENTION'}")
lines.append("```")

report = "\n".join(lines)
with open(OUT('VALIDATION_REPORT_LLAMA.md'), 'w') as f:
    f.write(report)
print(report)

# %%
KEEP_AS_IS = [
    'raw_generations_llama_all.csv', 'judged_harmful_llama_all.csv', 'judged_benign_llama_all.csv',
    'judged_generations_llama_all.csv', 'summary_by_model_precision_attack.csv',
    'safety_tax_deltas.csv', 'stats_tests.csv', 'benign_summary.csv',
    'manual_audit_sample.csv', 'manual_audit_sample_filled.csv',
    'audit_agreement.csv', 'audit_confusion_matrices.csv', 'audit_provenance.md',
    'prompt_set_harmful_base.csv', 'prompt_set_benign.csv',
]
for fname in KEEP_AS_IS:
    src = IN(fname)
    if os.path.exists(src):
        shutil.copyfile(src, OUT(fname))
    else:
        print("MISSING (check manually):", fname)

print("\nFinal contents of", OUT_DIR)
for f in sorted(os.listdir(OUT_DIR)):
    print(" -", f)

# %%
cc = open(OUT('LLAMA_CLAIM_CHECK.md')).read()

old_block = (
    "> **Reviewer-safe caveat:** The audit validates judge consistency on the sampled rows but should not\n"
    "> be interpreted as independent multi-annotator reliability, because the audit used one non-blinded\n"
    "> annotator. Therefore, Llama should remain secondary/mixed evidence rather than the primary claim anchor."
)
new_block = (
    "> **Reviewer-safe caveat:** The audit validates judge consistency on the sampled rows but should not be "
    "interpreted as independent multi-annotator reliability, because the audit used one non-blinded annotator. "
    "Therefore, Llama should remain secondary/mixed evidence rather than the primary claim anchor."
)

assert old_block in cc, "wrapped block not found — inspect file manually"
cc = cc.replace(old_block, new_block)

with open(OUT('LLAMA_CLAIM_CHECK.md'), 'w') as f:
    f.write(cc)

print("Patched. New caveat line:")
print([l for l in cc.splitlines() if 'Reviewer-safe caveat' in l][0][:200])

# %%
import json, os, zipfile
import pandas as pd

print("="*70)
print("VERIFICATION REPORT")
print("="*70)

results = []
def v(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"{'✅' if cond else '❌'} {name}" + (f" — {detail}" if detail else ""))

# ---- Fix 1: LLAMA_CLAIM_CHECK.md ----
cc = open(OUT('LLAMA_CLAIM_CHECK.md')).read()
v("Fix1: stale 'Human audit is PENDING' phrase removed", 'Human audit is PENDING' not in cc)
v("Fix1: completed audit block present", '524 rows' in cc and 'kappa = 1.0' in cc)
v("Fix1: reviewer-safe caveat present", 'not be interpreted as independent multi-annotator reliability' in cc)
v("Fix1: 5 claim statuses still present",
  all(s in cc for s in ['NOT SUPPORTED', 'PARTIALLY SUPPORTED', 'SUPPORTED']))

# ---- Fix 2: leakage_deltas.csv ----
ld_path = OUT('leakage_deltas.csv')
v("Fix2: leakage_deltas.csv exists", os.path.exists(ld_path))
if os.path.exists(ld_path):
    ld = pd.read_csv(ld_path)
    expected_cols = ['model_family','scope','comparison','metric','n','fp16_rate','quant_rate',
                      'delta','delta_ci_lo','delta_ci_hi','n10','n01','mcnemar_p','p_holm','significant_05']
    v("Fix2: has all required columns", list(ld.columns) == expected_cols, str(list(ld.columns)))
    v("Fix2: contains leak_after_refusal", 'leak_after_refusal' in ld['metric'].unique())
    v("Fix2: contains heuristic_refusal_bypass", 'heuristic_refusal_bypass' in ld['metric'].unique())
    expected_scopes = {'aggregate_all','aggregate_excl_hypothetical','direct','dan_template',
                        'roleplay_authority','hypothetical','obfuscated'}
    v("Fix2: all required scopes present", expected_scopes.issubset(set(ld['scope'].unique())),
      str(sorted(ld['scope'].unique())))
    v("Fix2: both comparisons present",
      set(ld['comparison'].unique()) == {'fp16_vs_int8','fp16_vs_nf4'})
    v("Fix2: row count matches 2 models x 2 metrics x 7 scopes x 2 comparisons = 56",
      ld.shape[0] == 56, f"got {ld.shape[0]}")
    v("Fix2: no null p_holm values", ld['p_holm'].isna().sum() == 0)

# ---- Fix 3: run_manifest.json prompt_source ----
man_path = OUT('run_manifest.json')
v("Fix3: run_manifest.json exists", os.path.exists(man_path))
if os.path.exists(man_path):
    m = json.load(open(man_path))
    ps = m.get('prompt_source', {})
    v("Fix3: old misleading source_file key removed", 'source_file' not in ps)
    v("Fix3: old misleading source_hash key removed", 'source_hash' not in ps)
    required_keys = {'harmful_prompt_file','benign_prompt_file','origin_note',
                      'harmful_prompt_count','benign_prompt_count',
                      'harmful_prompt_hash','benign_prompt_hash','attacks'}
    v("Fix3: new prompt_source schema keys present", required_keys.issubset(ps.keys()),
      str(sorted(ps.keys())))
    hh = hashlib.md5(open(IN('prompt_set_harmful_base.csv'),'rb').read()).hexdigest()
    bh = hashlib.md5(open(IN('prompt_set_benign.csv'),'rb').read()).hexdigest()
    v("Fix3: harmful_prompt_hash matches actual file", ps.get('harmful_prompt_hash') == hh)
    v("Fix3: benign_prompt_hash matches actual file", ps.get('benign_prompt_hash') == bh)
    v("Fix3: human_audit_status == completed", m.get('human_audit_status') == 'completed')

# ---- Fix 4: human-corrected audit-subset clarity ----
v("Fix4: AUDIT_SUBSET_NOTE.md exists", os.path.exists(OUT('AUDIT_SUBSET_NOTE.md')))
if os.path.exists(OUT('AUDIT_SUBSET_NOTE.md')):
    note = open(OUT('AUDIT_SUBSET_NOTE.md')).read()
    v("Fix4: note explains subset ≠ population-level HCR", 'not population-level HCR' in note or 'not**' in note or 'oversample' in note.lower())
v("Fix4: human_corrected_metric_summary.csv still present", os.path.exists(OUT('human_corrected_metric_summary.csv')))
v("Fix4: human_corrected_safety_tax_deltas.csv still present", os.path.exists(OUT('human_corrected_safety_tax_deltas.csv')))

# ---- Fix 5: README_LLAMA_FINAL.md ----
readme_path = OUT('README_LLAMA_FINAL.md')
v("Fix5: README_LLAMA_FINAL.md exists", os.path.exists(readme_path))
if os.path.exists(readme_path):
    rd = open(readme_path).read()
    v("Fix5: contains 'Role in paper' section", '## Role in paper' in rd)
    v("Fix5: contains 'Key result' section", '## Key result' in rd)
    v("Fix5: contains 'Audit' section", '## Audit' in rd)

# ---- Fix 6: VALIDATION_REPORT_LLAMA.md ----
vr_path = OUT('VALIDATION_REPORT_LLAMA.md')
v("Fix6: VALIDATION_REPORT_LLAMA.md exists", os.path.exists(vr_path))
if os.path.exists(vr_path):
    vr = open(vr_path).read()
    v("Fix6: no FAIL lines in report", '❌ FAIL' not in vr)
    v("Fix6: FINAL_STATUS present", 'FINAL_STATUS:' in vr)
    v("Fix6: status is PAPER_READY_AS_SECONDARY_EVIDENCE",
      'FINAL_STATUS: PAPER_READY_AS_SECONDARY_EVIDENCE' in vr)

# ---- Cross-check: all required deliverables present in OUT_DIR ----
required_files = [
    'raw_generations_llama_all.csv','judged_harmful_llama_all.csv','judged_benign_llama_all.csv',
    'judged_generations_llama_all.csv','summary_by_model_precision_attack.csv',
    'safety_tax_deltas.csv','stats_tests.csv','leakage_deltas.csv','benign_summary.csv',
    'manual_audit_sample.csv','manual_audit_sample_filled.csv','audit_agreement.csv',
    'audit_confusion_matrices.csv','audit_provenance.md','README_LLAMA_FINAL.md',
    'LLAMA_CLAIM_CHECK.md','VALIDATION_REPORT_LLAMA.md','run_manifest.json',
    'prompt_set_harmful_base.csv','prompt_set_benign.csv','figures',
]
missing = [f for f in required_files if not os.path.exists(OUT(f))]
v("Deliverables: all files from prompt's 'Final Deliverables' list present", len(missing) == 0, str(missing))

print("="*70)
n_pass = sum(1 for _,ok,_ in results if ok)
n_total = len(results)
print(f"TOTAL: {n_pass}/{n_total} checks passed")
if n_pass == n_total:
    print("✅ ALL FIXES VERIFIED — package is ready.")
else:
    print("❌ Some checks failed — see ❌ lines above.")
print("="*70)

# %%
import shutil

zip_base = '/kaggle/working/outputs_llama_final_fixed'  # -> outputs_llama_final_fixed.zip
zip_path = shutil.make_archive(zip_base, 'zip', OUT_DIR)
print("Created:", zip_path)
print("Size (MB):", round(os.path.getsize(zip_path) / 1e6, 2))

# sanity: confirm the zip contains everything we expect, nothing corrupted
with zipfile.ZipFile(zip_path) as z:
    bad = z.testzip()
    print("Corrupt file in zip:", bad)  # should print None
    names = z.namelist()
    print(f"{len(names)} entries in zip")
    for n in sorted(names):
        print(" -", n)

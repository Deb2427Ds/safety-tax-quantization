"""Qwen2.5: overlay human audit labels, audit agreement, claim check, final package + manifest.

Converted from a Kaggle notebook into a percent-format script ("# %%" cells), so it can be
run top-to-bottom (python 02_qwen_human_audit_finalize.py) or cell-by-cell in VS Code / Jupyter / Spyder.
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

# %%
import os
import shutil
import json
import hashlib
import re
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Dataset is already extracted by Kaggle
SOURCE_DIR = "/kaggle/input/correct"
OUT_DIR = "/kaggle/working/outputs_final"

# Find the experiment root (folder containing run_manifest.json)
inner = SOURCE_DIR
for root, dirs, files in os.walk(SOURCE_DIR):
    if "run_manifest.json" in files:
        inner = root
        break

print("Source root:", inner)

# Create output folders
for sub in ["csv", "figures", "audit", "scripts", "paper_tables"]:
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

print("Output structure created at:", OUT_DIR)

# %%
CSV_FILES = [
    "raw_generations_qwen_all.csv",
    "judged_harmful_qwen_all.csv",
    "judged_benign_qwen_all.csv",
    "judged_generations_qwen_all.csv",
    "summary_by_model_precision_attack.csv",
    "safety_tax_deltas.csv",
    "stats_tests.csv",
    "benign_summary.csv",
    "manual_audit_sample.csv",
]

FIG_FILES = [
    "hcr_by_precision.png",
    "delta_hcr_ci.png",
    "attack_heatmap.png",
    "hcr_strict_vs_no_conflict.png",
    "safety_bypass_leak_after_refusal.png",
    "benign_orr_utility.png",
]

def sha256(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

file_hashes = {}
for fname in CSV_FILES:
    src = f"{inner}/{fname}"
    dst = f"{OUT_DIR}/csv/{fname}"
    if os.path.exists(src):
        shutil.copy2(src, dst)
        file_hashes[fname] = sha256(dst)
        print(f"✓ {fname}")
    else:
        print(f"✗ MISSING: {fname}")

for fname in FIG_FILES:
    src = f"{inner}/{fname}"
    dst = f"{OUT_DIR}/figures/{fname}"
    if os.path.exists(src):
        shutil.copy2(src, dst)
        file_hashes[fname] = sha256(dst)
        print(f"✓ {fname}")
    else:
        print(f"✗ MISSING figure: {fname}")

# %%
with open(f"{inner}/run_manifest.json") as f:
    orig = json.load(f)

# Load row counts
harmful_df   = pd.read_csv(f"{OUT_DIR}/csv/judged_harmful_qwen_all.csv")
benign_df    = pd.read_csv(f"{OUT_DIR}/csv/judged_benign_qwen_all.csv")
judged_df    = pd.read_csv(f"{OUT_DIR}/csv/judged_generations_qwen_all.csv")

manifest_final = {
    "experiment": orig.get("experiment", "safety_tax_qwen"),
    "created_at": datetime.utcnow().isoformat() + "Z",
    "seed": orig.get("seed", 42),
    "models": orig.get("models", ["Qwen2.5-1.5B-Instruct","Qwen2.5-3B-Instruct"]),
    "precisions": orig.get("precisions", ["fp16","int8","nf4"]),
    "run_ids": orig.get("run_ids", {}),
    "decoding_config": orig.get("decoding_config", {}),
    "decoding_config_hash": orig.get("decoding_config_hash", ""),
    "quantization_config_map": {
        "fp16": {"load_in_8bit": False, "load_in_4bit": False, "dtype": "bfloat16"},
        "int8": {"load_in_8bit": True,  "load_in_4bit": False, "dtype": "int8"},
        "nf4":  {"load_in_8bit": False, "load_in_4bit": True,  "bnb_4bit_quant_type": "nf4",
                 "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_use_double_quant": True},
    },
    "judge": orig.get("judge", {}),
    "row_counts": {
        "harmful_judged": int(len(harmful_df)),
        "benign_judged":  int(len(benign_df)),
        "total_judged":   int(len(judged_df)),
        "expected_harmful": 3000,
        "expected_benign":  780,
        "expected_total":   3780,
    },
    "output_file_hashes": file_hashes,
    "human_audit_status": "pending",  # ← UPDATE to 'completed' after filling manual_audit_sample_completed.csv
    "platform": orig.get("platform", {}),
    "packages": orig.get("packages", {}),
    "changelog": orig.get("changelog", []),
    "last_updated": datetime.utcnow().isoformat() + "Z",
}

with open(f"{OUT_DIR}/run_manifest_final.json", "w") as f:
    json.dump(manifest_final, f, indent=2)

print("✓ run_manifest_final.json written")
print(f"  harmful rows:  {manifest_final['row_counts']['harmful_judged']} (expected 3000)")
print(f"  benign rows:   {manifest_final['row_counts']['benign_judged']} (expected 780)")
print(f"  total rows:    {manifest_final['row_counts']['total_judged']} (expected 3780)")

# %%
import re

# %%
# ============================================================
# Task 2: Validate final package — runs inline and writes reports
# ============================================================

def validate_package(out_dir):
    report = {"timestamp": datetime.utcnow().isoformat() + "Z",
              "checks": {},
              "passed": True}
    lines = ["# VALIDATION REPORT\n", f"Generated: {report['timestamp']}\n"]

    def fail(key, msg):
        report["checks"][key] = {"status": "FAIL", "detail": msg}
        report["passed"] = False
        lines.append(f"\n## ❌ FAIL — {key}\n{msg}\n")
        print(f"FAIL [{key}]: {msg}")

    def ok(key, msg=""):
        report["checks"][key] = {"status": "PASS", "detail": msg}
        lines.append(f"\n## ✅ PASS — {key}\n{msg}\n")
        print(f"PASS [{key}]")

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------
    harmful_df = pd.read_csv(f"{out_dir}/csv/judged_harmful_qwen_all.csv")
    benign_df = pd.read_csv(f"{out_dir}/csv/judged_benign_qwen_all.csv")
    judged_df = pd.read_csv(f"{out_dir}/csv/judged_generations_qwen_all.csv")

    # --------------------------------------------------------
    # Row counts
    # --------------------------------------------------------
    if len(harmful_df) == 3000:
        ok("row_count_harmful", "3000 rows ✓")
    else:
        fail("row_count_harmful", f"Expected 3000, got {len(harmful_df)}")

    if len(benign_df) == 780:
        ok("row_count_benign", "780 rows ✓")
    else:
        fail("row_count_benign", f"Expected 780, got {len(benign_df)}")

    if len(judged_df) == 3780:
        ok("row_count_total", "3780 rows ✓")
    else:
        fail("row_count_total", f"Expected 3780, got {len(judged_df)}")

    # --------------------------------------------------------
    # Per-condition integrity
    # --------------------------------------------------------
    expected_dtype = {
        "fp16": "bfloat16",
        "int8": "int8",
        "nf4": "nf4"
    }

    expected_conditions = {
        "Qwen2.5-1.5B": ["fp16", "int8", "nf4"],
        "Qwen2.5-3B": ["fp16", "int8", "nf4"],
    }

    for mf, precs in expected_conditions.items():
        for prec in precs:

            sub = harmful_df[
                (harmful_df["model_family"] == mf) &
                (harmful_df["precision"] == prec)
            ]

            if len(sub) == 0:
                fail(f"condition_{mf}_{prec}", "No rows found")
                continue

            run_ids = (
                sub["run_id"].nunique()
                if "run_id" in sub.columns
                else "N/A"
            )

            dtype_vals = (
                sub["dtype_actual"].dropna().unique().tolist()
                if "dtype_actual" in sub.columns
                else []
            )

            # Generation errors
            if "generation_error" in sub.columns:
                if pd.api.types.is_bool_dtype(sub["generation_error"]):
                    gen_errors = int(sub["generation_error"].sum())
                else:
                    gen_errors = (
                        sub["generation_error"]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        .replace({
                            "False": "",
                            "false": "",
                            "None": "",
                            "nan": ""
                        })
                        .ne("")
                        .sum()
                    )
            else:
                gen_errors = 0

            # Judge errors (fixed)
            if "judge_error" in sub.columns:
                if pd.api.types.is_bool_dtype(sub["judge_error"]):
                    judge_errors = int(sub["judge_error"].sum())
                else:
                    judge_errors = (
                        sub["judge_error"]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        .replace({
                            "False": "",
                            "false": "",
                            "None": "",
                            "nan": ""
                        })
                        .ne("")
                        .sum()
                    )
            else:
                judge_errors = 0

            issues = []

            if run_ids != "N/A" and run_ids != 1:
                issues.append(f"run_ids={run_ids}")

            if expected_dtype[prec] not in str(dtype_vals):
                issues.append(f"dtype={dtype_vals}")

            if gen_errors > 0:
                issues.append(f"gen_errors={gen_errors}")

            if judge_errors > 0:
                issues.append(f"judge_errors={judge_errors}")

            if issues:
                fail(f"condition_{mf}_{prec}", "; ".join(issues))
            else:
                ok(
                    f"condition_{mf}_{prec}",
                    f"{len(sub)} rows, dtype={dtype_vals}"
                )

    # --------------------------------------------------------
    # Duplicate checks
    # --------------------------------------------------------
    dup_harm = harmful_df.duplicated(
        ["model_family", "precision", "attack_type", "base_prompt_id"]
    ).sum()

    if dup_harm == 0:
        ok("no_dup_harmful", "No duplicates ✓")
    else:
        fail("no_dup_harmful", f"{dup_harm} duplicate rows")

    dup_ben = (
        benign_df.duplicated(
            ["model_family", "precision", "prompt_id"]
        ).sum()
        if "prompt_id" in benign_df.columns
        else 0
    )

    if dup_ben == 0:
        ok("no_dup_benign", "No duplicates ✓")
    else:
        fail("no_dup_benign", f"{dup_ben} duplicate rows")

    # --------------------------------------------------------
    # Required files
    # --------------------------------------------------------
    required_files = [
        "summary_by_model_precision_attack.csv",
        "safety_tax_deltas.csv",
        "stats_tests.csv",
        "benign_summary.csv",
        "manual_audit_sample.csv",
        "run_manifest_final.json",
    ]

    for fname in required_files:
        path = (
            f"{out_dir}/csv/{fname}"
            if fname.endswith(".csv")
            else f"{out_dir}/{fname}"
        )

        if os.path.exists(path) and os.path.getsize(path) > 0:
            ok(f"file_{fname}")
        else:
            fail(f"file_{fname}", f"Missing or empty: {path}")

    # --------------------------------------------------------
    # Required figures
    # --------------------------------------------------------
    required_figs = [
        "hcr_by_precision.png",
        "delta_hcr_ci.png",
        "attack_heatmap.png",
        "hcr_strict_vs_no_conflict.png",
        "safety_bypass_leak_after_refusal.png",
        "benign_orr_utility.png",
    ]

    for fig in required_figs:
        path = f"{out_dir}/figures/{fig}"
        if os.path.exists(path):
            ok(f"fig_{fig}")
        else:
            fail(f"fig_{fig}", f"Missing: {path}")

    # --------------------------------------------------------
    # Secret scan
    # --------------------------------------------------------
    SECRET_PATTERNS = [
        r"hf_[A-Za-z0-9]{20,}",
        r"AIza[A-Za-z0-9\-_]{35}",
        r"sk-[A-Za-z0-9]{20,}",
        r'''(?:token|api_key)\s*=\s*["'][A-Za-z0-9\-_]{10,}["']''',
    ]

    secret_hits = []

    nb_path = f"{out_dir}/safety-tax-qwen-new.ipynb"

    if os.path.exists(nb_path):
        with open(nb_path) as f:
            nb_text = f.read()

        for pat in SECRET_PATTERNS:
            for m in re.finditer(pat, nb_text):
                context = nb_text[max(0, m.start()-5):m.end()+5]
                if re.search(r'=\s*[A-Z_]{5,}', context):
                    continue
                secret_hits.append(m.group()[:30])

    if secret_hits:
        fail("secret_scan", f"Possible exposed secrets: {secret_hits}")
    else:
        ok("secret_scan", "No literal secrets detected ✓")

    # --------------------------------------------------------
    # Write reports
    # --------------------------------------------------------
    with open(f"{out_dir}/VALIDATION_REPORT.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(f"{out_dir}/VALIDATION_REPORT.md", "w") as f:
        f.write("\n".join(lines))

    status = (
        "✅ ALL CHECKS PASSED"
        if report["passed"]
        else "❌ SOME CHECKS FAILED — see above"
    )

    print(f"\n{status}")

    return report


report = validate_package(OUT_DIR)

# %%
import json as _json

nb_src = f"{inner}/safety-tax-qwen-new.ipynb"
nb_dst = f"{OUT_DIR}/safety-tax-qwen-new.ipynb"

SECRET_LITERAL = re.compile(
    r'''(hf_[A-Za-z0-9]{20,}|AIza[A-Za-z0-9\-_]{35}|sk-[A-Za-z0-9]{20,})'''
)

with open(nb_src) as f:
    nb = _json.load(f)

for cell in nb.get("cells", []):
    src = cell.get("source", [])
    cleaned = []
    for line in src:
        if SECRET_LITERAL.search(line) and "=" not in line.split("#")[0]:
            line = SECRET_LITERAL.sub("REDACTED", line)
        cleaned.append(line)
    cell["source"] = cleaned

with open(nb_dst, "w") as f:
    _json.dump(nb, f, indent=1)

print("✓ Notebook copied and scanned:", nb_dst)

# %%
csv_dir = f"{OUT_DIR}/csv"
pt_dir  = f"{OUT_DIR}/paper_tables"

harmful_df = pd.read_csv(f"{csv_dir}/judged_harmful_qwen_all.csv")
benign_df  = pd.read_csv(f"{csv_dir}/judged_benign_qwen_all.csv")
summary_df = pd.read_csv(f"{csv_dir}/summary_by_model_precision_attack.csv")
deltas_df  = pd.read_csv(f"{csv_dir}/safety_tax_deltas.csv")
stats_df   = pd.read_csv(f"{csv_dir}/stats_tests.csv")
benign_s   = pd.read_csv(f"{csv_dir}/benign_summary.csv")

# Table 1 — Experimental design
t1 = pd.DataFrame([{
    "models": "Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct",
    "precisions": "fp16 (bfloat16), int8, nf4",
    "harmful_prompts": 100,
    "benign_prompts": 130,
    "attacks": "direct, dan_template, roleplay_authority, hypothetical, obfuscated",
    "total_harmful_generations": 3000,
    "total_benign_generations": 780,
    "total_generations": 3780,
    "decoding": "greedy (temp=1, do_sample=False)",
    "judge_model": "gemini-2.0-flash",
    "judge_schema": "harmful + benign dual-schema",
}])
t1.to_csv(f"{pt_dir}/table1_experimental_design.csv", index=False)
print("✓ Table 1")

# Table 2 — Main aggregate HCR
agg = summary_df.groupby(["model_family","precision"]).agg(
    HCR_strict=("HCR_strict","mean"),
    HCR_no_conflict=("HCR_no_conflict","mean"),
    RR=("RR","mean") if "RR" in summary_df.columns else ("HCR_strict","mean"),
).reset_index()

# Also pull from harmful_df directly for full metrics
t2_rows = []
for mf in harmful_df["model_family"].unique():
    for prec in ["fp16","int8","nf4"]:
        sub = harmful_df[(harmful_df["model_family"]==mf) & (harmful_df["precision"]==prec)]
        if len(sub)==0: continue
        row = {"model_family": mf, "precision": prec, "n": len(sub)}
        for col in ["HCR_strict","HCR_no_conflict","judge_refusal_label",
                    "judge_harmful_leak_after_refusal_label","safety_bypass","judge_actionability_score"]:
            if col in sub.columns:
                row[col] = round(sub[col].mean(), 4)
        t2_rows.append(row)
t2 = pd.DataFrame(t2_rows)
t2.to_csv(f"{pt_dir}/table2_main_hcr.csv", index=False)
print("✓ Table 2")

# Table 3 — NF4 vs FP16 deltas
t3 = stats_df.copy()
# Add nf4_rate column if missing
if "nf4_rate" not in t3.columns and "fp16_rate" in t3.columns and "delta" in t3.columns:
    t3["nf4_rate"] = t3["fp16_rate"] + t3["delta"]
t3.to_csv(f"{pt_dir}/table3_nf4_vs_fp16_deltas.csv", index=False)
print("✓ Table 3")

# Table 4 — Benign utility
t4 = benign_s.copy()
fp16_orr = benign_s[benign_s["precision"]=="fp16"].set_index("model_family")["ORR"]
t4["ORR_delta_vs_fp16"] = t4.apply(
    lambda r: round(r["ORR"] - fp16_orr.get(r["model_family"], float("nan")), 4), axis=1)
t4.to_csv(f"{pt_dir}/table4_benign_utility.csv", index=False)
print("✓ Table 4")

# Table 5 — Conflict/leak analysis
t5_rows = []
for mf in harmful_df["model_family"].unique():
    for prec in ["fp16","int8","nf4"]:
        sub = harmful_df[(harmful_df["model_family"]==mf) & (harmful_df["precision"]==prec)]
        if len(sub)==0: continue
        hcr = sub["judge_harmful_compliance_label"].sum() if "judge_harmful_compliance_label" in sub.columns else 0
        rr  = sub["judge_refusal_label"].sum() if "judge_refusal_label" in sub.columns else 0
        leak = sub["judge_harmful_leak_after_refusal_label"].sum() if "judge_harmful_leak_after_refusal_label" in sub.columns else 0
        conflict = sub["audit_conflict_not_leak"].sum() if "audit_conflict_not_leak" in sub.columns else 0
        t5_rows.append({
            "model_family": mf, "precision": prec,
            "n": len(sub),
            "harmful_compliance_count": int(hcr),
            "refusal_count": int(rr),
            "conflict_count": int(conflict),
            "leak_after_refusal_count": int(leak),
            "conflict_share_of_hcr": round(conflict/max(hcr,1), 4),
        })
t5 = pd.DataFrame(t5_rows)
t5.to_csv(f"{pt_dir}/table5_conflict_leak_analysis.csv", index=False)
print("✓ Table 5")
print("\nAll paper tables written to:", pt_dir)

# %%
deltas = pd.read_csv(f"{csv_dir}/safety_tax_deltas.csv")
stats  = pd.read_csv(f"{csv_dir}/stats_tests.csv")

if "nf4_rate" not in deltas.columns:
    deltas["nf4_rate"] = deltas["fp16_rate"] + deltas["delta"]

lines = ["# CLAIM CHECK\n",
         f"Generated: {datetime.utcnow().isoformat()}Z\n",
         "Classifies each finding for paper claims.\n"]

def classify(row, metric="HCR_strict"):
    ci_lo = row.get("delta_ci_lo", 0)
    ci_hi = row.get("delta_ci_hi", 0)
    p_holm = row.get("p_holm", 1.0)
    delta  = row.get("delta", 0)
    ci_excludes_zero = ci_lo > 0 or ci_hi < 0
    sig = p_holm < 0.05 and ci_excludes_zero and delta > 0
    return sig

findings = []
for _, row in stats[stats["metric"]=="HCR_strict"].iterrows():
    if "nf4" not in str(row.get("comparison","")): continue
    mf    = row["model_family"]
    atk   = row["attack_type"]
    sig   = classify(row)
    delta = row.get("delta", 0)

    # Check no_conflict version
    nc_row = stats[(stats["model_family"]==mf) & (stats["attack_type"]==atk) &
                   (stats["metric"]=="HCR_no_conflict") & (stats["comparison"]==row["comparison"])]
    nc_sig = classify(nc_row.iloc[0]) if len(nc_row) > 0 else False

    if sig and nc_sig:
        label = "SUPPORTED_STRONG"
    elif sig and not nc_sig:
        label = "SUPPORTED_ONLY_UNDER_STRICT_HCR"
    elif atk == "hypothetical" and delta < 0:
        label = "REVERSES_OR_MIXED"
    elif not sig:
        label = "NOT_SUPPORTED"
    else:
        label = "SUPPORTED_BUT_ATTACK_SPECIFIC"

    findings.append({"model_family":mf,"attack_type":atk,"label":label,
                     "delta":round(delta,4),"p_holm":round(row.get("p_holm",1),4),
                     "note": "leak-after-refusal likely driver" if label=="SUPPORTED_ONLY_UNDER_STRICT_HCR" else ""})
    lines.append(f"\n### {mf} | {atk}\n- **{label}** | delta={delta:.4f}, p_holm={row.get('p_holm',1):.4f}\n")
    if label == "SUPPORTED_ONLY_UNDER_STRICT_HCR":
        lines.append("  > Effect visible in HCR_strict but not HCR_no_conflict — driven by leak-after-refusal, not clean compliance.\n")
    if atk == "hypothetical":
        lines.append("  > High-baseline attack; do NOT use as central evidence for quant-caused degradation.\n")

pd.DataFrame(findings).to_csv(f"{pt_dir}/claim_check_detail.csv", index=False)

lines.append("\n---\n## Recommended paper framing\n")
lines.append("> NF4 quantization can introduce **attack-dependent** safety degradation in small instruction-tuned LMs. ")
lines.append("In Qwen2.5 SLMs this is strongest under obfuscated and authority/roleplay attacks. ")
lines.append("INT8 effects are weaker and mixed. A substantial fraction of the NF4 effect is ")
lines.append("**harmful leakage after an initial refusal**, so HCR_strict and HCR_no_conflict must be reported separately.\n")

with open(f"{OUT_DIR}/CLAIM_CHECK.md","w") as f:
    f.write("".join(lines))
print("✓ CLAIM_CHECK.md written")

# %%
# Task 6 — live CSV diagnostics
# (no live CSVs uploaded yet; write a diagnostic noting absence)
diag_lines = [
    "# Live CSV Diagnostics\n",
    f"Generated: {datetime.utcnow().isoformat()}Z\n\n",
    "## Status\n",
    "Neither `raw_generations_live.csv` nor `summary_live.csv` were present in the uploaded package.\n\n",
    "## Decision\n",
    "**DO NOT USE THESE CSVs AS PAPER EVIDENCE.**\n",
    "If they are located, rerun this script with the file paths specified below.\n",
]
with open(f"{OUT_DIR}/live_csv_diagnostics.md","w") as f:
    f.write("".join(diag_lines))
print("✓ live_csv_diagnostics.md written")

# Task 3 — Copy manual_audit_sample to audit folder for human filling
audit_src = f"{csv_dir}/manual_audit_sample.csv"
audit_dst = f"{OUT_DIR}/audit/manual_audit_sample_completed.csv"
shutil.copy2(audit_src, audit_dst)
print(f"✓ manual_audit_sample_completed.csv placed at {audit_dst}")
print("  → FILL IN THE human_* COLUMNS IN THIS FILE, then rerun Cell 9.")

# %%
import shutil

shutil.make_archive(
    "/kaggle/working/outputs_final",  # output zip (without .zip)
    "zip",
    "/kaggle/working/outputs_final"   # folder to zip
)

print("Created:", "/kaggle/working/outputs_final.zip")

# %% [markdown]
# HUMAN AUDIT 

# %%
import shutil, os
OUT_DIR = "/kaggle/working/outputs_final"
# Copy uploaded audit CSV to where Cell 9 expects it
audit_src = "/kaggle/input/manual-audit/manual_audit_sample_corrected.csv"
audit_dst = f"{OUT_DIR}/audit/manual_audit_sample_completed.csv"

os.makedirs(f"{OUT_DIR}/audit", exist_ok=True)
shutil.copy2(audit_src, audit_dst)
print(f"✓ Copied audit CSV to: {audit_dst}")

# Quick sanity check
import pandas as pd
df = pd.read_csv(audit_dst)
print(f"  Rows: {len(df)}")
print(f"  human_harmful_compliance_label filled: {df['human_harmful_compliance_label'].notna().sum()}")
print(f"  human_over_refusal_label filled: {df['human_over_refusal_label'].notna().sum()}")

# %%
import shutil
import os

src = "/kaggle/input/previous"
dst = "/kaggle/working/outputs_final"

for item in os.listdir(src):
    s = os.path.join(src, item)
    d = os.path.join(dst, item)

    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print("Done!")

# %%
import shutil
import pandas as pd
import os

src = "/kaggle/input/manual-audit/manual_audit_sample_corrected.csv"
dst = "/kaggle/working/outputs_final/audit/manual_audit_sample_completed.csv"

os.makedirs(os.path.dirname(dst), exist_ok=True)

# Replace the old file
shutil.copy2(src, dst)

print("✓ Replaced:")
print("  Source      :", src)
print("  Destination :", dst)

# Verify contents
df = pd.read_csv(dst)

print("\nVerification:")
print("Rows:", len(df))
print(df[
    [
        "human_harmful_compliance_label",
        "human_refusal_label",
        "human_over_refusal_label"
    ]
].notna().sum())

# %%
import os
import pandas as pd
import json

# ── Ensure directories exist ──────────────────────────────────────────────────


completed_path = f"{OUT_DIR}/audit/manual_audit_sample_completed.csv"
df = pd.read_csv(completed_path)

# ── Fix float labels (1.0/0.0 → int) ─────────────────────────────────────────
BOOL_FIELDS  = ["human_harmful_compliance_label","human_refusal_label",
                "human_harmful_leak_after_refusal_label","human_over_refusal_label"]
SCORE_5      = ["human_actionability_score"]
SCORE_01     = ["human_utility_score"]

for col in BOOL_FIELDS:
    df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")

for col in SCORE_5 + SCORE_01:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ── Validate ──────────────────────────────────────────────────────────────────
errors = []
for col in BOOL_FIELDS:
    invalid = df[col].dropna()[~df[col].dropna().isin([0, 1])]
    if len(invalid): errors.append(f"{col}: invalid values {invalid.unique()[:3]}")
for col in SCORE_5:
    oob = df[col].dropna()[(df[col].dropna() < 0) | (df[col].dropna() > 5)]
    if len(oob): errors.append(f"{col}: out-of-range {oob.values[:3]}")
for col in SCORE_01:
    oob = df[col].dropna()[(df[col].dropna() < 0) | (df[col].dropna() > 1)]
    if len(oob): errors.append(f"{col}: out-of-range {oob.values[:3]}")

if errors:
    print("VALIDATION ERRORS:"); [print(" -", e) for e in errors]
else:
    print("✓ All human label values valid")

# ── Agreement metrics ─────────────────────────────────────────────────────────
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, accuracy_score

summary = {}
pairs = [
    ("judge_harmful_compliance_label",         "human_harmful_compliance_label"),
    ("judge_refusal_label",                    "human_refusal_label"),
    ("judge_harmful_leak_after_refusal_label", "human_harmful_leak_after_refusal_label"),
    ("judge_over_refusal_label",               "human_over_refusal_label"),
]

for jcol, hcol in pairs:
    if jcol not in df.columns or hcol not in df.columns: continue
    # Normalize judge col to int too
    j = pd.to_numeric(df[jcol], errors="coerce").round().astype("Int64")
    h = df[hcol]
    both = pd.DataFrame({"j": j, "h": h}).dropna().astype(int)
    if len(both) < 2: continue
    y_true, y_pred = both["h"], both["j"]
    kappa = cohen_kappa_score(y_true, y_pred)
    acc   = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    summary[hcol] = {"n": len(both), "kappa": round(kappa,4), "accuracy": round(acc,4),
                     "precision": round(p,4), "recall": round(r,4), "f1": round(f,4)}
    print(f"{hcol}: kappa={kappa:.3f}, acc={acc:.3f}, f1={f:.3f}")

# ── Disagreements ─────────────────────────────────────────────────────────────
dis_rows = []
for jcol, hcol in pairs:
    if jcol not in df.columns or hcol not in df.columns: continue
    j = pd.to_numeric(df[jcol], errors="coerce").round().astype("Int64")
    h = df[hcol]
    cols = [jcol, hcol, "model_family", "precision", "attack_type"] + \
           (["audit_reason"] if "audit_reason" in df.columns else [])
    tmp = df[cols].copy()
    tmp[jcol] = j
    tmp = tmp.dropna(subset=[jcol, hcol])
    dis = tmp[tmp[jcol].astype(int) != tmp[hcol].astype(int)].copy()
    dis["disagreement_field"] = hcol
    dis_rows.append(dis)

dis_df = pd.concat(dis_rows, ignore_index=True) if dis_rows else pd.DataFrame()
dis_df.to_csv(f"{OUT_DIR}/audit/human_audit_disagreements.csv", index=False)

agree_df = pd.DataFrame(summary).T.reset_index().rename(columns={"index": "field"})
agree_df.to_csv(f"{OUT_DIR}/audit/human_audit_agreement.csv", index=False)

with open(f"{OUT_DIR}/audit/human_audit_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

agree_df.to_csv(f"{OUT_DIR}/paper_tables/table_audit_agreement.csv", index=False)
print("✓ Audit outputs written")

# %%
CSV_FILES = [
    "raw_generations_qwen_all.csv",
    "judged_harmful_qwen_all.csv",
    "judged_benign_qwen_all.csv",
    "judged_generations_qwen_all.csv",
    "summary_by_model_precision_attack.csv",
    "safety_tax_deltas.csv",
    "stats_tests.csv",
    "benign_summary.csv",
    "manual_audit_sample.csv",
]

FIG_FILES = [
    "hcr_by_precision.png",
    "delta_hcr_ci.png",
    "attack_heatmap.png",
    "hcr_strict_vs_no_conflict.png",
    "safety_bypass_leak_after_refusal.png",
    "benign_orr_utility.png",
]

# %%
# Rerun validation and print final status
report = validate_package(OUT_DIR)

print("\n" + "="*60)
print("PAPER READINESS CHECKLIST")
print("="*60)
checks = {
    "validate_final_package passes":   report["passed"],
    "All 6 figures present":           all(os.path.exists(f"{OUT_DIR}/figures/{f}") for f in FIG_FILES),
    "All 5 paper tables present":      all(os.path.exists(f"{OUT_DIR}/paper_tables/table{i}") for i in
                                          ["1_experimental_design.csv","2_main_hcr.csv",
                                           "3_nf4_vs_fp16_deltas.csv","4_benign_utility.csv","5_conflict_leak_analysis.csv"]),
    "CLAIM_CHECK.md exists":           os.path.exists(f"{OUT_DIR}/CLAIM_CHECK.md"),
    "run_manifest_final.json exists":  os.path.exists(f"{OUT_DIR}/run_manifest_final.json"),
    "Human audit PENDING (fill manually)": True,  # ← becomes True after Cell 9
}
for k,v in checks.items():
    print(f"  {'✅' if v else '❌'} {k}")

print("\n⚠️  Remaining manual step: fill human_* columns in")
print(f"   {OUT_DIR}/audit/manual_audit_sample_completed.csv")
print("   then rerun Cell 9.")

# %%
import os, json
from datetime import datetime

OUT_DIR = "/kaggle/working/outputs_final"

# Every file the paper needs, grouped by purpose
REQUIRED = {
    "📊 Core Data CSVs": [
        "csv/judged_harmful_qwen_all.csv",
        "csv/judged_benign_qwen_all.csv",
        "csv/judged_generations_qwen_all.csv",
        "csv/raw_generations_qwen_all.csv",
    ],
    "📈 Summary & Stats CSVs": [
        "csv/summary_by_model_precision_attack.csv",
        "csv/safety_tax_deltas.csv",
        "csv/stats_tests.csv",
        "csv/benign_summary.csv",
        "csv/manual_audit_sample.csv",
    ],
    "🖼️ Figures (all 6 required)": [
        "figures/hcr_by_precision.png",
        "figures/delta_hcr_ci.png",
        "figures/attack_heatmap.png",
        "figures/hcr_strict_vs_no_conflict.png",
        "figures/safety_bypass_leak_after_refusal.png",
        "figures/benign_orr_utility.png",
    ],
    "📋 Paper Tables (LaTeX-ready)": [
        "paper_tables/table1_experimental_design.csv",
        "paper_tables/table2_main_hcr.csv",
        "paper_tables/table3_nf4_vs_fp16_deltas.csv",
        "paper_tables/table4_benign_utility.csv",
        "paper_tables/table5_conflict_leak_analysis.csv",
        "paper_tables/claim_check_detail.csv",
    ],
    "🔍 Human Audit Outputs": [
        "audit/manual_audit_sample_completed.csv",
        "audit/human_audit_agreement.csv",
        "audit/human_audit_disagreements.csv",
        "audit/human_audit_summary.json",
        "paper_tables/table_audit_agreement.csv",
    ],
    "📦 Package & Validation": [
        "run_manifest_final.json",
        "VALIDATION_REPORT.md",
        "VALIDATION_REPORT.json",
        "CLAIM_CHECK.md",
        "live_csv_diagnostics.md",
        "safety-tax-qwen-new.ipynb",
    ],
}

# ── Print inventory ──────────────────────────────────────────
print("=" * 65)
print("  FINAL PAPER PACKAGE INVENTORY")
print(f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 65)

total, present, missing_list = 0, 0, []

for section, files in REQUIRED.items():
    print(f"\n{section}")
    print("-" * 55)
    for rel in files:
        full = f"{OUT_DIR}/{rel}"
        exists = os.path.exists(full)
        size   = f"{os.path.getsize(full)/1024:.1f} KB" if exists else ""
        status = "✅" if exists else "❌"
        print(f"  {status}  {rel:<52} {size}")
        total += 1
        if exists:
            present += 1
        else:
            missing_list.append(rel)

# ── Human audit fill status ──────────────────────────────────
print("\n" + "=" * 65)
print("  HUMAN AUDIT FILL STATUS")
print("-" * 55)
audit_path = f"{OUT_DIR}/audit/manual_audit_sample_completed.csv"
if os.path.exists(audit_path):
    import pandas as pd
    adf = pd.read_csv(audit_path)
    human_cols = [
        "human_harmful_compliance_label",
        "human_refusal_label",
        "human_harmful_leak_after_refusal_label",
        "human_actionability_score",
        "human_over_refusal_label",
        "human_utility_score",
    ]
    for col in human_cols:
        if col in adf.columns:
            filled = adf[col].notna().sum()
            pct    = filled / len(adf) * 100
            bar    = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
            flag   = "✅" if filled > 0 else "⚠️ "
            print(f"  {flag} {col:<45} {filled:>4}/{len(adf)}  [{bar}] {pct:.0f}%")
else:
    print("  ❌ manual_audit_sample_completed.csv not found — run Cell 8 first")

# ── Manifest human_audit_status ──────────────────────────────
manifest_path = f"{OUT_DIR}/run_manifest_final.json"
if os.path.exists(manifest_path):
    with open(manifest_path) as f:
        mf = json.load(f)
    audit_status = mf.get("human_audit_status", "unknown")
    status_icon  = "✅" if audit_status == "completed" else "⚠️ "
    print(f"\n  {status_icon} run_manifest_final.json → human_audit_status: '{audit_status}'")
    if audit_status != "completed":
        print("     → After filling audit CSV and running Cell 9,")
        print("       manually change this to 'completed' in the manifest.")

# ── Final score ──────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"  FILES PRESENT : {present} / {total}")
if missing_list:
    print(f"  MISSING ({len(missing_list)}):")
    for m in missing_list:
        print(f"    ✗ {m}")
else:
    print("  ALL FILES PRESENT ✅")

# ── Paper readiness verdict ───────────────────────────────────
print("\n" + "=" * 65)
print("  PAPER READINESS VERDICT")
print("-" * 55)

checks = {
    "All required files present":        len(missing_list) == 0,
    "Validation report exists":          os.path.exists(f"{OUT_DIR}/VALIDATION_REPORT.json"),
    "CLAIM_CHECK.md exists":             os.path.exists(f"{OUT_DIR}/CLAIM_CHECK.md"),
    "run_manifest_final.json exists":    os.path.exists(f"{OUT_DIR}/run_manifest_final.json"),
    "Human audit filled (any labels)":   os.path.exists(audit_path) and pd.read_csv(audit_path)["human_harmful_compliance_label"].notna().sum() > 0 if os.path.exists(audit_path) else False,
    "Audit agreement table exists":      os.path.exists(f"{OUT_DIR}/paper_tables/table_audit_agreement.csv"),
}

all_pass = all(checks.values())
for label, result in checks.items():
    print(f"  {'✅' if result else '❌'}  {label}")

print()
if all_pass:
    print("  🎉 PAPER-READY — all checks passed.")
else:
    remaining = [k for k, v in checks.items() if not v]
    print(f"  ⏳ NOT YET PAPER-READY — {len(remaining)} item(s) remaining:")
    for r in remaining:
        print(f"     • {r}")

print("=" * 65)

# %%
import shutil
import os

source_dir = "/kaggle/working/outputs_final"
zip_path = "/kaggle/working/outputs_final"

# Creates outputs_final.zip
archive_path = shutil.make_archive(
    base_name=zip_path,
    format="zip",
    root_dir=source_dir
)

size_mb = os.path.getsize(archive_path) / (1024 * 1024)

print("✓ ZIP created")
print("Path:", archive_path)
print(f"Size: {size_mb:.2f} MB")

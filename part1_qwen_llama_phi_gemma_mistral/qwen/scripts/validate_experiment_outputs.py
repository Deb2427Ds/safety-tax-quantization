#!/usr/bin/env python3
"""
validate_experiment_outputs.py — final pipeline-level validation (prompt §8).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

EXPECTED_MODELS = ["Qwen2.5-1.5B", "Qwen2.5-3B"]
EXPECTED_PRECISIONS = ["fp16", "int8", "nf4"]
EXPECTED_HARMFUL_PER_CONDITION = 500
EXPECTED_BENIGN_PER_CONDITION = 130
EXPECTED_TOTAL_RAW = len(EXPECTED_MODELS) * len(EXPECTED_PRECISIONS) * (
    EXPECTED_HARMFUL_PER_CONDITION + EXPECTED_BENIGN_PER_CONDITION
)

PLOT_ALIASES = {
    "hcr_by_precision.png": ["hcr_by_precision.png", "plot1_hcr.png", "plot1_hcr_by_precision_attack.png"],
    "delta_hcr_ci.png": ["delta_hcr_ci.png", "plot_delta_hcr_ci.png", "plot3_delta_ci.png"],
    "attack_heatmap.png": ["attack_heatmap.png", "plot_attack_heatmap.png", "plot4_heatmap.png"],
    "safety_bypass_leak_after_refusal.png": [
        "safety_bypass_leak_after_refusal.png", "plot_safety_bypass.png", "plot5_safety_bypass.png",
    ],
    "benign_orr_utility.png": ["benign_orr_utility.png", "plot2_orr_corrected.png", "plot6_orr_utility.png"],
}

REQUIRED_SUMMARY_CSVS = [
    "summary_by_model_precision_attack.csv", "benign_summary.csv", "safety_tax_deltas.csv",
    "stats_tests.csv", "manual_audit_sample.csv", "run_manifest.json",
]


def fail(errors, msg):
    errors.append(msg)


def check_row_counts(out, errors):
    raw_path = out / "raw_generations_qwen_all.csv"
    if not raw_path.exists():
        fail(errors, f"missing {raw_path}")
        return
    raw = pd.read_csv(raw_path)
    if len(raw) != EXPECTED_TOTAL_RAW:
        fail(errors, f"raw_generations_qwen_all.csv has {len(raw)} rows, expected {EXPECTED_TOTAL_RAW}")
    for fam in EXPECTED_MODELS:
        for prec in EXPECTED_PRECISIONS:
            sub = raw[(raw.model_family == fam) & (raw.precision == prec)]
            n_harm = (sub.dataset_split == "harmful").sum()
            n_ben = (sub.dataset_split == "benign").sum()
            if n_harm != EXPECTED_HARMFUL_PER_CONDITION:
                fail(errors, f"{fam}/{prec}: {n_harm} harmful rows, expected {EXPECTED_HARMFUL_PER_CONDITION}")
            if n_ben != EXPECTED_BENIGN_PER_CONDITION:
                fail(errors, f"{fam}/{prec}: {n_ben} benign rows, expected {EXPECTED_BENIGN_PER_CONDITION}")


def check_single_run_id(out, errors):
    raw_path = out / "raw_generations_qwen_all.csv"
    if not raw_path.exists():
        return
    raw = pd.read_csv(raw_path)
    grp = raw.groupby(["model_family", "precision"])["run_id"].nunique()
    for (fam, prec), n in grp[grp > 1].items():
        ids = raw[(raw.model_family == fam) & (raw.precision == prec)]["run_id"].unique().tolist()
        fail(errors, f"{fam}/{prec} has {n} distinct run_ids in raw: {ids}")

    for fname in ["judged_harmful_qwen_all.csv", "judged_generations_qwen_all.csv"]:
        p = out / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        grp = df.groupby(["model_family", "precision"])["run_id"].nunique()
        for (fam, prec), n in grp[grp > 1].items():
            ids = df[(df.model_family == fam) & (df.precision == prec)]["run_id"].unique().tolist()
            fail(errors, f"{fname}: {fam}/{prec} has {n} distinct run_ids: {ids}")


def check_duplicates(out, errors):
    for fname, key, split in [
        ("judged_harmful_qwen_all.csv", ["model_family", "precision", "attack_type", "base_prompt_id"], "harmful"),
        ("judged_benign_qwen_all.csv", ["model_family", "precision", "prompt_id"], "benign"),
    ]:
        p = out / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "dataset_split" in df.columns:
            df = df[df["dataset_split"] == split]
        dupe_mask = df.duplicated(subset=key, keep=False)
        if dupe_mask.any():
            fail(errors, f"{fname}: {dupe_mask.sum()} duplicate rows on key {key}")


def check_judge_and_generation_errors(out, errors):
    for fname in ["judged_harmful_qwen_all.csv", "judged_benign_qwen_all.csv", "judged_generations_qwen_all.csv"]:
        p = out / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "judge_error" in df.columns:
            n_err = df["judge_error"].fillna(False).astype(str).str.lower().eq("true").sum()
            if n_err:
                fail(errors, f"{fname}: {n_err} rows with judge_error=True")
        if "generation_error" in df.columns:
            ge = df["generation_error"]
            bad = ge[~(ge.isna() | (ge == False) | (ge.astype(str).str.lower() == "false"))]
            if len(bad):
                fail(errors, f"{fname}: {len(bad)} rows with generation_error set")


def check_plots_and_summaries(out, errors):
    fig_dir = out / "figures"
    for slot, aliases in PLOT_ALIASES.items():
        if not any((fig_dir / a).exists() for a in aliases):
            fail(errors, f"missing required plot for '{slot}' (tried {aliases})")
    for fname in REQUIRED_SUMMARY_CSVS:
        if not (out / fname).exists():
            fail(errors, f"missing required summary file: {fname}")


def check_stats_aggregates(out, errors):
    p = out / "stats_tests.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    if "attack_type" not in df.columns:
        fail(errors, "stats_tests.csv missing 'attack_type' column")
        return
    vals = set(df["attack_type"].unique())
    for required in ("aggregate_all", "aggregate_excl_hypothetical"):
        if required not in vals:
            fail(errors, f"stats_tests.csv missing '{required}' aggregate rows")


def run_subscript(script_name, args, errors):
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        fail(errors, f"required script missing: {script_path}")
        return
    result = subprocess.run([sys.executable, str(script_path)] + args, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        fail(errors, f"{script_name} reported failures (see output above)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs/")
    args = parser.parse_args()
    out = Path(args.outputs)

    errors = []

    print("=== 1. Secret scan ===")
    run_subscript("scan_secrets.py", [str(out.parent)], errors)

    print("\n=== 2. Row-level metadata validation ===")
    run_subscript("validate_outputs.py", ["--outputs", str(out)], errors)

    print("\n=== 3. Row counts ===")
    check_row_counts(out, errors)

    print("\n=== 4. Single run_id per condition ===")
    check_single_run_id(out, errors)

    print("\n=== 5. Duplicate row checks ===")
    check_duplicates(out, errors)

    print("\n=== 6. Judge / generation error checks ===")
    check_judge_and_generation_errors(out, errors)

    print("\n=== 7. Plots and summary files ===")
    check_plots_and_summaries(out, errors)

    print("\n=== 8. Stats aggregate rows ===")
    check_stats_aggregates(out, errors)

    print("\n" + "=" * 60)
    if errors:
        print(f"❌ validate_experiment_outputs.py FAILED — {len(errors)} issue(s):\n")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("✅ validate_experiment_outputs.py: all checks passed.")
        print("\nNOTE: human audit must still be reviewed before paper drafting begins.")
        sys.exit(0)


if __name__ == "__main__":
    main()

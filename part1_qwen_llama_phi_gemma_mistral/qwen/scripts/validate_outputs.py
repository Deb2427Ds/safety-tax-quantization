#!/usr/bin/env python3
"""
validate_outputs.py — row-level metadata completeness checks (prompt §2).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

VALID_PRECISIONS = {"fp16", "int8", "nf4"}
VALID_SPLITS = {"harmful", "benign"}

REQUIRED_NONNULL_COLS = [
    "run_id", "model_family", "model_hf_id", "precision", "dtype_actual",
    "decoding_config_hash", "dataset_split", "prompt_id", "base_prompt_id", "response_text",
]

CANONICAL_QCONFIG = {
    "int8": json.dumps({"load_in_8bit": True, "compute_dtype": "bfloat16"}, sort_keys=True),
    "nf4": json.dumps(
        {"load_in_4bit": True, "quant_type": "nf4", "double_quant": True, "compute_dtype": "bfloat16"},
        sort_keys=True,
    ),
}

HARMFUL_JUDGE_COLS = [
    "judge_harmful_compliance_label", "judge_refusal_label",
    "judge_harmful_leak_after_refusal_label", "judge_actionability_score",
    "judge_harm_category", "judge_confidence",
]

BENIGN_JUDGE_COLS = ["judge_over_refusal_label", "judge_answered_label", "judge_utility_score"]


def normalize_qconfig(value):
    if pd.isna(value) or value in (None, "", "nan", "None"):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return "UNPARSEABLE:" + value
    return json.dumps(value, sort_keys=True)


def check_dataframe(df, name, errors, is_harmful=None, is_benign=None):
    n = len(df)
    print(f"\n--- {name} ({n} rows) ---")

    for col in REQUIRED_NONNULL_COLS:
        if col not in df.columns:
            errors.append(f"{name}: missing required column '{col}'")
            continue
        n_null = df[col].isna().sum()
        if n_null:
            errors.append(f"{name}: column '{col}' has {n_null} null values")

    if "precision" in df.columns:
        bad = set(df["precision"].dropna().unique()) - VALID_PRECISIONS
        if bad:
            errors.append(f"{name}: invalid precision values: {bad}")

    if {"dtype_actual", "model_family", "precision"}.issubset(df.columns):
        grp = df.groupby(["model_family", "precision"])["dtype_actual"].nunique()
        mixed = grp[grp > 1]
        for (fam, prec), n_distinct in mixed.items():
            vals = df[(df.model_family == fam) & (df.precision == prec)]["dtype_actual"].unique().tolist()
            errors.append(f"{name}: mixed dtype_actual for ({fam}, {prec}): {vals}")

    if {"quantization_config", "precision"}.issubset(df.columns):
        for prec in ("int8", "nf4"):
            sub = df[df["precision"] == prec]
            if len(sub) == 0:
                continue
            normed = sub["quantization_config"].apply(normalize_qconfig)
            n_null = normed.isna().sum()
            if n_null:
                errors.append(f"{name}: {n_null} {prec} rows missing quantization_config")
            bad_canon = normed.dropna()
            bad_canon = bad_canon[bad_canon != CANONICAL_QCONFIG[prec]]
            if len(bad_canon):
                errors.append(
                    f"{name}: {len(bad_canon)} {prec} rows have non-canonical quantization_config "
                    f"(expected {CANONICAL_QCONFIG[prec]})"
                )

    if "dataset_split" in df.columns:
        bad = set(df["dataset_split"].dropna().unique()) - VALID_SPLITS
        if bad:
            errors.append(f"{name}: invalid dataset_split values: {bad}")

    if "generation_error" in df.columns:
        ge = df["generation_error"]
        bad = ge[~(ge.isna() | (ge == False) | (ge.astype(str).str.lower() == "false"))]
        if len(bad):
            errors.append(f"{name}: {len(bad)} rows have generation_error set (truthy)")

    if is_harmful:
        for col in HARMFUL_JUDGE_COLS:
            if col not in df.columns:
                errors.append(f"{name}: missing harmful judge column '{col}'")
                continue
            n_null = df[col].isna().sum()
            if n_null:
                errors.append(f"{name}: harmful judge column '{col}' has {n_null} nulls")

    if is_benign:
        for col in BENIGN_JUDGE_COLS:
            if col not in df.columns:
                errors.append(f"{name}: missing benign judge column '{col}'")
                continue
            n_null = df[col].isna().sum()
            if n_null:
                errors.append(f"{name}: benign judge column '{col}' has {n_null} nulls")

    n_errors_here = len([e for e in errors if e.startswith(name)])
    print(f"  -> {n_errors_here} issue(s)" if n_errors_here else "  -> OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs/")
    args = parser.parse_args()
    out = Path(args.outputs)

    errors = []

    harmful_path = out / "judged_harmful_qwen_all.csv"
    benign_path = out / "judged_benign_qwen_all.csv"
    combined_path = out / "judged_generations_qwen_all.csv"

    if harmful_path.exists():
        check_dataframe(pd.read_csv(harmful_path), "judged_harmful_qwen_all.csv", errors, is_harmful=True, is_benign=False)
    else:
        errors.append(f"missing file: {harmful_path}")

    if benign_path.exists():
        check_dataframe(pd.read_csv(benign_path), "judged_benign_qwen_all.csv", errors, is_harmful=False, is_benign=True)
    else:
        errors.append(f"missing file: {benign_path}")

    if combined_path.exists():
        combined = pd.read_csv(combined_path)
        check_dataframe(combined, "judged_generations_qwen_all.csv", errors, is_harmful=False, is_benign=False)
        h = combined[combined["dataset_split"] == "harmful"]
        b = combined[combined["dataset_split"] == "benign"]
        for col in HARMFUL_JUDGE_COLS:
            if col in h.columns and h[col].isna().any():
                errors.append(f"judged_generations_qwen_all.csv: harmful rows missing '{col}' ({h[col].isna().sum()} nulls)")
        for col in BENIGN_JUDGE_COLS:
            if col in b.columns and b[col].isna().any():
                errors.append(f"judged_generations_qwen_all.csv: benign rows missing '{col}' ({b[col].isna().sum()} nulls)")
    else:
        errors.append(f"missing file: {combined_path}")

    print("\n" + "=" * 60)
    if errors:
        print(f"❌ validate_outputs.py FAILED — {len(errors)} issue(s):\n")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("✅ validate_outputs.py: all checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()

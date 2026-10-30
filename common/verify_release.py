"""Executable verification of the FINAL per-family result packages (run before building results/).

Every check is an assertion-style test computed from the data, not a hard-coded flag. A family is
'data' checks must all pass for a family's tables to be released; 'metadata' checks (manifest) decide only
whether the original run manifest is released. build_results_release.py enforces this.

Usage:
    PACKAGES_ROOT=/path/to/unzipped/packages python common/verify_release.py [--out results/VERIFICATION_REPORT.md]

PACKAGES_ROOT must contain the final release packages in the layout documented in PACKAGES below
(the same layout used by common/cross_family_analysis in its original form).
"""
import argparse, hashlib, json, os, re, sys
import numpy as np
import pandas as pd

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "prompts")
ATT = ["direct", "dan_template", "roleplay_authority", "hypothetical", "obfuscated"]
PREC = ["fp16", "int8", "nf4"]

# family -> (package dir relative to PACKAGES_ROOT, expected number of checkpoints)
PACKAGES = {
    "qwen":    ("final/csv", 2),
    "llama":   ("llama_final_fixed", 2),
    "phi":     ("phi_final_corrected/outputs_phi_final", 2),
    "gemma":   ("gemma_final_corrected_20260801_025157", 3),
    "mistral": ("mistral_final_corrected_v2_20260803_074315", 1),
    "smollm":  ("smollm_final_release/outputs_smollm_final_release", 1),
    "granite": ("outputs_granite_final_release_v2", 1),
}
# raw-generation file used for the generation-error check
RAW = {"qwen": "raw_generations_qwen_all.csv", "llama": "raw_generations_llama_all.csv", "phi": "raw_generations_phi_all.csv",
       "gemma": "raw_generations_gemma_all.csv", "mistral": "raw_generations_mistral_all.csv",
       "smollm": "raw_generations_smollm_normalized.csv", "granite": "raw_generations_granite_normalized.csv"}
# Qwen's release keeps its manifest/claim files one level above csv/
PKG_TOP = {"qwen": "final"}


def b(s):
    """Parse booleans safely (never astype(bool) on strings)."""
    m = {True: True, False: False, 1: True, 0: False, "1": True, "0": False, "true": True, "false": False,
         "True": True, "False": False, "TRUE": True, "FALSE": False, 1.0: True, 0.0: False}
    out = s.map(m)
    if out.isna().any():
        raise ValueError(f"unparseable booleans: {s[out.isna()].unique()[:5]}")
    return out.astype(bool)


def norm(t):
    return " ".join(str(t).split())


def attack_col(d):
    c = "canonical_attack_type" if "canonical_attack_type" in d else "attack_type"
    return d[c].replace({"dan": "dan_template", "roleplay": "roleplay_authority"})


def base_col(d):
    return d["canonical_base_prompt_id"] if "canonical_base_prompt_id" in d else d["base_prompt_id"]


def find_hashmap(obj):
    """Return {path: md5-or-sha256 hex} dicts found anywhere in a manifest (the study manifests use MD5)."""
    found = {}
    if isinstance(obj, dict):
        vals = list(obj.values())
        if vals and all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{32}|[0-9a-f]{64}", v) for v in vals):
            found.update(obj)
        else:
            for v in vals:
                found.update(find_hashmap(v))
    elif isinstance(obj, list):
        for v in obj:
            found.update(find_hashmap(v))
    return found


def verify(fam, root):
    rel, n_models = PACKAGES[fam]
    P = os.path.join(root, rel)
    R = []  # (check, passed, detail)

    def chk(name, cond, detail=""):
        R.append((name, bool(cond), str(detail)))

    H = pd.read_csv(os.path.join(P, f"judged_harmful_{fam}_all.csv"), low_memory=False)
    B = pd.read_csv(os.path.join(P, f"judged_benign_{fam}_all.csv"), low_memory=False)
    H["_att"], H["_base"] = attack_col(H), base_col(H)
    models = sorted(H.model_family.unique())

    chk("n_checkpoints", len(models) == n_models, f"{len(models)} == {n_models}: {models}")
    chk("harmful_rows", len(H) == 1500 * n_models, f"{len(H)} == {1500*n_models}")
    chk("benign_rows", len(B) == 390 * n_models, f"{len(B)} == {390*n_models}")
    chk("precisions_exact", set(H.precision) == set(PREC) and set(B.precision) == set(PREC), sorted(set(H.precision)))
    chk("attacks_canonical", set(H._att) == set(ATT), sorted(set(H._att)))

    hc = H.groupby(["model_family", "precision", "_att"]).size()
    chk("harmful_cells_all_100", (hc == 100).all() and len(hc) == 15 * n_models, f"{len(hc)} cells, min={hc.min()}, max={hc.max()}")
    bc = B.groupby(["model_family", "precision"]).size()
    chk("benign_cells_all_130", (bc == 130).all() and len(bc) == 3 * n_models, f"{len(bc)} cells")
    chk("harmful_keys_unique", not H.duplicated(["model_family", "precision", "_att", "_base"]).any())
    bk = "canonical_prompt_id" if "canonical_prompt_id" in B else "prompt_id"
    chk("benign_keys_unique", not B.duplicated(["model_family", "precision", bk]).any())
    # the SAME prompt text must be used in all three precisions (a paired comparison is meaningless otherwise)
    # uses the raw generations (the stored prompt_text is exactly what was fed to the model)
    Rw0 = pd.read_csv(os.path.join(P, RAW[fam]), low_memory=False, keep_default_na=False)
    Rw0 = Rw0[Rw0.dataset_split == "harmful"].copy()
    Rw0["_a"], Rw0["_b"], Rw0["_t"] = attack_col(Rw0), base_col(Rw0), Rw0.prompt_text.map(norm)
    nu = Rw0.groupby(["model_family", "_a", "_b"])._t.nunique()
    bad_pairs = nu[nu > 1].reset_index()
    bad_models = sorted(bad_pairs.model_family.unique())
    chk("prompts_identical_across_precisions", len(bad_pairs) == 0,
        f"{len(bad_pairs)} of {len(nu)} (checkpoint, attack, base-prompt) groups have different prompt text across precisions"
        + (f"; affected checkpoints: {bad_models}; attacks: {sorted(bad_pairs._a.unique())}" if len(bad_pairs) else ""))
    # pairing: every (model, attack, base) exists in all three precisions
    pv = H.groupby(["model_family", "_att", "_base"]).precision.nunique()
    chk("pairing_complete_3_precisions", (pv == 3).all(), f"{len(pv)} pairs")

    # generation errors (raw file) and judge errors
    Rw = pd.read_csv(os.path.join(P, RAW[fam]), low_memory=False)
    ge = Rw["generation_error"].fillna("").astype(str).str.strip().replace({"nan": "", "False": "", "0": ""})
    chk("generation_errors_zero", (ge == "").all(), f"{int((ge != '').sum())} errors / {len(Rw)} rows")
    FALSY = {"False", "false", "0", "0.0", "", "nan", "None"}
    for nm, D in (("harmful", H), ("benign", B)):
        # judge_error may hold booleans or an error string; anything not falsy is an error
        je = D["judge_error"].notna() & ~D["judge_error"].astype(str).str.strip().isin(FALSY) if "judge_error" in D else pd.Series(False, index=D.index)
        res = pd.Series(False, index=D.index)
        if "manual_resolution_flag" in D:
            res |= b(D["manual_resolution_flag"].fillna(False))
        if "judge_label_source" in D:
            res |= D["judge_label_source"].astype(str).eq("human_resolution")
        unresolved = int((je & ~res).sum())
        chk(f"judge_unresolved_errors_{nm}", unresolved == 0, f"{int(je.sum())} flagged, {int((je & res).sum())} resolved by human review, {unresolved} unresolved")
    # labels parse and are non-null
    for c in ("judge_harmful_compliance_label", "judge_refusal_label"):
        chk(f"{c}_parses_nonnull", H[c].notna().all() and b(H[c]).size == len(H))

    # decoding hash: one per checkpoint across its precisions
    dh = H.groupby("model_family").decoding_config_hash.nunique()
    chk("one_decoding_hash_per_checkpoint", (dh == 1).all(), dict(dh))

    # identical prompts to data/prompts
    hp = pd.read_csv(os.path.join(PROMPT_DIR, "harmful_base_prompts.csv"))
    bp = pd.read_csv(os.path.join(PROMPT_DIR, "benign_prompts.csv"))
    hs, bs = set(map(norm, hp.prompt_text)), set(map(norm, bp.prompt_text))
    d = H[(H._att == "direct")]
    hcol = "prompt_text_base" if "prompt_text_base" in H else "prompt_text"
    got_h = set(map(norm, d[hcol])) if hcol == "prompt_text" else set(map(norm, H[hcol]))
    chk("harmful_prompts_identical_to_data_prompts", got_h == hs, f"{len(got_h & hs)}/100")
    chk("benign_prompts_identical_to_data_prompts", set(map(norm, B.prompt_text)) == bs, f"{len(set(map(norm, B.prompt_text)) & bs)}/130")

    # stats table vs recomputation from rows (aggregate_all, strict HCR, NF4 and INT8 vs FP16)
    S = pd.read_csv(os.path.join(P, "stats_tests.csv"))
    mcol = "model_family" if "model_family" in S else None
    scol = "scope" if "scope" in S else "attack_type"
    metric = S["metric"].astype(str)
    sub = S[(S[scol].astype(str) == "aggregate_all") & (metric.str.upper().isin(["HCR_STRICT"]))]
    H["_C"] = b(H.judge_harmful_compliance_label)
    ok_all, notes = True, []
    for m in models:
        for q in ("int8", "nf4"):
            a = H[(H.model_family == m) & (H.precision == "fp16")].set_index(["_att", "_base"])["_C"]
            z = H[(H.model_family == m) & (H.precision == q)].set_index(["_att", "_base"])["_C"].reindex(a.index)
            delta = (z.astype(float) - a.astype(float)).mean()
            r = sub[sub.comparison.astype(str).str.contains(q)]
            if mcol:
                r = r[r[mcol] == m]
            if len(r) != 1:
                ok_all = False; notes.append(f"{m}/{q}: {len(r)} stats rows"); continue
            v = float(r.delta.iloc[0])
            match = min(abs(v - delta), abs(v - delta * 100)) < 1e-6
            ok_all &= match
            if not match:
                notes.append(f"{m}/{q}: table {v:.4f} vs rows {delta:.4f}")
    chk("stats_tests_aggregate_strict_delta_matches_rows", ok_all, "; ".join(notes) or "all NF4/INT8 deltas recomputed from rows")

    # human-corrected tables: n limits and no impossible per-cell counts
    hs_path = os.path.join(P, "human_corrected_stats_tests.csv")
    if os.path.exists(hs_path):
        HS = pd.read_csv(hs_path)
        sc = "scope" if "scope" in HS else "attack_type"
        lim = HS[sc].astype(str).map(lambda s: 500 if s == "aggregate_all" else 400 if s == "aggregate_excl_hypothetical" else 100)
        chk("human_corrected_stats_n_within_scope_size", (HS["n"] <= lim).all() and (HS["n"] > 0).all(),
            f"max n={HS['n'].max()}; scopes={sorted(HS[sc].astype(str).unique())}")
        chk("human_corrected_stats_n_exact", (HS["n"] == lim).all(), "n equals 500 / 400 / 100 for every row")
    ms_path = os.path.join(P, "human_corrected_metric_summary.csv")
    if os.path.exists(ms_path):
        MS = pd.read_csv(ms_path)
        for c in [c for c in MS.columns if c.startswith("n_human_corrected")]:
            chk(f"{c}_within_cell_size", (MS[c].fillna(0) <= 100).all(), f"max={MS[c].max()}")
        if "n" in MS:
            chk("human_corrected_summary_n_not_expanded", (MS["n"] <= 500).all(), f"max n={MS['n'].max()}")

    # manifest hashes
    mf = [f for f in os.listdir(os.path.join(root, PKG_TOP.get(fam, rel))) if f.startswith("run_manifest") and f.endswith(".json")]
    top = os.path.join(root, PKG_TOP.get(fam, rel))
    canon = None
    if mf:
        best = None
        for f in mf:
            hm = find_hashmap(json.load(open(os.path.join(top, f))))
            if hm and (best is None or len(hm) > len(best[1])):
                best = (f, hm)
        if best:
            f, hm = best
            canon = f
            bad, miss, legacy, selfref = [], 0, 0, 0
            def digest(path, hv):
                h = hashlib.md5() if len(hv) == 32 else hashlib.sha256()
                h.update(open(path, "rb").read())
                return h.hexdigest()
            for rp, hv in hm.items():
                if rp.startswith("/"):          # legacy absolute Kaggle path = hash of a pre-fix file version; not mappable to the release
                    legacy += 1; continue
                if os.path.basename(rp) == f:   # a manifest cannot contain a valid hash of itself
                    selfref += 1; continue
                cands = [os.path.join(base, sub, rp) for base in (top, P) for sub in ("", "figures")]
                fp = next((c for c in cands if os.path.isfile(c)), None)
                if fp is None:
                    miss += 1
                elif digest(fp, hv) != hv:
                    bad.append(rp)
            n_ok = len(hm) - legacy - selfref - miss - len(bad)
            chk("manifest_hashes_match_files", not bad and miss == 0 and n_ok > 0,
                f"{f}: {n_ok} current entries match; {len(bad)} mismatch {[os.path.basename(x) for x in bad[:4]]}; {miss} missing; "
                f"ignored: {legacy} legacy absolute-path entries (pre-fix hashes), {selfref} self-hash entry")
        else:
            chk("manifest_has_hash_map", False, f"{mf}: no sha256 map found")
    else:
        chk("manifest_present", False, "no run_manifest*.json")
    # manifest audit status must agree with the audit evidence in the package
    evidence = any(os.path.exists(os.path.join(d, f)) for d in (top, P) for f in ("audit_agreement.csv", "audit/human_audit_summary.json"))
    if canon is None and mf:
        canon = sorted(mf, key=lambda x: ("final" not in x, x))[0]
    status = json.load(open(os.path.join(top, canon))).get("human_audit_status") if canon else None
    if evidence and status is not None:
        chk("manifest_audit_status_consistent_with_audit_files", str(status).lower() == "completed",
            f"{canon} says human_audit_status={status!r}; package contains completed audit files")
    else:
        chk("manifest_audit_status_consistent_with_audit_files", True, "no status field in manifest" if status is None else "no audit evidence")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--families", nargs="*", default=list(PACKAGES))
    a = ap.parse_args()
    root = os.environ.get("PACKAGES_ROOT", "./packages")
    lines = ["# Verification report for the final per-family packages", "",
             "Generated by `common/verify_release.py`. Each check is computed from the data. 'data' checks gate the release of a family's result tables; 'metadata' checks (manifest hashes/status) gate only whether its original manifest is released.", ""]
    verdicts = {}
    for fam in a.families:
        try:
            res = verify(fam, root)
        except Exception as e:  # a crash is a failure, not a pass
            res = [("verification_crashed", False, repr(e))]
        meta = lambda n: n.startswith("manifest_")
        data_ok = all(p for n, p, _ in res if not meta(n))
        meta_ok = all(p for n, p, _ in res if meta(n))
        verdicts[fam] = {"data": data_ok, "metadata": meta_ok}
        label = "VERIFIED" if data_ok and meta_ok else ("DATA VERIFIED, manifest withheld (metadata issue)" if data_ok else "NOT VERIFIED")
        ok = data_ok and meta_ok
        lines += [f"## {fam}: {label}", "", "| check | kind | result | detail |", "|---|---|---|---|"]
        lines += [f"| {n} | {'metadata' if meta(n) else 'data'} | {'pass' if p else '**FAIL**'} | {d[:170]} |" for n, p, d in res]
        lines.append("")
        print(f"{fam:8} {label}  ({sum(p for _, p, _ in res)}/{len(res)} checks)")
        for n, p, d in res:
            if not p:
                print(f"    FAIL {n}: {d}")
    if a.out:
        open(a.out, "w").write("\n".join(lines) + "\n")
    json.dump(verdicts, open(os.path.join(os.path.dirname(a.out) or ".", "verified_families.json"), "w")) if a.out else None
    sys.exit(0 if all(v['data'] for v in verdicts.values()) else 1)


if __name__ == "__main__":
    main()

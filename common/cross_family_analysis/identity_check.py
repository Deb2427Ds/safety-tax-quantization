"""Metric identity check (paper Table `tab:identity`), computed row by row from the judged harmful tables.

Verifies, for every checkpoint, that strict compliance C equals no-conflict compliance NC plus
compliance-with-refusal (C and R) on every one of the 1,500 harmful rows, and quantifies how the separately
judged leakage label L differs from (C and R).

Writes metric_identity_check.csv and tab_identity_rows.tex (consumed by tables.py).
Run after load.py's inputs exist:   python identity_check.py   (reads ../../results by default)
"""
import pandas as pd
from load import load, ORDER

D = load()
rows, tex = [], []
for m in ORDER:
    d = D[D.model == m]
    cr = d.C & d.R
    gap = float((d.C.astype(int) - (d.NC.astype(int) + cr.astype(int))).abs().max())
    assert gap == 0, (m, "identity C = NC + (C&R) violated")
    r = dict(model=m, strict=round(d.C.mean() * 100, 3), NC=round(d.NC.mean() * 100, 3),
             **{"C&R": round(cr.mean() * 100, 3), "L": round(d.L.mean() * 100, 3)},
             identity_gap=gap, **{"L_only(L&~C&R)": int((d.L & ~cr).sum()), "CR_only(C&R&~L)": int((cr & ~d.L).sum())},
             rows=len(d))
    rows.append(r)
    tex.append(f"{m} & {d.C.mean()*100:.2f}\\% & {d.NC.mean()*100:.2f}\\% & {cr.mean()*100:.2f}\\% & "
               f"{d.L.mean()*100:.2f}\\% & {r['L_only(L&~C&R)']} & {r['CR_only(C&R&~L)']}\\\\")
pd.DataFrame(rows).to_csv("metric_identity_check.csv", index=False)
open("tab_identity_rows.tex", "w").write("\n".join(tex) + "\n")
print("identity holds on every row for all", len(rows), "checkpoints")

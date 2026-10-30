import numpy as np, pandas as pd
from load import load,ORDER
import sys
if len(ORDER)<12:
    sys.exit(f"tables.py builds the 12-checkpoint manuscript tables; only {len(ORDER)} checkpoints are verified/present. See results/VERIFICATION_REPORT.md.")
D=load(); R=pd.read_csv("cluster_inference_all.csv"); T=pd.read_csv("primary_family_results.csv")
def fp(p): return "$<$0.001" if p<0.001 else f"{p:.3f}"
def f1(x): return f"{x:+.1f}".replace("-","$-$")
def f2(x):
    v=x; return (f"{v:+.1f}" if abs(v*10-round(v*10))<1e-6 else f"{v:+.2f}").replace("-","$-$")
def rate(m,pr,met="C",mask=None):
    a=D[(D.model==m)&(D.precision==pr)]
    if mask is not None: a=a[mask(a)]
    return a[met].mean()*100
def get(m,comp,sc,met): 
    r=R[(R.model==m)&(R.comparison==comp)&(R.scope==sc)&(R.metric==met)]; assert len(r)==1; return r.iloc[0]
fam={"Qwen2.5-1.5B":"Qwen","Qwen2.5-3B":"Qwen","Llama-3.2-1B":"Llama","Llama-3.2-3B":"Llama","Gemma-3-1B":"Gemma","Gemma-2-2B":"Gemma","Gemma-3-4B":"Gemma","Phi-3.5-mini":"Phi","Phi-3-mini-4k":"Phi","Ministral-3-3B":"Mistral","SmolLM2-1.7B":"SmolLM","Granite-3.1-2B":"Granite"}
def bold(cells,sig): return ["\\textbf{"+c+"}" for c in cells] if sig else cells
# ---- Table main (cluster-aware, Holm-12)
rows=[]
for m in ORDER:
    r=get(m,"nf4","aggregate_all","HCR_strict"); sig=r.p_holm12_cluster<0.05
    rows.append(" & ".join(bold([fam[m],m,f"{rate(m,'fp16'):.1f}\\%",f"{rate(m,'nf4'):.1f}\\%",f1(r.delta_pp),f"[{f1(r.cl_lo)}, {f1(r.cl_hi)}]",fp(r.p_cluster_perm),fp(r.p_holm12_cluster)],sig))+"\\\\")
main=r"""\begin{table}[!htbp]
\centering
\caption{Primary analysis: aggregate strict harmful compliance (HCR), FP16 vs.\ NF4 ($n=500$ paired prompts per checkpoint, 100 base-prompt clusters). $\Delta$ is in percentage points; the 95\% interval resamples the 100 base prompts (all five attacks kept together); $p$ is a two-sided base-prompt sign-flip permutation test and $\pHolm$ is Holm-adjusted across the twelve checkpoints (primary family). Bold rows are significant at $\alpha=0.05$. The FP16 column is the FP16-labeled reference (mixed bfloat16/float16 runtime for Granite).}
\label{tab:main}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcccccc}
\toprule
Family & Model & FP16 & NF4 & $\Delta$ (pp) & 95\% CI (pp) & $p$ & $\pHolm$\\
\midrule
"""+"\n".join(rows)+r"""
\bottomrule
\end{tabular}
\end{table}
"""
# ---- INT8 table
rows=[]
for m in ORDER:
    r=get(m,"int8","aggregate_all","HCR_strict")
    rows.append(f"{m} & {rate(m,'fp16'):.1f}\\% & {rate(m,'int8'):.1f}\\% & {f1(r.delta_pp)} & [{f1(r.cl_lo)}, {f1(r.cl_hi)}] & {fp(r.p_cluster_perm)} & {fp(r.p_holm12_cluster)}\\\\")
int8=r"""\begin{table}[!htbp]
\centering
\caption{Aggregate strict HCR, FP16 vs.\ INT8, for all twelve checkpoints ($n=500$ paired prompts; 100 base-prompt clusters). Intervals and $p$ as in Table~\ref{tab:main}; $\pHolm$ is Holm-adjusted across the twelve checkpoints within the INT8 family. No row is significant.}
\label{tab:int8_all}
\small
\begin{tabular}{lcccccc}
\toprule
Model & FP16 & INT8 & $\Delta$ (pp) & 95\% CI (pp) & $p$ & $\pHolm$\\
\midrule
"""+"\n".join(rows)+r"""
\bottomrule
\end{tabular}
\end{table}
"""
# ---- excl-hyp table
rows=[]
nohyp=lambda a:a.attack!="hypothetical"
for m in ORDER:
    a=get(m,"nf4","aggregate_excl_hypothetical","HCR_strict"); b=get(m,"nf4","aggregate_excl_hypothetical","HCR_no_conflict")
    f=rate(m,"fp16","C",nohyp); n=rate(m,"nf4","C",nohyp)
    sig=a.p_holm12_cluster<0.05
    rows.append(" & ".join([m,f"{f:.2f}\\%",f"{n:.2f}\\%",f2(a.delta_pp),f"[{f1(a.cl_lo)}, {f1(a.cl_hi)}]",fp(a.p_holm12_cluster),f2(b.delta_pp),fp(b.p_holm12_cluster)])+"\\\\")
exh=r"""\begin{table}[!htbp]
\centering
\caption{Secondary analysis: aggregate NF4 versus FP16 excluding the hypothetical attack ($n=400$ paired prompts; 100 base-prompt clusters). $\Delta$ is in percentage points; the interval is the base-prompt cluster bootstrap; $\pHolm$ is Holm-adjusted across the twelve checkpoints within each metric.}
\label{tab:exclhyp_all}
\footnotesize
\begin{tabular}{lcccccc c}
\toprule
 & \multicolumn{5}{c}{Strict HCR} & \multicolumn{2}{c}{No-conflict HCR}\\
\cmidrule(lr){2-6}\cmidrule(lr){7-8}
Model & FP16 & NF4 & $\Delta$ (pp) & 95\% CI (pp) & $\pHolm$ & $\Delta$ (pp) & $\pHolm$\\
\midrule
"""+"\n".join(rows)+r"""
\bottomrule
\end{tabular}
\end{table}
"""
exh=exh.replace("lcccccc c","lccccccc")
# ---- decomposition table (adds leakage)
rows=[]
interp={"Qwen2.5-1.5B":"Leakage after refusal","Qwen2.5-3B":"Leakage after refusal","Llama-3.2-1B":"Leakage after refusal (hypothetical-driven)","Llama-3.2-3B":"No effect","Gemma-3-1B":"Leakage after refusal (dominant)","Gemma-2-2B":"Weak / mixed","Gemma-3-4B":"No effect","Phi-3.5-mini":"Leakage after refusal (100\\%)","Phi-3-mini-4k":"Leakage after refusal (directional)","Ministral-3-3B":"Null / mixed","SmolLM2-1.7B":"\\textbf{Direct-compliance shift} (refusal $-17.4$)","Granite-3.1-2B":"Null / mixed"}
star=lambda r:"$^{*}$" if r.p_holm12_cluster<0.05 else ""
for m in ORDER:
    s=get(m,"nf4","aggregate_all","HCR_strict"); n=get(m,"nf4","aggregate_all","HCR_no_conflict"); l=get(m,"nf4","aggregate_all","leakage")
    rows.append(f"{fam[m]} & {m} & {f1(s.delta_pp)}{star(s)} & {f1(n.delta_pp)}{star(n)} & {f1(l.delta_pp)}{star(l)} & {interp[m]}\\\\")
dec=r"""\begin{table}[!htbp]
\centering
\caption{Decomposition of the aggregate NF4$-$FP16 change (pp) into strict HCR, no-conflict HCR (direct compliance without refusal), and separately judged leakage after refusal, for all twelve checkpoints. $^{*}$: significant after Holm correction across the twelve checkpoints within each metric (base-prompt cluster permutation test). A large gap between strict and no-conflict change indicates that the effect is leakage after refusal.}
\label{tab:decomp}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabularx}{\linewidth}{@{}llrrr@{\hspace{10pt}}>{\raggedright\arraybackslash}X@{}}
\toprule
Family & Model & $\Delta$ strict & $\Delta$ no-conflict & $\Delta$ leakage & Interpretation\\
\midrule
"""+"\n".join(rows)+r"""
\bottomrule
\end{tabularx}
\end{table}
"""
# ---- SmolLM2 table
rows=[]
for met,lab in [("HCR_strict","Strict HCR"),("HCR_no_conflict","No-conflict HCR (direct)"),("leakage","Leakage after refusal"),("refusal_rate","Refusal rate")]:
    r=get("SmolLM2-1.7B","nf4","aggregate_all",met); code={"HCR_strict":"C","HCR_no_conflict":"NC","leakage":"L","refusal_rate":"R"}[met]
    sig=r.p_holm12_cluster<0.05
    rows.append(" & ".join(bold([lab,f"{rate('SmolLM2-1.7B','fp16',code):.1f}\\%",f"{rate('SmolLM2-1.7B','nf4',code):.1f}\\%",f1(r.delta_pp),f"[{f1(r.cl_lo)}, {f1(r.cl_hi)}]",fp(r.p_holm12_cluster)],sig))+"\\\\")
sm=r"""\begin{table}[!htbp]
\centering
\caption{SmolLM2-1.7B: aggregate FP16 vs.\ NF4 ($n=500$ paired prompts; 100 base-prompt clusters). Interval: base-prompt cluster bootstrap; $\pHolm$: Holm-adjusted across the twelve checkpoints within each metric. Bold rows are significant.}
\label{tab:smol_agg}
\small
\begin{tabular}{lccccc}
\toprule
Metric & FP16 & NF4 & $\Delta$ (pp) & 95\% CI (pp) & $\pHolm$\\
\midrule
"""+"\n".join(rows)+r"""
\bottomrule
\end{tabular}
\end{table}
"""
# ---- sensitivity (NF4) and INT8 appendix tables
hdr=r"""\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccccc}
\toprule
 & & \multicolumn{2}{c}{95\% CI (pp)} & \multicolumn{2}{c}{Prompt-level McNemar} & \multicolumn{2}{c}{Base-prompt clusters} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
Model & $\Delta$ (pp) & Prompt-level & Cluster & $\pHolm^{\text{within}}$ & $\pHolm^{(12)}$ & $p$ & $\pHolm^{(12)}$\\
\midrule
"""
def sens(comp):
    L=[]
    for x in T[T.comparison==comp].itertuples():
        cells=[x.model,f1(x.delta_pp),f"[{f1(x.row_ci_lo)}, {f1(x.row_ci_hi)}]",f"[{f1(x.cl_ci_lo)}, {f1(x.cl_ci_hi)}]",fp(x.p_holm_within_model),fp(x.p_holm12_mcnemar),fp(x.p_cluster_perm),fp(x.p_holm12_cluster)]
        L.append(" & ".join(bold(cells,x.p_holm12_cluster<0.05))+"\\\\")
    return "\n".join(L)
tail="\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
sens_nf4="\\begin{table}[!htbp]\n\\centering\n\\caption{Sensitivity of the primary NF4 result (aggregate strict HCR) to the unit of analysis and the testing family. \\emph{Prompt-level}: the original analysis treating the 500 prompt pairs as independent (paired bootstrap interval; McNemar test). $\\pHolm^{\\text{within}}$: Holm within model; $\\pHolm^{(12)}$: Holm across the twelve checkpoints. \\emph{Base-prompt clusters}: resampling the 100 base prompts (sign-flip permutation test). Bold: significant after cross-checkpoint Holm correction under the cluster analysis. The same four checkpoints are significant under all three approaches.}\n\\label{tab:primary_sens}\n"+hdr+sens("nf4")+tail
sens_int8="\\begin{table}[!htbp]\n\\centering\n\\caption{Sensitivity of the INT8 result (aggregate strict HCR), with the same columns as Table~\\ref{tab:primary_sens}. No checkpoint is significant under any approach.}\n\\label{tab:primary_sens_int8}\n"+hdr+sens("int8")+tail
ident=r"""\begin{table}[!htbp]
\centering
\caption{Metric identity check over all 1,500 harmful rows per checkpoint. Strict HCR equals no-conflict HCR plus the rate of compliance with refusal ($C\wedge R$) on every row (gap $=0$ for all checkpoints). ``L only'' counts rows with judged leakage but not $C\wedge R$; ``$C\wedge R$ only'' counts rows with compliance and refusal but no judged leakage.}
\label{tab:identity}
\footnotesize
\begin{tabular}{lccccrr}
\toprule
Model & Strict & No-conflict & $C\wedge R$ & Leakage $L$ & L only & $C\wedge R$ only\\
\midrule
"""+open("tab_identity_rows.tex").read()+r"""\bottomrule
\end{tabular}
\end{table}
"""
import json
json.dump({"main":main,"int8":int8,"exh":exh,"dec":dec,"sm":sm,"sens_nf4":sens_nf4,"sens_int8":sens_int8,"ident":ident},open("newtables.json","w"))
print("ok",len(main),len(sens_nf4))

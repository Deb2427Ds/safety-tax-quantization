import os
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
T=pd.read_csv("primary_family_results.csv"); ORDER=list(T[T.comparison=="nf4"].model)
OUT=os.environ.get("FIG_OUT","./figures/").rstrip("/")+"/"; os.makedirs(OUT,exist_ok=True)
# NF4 forest (cluster CI, Holm across 12 checkpoints)
n=T[T.comparison=="nf4"].set_index("model").reindex(ORDER)[::-1]
fig,ax=plt.subplots(figsize=(6.8,4.8))
for i,(m,r) in enumerate(n.iterrows()):
    sig=r.p_holm12_cluster<0.05; c="#1b9e77" if sig else "#888888"
    ax.plot([r.cl_ci_lo,r.cl_ci_hi],[i,i],color=c,lw=2); ax.scatter([r.delta_pp],[i],color=c,s=55,zorder=3)
ax.axvline(0,color="k",ls="--",lw=1); ax.set_yticks(range(len(n))); ax.set_yticklabels(n.index,fontsize=9)
ax.set_xlabel(r"$\Delta$ Strict HCR, NF4 $-$ FP16 (percentage points)",fontsize=9)
ax.set_title("NF4-induced change in strict HCR\n(base-prompt cluster bootstrap 95% CI, 100 clusters)",fontsize=9)
ax.legend(handles=[Line2D([0],[0],marker='o',color='#1b9e77',label='Significant (Holm across 12 checkpoints)',lw=2),Line2D([0],[0],marker='o',color='#888888',label='Not significant',lw=2)],fontsize=7,loc="lower right")
plt.tight_layout(); plt.savefig(OUT+"cross_family_delta_hcr.png",dpi=200); plt.close()
# INT8 vs NF4
fig,ax=plt.subplots(figsize=(7.0,5.0)); ys=np.arange(len(ORDER))[::-1]
for comp,c,off,lab in [("int8","#4c72b0",0.17,"INT8"),("nf4","#c44e52",-0.17,"NF4")]:
    t=T[T.comparison==comp].set_index("model").reindex(ORDER)
    for k,(y,m) in enumerate(zip(ys,ORDER)):
        r=t.loc[m]; sig=r.p_holm12_cluster<0.05
        ax.plot([r.cl_ci_lo,r.cl_ci_hi],[y+off]*2,color=c,lw=1.6)
        ax.scatter([r.delta_pp],[y+off],s=34,color=c if sig else "white",edgecolor=c,zorder=3,linewidth=1.4,label=lab if k==0 else None)
ax.axvline(0,color="k",ls="--",lw=1); ax.set_yticks(ys); ax.set_yticklabels(ORDER,fontsize=8)
ax.set_xlabel(r"$\Delta$ strict HCR vs FP16 (percentage points)",fontsize=9)
ax.set_title("INT8 and NF4 change in aggregate strict HCR\n(base-prompt cluster bootstrap 95% CI; filled = significant after Holm across 12 checkpoints)",fontsize=8.5)
ax.legend(fontsize=8,loc="lower right"); plt.tight_layout(); plt.savefig(OUT+"cross_family_int8_vs_nf4_delta.png",dpi=200)
print("ok")

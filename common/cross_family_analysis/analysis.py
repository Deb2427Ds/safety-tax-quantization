import numpy as np, pandas as pd
from load import load,arrays,ORDER,ATT
rng=np.random.default_rng(42)
B=10000; NPERM=200000
HYP=ATT.index("hypothetical")
scopes={"aggregate_all":[0,1,2,3,4],"aggregate_excl_hypothetical":[0,1,2,4]}
names={"C":"HCR_strict","NC":"HCR_no_conflict","R":"refusal_rate","L":"leakage"}
def holm(p):
    p=np.asarray(p,float); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for i,k in enumerate(o):
        run=max(run,(m-i)*p[k]); adj[k]=min(1,run)
    return adj
D=load(); rows=[]
idx_c=rng.integers(0,100,size=(B,100))              # cluster bootstrap: resample 100 base prompts
for m in ORDER:
    A=arrays(D,m)
    for comp in ["nf4","int8"]:
        for sc,cols in scopes.items():
            for met in ["C","NC","R","L"]:
                diff=A[comp][met][:,cols]-A["fp16"][met][:,cols]       # (100 bases, k attacks) paired differences
                d=diff.mean(axis=1)                                     # per-base-prompt mean difference (cluster level)
                obs=d.mean()
                bs=d[idx_c].mean(axis=1); lo,hi=np.percentile(bs,[2.5,97.5])
                # row-level paired bootstrap (original unit of resampling: individual prompt pairs)
                flat=diff.ravel(); n=len(flat); ir=rng.integers(0,n,size=(B,n)); bs_r=flat[ir].mean(axis=1); rlo,rhi=np.percentile(bs_r,[2.5,97.5])
                # cluster sign-flip permutation test (two-sided)
                s=rng.choice([-1.0,1.0],size=(NPERM,100)); perm=(s*d).mean(axis=1)
                p=(1+np.sum(np.abs(perm)>=abs(obs)-1e-12))/(1+NPERM)
                rows.append(dict(model=m,comparison=comp,scope=sc,metric=names[met],delta_pp=obs*100,cl_lo=lo*100,cl_hi=hi*100,row_lo=rlo*100,row_hi=rhi*100,p_cluster_perm=p,
                                 n_clusters=100,cl_width=(hi-lo)*100,row_width=(rhi-rlo)*100))
R=pd.DataFrame(rows)
# Holm correction across the 12 checkpoints, separately within each (comparison, scope, metric) family,
# applied to the cluster sign-flip permutation p-values (verified to reproduce the released column exactly).
from statsmodels.stats.multitest import multipletests
R["p_holm12_cluster"]=np.nan
for _,g in R.groupby(["comparison","scope","metric"]):
    R.loc[g.index,"p_holm12_cluster"]=multipletests(g.p_cluster_perm,method="holm")[1]
R.to_csv("cluster_inference_all.csv",index=False)
print(len(R))

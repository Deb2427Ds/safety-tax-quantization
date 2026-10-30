"""Primary-family analysis: Holm across the 12 checkpoints (NF4 and INT8 families kept separate).
Requires cluster_inference_all.csv (from analysis.py) and the result packages' stats_tests.csv (for the original McNemar p-values)."""
import os
import numpy as np, pandas as pd
E=os.environ.get("RESULTS_ROOT",os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","results")).rstrip("/")+"/"
def holm(p):
    p=np.asarray(p,float); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for i,k in enumerate(o):
        run=max(run,(m-i)*p[k]); adj[k]=min(1,run)
    return adj
spec=[("qwen/","attack_type","mcnemar_p"),("phi/","scope","mcnemar_p"),("llama/","scope","mcnemar_p"),
("gemma/","attack_type","mcnemar_exact_p"),("mistral/","scope","mcnemar_p"),
("smollm/","scope","mcnemar_exact_p"),("granite/","scope","mcnemar_exact_p")]
from load import ORDER   # checkpoints actually present (12 when every family is released)
spec=[x for x in spec if os.path.exists(E+x[0]+"stats_tests.csv")]
fr=[]
for p,sc,pc in spec:
    d=pd.read_csv((p if p.startswith('/') else E+p)+"stats_tests.csv").rename(columns={"model_family":"model"})
    if "model" not in d: d["model"]="SmolLM2-1.7B" if "smollm" in p else "Granite-3.1-2B"
    d["model"]=d["model"].replace({"Ministral-3-3B-Instruct-2512":"Ministral-3-3B"})
    d["scope_"]=d[sc]; d["comp"]=np.where(d["comparison"].str.contains("int8"),"int8","nf4")
    d["p_raw"]=pd.to_numeric(d[pc]); d["p_holm_within"]=pd.to_numeric(d["p_holm"] if "p_holm" in d else d["holm_p"])
    fr.append(d[["model","scope_","comp","metric","p_raw","p_holm_within","delta_ci_lo","delta_ci_hi"]])
O=pd.concat(fr); O=O[(O.scope_=="aggregate_all")&(O.metric=="HCR_strict")]
R=pd.read_csv("cluster_inference_all.csv"); R=R[(R.scope=="aggregate_all")&(R.metric=="HCR_strict")]
out=[]
for comp in ["nf4","int8"]:
    o=O[O.comp==comp].set_index("model").reindex(ORDER); r=R[R.comparison==comp].set_index("model").reindex(ORDER)
    out.append(pd.DataFrame({"comparison":comp,"delta_pp":r.delta_pp,"row_ci_lo":o.delta_ci_lo*100,"row_ci_hi":o.delta_ci_hi*100,"cl_ci_lo":r.cl_lo,"cl_ci_hi":r.cl_hi,
      "p_mcnemar_raw":o.p_raw,"p_holm_within_model":o.p_holm_within,"p_holm12_mcnemar":holm(o.p_raw.values),
      "p_cluster_perm":r.p_cluster_perm,"p_holm12_cluster":holm(r.p_cluster_perm.values)}).reset_index())
pd.concat(out).to_csv("primary_family_results.csv",index=False)

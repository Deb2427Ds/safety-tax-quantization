"""Load the row-level judged harmful labels of every verified checkpoint (12 when all families are released) into one frame and check pairing.

Reads the label-only files shipped in this repository (results/<family>/judged_harmful_<family>_labels.csv).
Set RESULTS_ROOT to point somewhere else (e.g. a freshly built results/ folder from
common/build_results_release.py); the default is <repo>/results.
"""
import os
import pandas as pd, numpy as np
E=os.environ.get("RESULTS_ROOT",os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","results")).rstrip("/")+"/"
FAMILIES={"qwen":["Qwen2.5-1.5B","Qwen2.5-3B"],"llama":["Llama-3.2-1B","Llama-3.2-3B"],"gemma":["Gemma-3-1B","Gemma-2-2B","Gemma-3-4B"],
 "phi":["Phi-3.5-mini","Phi-3-mini-4k"],"mistral":["Ministral-3-3B"],"smollm":["SmolLM2-1.7B"],"granite":["Granite-3.1-2B"]}
SINGLE={"smollm":"SmolLM2-1.7B","granite":"Granite-3.1-2B"}
P={f"{f}/judged_harmful_{f}_labels.csv":SINGLE.get(f) for f in FAMILIES if os.path.exists(E+f"{f}/judged_harmful_{f}_labels.csv")}

ATT=["direct","dan_template","roleplay_authority","hypothetical","obfuscated"]
ALL_ORDER=["Qwen2.5-1.5B","Qwen2.5-3B","Llama-3.2-1B","Llama-3.2-3B","Gemma-3-1B","Gemma-2-2B","Gemma-3-4B","Phi-3.5-mini","Phi-3-mini-4k","Ministral-3-3B","SmolLM2-1.7B","Granite-3.1-2B"]
_present={m for f in FAMILIES if os.path.exists(E+f"{f}/judged_harmful_{f}_labels.csv") for m in FAMILIES[f]}
ORDER=[m for m in ALL_ORDER if m in _present]
if len(ORDER)<12:
    import warnings
    warnings.warn(f"Only {len(ORDER)}/12 checkpoints are present in {E} (missing: {[m for m in ALL_ORDER if m not in _present]}). "
                  "Holm families are computed over the checkpoints present, so results will NOT match the 12-checkpoint manuscript tables. "
                  "See results/VERIFICATION_REPORT.md for why a family was withheld.",stacklevel=2)

def load():
    rows=[]
    for p,name in P.items():
        d=pd.read_csv(p if p.startswith('/') else E+p)
        ac='canonical_attack_type' if 'canonical_attack_type' in d else 'attack_type'
        bc='canonical_base_prompt_id' if 'canonical_base_prompt_id' in d else 'base_prompt_id'
        x=pd.DataFrame({"model":name if name else d["model_family"],"precision":d["precision"],"attack":d[ac].replace({"dan":"dan_template"}),"base":d[bc],
          "C":d["judge_harmful_compliance_label"].astype(bool),"R":d["judge_refusal_label"].astype(bool),"L":d["judge_harmful_leak_after_refusal_label"].fillna(False).astype(bool)})
        rows.append(x)
    D=pd.concat(rows,ignore_index=True)
    D["model"]=D["model"].replace({"Ministral-3-3B-Instruct-2512":"Ministral-3-3B"})
    D["NC"]=D.C&~D.R
    return D
def arrays(D,m):
    a=D[D.model==m]; out={}
    bases=sorted(a.base.unique())
    for pr in ["fp16","int8","nf4"]:
        s=a[a.precision==pr]
        assert not s.duplicated(["attack","base"]).any(),(m,pr)
        out[pr]={}
        for met in ["C","NC","R","L"]:
            piv=s.pivot(index="base",columns="attack",values=met).reindex(bases)[ATT]
            assert piv.shape==(100,5) and piv.notna().all().all(),(m,pr,met,piv.shape)
            out[pr][met]=piv.values.astype(float)
    return out
if __name__=="__main__":
    D=load(); print(D.groupby("model").size().reindex(ORDER).to_dict())
    for m in ORDER: A=arrays(D,m)
    print("all pairing checks passed (100 bases x 5 attacks x 3 precisions per checkpoint)")
    print((D[D.precision=="fp16"].groupby("model").C.mean().reindex(ORDER)*100).round(1).to_dict())
    print((D[D.precision=="nf4"].groupby("model").C.mean().reindex(ORDER)*100).round(1).to_dict())

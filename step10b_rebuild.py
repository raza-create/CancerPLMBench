#!/usr/bin/env python3
"""STEP 10b - rebuild the dbNSFP merge with consistent keys.
The cache mixes two key-naming schemes (16 genes cached under the pre-patch
scheme, 8 fetched after), so 12 predictors vanished. This re-derives every
column from the RAW cache uniformly and merges with an explicit suffix guard."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        s=" ".join(str(x) for x in a); print(s,flush=True); self.f.write(s+"\n")
    def close(self): self.f.close()

def auc(y,s):
    y=np.asarray(y,float); s=np.asarray(s,float); ok=~np.isnan(s)
    y=y[ok]; s=s[ok]
    return roc_auc_score(y,s) if len(np.unique(y))>1 else np.nan

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP10B_rebuild_report.txt")
    log("="*78); log("STEP 10b - REBUILDING THE dbNSFP MERGE"); log("="*78)
    log("\nPROBLEM: the cache was written under two different key-naming schemes")
    log("(16 genes before the patch, 8 after), so 12 predictors - including")
    log("MetaRNN, ClinPred, VEST4, PrimateAI, ESM1b, MutPred - were dropped.")

    cache=json.loads((od/"step10_dbnsfp_cache.json").read_text())
    log(f"\ncached genes: {len(cache)}")
    rows=[]
    for g,recs in cache.items():
        for r in recs: rows.append(r)
    D=pd.DataFrame(rows)
    log(f"raw rows: {len(D)}   distinct columns across cache: {len(D.columns)}")
    log(f"columns: {sorted(c for c in D.columns if c not in ('target_gene','position','wt_aa','mut_aa'))}")

    # canonicalise: strip suffixes so old/new schemes collapse together
    CANON={}
    for c in D.columns:
        if c in ("target_gene","position","wt_aa","mut_aa"): continue
        k=(c.lower().replace(".rankscore","").replace("_rankscore","")
             .replace(".converted_rankscore","").replace("_converted","")
             .replace(".score","").replace("_score","")
             .replace("++","pp").replace("-","_").replace(".","_"))
        CANON.setdefault(k,[]).append(c)
    log(f"\ncanonical predictor names: {len(CANON)}")
    merged={}
    for k,cols in sorted(CANON.items()):
        v=None
        for c in cols:
            s=pd.to_numeric(D[c],errors="coerce")
            v=s if v is None else v.fillna(s)
        merged[k]=v
        if len(cols)>1:
            log(f"    {k:<32} merged from {cols}")
    C=pd.DataFrame(merged)
    for c in ("target_gene","position","wt_aa","mut_aa"): C[c]=D[c]
    C["position"]=pd.to_numeric(C["position"],errors="coerce")
    C=C.dropna(subset=["position"]); C["position"]=C["position"].astype(int)
    C=C.drop_duplicates(subset=["target_gene","position","wt_aa","mut_aa"])
    log(f"\nunique variant rows: {len(C)}")

    # merge onto a CLEAN master (drop any previous s_new_ / bare dbnsfp columns)
    m=pd.read_csv(od/"master_scores.csv")
    drop=[c for c in m.columns if c.startswith("s_new_")]
    drop+=[c for c in m.columns if c.endswith(("_x","_y"))]
    drop+=[c for c in m.columns if c in CANON and not c.startswith("s_")]
    m=m.drop(columns=list(dict.fromkeys(drop)),errors="ignore")
    log(f"dropped {len(set(drop))} stale columns from master")

    pred=[c for c in C.columns if c not in ("target_gene","position","wt_aa","mut_aa")]
    C2=C.rename(columns={c:"s_new_"+c for c in pred})
    m2=m.merge(C2,on=["target_gene","position","wt_aa","mut_aa"],how="left")
    assert len(m2)==len(m)
    m2=m2[m2.analysis_ok==True] if "analysis_ok" in m2.columns else m2

    log("\n[1] COVERAGE AND DIRECTION")
    log(f"    {'predictor':<32}{'n':>7}{'cov%':>8}{'mean(P)':>10}{'mean(B)':>10}"
        f"{'dir':>5}{'AUC':>8}")
    keep=[]; res=[]
    for c in pred:
        col="s_new_"+c
        v=m2[col]; n=int(v.notna().sum())
        if n<200:
            log(f"    {c:<32}{n:>7}   (sparse - dropped)"); continue
        mp=v[m2.label==1].mean(); mb=v[m2.label==0].mean()
        d=m2.dropna(subset=[col])
        A=auc(d.label.values,d[col].values)
        log(f"    {c:<32}{n:>7}{100*n/len(m2):>7.1f}%{mp:>10.3f}{mb:>10.3f}"
            f"{'+' if mp>mb else '-':>5}{A:>8.3f}")
        keep.append(c); res.append(dict(predictor=c,n=n,
            coverage=round(100*n/len(m2),1),auc=A,direction="+" if mp>mb else "-"))
    log(f"\n    retained {len(keep)} predictors")
    bad=[r["predictor"] for r in res if r["direction"]=="-"]
    if bad: log(f"    DIRECTION WARNING (needs sign flip): {bad}")
    else: log("    all predictors correctly oriented")

    R=pd.DataFrame(res).sort_values("auc",ascending=False)
    R.to_csv(od/"step10b_new_predictors.csv",index=False)
    m2.to_csv(od/"master_scores.csv",index=False)
    log(f"\n    master_scores.csv rewritten: {m2.shape}")

    log("\n[2] KEY ADDITIONS PRESENT?")
    for want in ["metarnn","clinpred","vest4","primateai","esm1b","mutpred",
                 "varity_r","varity_er","phactboost","bayesdel_add_af","gmvp","mvp"]:
        hit=[k for k in keep if want in k]
        log(f"    {want:<20}{'OK  '+hit[0] if hit else 'STILL MISSING'}")

    log("\n[3] CONSERVATION MEASURES (for item 10)")
    cons=[k for k in keep if any(t in k for t in
          ("phylop","phastcons","gerp","siphy"))]
    for c in cons:
        d=m2.dropna(subset=["s_new_"+c])
        log(f"    {c:<32} n={len(d):<6} AUC={auc(d.label.values,d['s_new_'+c].values):.3f}")
    log(f"    {len(cons)} independent conservation measures available")

    log("\n[4] TOP PREDICTORS (native coverage - NOT a paired comparison)")
    log(R.head(15).to_string(index=False))
    log("\n    A paired shared-subset DeLong comparison is required before any")
    log("    ranking claim. That is the next step.")
    log("\nWROTE: step10b_new_predictors.csv, master_scores.csv")
    log.close()

if __name__=="__main__": main()

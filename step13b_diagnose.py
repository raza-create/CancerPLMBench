#!/usr/bin/env python3
"""STEP 13b - is the matched-pair disorder analysis trustworthy?
Spread of 0.37 AUC across methods on identical pairs suggests the estimates
are unstable. This checks effective sample size and per-method coverage
within the disordered stratum."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
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
    log=Tee(od/"STEP13B_diagnostic_report.txt")
    m=pd.read_csv(od/"master_scores.csv")
    if "analysis_ok" in m.columns: m=m[m.analysis_ok==True].copy()
    struct=pd.read_csv(od/"step06e_structural_variants.csv")[
        ["target_gene","position","plddt","rsa","pdb_aa"]]
    d=m.merge(struct,on=["target_gene","position"],how="inner")
    d=d[d.pdb_aa==d.wt_aa].copy()
    d["order"]=pd.cut(d.plddt,[0,70,90,100],labels=["disordered","flexible","ordered"])
    log("="*78); log("STEP 13b - IS THE MATCHED-PAIR ANALYSIS TRUSTWORTHY?"); log("="*78)
    log("\nCONCERN: step13 reported drops from -0.131 (phastCons) to +0.240")
    log("(MutationAssessor) on the SAME 494 matched pairs. A 0.37 AUC spread")
    log("across methods scoring identical variants indicates unstable estimates.")

    log("\n[1] HOW MANY PATHOGENIC VARIANTS ARE IN THE DISORDERED STRATUM?")
    dis=d[d.order=="disordered"]; ordr=d[d.order=="ordered"]
    log(f"    disordered: n={len(dis)}  P={int(dis.label.sum())}  "
        f"B={int((dis.label==0).sum())}")
    log(f"    ordered:    n={len(ordr)}  P={int(ordr.label.sum())}  "
        f"B={int((ordr.label==0).sum())}")
    log("\n    The matched design samples min(n_ordered, n_disordered) per")
    log("    gene x class. With few disordered pathogenic variants, the effective")
    log("    sample is tiny regardless of how many bootstrap draws are taken.")

    log("\n[2] PER-METHOD COVERAGE WITHIN THE DISORDERED PATHOGENIC CELL")
    log("    This is the binding constraint on every AUC in the disordered stratum.")
    OWN={"s_alphamissense":"AlphaMissense","s_saprot":"SaProt","s_esm2_650m":"ESM-2 650M",
         "s_esm1v":"ESM-1v","s_revel":"REVEL","s_eve":"EVE","s_sift":"SIFT"}
    NEW={c:c.replace("s_new_","") for c in d.columns if c.startswith("s_new_")}
    ALL={**OWN,**NEW}
    dp=dis[dis.label==1]
    log(f"    disordered pathogenic variants total: {len(dp)}")
    log(f"    {'method':<28}{'scored':>8}{'genes':>7}")
    cnt=[]
    for c,nm in sorted(ALL.items(),key=lambda kv:kv[1]):
        if c not in d.columns: continue
        s=dp.dropna(subset=[c])
        cnt.append(dict(method=nm,n=len(s),genes=s.target_gene.nunique()))
    C=pd.DataFrame(cnt).sort_values("n")
    for _,r in C.iterrows():
        flag="  <-- UNRELIABLE" if r.n<25 else ""
        log(f"    {r.method:<28}{r.n:>8}{r.genes:>7}{flag}")
    C.to_csv(od/"step13b_disordered_coverage.csv",index=False)
    log(f"\n    methods with <25 disordered pathogenic variants: "
        f"{int((C.n<25).sum())}/{len(C)}")

    log("\n[3] DOES THE REPORTED DROP CORRELATE WITH SAMPLE SIZE?")
    f=od/"step13_disorder_all_methods.csv"
    if f.exists():
        R=pd.read_csv(f).merge(C,on="method",how="inner")
        if len(R)>=8:
            r1,p1=stats.spearmanr(R.n,R["drop"])
            log(f"    Spearman(n disordered pathogenic, reported drop): "
                f"rho={r1:+.3f} p={p1:.4f} n={len(R)}")
            r2,p2=stats.spearmanr(R.n,(R.hi-R.lo))
            log(f"    Spearman(n, CI width): rho={r2:+.3f} p={p2:.4f}")
            log("    A strong relationship means the 'effect' partly tracks how many")
            log("    variants each method could score, not genuine degradation.")
            log(f"\n    {'method':<26}{'n_dis_path':>12}{'drop':>9}{'CI width':>10}")
            for _,r in R.sort_values("n").head(12).iterrows():
                log(f"    {r.method:<26}{r.n:>12}{r['drop']:>+9.4f}{r.hi-r.lo:>10.4f}")

    log("\n[4] LEAVE-ONE-GENE-OUT STABILITY OF THE MATCHED ESTIMATE")
    log("    If one gene drives the result, the estimate is not robust.")
    RNG=np.random.default_rng(5)
    for c,nm in [("s_saprot","SaProt"),("s_esm2_650m","ESM-2 650M"),
                 ("s_alphamissense","AlphaMissense"),
                 ("s_new_phastcons_470way_mammalian","phastCons 470way")]:
        if c not in d.columns: continue
        genes=sorted(d.target_gene.unique()); vals=[]
        for gx in genes:
            sub=d[d.target_gene!=gx]
            pr=[]
            for g,gd in sub.groupby("target_gene"):
                for lab in [0,1]:
                    o=gd[(gd.order=="ordered")&(gd.label==lab)].dropna(subset=[c])
                    u=gd[(gd.order=="disordered")&(gd.label==lab)].dropna(subset=[c])
                    n=min(len(o),len(u))
                    if n>=3: pr.append((n,o,u))
            if len(pr)<4: continue
            dv=[]
            for _ in range(50):
                O=pd.concat([o.sample(n,random_state=int(RNG.integers(1e9))) for n,o,_ in pr])
                U=pd.concat([u.sample(n,random_state=int(RNG.integers(1e9))) for n,_,u in pr])
                ao=auc(O.label.values,O[c].values); au_=auc(U.label.values,U[c].values)
                if not (np.isnan(ao) or np.isnan(au_)): dv.append(ao-au_)
            if dv: vals.append((gx,float(np.mean(dv))))
        if vals:
            v=np.array([x[1] for x in vals])
            log(f"\n    {nm}: LOGO drops range [{v.min():+.4f}, {v.max():+.4f}], "
                f"SD={v.std():.4f}")
            ext=sorted(vals,key=lambda x:x[1])
            log(f"      most negative when excluding {ext[0][0]} ({ext[0][1]:+.4f}); "
                f"most positive excluding {ext[-1][0]} ({ext[-1][1]:+.4f})")
            if v.std()>0.03:
                log(f"      => UNSTABLE: single genes shift the estimate by >0.03 AUC")

    log("\n[5] VERDICT")
    log("    If (a) many methods have <25 disordered pathogenic variants,")
    log("       (b) drop correlates with that count, and")
    log("       (c) leave-one-gene-out is unstable,")
    log("    then the matched-pair disorder analysis - INCLUDING Step 6g - cannot")
    log("    support a claim about differential degradation, and should be reported")
    log("    only as the distributional observation (pathogenic variants are")
    log("    concentrated in ordered regions), which is robust at chi2 p<1e-215.")
    log.close()

if __name__=="__main__": main()

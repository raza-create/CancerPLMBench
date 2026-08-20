#!/usr/bin/env python3
"""STEP 4b - FIX: two-sided significance test for class-balance matching.
Corrects the one-sided bug in step04 section 4."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

NICE={"revel":"REVEL","cadd":"CADD","sift":"SIFT","polyphen2":"PolyPhen-2",
      "eve":"EVE","esm2_650m":"ESM-2 650M","esm2_150m":"ESM-2 150M","esm1v":"ESM-1v"}
CLASSICAL=["revel","cadd","sift","polyphen2","eve"]
PLMS=["esm2_650m","esm2_150m","esm1v"]
RNG=np.random.default_rng(7)

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
    log=Tee(od/"STEP04B_balance_fixed_report.txt")
    m=pd.read_csv(od/"master_scores.csv"); m=m[m.analysis_ok==True].copy()
    m["is_onc"]=m.gene_type.str.lower().str.startswith("onco")
    V5=[f"s_{x}" for x in CLASSICAL]
    d5=m[m[V5].notna().all(axis=1)].copy()

    log("="*78); log("STEP 4b - CLASS-BALANCE MATCHING (TWO-SIDED, CORRECTED)"); log("="*78)
    log("\nBUG FIXED: previous version tested only oncogene_AUC > TSG_upper_bound,")
    log("so a significant gap in the OPPOSITE direction was mislabelled")
    log("'gap not significant'. This version tests both tails.")

    onc=d5[d5.is_onc]; tsg=d5[~d5.is_onc]
    tgt=onc.label.mean()
    log(f"\nONC: n={len(onc)} genes={onc.target_gene.nunique()} %P={100*tgt:.1f}")
    log(f"TSG: n={len(tsg)} genes={tsg.target_gene.nunique()} %P={100*tsg.label.mean():.1f}")

    tp_=tsg[tsg.label==1]; tb_=tsg[tsg.label==0]
    n_tot=min(len(tsg),1500); n_p=min(int(round(n_tot*tgt)),len(tp_))
    n_b=min(n_tot-n_p,len(tb_))
    log(f"resampling TSG to {n_p}P + {n_b}B, 500 iterations\n")

    log(f"  {'method':<14}{'family':<11}{'ONC':>9}{'TSG_match':>11}"
        f"{'CI_lo':>9}{'CI_hi':>9}{'diff':>9}  verdict")
    out=[]
    for meth in CLASSICAL+PLMS:
        c="s_"+meth
        if c not in d5.columns or d5[c].notna().sum()==0: continue
        vals=[]
        for _ in range(500):
            s=pd.concat([tp_.sample(n_p,replace=True,random_state=RNG.integers(1e9)),
                         tb_.sample(n_b,replace=True,random_state=RNG.integers(1e9))])
            vals.append(auc(s.label.values,s[c].values))
        vals=np.array(vals,dtype=float)
        lo,hi=np.nanpercentile(vals,2.5),np.nanpercentile(vals,97.5)
        mean=np.nanmean(vals)
        oa=auc(onc.label.values,onc[c].values)
        if oa>hi:   v="ONC BETTER (significant)"
        elif oa<lo: v="TSG BETTER (significant, REVERSED)"
        else:       v="no significant difference"
        fam="classical" if meth in CLASSICAL else "pLM"
        log(f"  {NICE[meth]:<14}{fam:<11}{oa:>9.4f}{mean:>11.4f}"
            f"{lo:>9.4f}{hi:>9.4f}{oa-mean:>+9.4f}  {v}")
        out.append(dict(method=NICE[meth],family=fam,onc_auc=oa,tsg_matched_mean=mean,
                        tsg_lo=lo,tsg_hi=hi,diff_onc_minus_tsg=oa-mean,verdict=v))
    res=pd.DataFrame(out)
    res.to_csv(od/"step04_balance_matched.csv",index=False)
    log(f"\n  overwrote step04_balance_matched.csv with corrected verdicts")

    log("\nSUMMARY BY FAMILY")
    for fam,g in res.groupby("family"):
        sig=g[g.verdict.str.contains("significant")&~g.verdict.str.contains("no ")]
        log(f"  {fam:<11} n={len(g)}  significant={len(sig)}  "
            f"median diff={g.diff_onc_minus_tsg.median():+.4f}")
    cl=res[res.family=="classical"].diff_onc_minus_tsg
    pl=res[res.family=="pLM"].diff_onc_minus_tsg
    log(f"\n  classical diffs: {[round(x,3) for x in cl]}")
    log(f"  pLM diffs:       {[round(x,3) for x in pl]}")
    if (cl>0).all() and (pl<0).all():
        log("\n  => SIGN REVERSAL CONFIRMED: every classical method favours the")
        log("     oncogene/RTK set; every pLM favours tumour suppressors.")
        log("     A difficulty confound cannot produce opposite signs on the")
        log("     SAME variants, so this reflects a genuine model-family difference.")
    else:
        log("\n  => signs are NOT cleanly separated; do not claim a family-level reversal.")
    log.close()

if __name__=="__main__": main()

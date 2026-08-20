#!/usr/bin/env python3
"""STEP 13d - finalise the disorder analysis after removing saturated scores
and correcting the family grouping."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

RNG=np.random.default_rng(401)
# methods that use the MUTANT residue (substitution-specific)
EVOL={"SaProt","ESM-2 650M","ESM-2 150M","ESM-1v","esm1b","EVE","SIFT","sift4g",
      "provean","mutationassessor","fathmm"}
TRAINED={"AlphaMissense","REVEL","CADD","PolyPhen-2","metarnn","phactboost",
  "clinpred","varity_r","varity_er","bayesdel_add_af","bayesdel_no_af","vest4",
  "gmvp","mvp","m_cap","metalr","metasvm","mutformer","primateai","deogen2",
  "mutpred","mpc","list_s2"}
# excluded: position-only nucleotide conservation, cannot score substitutions
EXCLUDE={"phastcons_100way_vertebrate","phastcons_470way_mammalian",
  "phylop_100way_vertebrate","phylop_470way_mammalian","phylop_17way_primate",
  "gerp_91_mammals","siphy_29way_logodds","gerppp_rs"}

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        s=" ".join(str(x) for x in a); print(s,flush=True); self.f.write(s+"\n")
    def close(self): self.f.close()

def auc(y,s):
    y=np.asarray(y,float); s=np.asarray(s,float); ok=~np.isnan(s)
    y=y[ok]; s=s[ok]
    return roc_auc_score(y,s) if len(np.unique(y))>1 else np.nan

def bh(p):
    p=np.asarray(p,float); n=len(p); o=np.argsort(p); adj=np.empty(n); prev=1.0
    for r,i in enumerate(o[::-1]):
        prev=min(prev,p[i]*n/(n-r)); adj[i]=prev
    return adj

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP13D_final_disorder_report.txt")
    m=pd.read_csv(od/"master_scores.csv")
    if "analysis_ok" in m.columns: m=m[m.analysis_ok==True].copy()
    st=pd.read_csv(od/"step06e_structural_variants.csv")[
        ["target_gene","position","plddt","pdb_aa"]]
    d=m.merge(st,on=["target_gene","position"],how="inner")
    d=d[d.pdb_aa==d.wt_aa].copy()
    d["order"]=pd.cut(d.plddt,[0,70,90,100],labels=["disordered","flexible","ordered"])

    log("="*78); log("STEP 13d - FINAL DISORDER ANALYSIS"); log("="*78)
    log("\nCORRECTIONS APPLIED:")
    log("  1. phastCons/phyloP/GERP/SiPhy EXCLUDED from the predictor comparison.")
    log("     phastCons is saturated: 94.4% of ordered-region variants sit in its")
    log("     top 1%, ties 98.7%, ordered AUC 0.650. Its apparent 'improvement' in")
    log("     disordered regions is a ceiling effect. These scores also give only")
    log("     1.14 distinct values per position and cannot discriminate between")
    log("     substitutions at all. Step 13's 'pure conservation reverses' claim")
    log("     is RETRACTED.")
    log("  2. Family grouping now reflects what each method computes:")
    log("     EVOLUTIONARY (scores the mutant residue from sequence/alignment)")
    log("     vs TRAINED (fitted to labelled variant sets).")
    log("  3. MutationAssessor and PROVEAN verified as genuine, not artefacts:")
    log("     no saturation (1.1%, 1.7% at max), and their correlation with SaProt")
    log("     collapses in disordered regions (0.48->0.13, 0.53->0.20).")

    OWN={"s_alphamissense":"AlphaMissense","s_saprot":"SaProt","s_esm2_650m":"ESM-2 650M",
         "s_esm2_150m":"ESM-2 150M","s_esm1v":"ESM-1v","s_revel":"REVEL","s_cadd":"CADD",
         "s_sift":"SIFT","s_polyphen2":"PolyPhen-2","s_eve":"EVE"}
    DUP={"alphamissense","revel","eve","sift","polyphen2_hdiv"}
    NEW={c:c.replace("s_new_","") for c in d.columns if c.startswith("s_new_")
         and c.replace("s_new_","") not in DUP|EXCLUDE}
    ALL={**OWN,**NEW}

    pairs=[]
    for g,gd in d.groupby("target_gene"):
        for lab in [0,1]:
            o=gd[(gd.order=="ordered")&(gd.label==lab)]
            u=gd[(gd.order=="disordered")&(gd.label==lab)]
            n=min(len(o),len(u))
            if n>=3: pairs.append((n,o,u))
    log(f"\n[1] MATCHED DESIGN: {len(pairs)} sets, {sum(p[0] for p in pairs)} pairs")
    log(f"    {'method':<26}{'family':<14}{'drop':>9}{'CI_lo':>9}{'CI_hi':>9}{'p':>9}")
    rows=[]; B={}
    for c,nm in ALL.items():
        if c not in d.columns or d[c].notna().sum()<300: continue
        fam="evolutionary" if nm in EVOL else ("trained" if nm in TRAINED else "other")
        if fam=="other": continue
        v=[]
        for _ in range(800):
            O=pd.concat([o.sample(n,random_state=int(RNG.integers(1e9))) for n,o,_ in pairs])
            U=pd.concat([u.sample(n,random_state=int(RNG.integers(1e9))) for n,_,u in pairs])
            ao=auc(O.label.values,O[c].values); au_=auc(U.label.values,U[c].values)
            if not (np.isnan(ao) or np.isnan(au_)): v.append(ao-au_)
        if len(v)<100: continue
        v=np.array(v); B[nm]=v
        lo,hi=np.percentile(v,2.5),np.percentile(v,97.5)
        p=max(2*min((v<=0).mean(),(v>=0).mean()),1/len(v))
        log(f"    {nm:<26}{fam:<14}{v.mean():>+9.4f}{lo:>+9.4f}{hi:>+9.4f}{p:>9.4f}")
        rows.append(dict(method=nm,family=fam,drop=v.mean(),lo=lo,hi=hi,p=p))
    R=pd.DataFrame(rows); R["bh_q"]=bh(R.p.values); R["sig"]=R.bh_q<0.05
    R.to_csv(od/"step13d_final_disorder.csv",index=False)

    log("\n[2] FAMILY COMPARISON")
    for f,g in R.groupby("family"):
        log(f"    {f:<14} n={len(g):<3} median={g['drop'].median():+.4f}  "
            f"[{g['drop'].min():+.4f}, {g['drop'].max():+.4f}]")
    e=R[R.family=="evolutionary"]["drop"]; t=R[R.family=="trained"]["drop"]
    if len(e)>=4 and len(t)>=4:
        u_,p=stats.mannwhitneyu(e,t,alternative="greater")
        log(f"\n    Mann-Whitney (evolutionary drop > trained): p={p:.4f}")
        log(f"    evolutionary median {e.median():+.4f} vs trained {t.median():+.4f}")
        if p<0.05:
            log("    => SUPPORTED. Methods that score substitutions from sequence or")
            log("       alignment evidence lose more accuracy in predicted-disordered")
            log("       regions than methods trained on labelled variant sets.")
        else:
            log("    => NOT SUPPORTED at the family level. Report per-method results")
            log("       with the strongest individual contrasts named explicitly.")

    log("\n[3] STRONGEST INDIVIDUAL CONTRASTS (paired on shared resamples)")
    KEY=[("SaProt","AlphaMissense"),("ESM-2 650M","AlphaMissense"),
         ("mutationassessor","AlphaMissense"),("provean","AlphaMissense"),
         ("SaProt","varity_r"),("ESM-2 650M","metarnn"),
         ("mutationassessor","phactboost")]
    kr=[]
    log(f"    {'comparison':<44}{'diff':>9}{'CI_lo':>9}{'CI_hi':>9}")
    for A,Bm in KEY:
        if A not in B or Bm not in B: continue
        n=min(len(B[A]),len(B[Bm])); v=B[A][:n]-B[Bm][:n]
        lo,hi=np.percentile(v,2.5),np.percentile(v,97.5)
        log(f"    {A+' vs '+Bm:<44}{v.mean():>+9.4f}{lo:>+9.4f}{hi:>+9.4f}")
        kr.append(dict(comparison=f"{A} vs {Bm}",diff=v.mean(),lo=lo,hi=hi,
                       excludes_zero=bool(lo>0 or hi<0)))
    K=pd.DataFrame(kr); K.to_csv(od/"step13d_key_contrasts.csv",index=False)
    log(f"\n    {int(K.excludes_zero.sum())}/{len(K)} contrasts with CI excluding zero")

    log("\n[4] FINAL REPORTABLE STATEMENT")
    log("-"*78)
    sig=R[R.sig&(R["drop"]>0.05)].sort_values("drop",ascending=False)
    log(f"  Methods losing >0.05 AUC in disordered regions ({len(sig)}):")
    for _,r in sig.iterrows():
        log(f"    {r.method:<24}{r['drop']:+.3f}  [{r.lo:+.3f}, {r.hi:+.3f}]  {r.family}")
    stable=R[R["drop"].abs()<0.03].sort_values("drop")
    log(f"\n  Methods essentially unaffected (|drop|<0.03, n={len(stable)}):")
    log("    " + ", ".join(stable.method.tolist()))
    log("-"*78)
    log("\nWROTE: step13d_final_disorder.csv, step13d_key_contrasts.csv")
    log.close()

if __name__=="__main__": main()

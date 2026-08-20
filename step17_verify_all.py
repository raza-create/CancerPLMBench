#!/usr/bin/env python3
"""STEP 17 - end-to-end verification. Regenerates EVERY headline number from
the current master_scores.csv and compares it with what earlier steps reported.
Fixes: (1) no reproducibility audit, (2) master table mutated ~10 times,
(3) disorder analysis run on two different variant sets."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

RNG=np.random.default_rng(1213)
NICE={"alphamissense":"AlphaMissense","saprot":"SaProt","revel":"REVEL",
      "esm2_650m":"ESM-2 650M","cadd":"CADD","sift":"SIFT","polyphen2":"PolyPhen-2",
      "esm2_150m":"ESM-2 150M","esm1v":"ESM-1v","eve":"EVE"}
TEN=list(NICE)

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        s=" ".join(str(x) for x in a); print(s,flush=True); self.f.write(s+"\n")
    def close(self): self.f.close()

def auc(y,s):
    y=np.asarray(y,float); s=np.asarray(s,float); ok=~np.isnan(s)
    y=y[ok]; s=s[ok]
    return roc_auc_score(y,s) if len(np.unique(y))>1 else np.nan

def chk(log,label,got,expected,tol=0.002):
    if expected is None:
        log(f"    {label:<46}{got:>12}   (no prior value)"); return True
    try:
        ok=abs(float(got)-float(expected))<=tol
        flag="OK" if ok else "*** MISMATCH"
        log(f"    {label:<46}{float(got):>12.4f}  vs {float(expected):>9.4f}  {flag}")
        return ok
    except Exception:
        ok=str(got)==str(expected)
        log(f"    {label:<46}{str(got):>12}  vs {str(expected):>9}  "
            f"{'OK' if ok else '*** MISMATCH'}")
        return ok

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP17_verification_report.txt")
    log("="*78); log("STEP 17 - END-TO-END VERIFICATION"); log("="*78)
    log("\nWHY: master_scores.csv was rewritten by steps 1,1b,1c,1d,1e,10,10b,12,")
    log("12b,13. Roughly ten bugs were found and patched during the analysis. No")
    log("one has confirmed that the CURRENT file still reproduces the numbers")
    log("earlier steps reported. Every headline value is recomputed here.")

    m=pd.read_csv(od/"master_scores.csv")
    log(f"\n[0] MASTER TABLE: {m.shape[0]} rows x {m.shape[1]} cols")
    fails=[]

    # ---------- integrity ----------
    log("\n[1] INTEGRITY CHECKS")
    if "analysis_ok" in m.columns:
        clean=m[m.analysis_ok==True].copy()
    else:
        log("    *** analysis_ok column MISSING"); fails.append("analysis_ok"); clean=m
    log(f"    clean benchmark n           = {len(clean)}")
    if not chk(log,"n variants (expect 4316)",len(clean),4316,0.5): fails.append("n")
    if not chk(log,"n genes (expect 24)",clean.target_gene.nunique(),24,0.5): fails.append("genes")
    if not chk(log,"pathogenic (expect 2240)",int((clean.label==1).sum()),2240,0.5): fails.append("P")
    if not chk(log,"benign (expect 2076)",int((clean.label==0).sum()),2076,0.5): fails.append("B")
    dup=clean.duplicated(subset=["target_gene","position","wt_aa","mut_aa"]).sum()
    log(f"    duplicate variant keys      = {dup}" + ("  *** SHOULD BE 0" if dup else "  OK"))
    if dup: fails.append("duplicates")

    log("\n    score orientation (mean pathogenic must exceed mean benign):")
    bad=[]
    for x in TEN:
        c="s_"+x
        if c not in clean.columns: log(f"      {NICE[x]:<16} MISSING"); bad.append(x); continue
        mp=clean.loc[clean.label==1,c].mean(); mb=clean.loc[clean.label==0,c].mean()
        if not (mp>mb): bad.append(x)
        log(f"      {NICE[x]:<16}{mp:>9.3f}{mb:>9.3f}   "
            f"{'OK' if mp>mb else '*** FLIPPED'}")
    if bad: fails.append("orientation:"+",".join(bad))

    log("\n    per-gene counts vs step01d Table 2:")
    t2=od/"step01d_table2_clean.csv"
    if t2.exists():
        T=pd.read_csv(t2)
        now=(clean.groupby("target_gene")["label"]
               .agg(P=lambda s:int((s==1).sum()),B=lambda s:int((s==0).sum())).reset_index())
        mg=T.merge(now,on="target_gene",suffixes=("_old","_new"))
        diff=mg[(mg.P_old!=mg.P_new)|(mg.B_old!=mg.B_new)]
        log(f"      genes differing: {len(diff)}/{len(mg)}")
        if len(diff):
            log(diff[["target_gene","P_old","P_new","B_old","B_new"]].to_string(index=False))
            fails.append("table2")
    else:
        log("      step01d_table2_clean.csv not found")

    # ---------- headline: shared subset ----------
    log("\n[2] SHARED-SUBSET RANKING (Step 2 / Step 9)")
    sc=[f"s_{x}" for x in TEN]
    sh=clean[clean[sc].notna().all(axis=1)]
    if not chk(log,"fully-shared n (expect 2080)",len(sh),2080,0.5): fails.append("shared_n")
    if not chk(log,"shared genes (expect 17)",sh.target_gene.nunique(),17,0.5): fails.append("shared_g")
    EXP={"alphamissense":0.975,"saprot":0.947,"revel":0.940,"esm2_650m":0.931,
         "cadd":0.885,"sift":0.865,"polyphen2":0.848,"esm2_150m":0.847,
         "esm1v":0.838,"eve":0.836}
    log(f"    {'method':<24}{'recomputed':>12}{'submitted':>12}")
    for x in TEN:
        v=auc(sh.label.values,sh["s_"+x].values)
        if not chk(log,f"  {NICE[x]}",v,EXP[x],0.004): fails.append("auc:"+x)

    # ---------- gene-cluster bootstrap ----------
    log("\n[3] GENE-CLUSTER BOOTSTRAP (Step 2)")
    genes=np.array(sorted(sh.target_gene.unique()))
    idx={g:sh.index[sh.target_gene==g].values for g in genes}
    B=np.full((1000,len(TEN)),np.nan)
    for b in range(1000):
        pick=RNG.choice(genes,size=len(genes),replace=True)
        s=sh.loc[np.concatenate([idx[g] for g in pick])]
        if s.label.nunique()<2: continue
        y=s.label.values
        for j,x in enumerate(TEN): B[b,j]=auc(y,s["s_"+x].values)
    log(f"    {'method':<20}{'AUC':>8}{'gene 95% CI':>22}{'width':>9}")
    for j,x in enumerate(TEN):
        lo,hi=np.nanpercentile(B[:,j],[2.5,97.5])
        log(f"    {NICE[x]:<20}{auc(sh.label.values,sh['s_'+x].values):>8.4f}"
            f"    [{lo:.3f}, {hi:.3f}]{hi-lo:>9.4f}")
    i1,i2=TEN.index("saprot"),TEN.index("esm2_650m")
    d=B[:,i1]-B[:,i2]; d=d[~np.isnan(d)]
    lo,hi=np.percentile(d,[2.5,97.5])
    log(f"\n    SaProt - ESM-2 650M: CI [{lo:+.4f}, {hi:+.4f}]  "
        f"crosses zero = {bool(lo<0<hi)}")
    log("    (retraction of the SaProt>ESM-2 claim requires crosses zero = True)")
    if not (lo<0<hi): fails.append("saprot_retraction")

    # ---------- disorder: WHICH set? ----------
    log("\n[4] DISORDER ANALYSIS - RECONCILING TWO VARIANT SETS")
    log("    Step 6g used 294 pairs / 15 genes (112 disordered pathogenic).")
    log("    Step 13d used 494 pairs / 24 sets (247 disordered pathogenic).")
    log("    They differ because WT1 was excluded and genes were added between")
    log("    runs. Recomputing from the CURRENT master table:")
    sp=od/"step06e_structural_variants.csv"
    if sp.exists():
        st=pd.read_csv(sp)[["target_gene","position","plddt","pdb_aa"]]
        d2=clean.merge(st,on=["target_gene","position"],how="inner")
        d2=d2[d2.pdb_aa==d2.wt_aa]
        d2["order"]=pd.cut(d2.plddt,[0,70,90,100],
                           labels=["disordered","flexible","ordered"])
        dis=d2[d2.order=="disordered"]; ordr=d2[d2.order=="ordered"]
        log(f"    structurally annotated: {len(d2)}  genes={d2.target_gene.nunique()}")
        log(f"    disordered: n={len(dis)} P={int(dis.label.sum())} "
            f"B={int((dis.label==0).sum())}")
        log(f"    ordered:    n={len(ordr)} P={int(ordr.label.sum())} "
            f"B={int((ordr.label==0).sum())}")
        pairs=[]
        for g,gd in d2.groupby("target_gene"):
            for lab in [0,1]:
                o=gd[(gd.order=="ordered")&(gd.label==lab)]
                u=gd[(gd.order=="disordered")&(gd.label==lab)]
                n=min(len(o),len(u))
                if n>=3: pairs.append((n,o,u))
        log(f"    matched pair sets: {len(pairs)}  total pairs: {sum(p[0] for p in pairs)}")
        log("\n    key methods, recomputed:")
        log(f"    {'method':<20}{'drop':>9}{'CI_lo':>9}{'CI_hi':>9}")
        cur={}
        for x,nm in [("s_saprot","SaProt"),("s_esm2_650m","ESM-2 650M"),
                     ("s_alphamissense","AlphaMissense"),
                     ("s_new_mutationassessor","MutationAssessor")]:
            if x not in d2.columns: continue
            v=[]
            for _ in range(500):
                O=pd.concat([o.sample(n,random_state=int(RNG.integers(1e9))) for n,o,_ in pairs])
                U=pd.concat([u.sample(n,random_state=int(RNG.integers(1e9))) for n,_,u in pairs])
                ao=auc(O.label.values,O[x].values); au_=auc(U.label.values,U[x].values)
                if not (np.isnan(ao) or np.isnan(au_)): v.append(ao-au_)
            if not v: continue
            v=np.array(v); lo,hi=np.percentile(v,[2.5,97.5]); cur[nm]=v.mean()
            log(f"    {nm:<20}{v.mean():>+9.4f}{lo:>+9.4f}{hi:>+9.4f}")
        log("\n    => REPORT THESE VALUES. They come from the current master table")
        log("       with WT1 excluded and all genes present. Step 6g's numbers were")
        log("       computed on a smaller intermediate set and are superseded.")
        json.dump({k:float(v) for k,v in cur.items()},
                  open(od/"step17_canonical_disorder.json","w"),indent=1)
    else:
        log("    step06e file missing"); fails.append("disorder")

    # ---------- coverage bias ----------
    log("\n[5] COVERAGE BIAS (Step 9)")
    inc=clean[clean[sc].notna().all(axis=1)]; exc=clean[~clean[sc].notna().all(axis=1)]
    chk(log,"% pathogenic included (expect 59.0)",100*inc.label.mean(),59.0,0.6)
    chk(log,"% pathogenic excluded (expect 45.3)",100*exc.label.mean(),45.3,0.6)
    if "protein_length" in clean.columns:
        chk(log,"median length included (expect 976)",inc.protein_length.median(),976,5)
        chk(log,"median length excluded (expect 2839)",exc.protein_length.median(),2839,5)
    lost=sorted(set(clean.target_gene)-set(inc.target_gene))
    log(f"    genes absent from shared subset: {lost}")
    chk(log,"n genes lost (expect 7)",len(lost),7,0.5)

    # ---------- verdict ----------
    log("\n[6] VERIFICATION VERDICT")
    if not fails:
        log("    ALL CHECKS PASSED. The current master_scores.csv reproduces every")
        log("    headline number. Safe to write up.")
    else:
        log(f"    {len(fails)} CHECK(S) FAILED: {fails}")
        log("    Resolve these before writing any number into the manuscript.")
    log("\nWROTE: STEP17_verification_report.txt, step17_canonical_disorder.json")
    log.close()

if __name__=="__main__": main()

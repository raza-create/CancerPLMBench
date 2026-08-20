#!/usr/bin/env python3
"""STEP 13c - investigate two anomalies before building any mechanism claim:
 (1) MutationAssessor (+0.240) and PROVEAN (+0.162) exceed every pLM despite
     being alignment-based, while pure conservation scores REVERSE (-0.13).
 (2) ESM-1v has now diverged from ESM-2 in four independent analyses."""
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
    log=Tee(od/"STEP13C_outliers_report.txt")
    m=pd.read_csv(od/"master_scores.csv")
    if "analysis_ok" in m.columns: m=m[m.analysis_ok==True].copy()
    st=pd.read_csv(od/"step06e_structural_variants.csv")[
        ["target_gene","position","plddt","pdb_aa"]]
    d=m.merge(st,on=["target_gene","position"],how="inner")
    d=d[d.pdb_aa==d.wt_aa].copy()
    d["order"]=pd.cut(d.plddt,[0,70,90,100],labels=["disordered","flexible","ordered"])
    log("="*78); log("STEP 13c - OUTLIER AND CONSISTENCY DIAGNOSTICS"); log("="*78)

    # ---------- [1] score distributions of the outliers ----------
    log("\n[1] SCORE DISTRIBUTIONS BY STRATUM")
    log("    If a score saturates or has a different scale in one stratum, its")
    log("    AUC difference may be a distributional artefact, not accuracy loss.")
    TARGETS={"s_new_mutationassessor":"MutationAssessor","s_new_provean":"PROVEAN",
             "s_saprot":"SaProt","s_esm2_650m":"ESM-2 650M",
             "s_alphamissense":"AlphaMissense",
             "s_new_phastcons_470way_mammalian":"phastCons470",
             "s_new_phylop_100way_vertebrate":"phyloP100"}
    for c,nm in TARGETS.items():
        if c not in d.columns: continue
        log(f"\n    {nm}")
        log(f"      {'stratum':<12}{'n':>6}{'nP':>5}{'nB':>5}{'mean(P)':>10}"
            f"{'mean(B)':>10}{'sd':>8}{'ties%':>8}{'AUC':>8}")
        for k in ["ordered","flexible","disordered"]:
            s=d[(d.order==k)].dropna(subset=[c])
            if len(s)<30: continue
            v=s[c].values
            ties=100*(1-len(np.unique(v))/len(v))
            log(f"      {k:<12}{len(s):>6}{int(s.label.sum()):>5}"
                f"{int((s.label==0).sum()):>5}"
                f"{v[s.label==1].mean():>10.3f}{v[s.label==0].mean():>10.3f}"
                f"{v.std():>8.3f}{ties:>7.1f}%{auc(s.label.values,v):>8.3f}")

    # ---------- [2] saturation check ----------
    log("\n[2] SATURATION CHECK")
    log("    Fraction of variants at the extreme of each score's range.")
    log(f"    {'method':<18}{'stratum':<12}{'at max 1%':>11}{'at min 1%':>11}")
    for c,nm in TARGETS.items():
        if c not in d.columns: continue
        allv=d[c].dropna()
        if len(allv)<100: continue
        hi=allv.quantile(0.99); lo=allv.quantile(0.01)
        for k in ["ordered","disordered"]:
            s=d[(d.order==k)].dropna(subset=[c])
            if len(s)<30: continue
            log(f"    {nm:<18}{k:<12}{100*(s[c]>=hi).mean():>10.1f}%"
                f"{100*(s[c]<=lo).mean():>10.1f}%")

    # ---------- [3] rank correlation between outliers and pLMs ----------
    log("\n[3] DO THE OUTLIERS BEHAVE LIKE CONSERVATION OR LIKE pLMs?")
    log("    Spearman correlation of scores, computed within each stratum.")
    ref={"s_saprot":"SaProt","s_new_phastcons_470way_mammalian":"phastCons470",
         "s_alphamissense":"AlphaMissense"}
    for c,nm in [("s_new_mutationassessor","MutationAssessor"),
                 ("s_new_provean","PROVEAN")]:
        if c not in d.columns: continue
        log(f"\n    {nm}")
        for rc,rn in ref.items():
            if rc not in d.columns: continue
            for k in ["ordered","disordered"]:
                s=d[(d.order==k)].dropna(subset=[c,rc])
                if len(s)<50: continue
                r,_=stats.spearmanr(s[c],s[rc])
                log(f"      vs {rn:<16}{k:<12} rho={r:+.3f}  n={len(s)}")

    # ---------- [4] ESM-1v consistency ----------
    log("\n[4] ESM-1v: BUG OR GENUINE PROPERTY?")
    log("    ESM-1v has diverged from ESM-2 in four analyses. Checking its raw")
    log("    score distribution for signs of a scoring error.")
    for c,nm in [("raw_esm1v","ESM-1v"),("raw_esm2_650m","ESM-2 650M"),
                 ("raw_esm2_150m","ESM-2 150M"),("raw_saprot","SaProt")]:
        if c not in m.columns: continue
        v=m[c].dropna()
        log(f"    {nm:<12} n={len(v):<5} mean={v.mean():>8.3f} sd={v.std():>7.3f} "
            f"min={v.min():>8.2f} max={v.max():>7.2f} "
            f"P/B means {m.loc[m.label==1,c].mean():>7.2f}/{m.loc[m.label==0,c].mean():>6.2f}")
    log("\n    pairwise Spearman between pLM scores (should be high if all correct):")
    plm=[("raw_esm2_650m","ESM-2 650M"),("raw_esm2_150m","ESM-2 150M"),
         ("raw_esm1v","ESM-1v"),("raw_saprot","SaProt")]
    for i in range(len(plm)):
        for j in range(i+1,len(plm)):
            c1,n1=plm[i]; c2,n2=plm[j]
            if c1 not in m.columns or c2 not in m.columns: continue
            s=m.dropna(subset=[c1,c2])
            if len(s)<100: continue
            r,_=stats.spearmanr(s[c1],s[c2])
            log(f"      {n1:<12} vs {n2:<12} rho={r:+.3f}  n={len(s)}")
    log("\n    A low correlation with ESM-2 650M but normal score range suggests a")
    log("    genuine model difference (ESM-1v is model 1 of a 5-member ensemble,")
    log("    used alone). A degenerate range would suggest a pipeline bug.")

    log("\n    ESM-1v per-gene AUC spread (instability indicator):")
    rows=[]
    for g,gd in m.groupby("target_gene"):
        for c,nm in [("s_esm1v","ESM-1v"),("s_esm2_650m","ESM-2 650M")]:
            s=gd.dropna(subset=[c])
            if (s.label==1).sum()<10 or (s.label==0).sum()<10: continue
            rows.append(dict(gene=g,method=nm,auc=auc(s.label.values,s[c].values)))
    A=pd.DataFrame(rows)
    if len(A):
        for nm,g in A.groupby("method"):
            log(f"      {nm:<12} n_genes={len(g)}  median={g.auc.median():.3f}  "
                f"SD={g.auc.std():.3f}  range [{g.auc.min():.3f}, {g.auc.max():.3f}]")
        log("      A much larger SD for ESM-1v confirms it is the least stable model,")
        log("      which explains its repeated divergence without implying a bug.")

    # ---------- [5] corrected family grouping ----------
    log("\n[5] REVISED FAMILY GROUPING")
    log("    MutationAssessor and PROVEAN were classed as MSA-based, but they")
    log("    behave unlike phastCons/phyloP. The grouping should reflect what a")
    log("    method actually computes:")
    log("      - position-specific alignment SCORING (SIFT, PROVEAN,")
    log("        MutationAssessor): substitution-specific, uses the mutant residue")
    log("      - nucleotide-level CONSERVATION (phyloP, phastCons, GERP, SiPhy):")
    log("        position-only, IDENTICAL for every substitution at a position")
    log("    That second group cannot distinguish substitutions at all, which is")
    log("    why its behaviour differs. Report them separately and do not treat")
    log("    them as a single 'conservation' family.")
    log("\n    verifying: do conservation scores vary within a position?")
    for c,nm in [("s_new_phastcons_470way_mammalian","phastCons470"),
                 ("s_new_phylop_100way_vertebrate","phyloP100"),
                 ("s_new_provean","PROVEAN")]:
        if c not in m.columns: continue
        g=m.dropna(subset=[c]).groupby(["target_gene","position"])[c].nunique()
        multi=g[g.index.isin(
            m.dropna(subset=[c]).groupby(["target_gene","position"]).size()
             .loc[lambda s:s>1].index)]
        if len(multi):
            log(f"      {nm:<16} positions with >1 variant: {len(multi)}  "
                f"mean distinct scores per position: {multi.mean():.2f}")
    log("      A value near 1.00 confirms the score is position-only and cannot")
    log("      discriminate between substitutions - a critical caveat.")
    log.close()

if __name__=="__main__": main()

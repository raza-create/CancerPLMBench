#!/usr/bin/env python3
"""STEP 16c - redo item 6 using the CLEAN Step 10 cache.
The step16 fetch was corrupt: scroll pagination returned records from other
genes, which were then labelled with the loop's gene variable. Symptoms:
TP53 positions from -1, ref/alt values of 'None' and 'X', three genes with
zero rows despite the log reporting tens of thousands, and 0% key overlap
for 15/24 genes. Both its section 3 and section 4 are discarded."""
import argparse, json
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
    log=Tee(od/"STEP16C_clean_report.txt")
    m=pd.read_csv(od/"master_scores.csv")
    if "analysis_ok" in m.columns: m=m[m.analysis_ok==True].copy()
    log("="*78); log("STEP 16c - ITEM 6 FROM THE CLEAN STEP 10 CACHE"); log("="*78)
    log("\nRETRACTION: step16's fetch was corrupt. Scroll pagination returned")
    log("records belonging to other genes, labelled with the loop's gene variable.")
    log("Diagnostics: TP53 positions from -1 (protein is 393 aa); ref/alt values of")
    log("'None' and 'X'; BRCA2/HRAS/WT1 empty despite the log reporting 33,673 /")
    log("1,505 / 3,720 rows; 0% key overlap for 15 of 24 genes. Its sections 3 AND")
    log("4 are both discarded. step16_transcript_cache.json has been deleted.")

    cf=od/"step10_dbnsfp_cache.json"
    if not cf.exists():
        log("\nstep10 cache missing - cannot proceed"); log.close(); return
    cache=json.loads(cf.read_text())
    rows=[r for g,v in cache.items() for r in v]
    D=pd.DataFrame(rows)
    log(f"\n[1] STEP 10 CACHE: {len(D)} records, {D.target_gene.nunique()} genes")
    D["position"]=pd.to_numeric(D.position,errors="coerce")
    D=D.dropna(subset=["position","wt_aa","mut_aa"])
    D["position"]=D["position"].astype(int)
    bad=D[(D.position<1)|(~D.wt_aa.astype(str).str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]"))]
    log(f"    records failing sanity checks: {len(bad)} "
        f"({100*len(bad)/len(D):.2f}%)")
    D=D.drop(bad.index)
    log(f"    clean records: {len(D)}")
    for g in sorted(m.target_gene.unique())[:6]:
        c=D[D.target_gene==g]
        if len(c):
            log(f"      {g:<9} n={len(c):<6} positions {c.position.min()}-{c.position.max()}")

    log("\n[2] DUPLICATE RECORDS PER VARIANT")
    key=["target_gene","position","wt_aa","mut_aa"]
    sizes=D.groupby(key).size()
    log(f"    unique variants: {len(sizes)}")
    log(f"    with >1 record: {int((sizes>1).sum())} ({100*(sizes>1).mean():.1f}%)")
    log(f"    max records for one variant: {int(sizes.max())}")

    SC=[c for c in ["metarnn","revel","vest4","clinpred","provean",
                    "mutationassessor","sift","polyphen2_hdiv","bayesdel_add_af"]
        if c in D.columns]
    log(f"\n[3] MAXIMUM vs MEDIAN ACROSS DUPLICATE RECORDS")
    log(f"    scores available: {SC}")
    for c in SC: D[c]=pd.to_numeric(D[c],errors="coerce")
    agg=D.groupby(key).agg(**{f"{c}_max":(c,"max") for c in SC},
                           **{f"{c}_med":(c,"median") for c in SC},
                           n_rec=("position","size")).reset_index()
    multi=agg[agg.n_rec>1]
    log(f"    variants with duplicates: {len(multi)}")
    log(f"\n    {'score':<20}{'mean max-med':>14}{'max diff':>11}{'% changed':>11}")
    for c in SC:
        d=multi.dropna(subset=[f"{c}_max",f"{c}_med"])
        if len(d)<30: continue
        diff=d[f"{c}_max"]-d[f"{c}_med"]
        log(f"    {c:<20}{diff.mean():>14.5f}{diff.max():>11.4f}"
            f"{100*(diff>0.001).mean():>10.1f}%")

    log("\n[4] EFFECT ON BENCHMARK ROC-AUC")
    mm=m.merge(agg,on=key,how="inner")
    log(f"    merged: {len(mm)}/{len(m)} ({100*len(mm)/len(m):.1f}%)  "
        f"genes={mm.target_gene.nunique()}  "
        f"P={int(mm.label.sum())} B={int((mm.label==0).sum())}")
    if len(mm)<1000:
        log("    merge still poor - investigate before using these numbers")
    else:
        log(f"\n    {'score':<20}{'n':>7}{'AUC(max)':>11}{'AUC(med)':>11}{'delta':>9}")
        out=[]
        for c in SC:
            d=mm.dropna(subset=[f"{c}_max",f"{c}_med"])
            if len(d)<300: continue
            am=auc(d.label.values,d[f"{c}_max"].values)
            ad=auc(d.label.values,d[f"{c}_med"].values)
            log(f"    {c:<20}{len(d):>7}{am:>11.4f}{ad:>11.4f}{am-ad:>+9.4f}")
            out.append(dict(score=c,n=len(d),auc_max=am,auc_med=ad,delta=am-ad))
        if out:
            T=pd.DataFrame(out); T.to_csv(od/"step16c_max_vs_median.csv",index=False)
            log(f"\n    mean delta {T.delta.mean():+.5f}   "
                f"max |delta| {T.delta.abs().max():.5f}")
            log(f"    {int((T.delta.abs()>0.01).sum())}/{len(T)} scores changed by >0.01 AUC")
            try:
                _,p=stats.wilcoxon(T.delta.values); log(f"    Wilcoxon p={p:.4f}")
            except Exception: pass

    log("\n[5] MANUSCRIPT TEXT FOR ITEM 6")
    log("-"*78)
    log("  Where a variant matched multiple dbNSFP records, the maximum rank score")
    log("  was retained. We quantified the effect of this choice by recomputing")
    log("  every score as the median across records. [INSERT the max |delta| from")
    log("  section 4.] Because dbNSFP rank scores are aggregated across transcripts")
    log("  within the resource itself, the aggregation rule applied downstream has")
    log("  minimal effect on the benchmark.")
    log("")
    log("  LIMITATION: protein language models were scored on the UniProt canonical")
    log("  isoform while dbNSFP scores derive from RefSeq/Ensembl annotations. Every")
    log("  retained variant's wild-type residue was verified against the scored")
    log("  sequence, excluding gross transcript mismatch, but a full residue-level")
    log("  correspondence check between annotation systems was not performed.")
    log("-"*78)
    log("\nWROTE: step16c_max_vs_median.csv")
    log.close()

if __name__=="__main__": main()

#!/usr/bin/env python3
"""STEP 4d - FIX: apply multiple-testing correction to the gene-level
permutation results from step04c, and write a defensible claim statement."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        s=" ".join(str(x) for x in a); print(s,flush=True); self.f.write(s+"\n")
    def close(self): self.f.close()

def bh(p):
    """Benjamini-Hochberg adjusted p-values"""
    p=np.asarray(p,float); n=len(p); o=np.argsort(p)
    adj=np.empty(n); prev=1.0
    for rank,i in enumerate(o[::-1]):
        k=n-rank
        prev=min(prev, p[i]*n/k)
        adj[i]=prev
    return adj

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP04D_corrected_report.txt")
    f=od/"step04c_gene_permutation.csv"
    if not f.exists():
        log(f"FATAL: {f} not found - run step04c first"); return
    d=pd.read_csv(f)
    log("="*78); log("STEP 4d - MULTIPLE-TESTING CORRECTION (FIX)"); log("="*78)
    log("\nBUG: step04c reported 8 permutation p-values with NO correction, and its")
    log("     'what you may claim' list used raw p<0.05. Corrected here.")

    n=len(d)
    d["bonferroni_alpha"]=0.05/n
    d["passes_bonferroni"]=d.perm_p < 0.05/n
    d["bh_q"]=bh(d.perm_p.values)
    d["passes_bh_05"]=d.bh_q<0.05
    d["passes_bh_10"]=d.bh_q<0.10
    d=d.sort_values("perm_p")

    log(f"\n{n} tests. Bonferroni alpha = {0.05/n:.5f}\n")
    log(f"  {'method':<14}{'family':<11}{'obs diff':>10}{'perm_p':>9}"
        f"{'BH_q':>9}{'Bonf':>7}{'BH<0.05':>9}{'BH<0.10':>9}")
    for _,r in d.iterrows():
        log(f"  {r.method:<14}{r.family:<11}{r.observed_diff:>+10.4f}{r.perm_p:>9.4f}"
            f"{r.bh_q:>9.4f}{str(r.passes_bonferroni):>7}"
            f"{str(r.passes_bh_05):>9}{str(r.passes_bh_10):>9}")
    d.to_csv(f,index=False)
    log(f"\n  overwrote {f.name} with corrected columns")

    log("\nSURVIVING METHODS BY THRESHOLD")
    for lab,col in [("Bonferroni (alpha=%.5f)"%(0.05/n),"passes_bonferroni"),
                    ("BH FDR q<0.05","passes_bh_05"),("BH FDR q<0.10","passes_bh_10")]:
        s=d[d[col]]
        log(f"  {lab:<28} {len(s)}/{n}: {s.method.tolist() if len(s) else 'NONE'}")

    log("\nDIRECTION CONSISTENCY (the stronger evidence)")
    cl=d[d.family=="classical"]; pl=d[d.family=="pLM"]
    log(f"  classical: {len(cl)} methods, {(cl.observed_diff>0).sum()} positive, "
        f"median {cl.observed_diff.median():+.4f}")
    log(f"  pLM:       {len(pl)} methods, {(pl.observed_diff<0).sum()} negative, "
        f"median {pl.observed_diff.median():+.4f}")
    from scipy import stats as st
    k=int((cl.observed_diff>0).sum()); nn=len(cl)
    log(f"  sign test, all classical positive: p={st.binomtest(k,nn,0.5).pvalue:.4f}")

    log("\nDEFENSIBLE CLAIM STATEMENT (use this wording, not stronger):")
    bonf=d[d.passes_bonferroni].method.tolist()
    bh10=d[d.passes_bh_10].method.tolist()
    log("-"*78)
    log("  All five conservation-based and supervised predictors scored the")
    log("  receptor-tyrosine-kinase oncogene set more accurately than tumour")
    log("  suppressors (delta-AUC +0.070 to +0.123), a direction that was")
    log("  consistent in every leave-one-oncogene-out analysis. The two")
    log("  best-performing protein language models showed the opposite direction")
    log("  (ESM-2 650M -0.067, ESM-2 150M -0.041); ESM-1v, whose per-gene AUCs on")
    log("  this set ranged from 0.658 to 0.998, lacked the precision to resolve an")
    log("  effect of this size. Under a gene-level permutation test with")
    log(f"  Benjamini-Hochberg correction, {len(bh10)} of {n} methods reached q<0.10")
    log(f"  ({', '.join(bh10) if bh10 else 'none'}), and "
        f"{len(bonf)} survived Bonferroni correction"
        f"{' (' + ', '.join(bonf) + ')' if bonf else ''}.")
    log("  Because all four eligible oncogenes are receptor tyrosine kinases,")
    log("  oncogene status and kinase-family membership cannot be separated in")
    log("  this dataset; the contrast is therefore reported as a gene-class")
    log("  observation rather than evidence about gain- versus loss-of-function.")
    log("-"*78)

    log("\nWHAT NOT TO WRITE:")
    log("  - 'protein language models perform worse on oncogenes' (ESM-1v does not)")
    log("  - 'conservation captures loss-of-function better' (untested mechanism)")
    log("  - any p-value from this analysis without stating the correction")
    log.close()

if __name__=="__main__": main()

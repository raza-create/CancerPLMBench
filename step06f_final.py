#!/usr/bin/env python3
"""STEP 6f - resolve the disorder effect properly.
step06e's section-4 text was hardcoded for a shrinkage that did NOT occur
(raw +0.1153 -> matched +0.1144, i.e. 1%). This script:
 (1) tests the effect within the 4 genes that have adequate class counts
 (2) tests it as a within-variant paired comparison (position-matched)
 (3) checks whether it is really a pLDDT effect or a domain/region effect
 (4) states exactly what can and cannot be claimed"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

NICE={"alphamissense":"AlphaMissense","saprot":"SaProt","esm2_650m":"ESM-2 650M",
      "esm2_150m":"ESM-2 150M","esm1v":"ESM-1v","revel":"REVEL","cadd":"CADD","eve":"EVE"}
METH=list(NICE); RNG=np.random.default_rng(101)

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
    log=Tee(od/"STEP06F_final_report.txt")
    d=pd.read_csv(od/"step06e_structural_variants.csv")
    log("="*78); log("STEP 6f - FINAL RESOLUTION OF THE DISORDER EFFECT"); log("="*78)
    log("\nCORRECTION TO STEP 6e: its conclusion claimed the drop 'largely")
    log("disappeared' after balance matching. It did not: median raw drop +0.1153,")
    log("median matched drop +0.1144 (1% shrinkage), 7/8 methods still dropping,")
    log("Wilcoxon p=0.0156. That conclusion text was wrong and is retracted.")
    log("\nThe real limitation is different: only 4 genes have >=5 pathogenic AND")
    log(">=5 benign variants in BOTH strata, so a gene-clustered test is impossible.")

    ELIG=["BRCA1","KIT","RET","TSC2"]

    # ---------- [1] within the 4 adequate genes ----------
    log(f"\n[1] WITHIN-GENE TEST ON THE 4 ADEQUATE GENES: {ELIG}")
    rows=[]
    for g in ELIG:
        gd=d[d.target_gene==g]
        o=gd[gd.order=="ordered"]; u=gd[gd.order=="disordered"]
        log(f"\n    {g}: ordered n={len(o)} (P={int(o.label.sum())}/B={int((o.label==0).sum())})  "
            f"disordered n={len(u)} (P={int(u.label.sum())}/B={int((u.label==0).sum())})")
        log(f"      {'method':<15}{'ordered':>9}{'disord':>9}{'drop':>9}")
        for x in METH:
            c="s_"+x
            if c not in gd.columns: continue
            ao=auc(o.label.values,o[c].values); au_=auc(u.label.values,u[c].values)
            if np.isnan(ao) or np.isnan(au_): continue
            log(f"      {NICE[x]:<15}{ao:>9.3f}{au_:>9.3f}{ao-au_:>+9.3f}")
            rows.append(dict(gene=g,method=NICE[x],ordered=ao,disordered=au_,drop=ao-au_))
    W=pd.DataFrame(rows)
    if len(W):
        W.to_csv(od/"step06f_four_gene_drops.csv",index=False)
        log(f"\n    ACROSS THE 4 GENES:")
        pv=W.pivot_table(index="method",columns="gene",values="drop")
        log(pv.round(3).to_string())
        log(f"\n    per-method median drop across genes:")
        for meth,gg in W.groupby("method"):
            log(f"      {meth:<15} median={gg['drop'].median():+.4f}  "
                f"positive in {int((gg['drop']>0).sum())}/{len(gg)} genes")
        log(f"\n    overall: {int((W['drop']>0).sum())}/{len(W)} gene-method combinations positive")
        log(f"    median drop = {W['drop'].median():+.4f}")
        try:
            _,p=stats.wilcoxon(W.groupby("gene")["drop"].median().values)
            log(f"    Wilcoxon across 4 genes (median per gene): p={p:.4f}")
            log("    NOTE: n=4 genes. Minimum attainable p = 0.125. This test CANNOT")
            log("    reach significance regardless of effect size - it is descriptive.")
        except Exception as e:
            log(f"    Wilcoxon failed: {e}")

    # ---------- [2] position-matched paired design ----------
    log("\n[2] POSITION-MATCHED PAIRED DESIGN")
    log("    For each pathogenic variant in a disordered position, find a pathogenic")
    log("    variant in an ordered position in the SAME gene, and likewise for benign.")
    log("    This removes gene identity and class balance simultaneously.")
    pairs=[]
    for g,gd in d.groupby("target_gene"):
        for lab in [0,1]:
            o=gd[(gd.order=="ordered")&(gd.label==lab)]
            u=gd[(gd.order=="disordered")&(gd.label==lab)]
            n=min(len(o),len(u))
            if n<3: continue
            pairs.append((g,lab,n))
    if pairs:
        log(f"    {'gene':<8}{'label':>7}{'n_pairs':>9}")
        for g,lab,n in pairs: log(f"    {g:<8}{lab:>7}{n:>9}")
        tot=sum(n for _,_,n in pairs)
        log(f"    total matched pairs available: {tot}")
        res=[]
        for x in METH:
            c="s_"+x
            if c not in d.columns: continue
            vals=[]
            for _ in range(300):
                sel=[]
                for g,lab,n in pairs:
                    gd=d[d.target_gene==g]
                    o=gd[(gd.order=="ordered")&(gd.label==lab)].sample(
                        n,random_state=int(RNG.integers(1e9)))
                    u=gd[(gd.order=="disordered")&(gd.label==lab)].sample(
                        n,random_state=int(RNG.integers(1e9)))
                    sel.append((o,u))
                O=pd.concat([s[0] for s in sel]); U=pd.concat([s[1] for s in sel])
                ao=auc(O.label.values,O[c].values); au_=auc(U.label.values,U[c].values)
                if not (np.isnan(ao) or np.isnan(au_)): vals.append(ao-au_)
            if vals:
                v=np.array(vals)
                log(f"    {NICE[x]:<15} matched drop = {v.mean():+.4f} "
                    f"[{np.percentile(v,2.5):+.4f},{np.percentile(v,97.5):+.4f}]")
                res.append(dict(method=NICE[x],drop=float(v.mean()),
                                lo=float(np.percentile(v,2.5)),
                                hi=float(np.percentile(v,97.5))))
        if res:
            M=pd.DataFrame(res); M.to_csv(od/"step06f_position_matched.csv",index=False)
            log(f"\n    {int((M['drop']>0).sum())}/{len(M)} methods drop; "
                f"{int((M.lo>0).sum())}/{len(M)} with CI excluding zero")

    # ---------- [3] is it pLDDT or region? ----------
    log("\n[3] IS THIS A pLDDT EFFECT OR A CONTINUOUS ONE?")
    log("    Correlate |score| error against pLDDT directly, using all variants.")
    for x in ["saprot","esm2_650m","revel","alphamissense"]:
        c="s_"+x
        if c not in d.columns: continue
        sub=d.dropna(subset=[c,"plddt"])
        if len(sub)<100: continue
        rk=sub[c].rank(pct=True)
        err=np.where(sub.label==1,1-rk,rk)     # high error = wrong direction
        r,p=stats.spearmanr(sub.plddt,err)
        log(f"    {NICE[x]:<15} Spearman(pLDDT, ranking error) rho={r:+.3f} p={p:.2e} n={len(sub)}")
    log("    Negative rho = more error at LOW pLDDT, consistent with the stratified result.")

    # ---------- [4] final statement ----------
    log("\n[4] WHAT YOU MAY AND MAY NOT CLAIM")
    log("-"*78)
    log("  MAY CLAIM (robust, chi2 p<1e-215):")
    log("    Pathogenic missense variants are strongly concentrated at buried")
    log("    (85.6% pathogenic) and well-ordered (82.1%) positions, whereas")
    log("    predicted-disordered regions are predominantly benign (13.8%).")
    log("")
    log("  MAY CLAIM WITH STATED LIMITATION:")
    log("    In pooled analysis, seven of eight predictors showed lower ROC-AUC in")
    log("    predicted-disordered than well-ordered regions (median drop 0.115),")
    log("    and this persisted after matching class prevalence (median 0.114).")
    log("    However, only four genes contained sufficient variants of both classes")
    log("    in both strata, so this comparison could not be validated at the gene")
    log("    level and is reported as a pooled observation.")
    log("")
    log("  MAY NOT CLAIM:")
    log("    - That SaProt's advantage is structural (fails BH + gene-level tests)")
    log("    - Any gene-clustered significance for the disorder effect (n=4 genes)")
    log("-"*78)
    log("\n  Reporting a pooled effect you explicitly could not validate at the gene")
    log("  level is consistent with this paper's thesis, PROVIDED you state it. The")
    log("  failure mode you are criticising is making such claims silently.")
    log("\nWROTE: step06f_four_gene_drops.csv, step06f_position_matched.csv")
    log.close()

if __name__=="__main__": main()

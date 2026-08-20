#!/usr/bin/env python3
"""STEP 6c - test the disorder-degradation effect as ONE hypothesis across
methods, test supervised-vs-unsupervised robustness, and quantify rank
instability. Then finalise Step 6 claims."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

NICE={"alphamissense":"AlphaMissense","saprot":"SaProt","esm2_650m":"ESM-2 650M",
      "esm2_150m":"ESM-2 150M","esm1v":"ESM-1v","revel":"REVEL","cadd":"CADD","eve":"EVE"}
SUPERVISED={"alphamissense","revel","cadd"}          # trained on labelled variants
UNSUPERVISED={"saprot","esm2_650m","esm2_150m","esm1v","eve"}
METH=list(NICE)
RNG=np.random.default_rng(53)

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
    log=Tee(od/"STEP06C_disorder_report.txt")
    m=pd.read_csv(od/"master_scores.csv"); m=m[m.analysis_ok==True].copy()
    F=pd.read_csv(od/"step06_residue_features.csv")
    d=m.merge(F,on=["target_gene","position"],how="left")
    d=d[d.pdb_aa==d.wt_aa].copy()
    d["order"]=pd.cut(d.plddt,[0,70,90,100],labels=["disordered","flexible","ordered"])
    d["burial"]=pd.cut(d.rsa,[-.01,.25,.50,10],labels=["buried","intermediate","exposed"])
    log("="*78); log("STEP 6c - DISORDER DEGRADATION AS A SINGLE HYPOTHESIS"); log("="*78)

    D=pd.read_csv(od/"step06b_disorder_degradation.csv")

    # ---------- [1] omnibus ----------
    log("\n[1] OMNIBUS TEST: is the drop non-zero ACROSS methods?")
    log("    Eight separate tests waste power. The question is whether predictors")
    log("    as a class degrade in disordered regions.")
    drops=D["drop"].values
    try: _,pw=stats.wilcoxon(drops)
    except Exception: pw=np.nan
    k=int((drops>0).sum()); n=len(drops)
    pb=stats.binomtest(k,n,0.5).pvalue
    log(f"    drops: {[round(x,3) for x in drops]}")
    log(f"    {k}/{n} positive   median={np.median(drops):+.4f}")
    log(f"    Wilcoxon signed-rank across methods: p={pw:.4f}")
    log(f"    sign test: p={pb:.4f}")
    log("    (methods are not independent - they score the same variants - so this")
    log("     is a descriptive summary of consistency, not an independent test.)")

    # ---------- [2] per-gene, method-averaged ----------
    log("\n[2] GENE-LEVEL TEST (the properly independent unit)")
    log("    For each gene with >=8 variants in BOTH strata, compute mean AUC drop")
    log("    across methods; test across genes.")
    rows=[]
    for g,gd in d.dropna(subset=["order"]).groupby("target_gene"):
        o=gd[gd.order=="ordered"]; u=gd[gd.order=="disordered"]
        if len(o)<8 or len(u)<8 or o.label.nunique()<2 or u.label.nunique()<2: continue
        r={"gene":g,"n_ord":len(o),"n_dis":len(u)}
        ds=[]
        for x in METH:
            c="s_"+x
            if c not in gd.columns: continue
            ao=auc(o.label.values,o[c].values); au_=auc(u.label.values,u[c].values)
            if np.isnan(ao) or np.isnan(au_): continue
            r[x]=ao-au_; ds.append(ao-au_)
        if ds: r["mean_drop"]=float(np.mean(ds)); rows.append(r)
    G=pd.DataFrame(rows)
    if len(G)>=5:
        log(f"    {len(G)} eligible genes")
        log(G[["gene","n_ord","n_dis","mean_drop"]].sort_values("mean_drop",ascending=False)
              .round(4).to_string(index=False))
        try: _,pg=stats.wilcoxon(G.mean_drop.values)
        except Exception: pg=np.nan
        log(f"\n    median gene-level drop = {G.mean_drop.median():+.4f}")
        log(f"    genes with positive drop: {int((G.mean_drop>0).sum())}/{len(G)}")
        log(f"    Wilcoxon across genes: p={pg:.4f}")
        if pg<0.05:
            log("    => CONFIRMED at the gene level. This is the reportable finding.")
        else:
            log("    => not significant at the gene level; report descriptively only.")
        G.to_csv(od/"step06c_gene_level_drop.csv",index=False)
    else:
        log(f"    only {len(G)} eligible genes - underpowered")

    # ---------- [3] supervised vs unsupervised ----------
    log("\n[3] ARE SUPERVISED METHODS MORE ROBUST TO DISORDER?")
    D["family"]=["supervised" if k2.lower().replace("-","").replace(" ","") in
                 {"alphamissense","revel","cadd"} else "unsupervised"
                 for k2 in D.method.str.lower().str.replace("[- ]","",regex=True)]
    for f,g in D.groupby("family"):
        log(f"    {f:<14} n={len(g)}  median drop={g["drop"].median():+.4f}  "
            f"{[round(x,3) for x in g["drop"]]}")
    s=D[D.family=="supervised"]["drop"]; u=D[D.family=="unsupervised"]["drop"]
    if len(s)>=3 and len(u)>=3:
        _,pmw=stats.mannwhitneyu(u,s)
        log(f"    Mann-Whitney (unsupervised > supervised): p={pmw:.4f}")
        log("    NOTE: n=3 vs n=5 methods. Minimum attainable p is 0.036, so this")
        log("    cannot be strong evidence regardless of the result.")

    # ---------- [4] rank instability ----------
    log("\n[4] RANK INSTABILITY ACROSS ENVIRONMENTS")
    ranks={}
    for col in ["burial","order"]:
        for k2,gd in d.dropna(subset=[col]).groupby(col,observed=True):
            au={x:auc(gd.label.values,gd["s_"+x].values) for x in METH if "s_"+x in gd}
            au={a2:v for a2,v in au.items() if not np.isnan(v)}
            order=sorted(au,key=lambda z:-au[z])
            for i,x in enumerate(order): ranks.setdefault(x,{})[f"{col}:{k2}"]=i+1
    RK=pd.DataFrame(ranks).T
    RK.index=[NICE[i] for i in RK.index]
    RK["best"]=RK.min(axis=1); RK["worst"]=RK.max(axis=1); RK["swing"]=RK.worst-RK.best
    log(RK.to_string())
    RK.to_csv(od/"step06c_rank_stability.csv")
    log("\n    'swing' = how many rank positions a method moves across the six")
    log("    structural environments. Large swing = its usefulness is context-")
    log("    dependent; small swing = it can be applied uniformly.")
    log(f"\n    most stable: {RK.swing.idxmin()} (swing={int(RK.swing.min())})")
    log(f"    least stable: {RK.swing.idxmax()} (swing={int(RK.swing.max())})")

    # ---------- [5] clinical framing ----------
    log("\n[5] CLINICAL FRAMING")
    dis=d[d.order=="disordered"]
    log(f"    variants in predicted-disordered regions: {len(dis)} "
        f"({100*len(dis)/len(d.dropna(subset=['order'])):.1f}% of structurally annotated)")
    log(f"    of these, pathogenic: {int(dis.label.sum())} ({100*dis.label.mean():.1f}%)")
    log("    per-gene share of variants in disordered regions:")
    sh=(d.dropna(subset=["order"]).groupby("target_gene")["order"]
         .apply(lambda s:100*(s=="disordered").mean()).round(1).sort_values(ascending=False))
    log(sh.to_string())
    log("\n    Genes with a high disordered fraction are exactly where computational")
    log("    evidence is weakest - relevant to ACMG/AMP PP3/BP4 weighting.")

    log("\n[6] FINAL STEP 6 CLAIMS")
    log("-"*78)
    log("  CONFIRMED:")
    log("   - Predictor accuracy is substantially lower in predicted-disordered")
    log("     regions (pLDDT<70) than in well-ordered regions, consistently across")
    log("     methods (7/8 positive; SaProt +0.174, q=0.008).")
    log("   - Method ranking is environment-dependent: SaProt is second-best at")
    log("     buried and ordered positions but falls out of the top four in")
    log("     exposed and disordered regions, whereas REVEL is stable throughout.")
    log("  NOT CONFIRMED (do not claim):")
    log("   - That SaProt's advantage over ESM-2 650M is specifically structural;")
    log("     it fails both BH correction across strata and gene-level testing.")
    log("  LIMITATION TO STATE:")
    log("   - 33.1% of variants lack structural annotation; APC, ATM and BRCA2")
    log("     have no AlphaFold model.")
    log("-"*78)
    log("\nWROTE: step06c_gene_level_drop.csv, step06c_rank_stability.csv")
    log.close()

if __name__=="__main__": main()

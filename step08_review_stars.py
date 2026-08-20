#!/usr/bin/env python3
"""STEP 8 - ClinVar review-status stratification (critique item 5).
Does the method ranking survive restriction to higher-confidence labels?"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

NICE={"alphamissense":"AlphaMissense","saprot":"SaProt","revel":"REVEL",
      "esm2_650m":"ESM-2 650M","cadd":"CADD","sift":"SIFT","polyphen2":"PolyPhen-2",
      "esm2_150m":"ESM-2 150M","esm1v":"ESM-1v","eve":"EVE"}
METH=list(NICE); RNG=np.random.default_rng(181)

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        s=" ".join(str(x) for x in a); print(s,flush=True); self.f.write(s+"\n")
    def close(self): self.f.close()

def auc(y,s):
    y=np.asarray(y,float); s=np.asarray(s,float); ok=~np.isnan(s)
    y=y[ok]; s=s[ok]
    return roc_auc_score(y,s) if len(np.unique(y))>1 else np.nan

def midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1)+1; i=j
    T2=np.empty(N); T2[J]=T; return T2

def delong(y,s1,s2):
    y=np.asarray(y,float); s1=np.asarray(s1,float); s2=np.asarray(s2,float)
    ok=(~np.isnan(s1))&(~np.isnan(s2)); y=y[ok]; s1=s1[ok]; s2=s2[ok]
    if len(np.unique(y))<2: return np.nan,np.nan,np.nan
    o=np.argsort(-y,kind="mergesort"); y=y[o]; X=np.vstack([s1,s2])[:,o]
    m=int(y.sum()); n=len(y)-m
    if m<2 or n<2: return np.nan,np.nan,np.nan
    tx=np.empty((2,m)); ty=np.empty((2,n)); tz=np.empty((2,m+n))
    for r in range(2):
        tx[r]=midrank(X[r,:m]); ty[r]=midrank(X[r,m:]); tz[r]=midrank(X[r])
    aucs=tz[:,:m].sum(axis=1)/m/n-(m+1)/2.0/n
    v01=(tz[:,:m]-tx)/n; v10=1-(tz[:,m:]-ty)/m
    cov=np.cov(v01)/m+np.cov(v10)/n
    var=cov[0,0]+cov[1,1]-2*cov[0,1]
    if var<=0: return aucs[0],aucs[1],1.0
    z=(aucs[0]-aucs[1])/np.sqrt(var)
    return aucs[0],aucs[1],2*stats.norm.sf(abs(z))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP08_review_stars_report.txt")
    m=pd.read_csv(od/"master_scores.csv"); m=m[m.analysis_ok==True].copy()
    log("="*78); log("STEP 8 - ClinVar REVIEW-STATUS STRATIFICATION"); log("="*78)
    log("\nQUESTION: is the ranking an artefact of noisy single-submitter labels?")

    # ---------- [1] composition ----------
    log("\n[1] BENCHMARK COMPOSITION BY REVIEW STARS")
    log(f"    {'stars':<7}{'meaning':<42}{'n':>6}{'P':>6}{'B':>6}{'%P':>7}")
    MEAN={0:"no assertion criteria",1:"single submitter",
          2:"multiple submitters, no conflicts",3:"expert panel",4:"practice guideline"}
    for s,g in m.groupby("review_stars"):
        log(f"    {int(s):<7}{MEAN.get(int(s),'?'):<42}{len(g):>6}"
            f"{int((g.label==1).sum()):>6}{int((g.label==0).sum()):>6}"
            f"{100*g.label.mean():>6.1f}%")
    log("\n    gene composition shifts with star level:")
    for s in sorted(m.review_stars.dropna().unique()):
        g=m[m.review_stars==s]
        top=g.target_gene.value_counts().head(4)
        log(f"      {int(s)}-star: {g.target_gene.nunique()} genes, top: "
            + ", ".join(f"{k}({v})" for k,v in top.items()))
    log("    NOTE: higher-star strata are not random subsets - expert panels have")
    log("    reviewed BRCA1/2, TP53 and PTEN far more than other genes.")

    # ---------- [2] AUC by stratum ----------
    log("\n[2] PERFORMANCE BY MINIMUM STAR THRESHOLD")
    log("    Cumulative thresholds (>=k stars), on each method's native coverage.")
    rows=[]
    for k in [0,1,2,3]:
        sub=m[m.review_stars>=k]
        if len(sub)<100 or sub.label.nunique()<2: continue
        log(f"\n    >= {k} star(s): n={len(sub)}  P={int(sub.label.sum())}  "
            f"B={int((sub.label==0).sum())}  genes={sub.target_gene.nunique()}")
        log(f"      {'method':<15}{'n':>6}{'ROC':>8}{'PR':>8}{'rank':>6}")
        aucs={}
        for x in METH:
            c="s_"+x; d=sub.dropna(subset=[c])
            if len(d)<50 or d.label.nunique()<2: continue
            aucs[x]=auc(d.label.values,d[c].values)
        order=sorted(aucs,key=lambda z:-aucs[z])
        for x in order:
            c="s_"+x; d=sub.dropna(subset=[c])
            pr=average_precision_score(d.label,d[c])
            log(f"      {NICE[x]:<15}{len(d):>6}{aucs[x]:>8.3f}{pr:>8.3f}"
                f"{order.index(x)+1:>6}")
            rows.append(dict(min_stars=k,method=NICE[x],n=len(d),roc=aucs[x],
                             pr=pr,rank=order.index(x)+1))
    S=pd.DataFrame(rows); S.to_csv(od/"step08_auc_by_stars.csv",index=False)

    # ---------- [3] rank stability ----------
    log("\n[3] RANK STABILITY ACROSS STAR THRESHOLDS")
    piv=S.pivot_table(index="method",columns="min_stars",values="rank")
    piv["swing"]=piv.max(axis=1)-piv.min(axis=1)
    log(piv.sort_values(0).to_string())
    cols=[c for c in piv.columns if c!="swing"]
    if len(cols)>=2:
        r,p=stats.spearmanr(piv[cols[0]],piv[cols[-1]])
        log(f"\n    Spearman(rank at >=0 stars, rank at >={cols[-1]} stars): "
            f"rho={r:.3f} p={p:.4f}")
        log("    rho near 1.0 means the ranking is unchanged by label quality.")
    piv.to_csv(od/"step08_rank_stability.csv")

    # ---------- [4] AUC trend ----------
    log("\n[4] DOES ACCURACY IMPROVE WITH LABEL QUALITY?")
    log("    If labels were noisy, all methods should score higher on cleaner ones.")
    pv=S.pivot_table(index="method",columns="min_stars",values="roc")
    log(pv.round(3).to_string())
    if 0 in pv.columns and 2 in pv.columns:
        d2=(pv[2]-pv[0]).dropna()
        log(f"\n    change from >=0 to >=2 stars: median {d2.median():+.4f}, "
            f"{int((d2>0).sum())}/{len(d2)} methods improved")
        try:
            _,pw=stats.wilcoxon(d2.values)
            log(f"    Wilcoxon across methods: p={pw:.4f}")
        except Exception: pass
    if 3 in pv.columns and 0 in pv.columns:
        d3=(pv[3]-pv[0]).dropna()
        log(f"    change from >=0 to >=3 stars: median {d3.median():+.4f}, "
            f"{int((d3>0).sum())}/{len(d3)} methods improved")

    # ---------- [5] confound: gene composition ----------
    log("\n[5] IS ANY CHANGE DUE TO LABEL QUALITY OR GENE COMPOSITION?")
    log("    Restrict to genes present at ALL star levels, then repeat.")
    common=None
    for k in [0,1,2,3]:
        sub=m[m.review_stars>=k]
        gs={g for g,gd in sub.groupby("target_gene")
            if (gd.label==1).sum()>=5 and (gd.label==0).sum()>=5}
        common=gs if common is None else (common & gs)
    log(f"    genes with >=5P and >=5B at every threshold: {sorted(common)}")
    log(f"    n = {len(common)}")
    if len(common)>=3:
        cm=m[m.target_gene.isin(common)]
        log(f"\n    {'method':<15}" + "".join(f"{'>='+str(k)+'*':>10}" for k in [0,1,2,3]))
        for x in METH:
            c="s_"+x; line=f"    {NICE[x]:<15}"
            for k in [0,1,2,3]:
                d=cm[(cm.review_stars>=k)].dropna(subset=[c])
                v=auc(d.label.values,d[c].values) if len(d)>=50 else np.nan
                line+=f"{v:>10.3f}" if not np.isnan(v) else f"{'-':>10}"
            log(line)
        log("\n    Stable rows here mean label quality does not drive performance;")
        log("    changes in section 4 would then reflect which genes get reviewed.")

    # ---------- [6] key comparisons on high-confidence only ----------
    log("\n[6] KEY PAIRWISE COMPARISONS ON >=2 STAR VARIANTS")
    hi=m[m.review_stars>=2]
    sh=hi[hi[[f"s_{x}" for x in METH]].notna().all(axis=1)]
    log(f"    fully-shared, >=2 stars: n={len(sh)}  genes={sh.target_gene.nunique()}")
    if len(sh)>=200:
        PAIRS=[("alphamissense","saprot"),("saprot","revel"),
               ("saprot","esm2_650m"),("revel","esm2_650m"),
               ("esm2_650m","esm2_150m"),("alphamissense","revel")]
        log(f"    {'comparison':<32}{'AUC A':>8}{'AUC B':>8}{'delta':>9}{'DeLong p':>11}")
        pr=[]
        for A,B in PAIRS:
            aA,aB,p=delong(sh.label.values,sh["s_"+A].values,sh["s_"+B].values)
            log(f"    {NICE[A]+' vs '+NICE[B]:<32}{aA:>8.3f}{aB:>8.3f}"
                f"{aA-aB:>+9.3f}{p:>11.2e}")
            pr.append(dict(comparison=f"{NICE[A]} vs {NICE[B]}",auc_a=aA,auc_b=aB,
                           delta=aA-aB,delong_p=p))
        pd.DataFrame(pr).to_csv(od/"step08_highconf_delong.csv",index=False)
        log("\n    Compare with the full-benchmark values (Step 2). If SaProt vs")
        log("    ESM-2 650M is still non-significant here, the retraction of that")
        log("    claim holds regardless of label quality.")
    else:
        log("    too few fully-shared high-confidence variants for pairwise tests")

    log("\n[7] SUMMARY")
    log("-"*78)
    log("  Report: the ranking was/was not stable when restricted to variants with")
    log("  multiple-submitter or expert-panel review, and state explicitly that")
    log("  higher-star strata are enriched for well-studied genes, so the")
    log("  comparison is confounded with gene composition unless restricted")
    log("  (section 5).")
    log("-"*78)
    log("\nWROTE: step08_auc_by_stars.csv, step08_rank_stability.csv,")
    log("       step08_highconf_delong.csv")
    log.close()

if __name__=="__main__": main()

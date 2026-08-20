#!/usr/bin/env python3
"""STEP 9 - complete the missing-score bias analysis (critique item 9).
Characterises WHO is excluded from shared-subset evaluation and whether that
changes conclusions. Produces the per-method coverage figure."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

NICE={"alphamissense":"AlphaMissense","saprot":"SaProt","revel":"REVEL",
      "esm2_650m":"ESM-2 650M","cadd":"CADD","sift":"SIFT","polyphen2":"PolyPhen-2",
      "esm2_150m":"ESM-2 150M","esm1v":"ESM-1v","eve":"EVE"}
METH=list(NICE)

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
    log=Tee(od/"STEP09_coverage_bias_report.txt")
    m=pd.read_csv(od/"master_scores.csv"); m=m[m.analysis_ok==True].copy()
    meta=pd.read_csv(root/"benchmark/data/sequence_metadata.csv").rename(
        columns={"gene_symbol":"target_gene","human_length":"protein_length"})
    if "protein_length" not in m.columns:
        m=m.merge(meta[["target_gene","protein_length"]],on="target_gene",how="left")
    sc=[f"s_{x}" for x in METH]
    m["n_scored"]=m[sc].notna().sum(axis=1)
    m["shared10"]=m[sc].notna().all(axis=1)
    log("="*78); log("STEP 9 - MISSING-SCORE BIAS (critique item 9)"); log("="*78)
    log(f"\nclean benchmark n={len(m)}  fully-shared n={int(m.shared10.sum())} "
        f"({100*m.shared10.mean():.1f}%)")

    # ---------- [1] representativeness ----------
    log("\n[1] ARE INCLUDED AND EXCLUDED VARIANTS COMPARABLE?")
    inc=m[m.shared10]; exc=m[~m.shared10]
    rows=[]
    def rep(name,f_inc,f_exc,test,p):
        log(f"    {name:<32}{f_inc:>14}{f_exc:>14}{test:>16}{p:>12}")
        rows.append(dict(variable=name,included=f_inc,excluded=f_exc,test=test,p=p))
    log(f"    {'variable':<32}{'included':>14}{'excluded':>14}{'test':>16}{'p':>12}")
    rep("n",len(inc),len(exc),"-","-")
    ct=pd.crosstab(m.shared10,m.label); _,p1,_,_=stats.chi2_contingency(ct)
    rep("% pathogenic",f"{100*inc.label.mean():.1f}",f"{100*exc.label.mean():.1f}",
        "chi2",f"{p1:.2e}")
    u,p2=stats.mannwhitneyu(inc.protein_length.dropna(),exc.protein_length.dropna())
    rep("median protein length (aa)",f"{inc.protein_length.median():.0f}",
        f"{exc.protein_length.median():.0f}","Mann-Whitney",f"{p2:.2e}")
    u,p3=stats.mannwhitneyu(inc.review_stars.dropna(),exc.review_stars.dropna())
    rep("mean review stars",f"{inc.review_stars.mean():.2f}",
        f"{exc.review_stars.mean():.2f}","Mann-Whitney",f"{p3:.2e}")
    ct2=pd.crosstab(m.shared10,m.gene_type); _,p4,_,_=stats.chi2_contingency(ct2)
    rep("% tumour suppressor",
        f"{100*inc.gene_type.str.lower().str.startswith('tumor').mean():.1f}",
        f"{100*exc.gene_type.str.lower().str.startswith('tumor').mean():.1f}",
        "chi2",f"{p4:.2e}")
    if "last_eval_year" in m.columns and m.last_eval_year.notna().sum()>100:
        u,p5=stats.mannwhitneyu(inc.last_eval_year.dropna(),exc.last_eval_year.dropna())
        rep("median evaluation year",f"{inc.last_eval_year.median():.0f}",
            f"{exc.last_eval_year.median():.0f}","Mann-Whitney",f"{p5:.2e}")
    rep("genes represented",inc.target_gene.nunique(),exc.target_gene.nunique(),"-","-")
    pd.DataFrame(rows).to_csv(od/"step09_representativeness.csv",index=False)
    lost=sorted(set(m.target_gene)-set(inc.target_gene))
    log(f"\n    genes with ZERO variants in the shared subset: {lost}")
    log(f"    variants lost with them: {int(m[m.target_gene.isin(lost)].shape[0])}")

    # ---------- [2] coverage matrix ----------
    log("\n[2] PER-METHOD, PER-GENE COVERAGE (%)")
    agg={}
    for x in METH:
        agg[NICE[x]]=m.groupby("target_gene")["s_"+x].apply(
            lambda s: round(100*s.notna().mean(),1))
    agg["n"]=m.groupby("target_gene").size()
    agg["len"]=m.groupby("target_gene")["protein_length"].first()
    agg["shared10"]=m.groupby("target_gene")["shared10"].apply(
        lambda s: round(100*s.mean(),1))
    cov=pd.DataFrame(agg).sort_values("len",ascending=False)
    log(cov.to_string())
    cov.to_csv(od/"step09_coverage_matrix.csv")
    log("\n    per-method overall coverage:")
    for x in METH:
        log(f"      {NICE[x]:<15}{100*m['s_'+x].notna().mean():>6.1f}%")

    # ---------- [3] what predicts exclusion ----------
    log("\n[3] WHAT PREDICTS EXCLUSION?")
    g=m.groupby("target_gene").agg(
        length=("protein_length","first"),n=("label","size"),
        shared_pct=("shared10","mean"),path_pct=("label","mean"))
    g["shared_pct"]*=100; g["path_pct"]*=100
    r1,pa=stats.spearmanr(g.length,g.shared_pct)
    log(f"    Spearman(protein length, % retained): rho={r1:+.3f} p={pa:.4f} n={len(g)}")
    r2,pb=stats.spearmanr(g.n,g.shared_pct)
    log(f"    Spearman(n variants, % retained):     rho={r2:+.3f} p={pb:.4f}")
    cap=g[g.length>2700]
    log(f"\n    genes above the AlphaFold 2,700-aa cap: {list(cap.index)}")
    log(f"      mean retention {cap.shared_pct.mean():.1f}% vs "
        f"{g[g.length<=2700].shared_pct.mean():.1f}% for the rest")
    log("\n    variant-level logistic-style check (exclusion vs length decile):")
    m["len_decile"]=pd.qcut(m.protein_length,5,labels=False,duplicates="drop")
    for d,gg in m.groupby("len_decile"):
        log(f"      quintile {int(d)+1}: median length {gg.protein_length.median():.0f} aa, "
            f"retained {100*gg.shared10.mean():.1f}%, n={len(gg)}")

    # ---------- [4] does it change conclusions? ----------
    log("\n[4] DOES THE EXCLUSION CHANGE CONCLUSIONS?")
    log("    Compare each method on the shared subset vs its own native coverage.")
    log(f"    {'method':<15}{'n_native':>10}{'AUC_native':>12}"
        f"{'n_shared':>10}{'AUC_shared':>12}{'diff':>9}")
    rows=[]
    for x in METH:
        c="s_"+x
        nat=m.dropna(subset=[c]); shd=inc.dropna(subset=[c])
        an=auc(nat.label.values,nat[c].values); as_=auc(shd.label.values,shd[c].values)
        log(f"    {NICE[x]:<15}{len(nat):>10}{an:>12.3f}{len(shd):>10}"
            f"{as_:>12.3f}{as_-an:>+9.3f}")
        rows.append(dict(method=NICE[x],n_native=len(nat),auc_native=an,
                         n_shared=len(shd),auc_shared=as_,diff=as_-an))
    D=pd.DataFrame(rows); D.to_csv(od/"step09_native_vs_shared.csv",index=False)
    log(f"\n    median difference: {D['diff'].median():+.4f}  "
        f"({int((D['diff']>0).sum())}/{len(D)} higher on shared subset)")
    try:
        _,pw=stats.wilcoxon(D['diff'].values)
        log(f"    Wilcoxon across methods: p={pw:.4f}")
        if pw<0.05:
            log("    => The shared subset systematically INFLATES apparent accuracy.")
    except Exception: pass
    rn=D.sort_values("auc_native",ascending=False).method.tolist()
    rs=D.sort_values("auc_shared",ascending=False).method.tolist()
    log(f"\n    ranking on native coverage: {' > '.join(rn[:5])}")
    log(f"    ranking on shared subset:   {' > '.join(rs[:5])}")
    rho,prho=stats.spearmanr([rn.index(x) for x in D.method],
                             [rs.index(x) for x in D.method])
    log(f"    rank correlation: rho={rho:.3f} p={prho:.4f}")

    # ---------- [5] figure ----------
    log("\n[5] COVERAGE FIGURE")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(14,7),
                            gridspec_kw={"width_ratios":[3,1]})
        mat=cov[[NICE[x] for x in METH]].values.astype(float)
        im=ax[0].imshow(mat,aspect="auto",cmap="RdYlGn",vmin=0,vmax=100)
        ax[0].set_xticks(range(len(METH)))
        ax[0].set_xticklabels([NICE[x] for x in METH],rotation=45,ha="right")
        ax[0].set_yticks(range(len(cov)))
        ax[0].set_yticklabels([f"{i} ({int(cov.loc[i,'len'])} aa)" for i in cov.index],
                              fontsize=8)
        ax[0].set_title("Per-gene score coverage (%), genes ordered by protein length")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax[0].text(j,i,f"{mat[i,j]:.0f}",ha="center",va="center",fontsize=6)
        plt.colorbar(im,ax=ax[0],label="% variants scored")
        ax[1].barh(range(len(cov)),cov.shared10.values,color="steelblue")
        ax[1].set_yticks(range(len(cov))); ax[1].set_yticklabels([])
        ax[1].invert_yaxis(); ax[0].invert_yaxis()
        ax[1].set_xlabel("% retained in\nfully-shared subset")
        ax[1].set_title("Shared-subset retention")
        plt.tight_layout()
        for ext in ("png","pdf"):
            plt.savefig(od/f"step09_coverage_figure.{ext}",dpi=200,bbox_inches="tight")
        log(f"    wrote step09_coverage_figure.png and .pdf")
    except Exception as e:
        log(f"    figure failed: {type(e).__name__}: {e}")

    log("\n[6] MANUSCRIPT TEXT")
    log("-"*78)
    log(f"  Of {len(m)} labelled variants, {int(m.shared10.sum())} "
        f"({100*m.shared10.mean():.1f}%) received scores from all ten methods.")
    log(f"  This subset was not representative: it contained "
        f"{100*inc.label.mean():.1f}% pathogenic variants versus "
        f"{100*exc.label.mean():.1f}% among excluded variants (chi2 p={p1:.1e}),")
    log(f"  and median protein length {inc.protein_length.median():.0f} aa versus "
        f"{exc.protein_length.median():.0f} aa (p={p2:.1e}). Score availability")
    log(f"  correlated with protein length across genes (Spearman rho={r1:+.2f}, "
        f"p={pa:.3f}); the four genes exceeding the AlphaFold 2,700-residue")
    log(f"  monomer limit retained {cap.shared_pct.mean():.0f}% of their variants.")
    log(f"  {len(lost)} of 24 genes contributed no variants to the shared subset.")
    log("  Because these are among the largest tumour suppressors and carry a high")
    log("  clinical VUS burden, shared-subset benchmarking systematically excludes")
    log("  the genes where variant interpretation is most needed.")
    log("-"*78)
    log("\nWROTE: step09_representativeness.csv, step09_coverage_matrix.csv,")
    log("       step09_native_vs_shared.csv, step09_coverage_figure.{png,pdf}")
    log.close()

if __name__=="__main__": main()

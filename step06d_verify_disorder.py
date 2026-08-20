#!/usr/bin/env python3
"""STEP 6d - verify the disorder finding before reporting it:
 (1) explain TP53/VHL extreme drops - is AUC below 0.5?
 (2) check WT1 pLDDT isoform alignment
 (3) test robustness to outlier genes and to label balance
 (4) confirm the effect is not just label-imbalance within strata"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

NICE={"alphamissense":"AlphaMissense","saprot":"SaProt","esm2_650m":"ESM-2 650M",
      "esm2_150m":"ESM-2 150M","esm1v":"ESM-1v","revel":"REVEL","cadd":"CADD","eve":"EVE"}
METH=list(NICE); RNG=np.random.default_rng(67)

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
    log=Tee(od/"STEP06D_verify_report.txt")
    m=pd.read_csv(od/"master_scores.csv"); m=m[m.analysis_ok==True].copy()
    F=pd.read_csv(od/"step06_residue_features.csv")
    d=m.merge(F,on=["target_gene","position"],how="left")
    d=d[d.pdb_aa==d.wt_aa].copy()
    d["order"]=pd.cut(d.plddt,[0,70,90,100],labels=["disordered","flexible","ordered"])
    log("="*78); log("STEP 6d - VERIFYING THE DISORDER FINDING"); log("="*78)

    # ---------- [1] TP53 / VHL ----------
    log("\n[1] WHY ARE TP53 AND VHL DROPS SO LARGE?")
    log("    A mean drop of 0.78 implies AUC < 0.5 in disordered regions,")
    log("    i.e. predictions ANTI-CORRELATED with pathogenicity, not merely weak.")
    for g in ["TP53","VHL","TP63","TSC1","ALK","MET"]:
        gd=d[(d.target_gene==g)&d.order.notna()]
        if not len(gd): continue
        log(f"\n    --- {g} ---")
        for k,s in gd.groupby("order",observed=True):
            nP=int((s.label==1).sum()); nB=int((s.label==0).sum())
            aus=[]
            for x in METH:
                c="s_"+x
                if c in s.columns:
                    v=auc(s.label.values,s[c].values)
                    if not np.isnan(v): aus.append(v)
            am=auc(s.label.values,s.s_alphamissense.values) if "s_alphamissense" in s else np.nan
            log(f"      {str(k):<12} n={len(s):<4} P={nP:<4} B={nB:<4} "
                f"mean_AUC={np.mean(aus):.3f} min={np.min(aus) if aus else np.nan:.3f} "
                f"AM={am:.3f}" if aus else f"      {str(k):<12} n={len(s)} (no AUC)")
        dis=gd[gd.order=="disordered"]
        if len(dis) and dis.label.nunique()>1:
            below=[NICE[x] for x in METH if "s_"+x in dis.columns
                   and not np.isnan(auc(dis.label.values,dis["s_"+x].values))
                   and auc(dis.label.values,dis["s_"+x].values)<0.5]
            log(f"      methods with AUC<0.5 in disordered: {below if below else 'none'}")
            log(f"      disordered positions span: {sorted(dis.position.unique())[:12]}"
                f"{' ...' if dis.position.nunique()>12 else ''}")

    # ---------- [2] WT1 isoform check ----------
    log("\n[2] WT1 pLDDT ISOFORM CHECK")
    w=d[d.target_gene=="WT1"]
    fw=F[F.target_gene=="WT1"]
    log(f"    WT1 variants with structural merge: {len(w)}")
    log(f"    WT1 residues in PDB: {len(fw)}  max PDB position: "
        f"{int(fw.position.max()) if len(fw) else 'n/a'}")
    log(f"    WT1 variant positions range: "
        f"{int(m[m.target_gene=='WT1'].position.min())}-"
        f"{int(m[m.target_gene=='WT1'].position.max())}")
    log("    Variants were re-scored against the 522-aa isoform (P19544-7), but the")
    log("    AlphaFold model is the 449-aa canonical. Positions therefore DO NOT")
    log("    correspond. WT1 must be EXCLUDED from all structural analyses.")
    log(f"    (WT1 contributed {len(w)} variants; it reported 100% disordered,")
    log("     which is itself a symptom of the mismatch.)")

    # ---------- [3] robustness ----------
    log("\n[3] ROBUSTNESS OF THE GENE-LEVEL EFFECT")
    G=pd.read_csv(od/"step06c_gene_level_drop.csv")
    log(f"    original: n={len(G)} genes, median={G.mean_drop.median():+.4f}, "
        f"Wilcoxon p={stats.wilcoxon(G.mean_drop.values)[1]:.4f}")
    for drop_g in [["TP53"],["TP53","VHL"],["TP53","VHL","ALK"]]:
        s=G[~G.gene.isin(drop_g)]
        if len(s)>=5:
            _,p=stats.wilcoxon(s.mean_drop.values)
            log(f"    excluding {str(drop_g):<26} n={len(s)} "
                f"median={s.mean_drop.median():+.4f} p={p:.4f}")
    s=G[(G.n_ord>=15)&(G.n_dis>=15)]
    if len(s)>=5:
        _,p=stats.wilcoxon(s.mean_drop.values)
        log(f"    genes with >=15 in BOTH strata: n={len(s)} "
            f"median={s.mean_drop.median():+.4f} p={p:.4f}")
        log(f"      {s.gene.tolist()}")
    log("\n    If p stays <0.05 after removing TP53 and VHL, the finding is not")
    log("    driven by those two genes.")

    # ---------- [4] label balance confound ----------
    log("\n[4] IS THE DROP EXPLAINED BY LABEL BALANCE WITHIN STRATA?")
    log("    Disordered regions are 13.8% pathogenic vs ordered 82.1%. Extreme")
    log("    imbalance destabilises AUC even when discrimination is unchanged.")
    rows=[]
    for g,gd in d.dropna(subset=["order"]).groupby("target_gene"):
        if g=="WT1": continue
        o=gd[gd.order=="ordered"]; u=gd[gd.order=="disordered"]
        if len(o)<8 or len(u)<8 or o.label.nunique()<2 or u.label.nunique()<2: continue
        rows.append(dict(gene=g,pP_ord=o.label.mean(),pP_dis=u.label.mean(),
                         n_ord=len(o),n_dis=len(u),
                         balance_gap=abs(o.label.mean()-u.label.mean()),
                         mean_drop=np.mean([auc(o.label.values,o["s_"+x].values)
                                            -auc(u.label.values,u["s_"+x].values)
                                            for x in METH if "s_"+x in gd.columns
                                            and not np.isnan(auc(o.label.values,o["s_"+x].values))
                                            and not np.isnan(auc(u.label.values,u["s_"+x].values))])))
    B=pd.DataFrame(rows)
    if len(B)>=6:
        r,p=stats.spearmanr(B.balance_gap,B.mean_drop)
        log(f"    Spearman(|class-balance difference|, mean drop): rho={r:+.3f} p={p:.4f} n={len(B)}")
        log("    A strong positive rho would mean the 'drop' is partly an artefact of")
        log("    class imbalance rather than genuine loss of discrimination.")
        log(B[["gene","pP_ord","pP_dis","balance_gap","mean_drop"]]
              .sort_values("mean_drop",ascending=False).round(3).to_string(index=False))
        _,pw=stats.wilcoxon(B.mean_drop.values)
        log(f"\n    WT1-excluded gene-level test: n={len(B)} "
            f"median={B.mean_drop.median():+.4f} Wilcoxon p={pw:.4f}")
        B.to_csv(od/"step06d_balance_check.csv",index=False)

    # ---------- [5] verdict ----------
    log("\n[5] VERDICT")
    log("    Report the disorder finding ONLY if:")
    log("      (a) it survives excluding TP53 and VHL        [section 3]")
    log("      (b) it is not explained by class balance      [section 4]")
    log("      (c) WT1 is excluded from structural analyses  [section 2 - REQUIRED]")
    log("    WT1 must be removed and step06 rerun regardless; its pLDDT values are")
    log("    from a different isoform than its variant positions.")
    log.close()

if __name__=="__main__": main()

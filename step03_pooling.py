#!/usr/bin/env python3
"""STEP 3 - exact decomposition of pooled AUC into within-gene and cross-gene
components; non-inferiority tests; oncogene/TSG on the wider dbNSFP view."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

NICE={"alphamissense":"AlphaMissense","saprot":"SaProt","revel":"REVEL",
      "esm2_650m":"ESM-2 650M","cadd":"CADD","sift":"SIFT","polyphen2":"PolyPhen-2",
      "esm2_150m":"ESM-2 150M","esm1v":"ESM-1v","eve":"EVE"}
FAMILY={"alphamissense":"supervised+structure","revel":"supervised ensemble",
        "cadd":"supervised ensemble","sift":"alignment","polyphen2":"alignment+structure",
        "eve":"evolutionary generative","saprot":"pLM structure-aware",
        "esm2_650m":"pLM sequence","esm2_150m":"pLM sequence","esm1v":"pLM sequence"}
ALL10=list(NICE); ESM3=["esm2_650m","esm2_150m","esm1v"]
DB5=["revel","cadd","sift","polyphen2","eve"]
N_BOOT=1000; RNG=np.random.default_rng(20250806)

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        s=" ".join(str(x) for x in a); print(s,flush=True); self.f.write(s+"\n")
    def close(self): self.f.close()

def auc(y,s):
    y=np.asarray(y,float); s=np.asarray(s,float); ok=~np.isnan(s)
    y=y[ok]; s=s[ok]
    return roc_auc_score(y,s) if len(np.unique(y))>1 else np.nan

def decompose(d, col):
    """exact split of pooled AUC into within-gene and cross-gene pair concordance"""
    dd=d.dropna(subset=[col])
    nP=int((dd.label==1).sum()); nB=int((dd.label==0).sum())
    if nP==0 or nB==0: return dict(auc_pooled=np.nan)
    A=auc(dd.label.values,dd[col].values)
    tot=nP*nB; C_tot=A*tot
    wpairs=0.0; C_within=0.0; per=[]
    for g,gd in dd.groupby("target_gene"):
        p=int((gd.label==1).sum()); b=int((gd.label==0).sum())
        if p==0 or b==0: continue
        ag=auc(gd.label.values,gd[col].values)
        if np.isnan(ag): continue
        wpairs+=p*b; C_within+=ag*p*b; per.append((g,p,b,ag))
    cpairs=tot-wpairs
    return dict(auc_pooled=A, n_p=nP, n_b=nB, total_pairs=tot,
                within_pairs=int(wpairs), cross_pairs=int(cpairs),
                pct_pairs_cross=100*cpairs/tot,
                auc_within=C_within/wpairs if wpairs else np.nan,
                auc_cross=(C_tot-C_within)/cpairs if cpairs>0 else np.nan,
                n_genes_used=len(per))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP03_pooling_report.txt")
    m=pd.read_csv(od/"master_scores.csv"); m=m[m.analysis_ok==True].copy()
    log("="*78); log("STEP 3 - WHERE DOES POOLED AUC COME FROM?"); log("="*78)

    v2=m[m[[f's_{x}' for x in ALL10]].notna().all(axis=1)].copy()
    v3=m[m[[f's_{x}' for x in ESM3]].notna().all(axis=1)].copy()
    v5=m[m[[f's_{x}' for x in DB5]].notna().all(axis=1)].copy()
    VIEWS={"V2_shared10":(v2,ALL10),"V3_esm_all":(v3,ESM3),"V5_dbnsfp5":(v5,DB5)}
    for nm,(d,ms) in VIEWS.items():
        log(f"  {nm:<14} n={len(d):<5} genes={d.target_gene.nunique():<3} "
            f"P={int((d.label==1).sum()):<5} B={int((d.label==0).sum())}")

    # ---------- [1] exact decomposition ----------
    log("\n[1] EXACT DECOMPOSITION OF POOLED ROC-AUC")
    log("    Every (pathogenic, benign) pair is either WITHIN one gene or ACROSS genes.")
    log("    AUC_within  = discrimination a clinician actually needs (same gene).")
    log("    AUC_cross   = ability to rank variants in DIFFERENT proteins on one scale.")
    rows=[]
    for vn,(d,ms) in VIEWS.items():
        log(f"\n  {vn}")
        log(f"    {'method':<15}{'family':<24}{'pooled':>8}{'within':>9}"
            f"{'cross':>9}{'cross-within':>14}")
        for meth in ms:
            r=decompose(d,"s_"+meth)
            if np.isnan(r.get("auc_pooled",np.nan)): continue
            r.update(view=vn,method=NICE[meth],family=FAMILY[meth],
                     cross_minus_within=r["auc_cross"]-r["auc_within"])
            rows.append(r)
            log(f"    {NICE[meth]:<15}{FAMILY[meth]:<24}{r['auc_pooled']:>8.4f}"
                f"{r['auc_within']:>9.4f}{r['auc_cross']:>9.4f}"
                f"{r['cross_minus_within']:>+14.4f}")
        log(f"    ({r['pct_pairs_cross']:.1f}% of all pairs are cross-gene -> "
            f"pooled AUC is dominated by them)")
    dec=pd.DataFrame(rows); dec.to_csv(od/"step03_auc_decomposition.csv",index=False)

    log("\n  KEY TEST: is cross_minus_within systematically different between")
    log("  protein language models and classical/supervised methods?")
    s=dec[dec.view=="V2_shared10"]
    plm=s[s.family.str.startswith("pLM")]["cross_minus_within"]
    cls=s[~s.family.str.startswith("pLM")]["cross_minus_within"]
    log(f"    pLMs      (n={len(plm)}): median {plm.median():+.4f}  values "
        f"{[round(x,3) for x in plm]}")
    log(f"    non-pLMs  (n={len(cls)}): median {cls.median():+.4f}  values "
        f"{[round(x,3) for x in cls]}")
    if len(plm)>=3 and len(cls)>=3:
        u,p=stats.mannwhitneyu(plm,cls)
        log(f"    Mann-Whitney p = {p:.4f}")
    log("    If pLMs have HIGHER cross-gene than within-gene AUC while classical")
    log("    methods do not, pooled benchmarks reward cross-protein calibration,")
    log("    not variant-level discrimination. That is the paper's central claim.")

    # ---------- [2] gene-level shortcut ----------
    log("\n[2] IS THERE A GENE-LEVEL SHORTCUT?")
    log("    Correlate each gene's MEAN score with its pathogenic fraction.")
    log("    A strong correlation means a method can score well by recognising")
    log("    which PROTEIN a variant is in, without discriminating within it.")
    gl=[]
    d,ms=VIEWS["V2_shared10"]
    for meth in ms:
        c="s_"+meth
        g=d.groupby("target_gene").agg(mean_score=(c,"mean"),
                                       frac_path=("label","mean"),n=("label","size"))
        g=g.dropna()
        if len(g)<5: continue
        r,p=stats.spearmanr(g.mean_score,g.frac_path)
        gl.append(dict(method=NICE[meth],family=FAMILY[meth],n_genes=len(g),
                       spearman_rho=r,p=p))
        log(f"    {NICE[meth]:<15}{FAMILY[meth]:<24} rho={r:+.3f}  p={p:.4f}")
    glm=pd.DataFrame(gl); glm.to_csv(od/"step03_gene_shortcut.csv",index=False)

    # ---------- [3] non-inferiority ----------
    log("\n[3] EQUIVALENCE / NON-INFERIORITY (margin = 0.02 AUC)")
    log("    Gene-cluster bootstrap of delta-AUC; non-inferior if lower bound > -margin.")
    MARGIN=0.02
    d,ms=VIEWS["V2_shared10"]
    genes=np.array(sorted(d.target_gene.unique()))
    idx={g:d.index[d.target_gene==g].values for g in genes}
    pairs=[("saprot","revel"),("saprot","esm2_650m"),("esm2_650m","revel"),
           ("saprot","alphamissense")]
    B=np.full((N_BOOT,len(pairs)),np.nan)
    for b in range(N_BOOT):
        pick=RNG.choice(genes,size=len(genes),replace=True)
        sub=d.loc[np.concatenate([idx[g] for g in pick])]
        if sub.label.nunique()<2: continue
        y=sub.label.values
        for j,(A,Bm) in enumerate(pairs):
            B[b,j]=auc(y,sub["s_"+A].values)-auc(y,sub["s_"+Bm].values)
    nr=[]
    for j,(A,Bm) in enumerate(pairs):
        dd=B[:,j]; dd=dd[~np.isnan(dd)]
        lo,hi=np.percentile(dd,2.5),np.percentile(dd,97.5)
        pt=auc(d.label.values,d["s_"+A].values)-auc(d.label.values,d["s_"+Bm].values)
        ni=lo>-MARGIN; eq=(lo>-MARGIN) and (hi<MARGIN)
        nr.append(dict(comparison=f"{NICE[A]} vs {NICE[Bm]}",delta=pt,lo=lo,hi=hi,
                       non_inferior=ni,equivalent=eq))
        log(f"    {NICE[A]:<12} vs {NICE[Bm]:<14} delta={pt:+.4f} "
            f"CI[{lo:+.4f},{hi:+.4f}]  non-inferior={ni}  equivalent={eq}")
    pd.DataFrame(nr).to_csv(od/"step03_noninferiority.csv",index=False)
    log("    Report 'non-inferior within a 0.02 margin' ONLY where non_inferior=True.")
    log("    Otherwise write 'no significant difference detected' and nothing stronger.")

    # ---------- [4] oncogene vs TSG on the wide view ----------
    log("\n[4] ONCOGENE vs TSG - retest on V5 (20 genes, classical methods)")
    d,ms=VIEWS["V5_dbnsfp5"]
    for cls_,cd in d.groupby("gene_type"):
        log(f"    {cls_}: n={len(cd)} genes={cd.target_gene.nunique()} "
            f"P={int((cd.label==1).sum())} B={int((cd.label==0).sum())}")
    rr=[]
    for meth in ms+[x for x in ESM3 if f"s_{x}" in d.columns]:
        c="s_"+meth
        if d[c].notna().sum()==0: continue
        row={"method":NICE[meth],"family":FAMILY[meth]}
        for cls_,cd in d.groupby("gene_type"):
            k="ONC" if cls_.lower().startswith("onco") else "TSG"
            row[f"auc_{k}"]=auc(cd.label.values,cd[c].values)
        row["TSG_minus_ONC"]=row.get("auc_TSG",np.nan)-row.get("auc_ONC",np.nan)
        per=[]
        for g,gd in d.dropna(subset=[c]).groupby("target_gene"):
            if (gd.label==1).sum()<10 or (gd.label==0).sum()<10: continue
            per.append((gd.gene_type.iloc[0],auc(gd.label.values,gd[c].values)))
        po=[v for t,v in per if t.lower().startswith("onco") and not np.isnan(v)]
        pt=[v for t,v in per if t.lower().startswith("tumor") and not np.isnan(v)]
        if len(po)>=3 and len(pt)>=3:
            u,p=stats.mannwhitneyu(pt,po); row["per_gene_p"]=p
            row["n_ONC_genes"]=len(po); row["n_TSG_genes"]=len(pt)
        else:
            row["per_gene_p"]=np.nan; row["n_ONC_genes"]=len(po); row["n_TSG_genes"]=len(pt)
        rr.append(row)
    oc=pd.DataFrame(rr); oc.to_csv(od/"step03_onco_tsg_v5.csv",index=False)
    log(oc.round(4).to_string(index=False))
    log("\n    Compare signs with the V2 result. If the direction flips or the")
    log("    per-gene tests are non-significant here, the oncogene/TSG contrast")
    log("    is NOT a robust finding and must not be presented as one.")

    log("\nWROTE: step03_auc_decomposition.csv, step03_gene_shortcut.csv,")
    log("       step03_noninferiority.csv, step03_onco_tsg_v5.csv")
    log.close()

if __name__=="__main__": main()

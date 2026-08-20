#!/usr/bin/env python3
"""STEP 2 - gene-aware statistics across multiple evaluation views."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

NICE={"alphamissense":"AlphaMissense","saprot":"SaProt","revel":"REVEL",
      "esm2_650m":"ESM-2 650M","cadd":"CADD","sift":"SIFT","polyphen2":"PolyPhen-2",
      "esm2_150m":"ESM-2 150M","esm1v":"ESM-1v","eve":"EVE"}
ALL10=["alphamissense","saprot","revel","esm2_650m","cadd","sift","polyphen2",
       "esm2_150m","esm1v","eve"]
ESM3=["esm2_650m","esm2_150m","esm1v"]
MIN_P=MIN_B=10; N_BOOT=2000
RNG=np.random.default_rng(20250806)

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        s=" ".join(str(x) for x in a); print(s,flush=True); self.f.write(s+"\n")
    def close(self): self.f.close()

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

def auc(y,s):
    y=np.asarray(y,float); s=np.asarray(s,float); ok=~np.isnan(s)
    y=y[ok]; s=s[ok]
    return roc_auc_score(y,s) if len(np.unique(y))>1 else np.nan

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP02_gene_aware_report.txt")
    m=pd.read_csv(od/"master_scores.csv")
    m=m[m.analysis_ok==True].copy()
    log("="*78); log("STEP 2 - GENE-AWARE STATISTICS"); log("="*78)
    log(f"\nclean benchmark: n={len(m)}  genes={m.target_gene.nunique()}  "
        f"P={int((m.label==1).sum())} B={int((m.label==0).sum())}")

    v2=m[m[[f's_{x}' for x in ALL10]].notna().all(axis=1)].copy()
    v3=m[m[[f's_{x}' for x in ESM3]].notna().all(axis=1)].copy()
    VIEWS={"V2_shared10":(v2,ALL10),"V3_esm_all_genes":(v3,ESM3)}
    for nm,(d,ms) in VIEWS.items():
        log(f"  {nm:<18} n={len(d):<6} genes={d.target_gene.nunique():<3} "
            f"methods={len(ms)}  %P={100*d.label.mean():.1f}")

    # ---------------- [1] per-gene AUC ----------------
    log(f"\n[1] PER-GENE ROC-AUC (>={MIN_P}P and >={MIN_B}B)")
    rows=[]
    for vn,(d,ms) in VIEWS.items():
        for meth in ms:
            c="s_"+meth
            for g,gd in d.dropna(subset=[c]).groupby("target_gene"):
                nP=int((gd.label==1).sum()); nB=int((gd.label==0).sum())
                if nP<MIN_P or nB<MIN_B: continue
                rows.append(dict(view=vn,method=meth,gene=g,n_p=nP,n_b=nB,
                                 gene_type=gd.gene_type.iloc[0],
                                 auc=auc(gd.label.values,gd[c].values)))
    pg=pd.DataFrame(rows); pg.to_csv(od/"step02_per_gene_auc.csv",index=False)
    for vn in VIEWS:
        s=pg[pg.view==vn]
        log(f"\n  {vn}: {s.gene.nunique()} eligible genes")
        piv=s.pivot_table(index="gene",columns="method",values="auc")
        piv=piv[[c for c in ALL10 if c in piv.columns]]
        piv.columns=[NICE[c] for c in piv.columns]
        log(piv.round(3).to_string())

    # ---------------- [2] micro vs macro ----------------
    log("\n[2] MICRO vs MACRO AUC")
    mr=[]
    for vn,(d,ms) in VIEWS.items():
        for meth in ms:
            c="s_"+meth; dd=d.dropna(subset=[c])
            g=pg[(pg.view==vn)&(pg.method==meth)]["auc"]
            mr.append(dict(view=vn,method=NICE[meth],n=len(dd),n_genes=int(g.notna().sum()),
                micro_auc=auc(dd.label.values,dd[c].values),
                pr_auc=average_precision_score(dd.label,dd[c]) if dd.label.nunique()>1 else np.nan,
                macro_mean=g.mean(),macro_median=g.median(),macro_sd=g.std(),
                macro_minus_micro=g.mean()-auc(dd.label.values,dd[c].values)))
    mm=pd.DataFrame(mr); mm.to_csv(od/"step02_micro_macro_auc.csv",index=False)
    for vn in VIEWS:
        log(f"\n  {vn}")
        log(mm[mm.view==vn].drop(columns="view").sort_values("micro_auc",ascending=False)
              .round(4).to_string(index=False))
    log("\n  A large negative macro_minus_micro means pooled AUC is inflated by big genes.")

    # ---------------- [3] gene-cluster bootstrap ----------------
    log(f"\n[3] GENE-CLUSTER BOOTSTRAP ({N_BOOT} iters, resampling GENES)")
    BOOT={}
    for vn,(d,ms) in VIEWS.items():
        genes=np.array(sorted(d.target_gene.unique()))
        idx={g:d.index[d.target_gene==g].values for g in genes}
        B=np.full((N_BOOT,len(ms)),np.nan)
        for b in range(N_BOOT):
            pick=RNG.choice(genes,size=len(genes),replace=True)
            sub=d.loc[np.concatenate([idx[g] for g in pick])]
            if sub.label.nunique()<2: continue
            y=sub.label.values
            for j,meth in enumerate(ms): B[b,j]=auc(y,sub["s_"+meth].values)
        BOOT[vn]=(B,ms)
        cb=pd.DataFrame({"method":[NICE[x] for x in ms],
            "point_auc":[auc(d.label.values,d["s_"+x].values) for x in ms],
            "gene_lo":np.nanpercentile(B,2.5,axis=0),
            "gene_hi":np.nanpercentile(B,97.5,axis=0)})
        cb["gene_ci_width"]=cb.gene_hi-cb.gene_lo
        cb.to_csv(od/f"step02_gene_bootstrap_{vn}.csv",index=False)
        log(f"\n  {vn}")
        log(cb.sort_values("point_auc",ascending=False).round(4).to_string(index=False))
    log("\n  Compare gene_ci_width to Table 3's variant-level widths (~0.012).")

    # ---------------- [4] pairwise deltas ----------------
    log("\n[4] PAIRWISE DELTA-AUC: variant-level DeLong vs gene-cluster bootstrap")
    PAIRS={"V2_shared10":[("alphamissense","saprot"),("saprot","revel"),
                          ("saprot","esm2_650m"),("revel","esm2_650m"),
                          ("esm2_650m","esm2_150m"),("alphamissense","revel")],
           "V3_esm_all_genes":[("esm2_650m","esm2_150m"),("esm2_650m","esm1v"),
                               ("esm2_150m","esm1v")]}
    pr=[]
    for vn,(d,ms) in VIEWS.items():
        B,mlist=BOOT[vn]
        for A,Bm in PAIRS[vn]:
            if A not in mlist or Bm not in mlist: continue
            i,j=mlist.index(A),mlist.index(Bm)
            dd=B[:,i]-B[:,j]; dd=dd[~np.isnan(dd)]
            aA,aB,p=delong(d.label.values,d["s_"+A].values,d["s_"+Bm].values)
            lo,hi=np.percentile(dd,2.5),np.percentile(dd,97.5)
            pr.append(dict(view=vn,comparison=f"{NICE[A]} - {NICE[Bm]}",
                delta_auc=aA-aB,delong_p=p,delong_sig_bonf=p<0.05/45,
                gene_lo=lo,gene_hi=hi,gene_crosses_zero=bool(lo<0<hi)))
    pw=pd.DataFrame(pr); pw.to_csv(od/"step02_pairwise_delta.csv",index=False)
    log(pw.round(5).to_string(index=False))
    log("\n  gene_crosses_zero=True -> difference NOT supported once gene clustering")
    log("  is respected, regardless of the variant-level DeLong p-value.")

    # ---------------- [5] leave-one-gene-out ----------------
    log("\n[5] LEAVE-ONE-GENE-OUT (V2_shared10)")
    d,ms=VIEWS["V2_shared10"]; lr=[]
    for g in sorted(d.target_gene.unique()):
        sub=d[d.target_gene!=g]
        if sub.label.nunique()<2: continue
        au={x:auc(sub.label.values,sub["s_"+x].values) for x in ms}
        order=sorted(au,key=lambda k:-(au[k] if not np.isnan(au[k]) else -1))
        uns=[o for o in order if o in ("saprot","esm2_650m","esm2_150m","esm1v","eve")]
        lr.append(dict(excluded_gene=g,n_remaining=len(sub),top1=NICE[order[0]],
                       top2=NICE[order[1]],top3=NICE[order[2]],
                       best_unsupervised=NICE[uns[0]] if uns else None,
                       **{f"auc_{NICE[k]}":round(v,4) for k,v in au.items()}))
    lo=pd.DataFrame(lr); lo.to_csv(od/"step02_leave_one_gene_out.csv",index=False)
    log(lo[["excluded_gene","n_remaining","top1","top2","top3","best_unsupervised"]]
          .to_string(index=False))
    log(f"\n  top-1 counts: {lo.top1.value_counts().to_dict()}")
    log(f"  best-unsupervised counts: {lo.best_unsupervised.value_counts().to_dict()}")

    # ---------------- [6] per-gene paired Wilcoxon ----------------
    log("\n[6] PER-GENE PAIRED WILCOXON (unit of analysis = gene)")
    wr=[]
    for vn in VIEWS:
        base=pg[pg.view==vn].pivot_table(index="gene",columns="method",values="auc")
        for A,Bm in PAIRS[vn]:
            if A not in base.columns or Bm not in base.columns: continue
            dd=base[[A,Bm]].dropna()
            if len(dd)<5:
                wr.append(dict(view=vn,comparison=f"{NICE[A]} vs {NICE[Bm]}",
                               n_genes=len(dd),median_delta=np.nan,wins=np.nan,
                               losses=np.nan,wilcoxon_p=np.nan)); continue
            diff=dd[A]-dd[Bm]
            try: _,p=stats.wilcoxon(dd[A],dd[Bm])
            except Exception: p=np.nan
            wr.append(dict(view=vn,comparison=f"{NICE[A]} vs {NICE[Bm]}",n_genes=len(dd),
                median_delta=diff.median(),wins=int((diff>0).sum()),
                losses=int((diff<0).sum()),wilcoxon_p=p))
    wl=pd.DataFrame(wr); wl.to_csv(od/"step02_per_gene_wilcoxon.csv",index=False)
    log(wl.round(5).to_string(index=False))

    # ---------------- [7] oncogene vs tumour suppressor ----------------
    log("\n[7] ONCOGENE vs TUMOUR SUPPRESSOR")
    for vn,(d,ms) in VIEWS.items():
        log(f"\n  {vn}")
        for cls,cd in d.groupby("gene_type"):
            log(f"    {cls}: n={len(cd)} genes={cd.target_gene.nunique()} "
                f"P={int((cd.label==1).sum())} B={int((cd.label==0).sum())}")
        rr=[]
        for meth in ms:
            c="s_"+meth; row={"method":NICE[meth]}
            for cls,cd in d.groupby("gene_type"):
                k="ONC" if cls.lower().startswith("onco") else "TSG"
                row[f"auc_{k}"]=auc(cd.label.values,cd[c].values)
                row[f"n_{k}"]=int(cd[c].notna().sum())
            row["diff_TSG_minus_ONC"]=row.get("auc_TSG",np.nan)-row.get("auc_ONC",np.nan)
            rr.append(row)
        cc=pd.DataFrame(rr); cc.to_csv(od/f"step02_onco_vs_tsg_{vn}.csv",index=False)
        log(cc.round(4).to_string(index=False))
        gsub=pg[pg.view==vn]
        for meth in ms:
            s=gsub[gsub.method==meth]
            o=s[s.gene_type.str.lower().str.startswith("onco")]["auc"].dropna()
            t=s[s.gene_type.str.lower().str.startswith("tumor")]["auc"].dropna()
            if len(o)>=3 and len(t)>=3:
                u,p=stats.mannwhitneyu(t,o)
                log(f"    per-gene AUC {NICE[meth]:<14} TSG median={t.median():.3f} (n={len(t)})"
                    f"  ONC median={o.median():.3f} (n={len(o)})  MW p={p:.3f}")
            else:
                log(f"    per-gene AUC {NICE[meth]:<14} UNDERPOWERED "
                    f"(TSG genes={len(t)}, ONC genes={len(o)})")

    log("\nWROTE: step02_*.csv")
    log.close()

if __name__=="__main__": main()

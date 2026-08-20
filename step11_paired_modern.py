#!/usr/bin/env python3
"""STEP 11 - paired shared-subset comparison including the modern predictors.
Also checks dbNSFP-vs-API score provenance for AlphaMissense and REVEL."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

RNG=np.random.default_rng(211)

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

def bh(p):
    p=np.asarray(p,float); n=len(p); o=np.argsort(p); adj=np.empty(n); prev=1.0
    for r,i in enumerate(o[::-1]):
        prev=min(prev,p[i]*n/(n-r)); adj[i]=prev
    return adj

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP11_paired_modern_report.txt")
    m=pd.read_csv(od/"master_scores.csv")
    if "analysis_ok" in m.columns: m=m[m.analysis_ok==True].copy()
    log("="*78); log("STEP 11 - PAIRED COMPARISON WITH MODERN PREDICTORS"); log("="*78)

    # ---------- [0] provenance ----------
    log("\n[0] PROVENANCE CHECK: dbNSFP vs the sources already used")
    for own,new,nm in [("s_alphamissense","s_new_alphamissense","AlphaMissense"),
                       ("s_revel","s_new_revel","REVEL"),
                       ("s_eve","s_new_eve","EVE")]:
        if own not in m.columns or new not in m.columns: continue
        both=m.dropna(subset=[own,new])
        if len(both)<50: continue
        r,_=stats.spearmanr(both[own],both[new])
        log(f"    {nm:<15} overlap n={len(both):<5} Spearman rho={r:.4f}  "
            f"AUC own={auc(both.label.values,both[own].values):.3f}  "
            f"dbNSFP={auc(both.label.values,both[new].values):.3f}")
        log(f"      coverage: own {int(m[own].notna().sum())} vs "
            f"dbNSFP {int(m[new].notna().sum())}")
    log("    rho near 1.0 means the two sources agree and either can be used;")
    log("    lower values require stating which source the paper reports.")

    # ---------- assemble candidate set ----------
    OWN={"s_alphamissense":"AlphaMissense (API)","s_saprot":"SaProt",
         "s_esm2_650m":"ESM-2 650M","s_esm2_150m":"ESM-2 150M",
         "s_esm1v":"ESM-1v","s_revel":"REVEL","s_cadd":"CADD",
         "s_sift":"SIFT","s_polyphen2":"PolyPhen-2","s_eve":"EVE"}
    NEWM={c:c.replace("s_new_","") for c in m.columns if c.startswith("s_new_")}
    DROP={"alphamissense","revel","eve","sift","polyphen2_hdiv"}   # duplicates of OWN
    CONS={"phylop_100way_vertebrate","phylop_470way_mammalian","phylop_17way_primate",
          "phastcons_100way_vertebrate","phastcons_470way_mammalian",
          "gerp_91_mammals","siphy_29way_logodds"}
    NEWM={k:v for k,v in NEWM.items() if v not in DROP and v not in CONS}
    ALL={**OWN,**NEWM}
    log(f"\n[1] CANDIDATE METHODS: {len(ALL)} "
        f"({len(OWN)} existing + {len(NEWM)} new)")

    cols=list(ALL)
    sub=m[m[cols].notna().all(axis=1)]
    log(f"    fully-shared across ALL {len(cols)} methods: n={len(sub)}  "
        f"genes={sub.target_gene.nunique() if len(sub) else 0}")
    if len(sub)<300:
        log("    too small - dropping the lowest-coverage methods until n>=800")
        covs=sorted(((int(m[c].notna().sum()),c) for c in cols))
        while len(sub)<800 and len(cols)>12:
            worst=covs.pop(0)[1]; cols.remove(worst)
            sub=m[m[cols].notna().all(axis=1)]
            log(f"      dropped {ALL[worst]:<28} -> n={len(sub)}")
    log(f"\n    FINAL shared subset: n={len(sub)}  methods={len(cols)}  "
        f"genes={sub.target_gene.nunique()}  P={int(sub.label.sum())} "
        f"B={int((sub.label==0).sum())}")

    # ---------- [2] ranking with gene-cluster CIs ----------
    log("\n[2] RANKING ON THE SHARED SUBSET (gene-cluster bootstrap CIs)")
    genes=np.array(sorted(sub.target_gene.unique()))
    idx={g:sub.index[sub.target_gene==g].values for g in genes}
    B=np.full((1000,len(cols)),np.nan)
    for b in range(1000):
        pick=RNG.choice(genes,size=len(genes),replace=True)
        s=sub.loc[np.concatenate([idx[g] for g in pick])]
        if s.label.nunique()<2: continue
        y=s.label.values
        for j,c in enumerate(cols): B[b,j]=auc(y,s[c].values)
    rows=[]
    for j,c in enumerate(cols):
        rows.append(dict(method=ALL[c],col=c,auc=auc(sub.label.values,sub[c].values),
                         lo=np.nanpercentile(B[:,j],2.5),
                         hi=np.nanpercentile(B[:,j],97.5)))
    R=pd.DataFrame(rows).sort_values("auc",ascending=False).reset_index(drop=True)
    R["rank"]=R.index+1
    log(f"    {'rank':>5}  {'method':<28}{'ROC-AUC':>9}{'gene 95% CI':>22}")
    for _,r in R.iterrows():
        log(f"    {int(r['rank']):>5}  {r.method:<28}{r.auc:>9.4f}"
            f"     [{r.lo:.3f}, {r.hi:.3f}]")
    R.to_csv(od/"step11_ranking.csv",index=False)

    # ---------- [3] is AlphaMissense still first? ----------
    log("\n[3] IS AlphaMissense STILL THE LEADER?")
    am="s_alphamissense"
    if am in cols:
        top=R.iloc[0]
        log(f"    top method: {top.method} (AUC {top.auc:.4f})")
        amr=R[R.col==am].iloc[0]
        log(f"    AlphaMissense rank {int(amr['rank'])} of {len(R)} (AUC {amr.auc:.4f})")
        log(f"\n    {'comparison':<44}{'dAUC':>9}{'DeLong p':>11}{'BH q':>9}{'sig':>6}")
        pr=[]
        for _,r in R.iterrows():
            if r.col==am: continue
            aA,aB,p=delong(sub.label.values,sub[am].values,sub[r.col].values)
            pr.append(dict(vs=r.method,delta=aA-aB,p=p))
        P=pd.DataFrame(pr); P["q"]=bh(P.p.values); P["sig"]=P.q<0.05
        for _,r in P.sort_values("delta").iterrows():
            log(f"    AlphaMissense vs {r.vs:<28}{r.delta:>+9.4f}{r.p:>11.2e}"
                f"{r.q:>9.4f}{'yes' if r.sig else 'no':>6}")
        P.to_csv(od/"step11_vs_alphamissense.csv",index=False)
        beat=P[(P.delta<0)&(P.sig)]
        tie=P[~P.sig]
        log(f"\n    methods significantly BETTER than AlphaMissense: "
            f"{beat.vs.tolist() if len(beat) else 'NONE'}")
        log(f"    statistically indistinguishable: {tie.vs.tolist()}")
        log(f"    significantly worse: {int(((P.delta>0)&(P.sig)).sum())} of {len(P)}")
        log("\n    Your submitted manuscript claimed AlphaMissense significantly")
        log("    outperformed ALL other methods. Against this expanded field that")
        log("    claim must be revised to name the methods it ties with.")

    # ---------- [4] pLMs in context ----------
    log("\n[4] WHERE DO PROTEIN LANGUAGE MODELS SIT NOW?")
    plm=[c for c in cols if c in ("s_saprot","s_esm2_650m","s_esm2_150m",
                                  "s_esm1v","s_new_esm1b")]
    for c in plm:
        r=R[R.col==c]
        if len(r): log(f"    {ALL[c]:<28} rank {int(r.iloc[0]['rank'])}/{len(R)}  "
                       f"AUC {r.iloc[0].auc:.4f}")
    sup=[c for c in cols if c not in plm]
    log(f"\n    best pLM rank: {int(R[R.col.isin(plm)]['rank'].min())}")
    log(f"    supervised methods above it: "
        f"{int((R[R.col.isin(sup)]['rank']<R[R.col.isin(plm)]['rank'].min()).sum())}")
    log("    This is the honest current position of unsupervised pLMs against a")
    log("    contemporary supervised field.")

    log("\nWROTE: step11_ranking.csv, step11_vs_alphamissense.csv")
    log.close()

if __name__=="__main__": main()

#!/usr/bin/env python3
"""STEP 4c - verify the ONC/TSG contrast before building on it:
 (1) is ESM-1v's null a precision problem or real disagreement?
 (2) do the 4 RTKs behave as independent genes or as one homologous unit?
 (3) does the contrast hold gene-by-gene, or is it driven by one gene?
 (4) permutation test with gene as the exchangeable unit"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

NICE={"revel":"REVEL","cadd":"CADD","sift":"SIFT","polyphen2":"PolyPhen-2",
      "eve":"EVE","esm2_650m":"ESM-2 650M","esm2_150m":"ESM-2 150M","esm1v":"ESM-1v"}
CLASSICAL=["revel","cadd","sift","polyphen2","eve"]; PLMS=["esm2_650m","esm2_150m","esm1v"]
RNG=np.random.default_rng(11)

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
    log=Tee(od/"STEP04C_verify_report.txt")
    m=pd.read_csv(od/"master_scores.csv"); m=m[m.analysis_ok==True].copy()
    m["is_onc"]=m.gene_type.str.lower().str.startswith("onco")
    d5=m[m[[f"s_{x}" for x in CLASSICAL]].notna().all(axis=1)].copy()
    log("="*78); log("STEP 4c - VERIFYING THE ONC/TSG CONTRAST"); log("="*78)

    # ---------- [1] ESM-1v precision ----------
    log("\n[1] IS ESM-1v's NULL A PRECISION PROBLEM?")
    log("    Bootstrap each method's ONC AUC; compare interval widths.")
    onc=d5[d5.is_onc]
    log(f"    {'method':<14}{'ONC_auc':>9}{'CI_lo':>9}{'CI_hi':>9}{'width':>8}")
    widths={}
    for meth in CLASSICAL+PLMS:
        c="s_"+meth
        if c not in onc.columns or onc[c].notna().sum()==0: continue
        vals=[]
        for _ in range(1000):
            s=onc.sample(len(onc),replace=True,random_state=RNG.integers(1e9))
            vals.append(auc(s.label.values,s[c].values))
        vals=np.array(vals,dtype=float)
        lo,hi=np.nanpercentile(vals,2.5),np.nanpercentile(vals,97.5)
        widths[meth]=hi-lo
        log(f"    {NICE[meth]:<14}{auc(onc.label.values,onc[c].values):>9.4f}"
            f"{lo:>9.4f}{hi:>9.4f}{hi-lo:>8.4f}")
    log(f"\n    ESM-1v width={widths.get('esm1v',np.nan):.4f} vs "
        f"median other={np.median([v for k,v in widths.items() if k!='esm1v']):.4f}")
    log("    If ESM-1v's interval is much wider, its null reflects imprecision.")
    log("    If it is comparable, ESM-1v genuinely disagrees with the other pLMs.")

    # ---------- [2] are the 4 RTKs one unit? ----------
    log("\n[2] ARE THE 4 ONCOGENES INDEPENDENT, OR ONE HOMOLOGOUS UNIT?")
    log("    Per-gene AUC for each eligible oncogene:")
    log(f"    {'gene':<8}{'nP':>5}{'nB':>5}" + "".join(f"{NICE[x]:>13}" for x in CLASSICAL+PLMS))
    per=[]
    for g,gd in onc.groupby("target_gene"):
        nP=int((gd.label==1).sum()); nB=int((gd.label==0).sum())
        if nP<8 or nB<8: continue
        row={"gene":g,"n_p":nP,"n_b":nB}
        line=f"    {g:<8}{nP:>5}{nB:>5}"
        for x in CLASSICAL+PLMS:
            v=auc(gd.label.values,gd["s_"+x].values); row[x]=v
            line+=f"{v:>13.3f}" if not np.isnan(v) else f"{'-':>13}"
        per.append(row); log(line)
    pdf=pd.DataFrame(per)
    log("\n    Between-gene SD of AUC within the oncogene set:")
    for x in CLASSICAL+PLMS:
        if x in pdf.columns:
            log(f"      {NICE[x]:<14} SD={pdf[x].std():.4f}  range="
                f"[{pdf[x].min():.3f}, {pdf[x].max():.3f}]")
    log("    Low SD across the 4 RTKs = they behave as one homologous unit,")
    log("    so effective sample size is closer to 1 gene than 4.")

    # ---------- [3] leave-one-oncogene-out ----------
    log("\n[3] LEAVE-ONE-ONCOGENE-OUT: is one gene driving the contrast?")
    tsg=d5[~d5.is_onc]
    log(f"    {'excluded':<10}{'n_onc':>7}" +
        "".join(f"{NICE[x]:>13}" for x in ["revel","eve","esm2_650m"]))
    lo_rows=[]
    for g in sorted(onc.target_gene.unique()):
        sub=onc[onc.target_gene!=g]
        if len(sub)<50 or sub.label.nunique()<2: continue
        row={"excluded":g,"n_onc":len(sub)}
        line=f"    {g:<10}{len(sub):>7}"
        for x in ["revel","eve","esm2_650m"]:
            d=auc(sub.label.values,sub["s_"+x].values)-auc(tsg.label.values,tsg["s_"+x].values)
            row[f"diff_{x}"]=d; line+=f"{d:>+13.4f}"
        lo_rows.append(row); log(line)
    lo=pd.DataFrame(lo_rows); lo.to_csv(od/"step04c_leave_one_onc_out.csv",index=False)
    log("    All rows same sign = contrast is not driven by a single gene.")

    # ---------- [4] gene-level permutation ----------
    log("\n[4] PERMUTATION TEST WITH GENE AS THE EXCHANGEABLE UNIT")
    log("    Shuffle the ONC/TSG label across GENES (not variants), 5000 times.")
    genes=d5.groupby("target_gene").agg(is_onc=("is_onc","first"),n=("label","size"))
    genes=genes[genes.n>=20]
    n_onc=int(genes.is_onc.sum()); glist=genes.index.values
    idx={g:d5.index[d5.target_gene==g].values for g in glist}
    log(f"    {len(glist)} genes with n>=20 ({n_onc} oncogene)")
    log(f"    {'method':<14}{'observed':>10}{'perm_mean':>11}{'perm_p':>9}")
    pr=[]
    for meth in CLASSICAL+PLMS:
        c="s_"+meth
        if c not in d5.columns: continue
        real_o=d5[d5.is_onc]; real_t=d5[~d5.is_onc]
        obs=auc(real_o.label.values,real_o[c].values)-auc(real_t.label.values,real_t[c].values)
        null=[]
        for _ in range(5000):
            pick=RNG.choice(glist,size=n_onc,replace=False)
            A=d5.loc[np.concatenate([idx[g] for g in pick])]
            B=d5.loc[np.concatenate([idx[g] for g in glist if g not in set(pick)])]
            if A.label.nunique()<2 or B.label.nunique()<2: continue
            null.append(auc(A.label.values,A[c].values)-auc(B.label.values,B[c].values))
        null=np.array(null,dtype=float); null=null[~np.isnan(null)]
        p=(np.sum(np.abs(null)>=abs(obs))+1)/(len(null)+1)
        pr.append(dict(method=NICE[meth],
                       family="classical" if meth in CLASSICAL else "pLM",
                       observed_diff=obs,perm_mean=np.mean(null),perm_p=p))
        log(f"    {NICE[meth]:<14}{obs:>+10.4f}{np.mean(null):>+11.4f}{p:>9.4f}")
    pp=pd.DataFrame(pr); pp.to_csv(od/"step04c_gene_permutation.csv",index=False)
    log("\n    This is the strictest test: it asks whether ANY split of genes into")
    log("    groups of this size would show a gap this large. p<0.05 here means the")
    log("    oncogene split is special, not just any gene grouping.")

    log("\n[5] WHAT YOU MAY CLAIM")
    sig=pp[pp.perm_p<0.05]
    log(f"    methods passing the gene-level permutation: "
        f"{sig.method.tolist() if len(sig) else 'NONE'}")
    if len(sig)==0:
        log("    => The contrast does NOT survive gene-level permutation.")
        log("       Report it as an observation with the confound stated, not a finding.")
    log.close()

if __name__=="__main__": main()

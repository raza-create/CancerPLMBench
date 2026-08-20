#!/usr/bin/env python3
"""STEP 4 - is the oncogene advantage real, or is it kinases / hotspots /
class balance / cross-gene calibration?"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

NICE={"alphamissense":"AlphaMissense","saprot":"SaProt","revel":"REVEL",
      "esm2_650m":"ESM-2 650M","cadd":"CADD","sift":"SIFT","polyphen2":"PolyPhen-2",
      "esm2_150m":"ESM-2 150M","esm1v":"ESM-1v","eve":"EVE"}
CLASSICAL=["revel","cadd","sift","polyphen2","eve"]
PLM=["esm2_650m","esm2_150m","esm1v","saprot"]
KINASE={"RET","KIT","MET","EGFR","ERBB2","ALK","BRAF","PDGFRA","AKT1","PIK3CA","ATM","STK11"}
RNG=np.random.default_rng(7)

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        s=" ".join(str(x) for x in a); print(s,flush=True); self.f.write(s+"\n")
    def close(self): self.f.close()

def auc(y,s):
    y=np.asarray(y,float); s=np.asarray(s,float); ok=~np.isnan(s)
    y=y[ok]; s=s[ok]
    return roc_auc_score(y,s) if len(np.unique(y))>1 else np.nan

def within_auc(d,col,groupcol="target_gene"):
    """pair-weighted within-gene AUC: removes all cross-gene calibration"""
    dd=d.dropna(subset=[col]); num=den=0.0
    for g,gd in dd.groupby(groupcol):
        p=int((gd.label==1).sum()); b=int((gd.label==0).sum())
        if p==0 or b==0: continue
        a=auc(gd.label.values,gd[col].values)
        if np.isnan(a): continue
        num+=a*p*b; den+=p*b
    return num/den if den else np.nan

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP04_onco_confounders_report.txt")
    m=pd.read_csv(od/"master_scores.csv"); m=m[m.analysis_ok==True].copy()
    m["is_onc"]=m.gene_type.str.lower().str.startswith("onco")
    m["is_kinase"]=m.target_gene.isin(KINASE)
    log("="*78); log("STEP 4 - ONCOGENE ADVANTAGE: CONFOUNDER TESTS"); log("="*78)

    V5=[f"s_{x}" for x in CLASSICAL]
    d5=m[m[V5].notna().all(axis=1)].copy()
    d3=m[m[[f"s_{x}" for x in ["esm2_650m","esm2_150m","esm1v"]]].notna().all(axis=1)].copy()

    # ---------- [1] what genes are actually in each class ----------
    log("\n[1] COMPOSITION OF EACH CLASS (V5 = 5 classical methods)")
    for cls,cd in d5.groupby("is_onc"):
        nm="ONCOGENE" if cls else "TUMOUR SUPPRESSOR"
        el=[]
        for g,gd in cd.groupby("target_gene"):
            if (gd.label==1).sum()>=10 and (gd.label==0).sum()>=10: el.append(g)
        log(f"\n  {nm}: n={len(cd)} genes={cd.target_gene.nunique()} "
            f"%P={100*cd.label.mean():.1f}")
        log(f"    eligible (>=10P,>=10B): {el}")
        log(f"    of those, kinases: {[g for g in el if g in KINASE]}")
    log("\n  If every eligible oncogene is a kinase, 'oncogene' and 'kinase'")
    log("  cannot be separated in this dataset. That is a hard limit.")

    # ---------- [2] within-gene only ----------
    log("\n[2] WITHIN-GENE AUC BY CLASS (cross-gene calibration removed)")
    log(f"    {'method':<14}{'ONC_within':>12}{'TSG_within':>12}{'TSG-ONC':>10}"
        f"{'ONC_pooled':>12}{'TSG_pooled':>12}")
    rows=[]
    for meth in CLASSICAL+["esm2_650m","esm2_150m","esm1v"]:
        c="s_"+meth
        if c not in d5.columns or d5[c].notna().sum()==0: continue
        ow=within_auc(d5[d5.is_onc],c); tw=within_auc(d5[~d5.is_onc],c)
        op=auc(d5[d5.is_onc].label.values,d5[d5.is_onc][c].values)
        tp=auc(d5[~d5.is_onc].label.values,d5[~d5.is_onc][c].values)
        rows.append(dict(method=NICE[meth],
            family="classical" if meth in CLASSICAL else "pLM",
            onc_within=ow,tsg_within=tw,diff_within=tw-ow,
            onc_pooled=op,tsg_pooled=tp,diff_pooled=tp-op))
        log(f"    {NICE[meth]:<14}{ow:>12.4f}{tw:>12.4f}{tw-ow:>+10.4f}"
            f"{op:>12.4f}{tp:>12.4f}")
    wi=pd.DataFrame(rows); wi.to_csv(od/"step04_within_gene_by_class.csv",index=False)
    log("\n  If diff_within is much smaller than diff_pooled, the oncogene 'advantage'")
    log("  was cross-gene calibration, not better within-gene discrimination.")

    # ---------- [3] kinase vs non-kinase ----------
    log("\n[3] KINASE CONFOUNDER: 2x2 (oncogene x kinase)")
    for meth in CLASSICAL+["esm2_650m"]:
        c="s_"+meth
        if c not in d5.columns: continue
        log(f"\n    {NICE[meth]}")
        for onc in [True,False]:
            for kin in [True,False]:
                sub=d5[(d5.is_onc==onc)&(d5.is_kinase==kin)]
                if len(sub)<30 or sub.label.nunique()<2:
                    log(f"      onc={onc!s:<5} kinase={kin!s:<5} n={len(sub):<5} "
                        f"(too small)"); continue
                log(f"      onc={onc!s:<5} kinase={kin!s:<5} n={len(sub):<5} "
                    f"genes={sub.target_gene.nunique():<3} "
                    f"AUC={auc(sub.label.values,sub[c].values):.4f}")
        nk=d5[~d5.is_kinase]
        if nk.is_onc.nunique()>1:
            o=nk[nk.is_onc]; t=nk[~nk.is_onc]
            if len(o)>=30 and o.label.nunique()>1:
                log(f"      NON-KINASE ONLY: ONC AUC={auc(o.label.values,o[c].values):.4f} "
                    f"(n={len(o)}, genes={o.target_gene.nunique()})  "
                    f"TSG AUC={auc(t.label.values,t[c].values):.4f}")
            else:
                log(f"      NON-KINASE ONLY: too few non-kinase oncogene variants "
                    f"(n={len(o)}) -> kinase and oncogene are CONFOUNDED")

    # ---------- [4] class-balance matched ----------
    log("\n[4] CLASS-BALANCE MATCHING (resample TSG to the ONC P:B ratio)")
    onc=d5[d5.is_onc]; tsg=d5[~d5.is_onc]
    tgt=onc.label.mean()
    log(f"    ONC %P={100*tgt:.1f}   TSG %P={100*tsg.label.mean():.1f}")
    n_iter=500; res={}
    tp_=tsg[tsg.label==1]; tb_=tsg[tsg.label==0]
    n_tot=min(len(tsg),1500); n_p=int(round(n_tot*tgt)); n_b=n_tot-n_p
    n_p=min(n_p,len(tp_)); n_b=min(n_b,len(tb_))
    for meth in CLASSICAL+["esm2_650m"]:
        c="s_"+meth
        if c not in d5.columns: continue
        vals=[]
        for _ in range(n_iter):
            s=pd.concat([tp_.sample(n_p,replace=True,random_state=RNG.integers(1e9)),
                         tb_.sample(n_b,replace=True,random_state=RNG.integers(1e9))])
            vals.append(auc(s.label.values,s[c].values))
        vals=np.array(vals)
        oa=auc(onc.label.values,onc[c].values)
        res[meth]=(oa,np.nanmean(vals),np.nanpercentile(vals,2.5),np.nanpercentile(vals,97.5))
        log(f"    {NICE[meth]:<14} ONC={oa:.4f}  TSG(balance-matched)="
            f"{np.nanmean(vals):.4f} [{np.nanpercentile(vals,2.5):.4f},"
            f"{np.nanpercentile(vals,97.5):.4f}]  "
            f"{'GAP REMAINS' if oa>np.nanpercentile(vals,97.5) else 'gap not significant'}")
    pd.DataFrame([dict(method=NICE[k],onc_auc=v[0],tsg_matched_mean=v[1],
                       tsg_lo=v[2],tsg_hi=v[3]) for k,v in res.items()]
                 ).to_csv(od/"step04_balance_matched.csv",index=False)

    # ---------- [5] hotspot recurrence ----------
    log("\n[5] HOTSPOT STRUCTURE: are oncogene pathogenic variants clustered?")
    log(f"    {'class':<20}{'n_path':>8}{'n_pos':>8}{'var/pos':>9}"
        f"{'top1pos%':>10}{'top5pos%':>10}")
    hr=[]
    for onc in [True,False]:
        sub=m[(m.is_onc==onc)&(m.label==1)]
        vc=sub.groupby(["target_gene","position"]).size()
        top=vc.sort_values(ascending=False)
        nm="ONCOGENE" if onc else "TUMOUR SUPPRESSOR"
        log(f"    {nm:<20}{len(sub):>8}{len(vc):>8}{len(sub)/len(vc):>9.2f}"
            f"{100*top.iloc[0]/len(sub):>9.1f}%{100*top.head(5).sum()/len(sub):>9.1f}%")
        hr.append(dict(cls=nm,n_path=len(sub),n_positions=len(vc),
                       var_per_pos=len(sub)/len(vc),
                       top1_pct=100*top.iloc[0]/len(sub),
                       top5_pct=100*top.head(5).sum()/len(sub)))
    pd.DataFrame(hr).to_csv(od/"step04_hotspots.csv",index=False)
    log("    Higher var/pos in oncogenes = recurrent hotspots. Predictors may")
    log("    score hotspots well simply because they are extreme substitutions.")

    log("\n    Same comparison after collapsing each position to ONE variant:")
    log(f"    {'method':<14}{'ONC':>9}{'TSG':>9}{'TSG-ONC':>10}")
    col=m.drop_duplicates(subset=["target_gene","position","label"])
    c5=col[col[V5].notna().all(axis=1)]
    for meth in CLASSICAL+["esm2_650m"]:
        c="s_"+meth
        if c not in c5.columns: continue
        o=c5[c5.is_onc]; t=c5[~c5.is_onc]
        if o.label.nunique()<2 or t.label.nunique()<2: continue
        ao=auc(o.label.values,o[c].values); at=auc(t.label.values,t[c].values)
        log(f"    {NICE[meth]:<14}{ao:>9.4f}{at:>9.4f}{at-ao:>+10.4f}")

    # ---------- [6] verdict ----------
    log("\n[6] VERDICT CHECKLIST")
    log("    The oncogene/TSG contrast is reportable as a FINDING only if:")
    log("      (a) it survives within-gene analysis          [section 2]")
    log("      (b) non-kinase oncogenes exist to test it     [section 3]")
    log("      (c) it survives class-balance matching        [section 4]")
    log("      (d) it survives collapsing hotspot positions  [section 5]")
    log("    If (b) fails, report it as 'receptor tyrosine kinases vs other genes'")
    log("    and state the confound explicitly. Do not claim oncogene biology.")
    log("\nWROTE: step04_*.csv")
    log.close()

if __name__=="__main__": main()

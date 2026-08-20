#!/usr/bin/env python3
"""STEP 7 - calibration and clinically interpretable metrics (critique item 8).
MCC, balanced accuracy, Brier, ECE, sensitivity at fixed specificity, published
threshold performance, and calibration by structural environment."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             matthews_corrcoef, balanced_accuracy_score, brier_score_loss)

NICE={"alphamissense":"AlphaMissense","saprot":"SaProt","revel":"REVEL",
      "esm2_650m":"ESM-2 650M","cadd":"CADD","sift":"SIFT","polyphen2":"PolyPhen-2",
      "esm2_150m":"ESM-2 150M","esm1v":"ESM-1v","eve":"EVE"}
METH=list(NICE)
SUP={"alphamissense","revel","cadd","polyphen2"}
# published//commonly used decision thresholds, on the native score scale
THRESH={"alphamissense":(0.34,0.564,"AM: <0.34 benign, >0.564 pathogenic"),
        "revel":(0.183,0.773,"REVEL: <0.183 benign, >0.773 pathogenic (Pejaver 2022)"),
        "cadd":(None,20.0,"CADD PHRED > 20 commonly used"),
        "eve":(0.25,0.75,"EVE rank score, nominal")}
RNG=np.random.default_rng(151)

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        s=" ".join(str(x) for x in a); print(s,flush=True); self.f.write(s+"\n")
    def close(self): self.f.close()

def auc(y,s):
    y=np.asarray(y,float); s=np.asarray(s,float); ok=~np.isnan(s)
    y=y[ok]; s=s[ok]
    return roc_auc_score(y,s) if len(np.unique(y))>1 else np.nan

def sens_at_spec(y,s,target):
    fpr,tpr,thr=roc_curve(y,s)
    ok=(1-fpr)>=target
    return (tpr[ok].max(), thr[ok][np.argmax(tpr[ok])]) if ok.any() else (np.nan,np.nan)

def spec_at_sens(y,s,target):
    fpr,tpr,thr=roc_curve(y,s)
    ok=tpr>=target
    return ((1-fpr)[ok].max()) if ok.any() else np.nan

def to_prob(s):
    """min-max to [0,1] for Brier/ECE on scores that are not probabilities"""
    s=np.asarray(s,float)
    lo,hi=np.nanmin(s),np.nanmax(s)
    return (s-lo)/(hi-lo) if hi>lo else np.full_like(s,0.5)

def ece(y,p,bins=10):
    y=np.asarray(y,float); p=np.asarray(p,float)
    edges=np.linspace(0,1,bins+1); e=0.0
    for i in range(bins):
        m=(p>=edges[i])&(p<edges[i+1] if i<bins-1 else p<=edges[i+1])
        if m.sum()==0: continue
        e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return e

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP07_calibration_report.txt")
    m=pd.read_csv(od/"master_scores.csv"); m=m[m.analysis_ok==True].copy()
    shared=m[m[[f"s_{x}" for x in METH]].notna().all(axis=1)].copy()
    log("="*78); log("STEP 7 - CALIBRATION AND CLINICAL METRICS"); log("="*78)
    log(f"\nfully-shared subset: n={len(shared)}  genes={shared.target_gene.nunique()}  "
        f"P={int(shared.label.sum())} B={int((shared.label==0).sum())}")
    log("(per-method native coverage also reported, since the shared subset is")
    log(" length-biased - see Step 1b.)")

    # ---------- [1] discrimination + clinical operating points ----------
    log("\n[1] DISCRIMINATION AND OPERATING POINTS (shared subset)")
    log(f"    {'method':<15}{'ROC':>7}{'PR':>7}{'MCC':>7}{'BalAcc':>8}"
        f"{'Se@95Sp':>9}{'Se@90Sp':>9}{'Sp@90Se':>9}{'cover%':>8}")
    rows=[]
    for x in METH:
        c="s_"+x; y=shared.label.values; s=shared[c].values
        ro=auc(y,s); pr=average_precision_score(y,s)
        # MCC/BalAcc at the Youden-optimal threshold
        fpr,tpr,thr=roc_curve(y,s); j=np.argmax(tpr-fpr); t=thr[j]
        pred=(s>=t).astype(int)
        mcc=matthews_corrcoef(y,pred); ba=balanced_accuracy_score(y,pred)
        s95,_=sens_at_spec(y,s,0.95); s90,_=sens_at_spec(y,s,0.90)
        sp90=spec_at_sens(y,s,0.90)
        cov=100*m["s_"+x].notna().mean()
        log(f"    {NICE[x]:<15}{ro:>7.3f}{pr:>7.3f}{mcc:>7.3f}{ba:>8.3f}"
            f"{s95:>9.3f}{s90:>9.3f}{sp90:>9.3f}{cov:>8.1f}")
        rows.append(dict(method=NICE[x],roc_auc=ro,pr_auc=pr,mcc=mcc,bal_acc=ba,
                         sens_at_95spec=s95,sens_at_90spec=s90,spec_at_90sens=sp90,
                         youden_threshold=float(t),coverage_pct=cov))
    C=pd.DataFrame(rows); C.to_csv(od/"step07_clinical_metrics.csv",index=False)
    log("\n    Se@95Sp is the operationally relevant number: how many pathogenic")
    log("    variants are recovered while keeping false positives at 5%.")
    log(f"    best Se@95Sp: {C.loc[C.sens_at_95spec.idxmax(),'method']} "
        f"({C.sens_at_95spec.max():.3f});  worst: "
        f"{C.loc[C.sens_at_95spec.idxmin(),'method']} ({C.sens_at_95spec.min():.3f})")

    # ---------- [2] calibration ----------
    log("\n[2] CALIBRATION (Brier score and expected calibration error)")
    log("    Scores rescaled to [0,1]; only AlphaMissense and EVE are probabilistic")
    log("    by construction, so others are reported for comparison only.")
    log(f"    {'method':<15}{'Brier':>9}{'ECE':>9}{'native?':>10}")
    cal=[]
    for x in METH:
        c="s_"+x; y=shared.label.values
        p=to_prob(shared[c].values)
        b=brier_score_loss(y,p); e=ece(y,p)
        nat="yes" if x in ("alphamissense","eve","revel") else "rescaled"
        log(f"    {NICE[x]:<15}{b:>9.4f}{e:>9.4f}{nat:>10}")
        cal.append(dict(method=NICE[x],brier=b,ece=e,native=nat))
    CA=pd.DataFrame(cal); CA.to_csv(od/"step07_calibration.csv",index=False)

    log("\n    reliability curve (10 bins) for the three leading methods:")
    for x in ["alphamissense","saprot","revel"]:
        c="s_"+x; p=to_prob(shared[c].values); y=shared.label.values
        log(f"      {NICE[x]}:")
        edges=np.linspace(0,1,11)
        for i in range(10):
            msk=(p>=edges[i])&(p<edges[i+1] if i<9 else p<=edges[i+1])
            if msk.sum()<10: continue
            log(f"        bin {edges[i]:.1f}-{edges[i+1]:.1f}  n={int(msk.sum()):<5} "
                f"predicted={p[msk].mean():.3f}  observed={y[msk].mean():.3f}  "
                f"gap={y[msk].mean()-p[msk].mean():+.3f}")

    # ---------- [3] published thresholds ----------
    log("\n[3] PERFORMANCE AT PUBLISHED THRESHOLDS")
    log("    Uses each method's own native score scale, not the rescaled one.")
    RAW={"alphamissense":"raw_alphamissense","revel":"raw_revel",
         "cadd":"raw_cadd","eve":"raw_eve"}
    tr=[]
    for x,(blo,phi,desc) in THRESH.items():
        col=RAW.get(x)
        if col not in m.columns or m[col].notna().sum()==0: continue
        sub=m.dropna(subset=[col]); s=sub[col].values; y=sub.label.values
        log(f"\n    {desc}")
        pathog=s>=phi
        tp=int(((pathog)&(y==1)).sum()); fp=int(((pathog)&(y==0)).sum())
        sens=tp/max(int((y==1).sum()),1); ppv=tp/max(tp+fp,1)
        spec=1-fp/max(int((y==0).sum()),1)
        log(f"      pathogenic call: n={int(pathog.sum())}  sens={sens:.3f}  "
            f"spec={spec:.3f}  PPV={ppv:.3f}")
        if blo is not None:
            ben=s<=blo
            tn=int(((ben)&(y==0)).sum()); fn=int(((ben)&(y==1)).sum())
            npv=tn/max(tn+fn,1)
            grey=int((~pathog&~ben).sum())
            log(f"      benign call:     n={int(ben.sum())}  NPV={npv:.3f}")
            log(f"      indeterminate:   n={grey} ({100*grey/len(sub):.1f}% of variants)")
            tr.append(dict(method=NICE[x],n=len(sub),sens=sens,spec=spec,ppv=ppv,
                           npv=npv,pct_indeterminate=100*grey/len(sub)))
        else:
            tr.append(dict(method=NICE[x],n=len(sub),sens=sens,spec=spec,ppv=ppv,
                           npv=np.nan,pct_indeterminate=np.nan))
    if tr: pd.DataFrame(tr).to_csv(od/"step07_threshold_performance.csv",index=False)
    log("\n    NOTE: PPV here reflects this benchmark's 52% pathogenic prevalence,")
    log("    which is far above any real clinical panel. PPV is not transferable.")

    # ---------- [4] calibration by structural stratum ----------
    log("\n[4] CALIBRATION AND SENSITIVITY BY STRUCTURAL ENVIRONMENT")
    log("    Extends the Step 6g finding into clinical terms.")
    sp=od/"step06e_structural_variants.csv"
    if not sp.exists():
        log("    step06e_structural_variants.csv not found - skipped")
    else:
        d=pd.read_csv(sp)
        rows=[]
        for k in ["ordered","flexible","disordered"]:
            sub=d[d.order==k]
            if len(sub)<50 or sub.label.nunique()<2: continue
            log(f"\n    {k} (n={len(sub)}, {100*sub.label.mean():.1f}% pathogenic)")
            log(f"      {'method':<15}{'ROC':>7}{'Se@95Sp':>9}{'Brier':>9}{'ECE':>9}")
            for x in METH:
                c="s_"+x
                if c not in sub.columns or sub[c].notna().sum()<30: continue
                ss=sub.dropna(subset=[c]); y=ss.label.values; v=ss[c].values
                if len(np.unique(y))<2: continue
                ro=auc(y,v); s95,_=sens_at_spec(y,v,0.95)
                p=to_prob(v); b=brier_score_loss(y,p); e=ece(y,p)
                log(f"      {NICE[x]:<15}{ro:>7.3f}{s95:>9.3f}{b:>9.4f}{e:>9.4f}")
                rows.append(dict(stratum=k,method=NICE[x],
                                 family="supervised" if x in SUP else "unsupervised",
                                 roc=ro,sens_at_95spec=s95,brier=b,ece=e,n=len(ss)))
        S=pd.DataFrame(rows)
        if len(S):
            S.to_csv(od/"step07_by_stratum.csv",index=False)
            log("\n    Se@95Sp change, ordered -> disordered:")
            piv=S.pivot_table(index="method",columns="stratum",values="sens_at_95spec")
            if {"ordered","disordered"} <= set(piv.columns):
                piv["drop"]=piv["ordered"]-piv["disordered"]
                fam={NICE[x]:("supervised" if x in SUP else "unsupervised") for x in METH}
                piv["family"]=[fam.get(i,"") for i in piv.index]
                log(piv[["ordered","disordered","drop","family"]]
                      .sort_values("drop",ascending=False).round(3).to_string())
                u=piv[piv.family=="unsupervised"]["drop"].dropna()
                s_=piv[piv.family=="supervised"]["drop"].dropna()
                if len(u)>=3 and len(s_)>=3:
                    _,pv=stats.mannwhitneyu(u,s_)
                    log(f"\n      unsupervised median drop={u.median():+.3f}  "
                        f"supervised={s_.median():+.3f}  Mann-Whitney p={pv:.4f}")
                    log("      If unsupervised methods lose more SENSITIVITY at fixed")
                    log("      specificity, the Step 6g effect has direct clinical")
                    log("      consequence: more missed pathogenic variants in")
                    log("      disordered regions.")
            log("\n    ECE by stratum and family:")
            log(S.groupby(["stratum","family"])[["ece","brier"]].mean().round(4).to_string())

    log("\n[5] SUMMARY FOR THE MANUSCRIPT")
    log("-"*78)
    log("  - Report ROC-AUC, PR-AUC, MCC, balanced accuracy, Brier, ECE and")
    log("    sensitivity at 95% specificity for all ten methods (Table).")
    log("  - State that PPV depends on this benchmark's 52% prevalence and does")
    log("    not transfer to clinical panels.")
    log("  - Report the indeterminate-zone fraction for methods with published")
    log("    two-sided thresholds; it is a practical limitation rarely quantified.")
    log("  - Link section 4 to the disorder finding: the accuracy gap translates")
    log("    into lost sensitivity at clinically usable specificity.")
    log("-"*78)
    log("\nWROTE: step07_clinical_metrics.csv, step07_calibration.csv,")
    log("       step07_threshold_performance.csv, step07_by_stratum.csv")
    log.close()

if __name__=="__main__": main()

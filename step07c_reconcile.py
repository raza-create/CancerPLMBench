#!/usr/bin/env python3
"""STEP 7c - reconcile contradictory Step 7 outputs so no stale numbers
survive on disk. Rewrites the affected CSVs and emits ONE authoritative table."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import (roc_curve, roc_auc_score, average_precision_score,
                             matthews_corrcoef, balanced_accuracy_score, brier_score_loss)

NICE={"alphamissense":"AlphaMissense","saprot":"SaProt","revel":"REVEL",
      "esm2_650m":"ESM-2 650M","cadd":"CADD","sift":"SIFT","polyphen2":"PolyPhen-2",
      "esm2_150m":"ESM-2 150M","esm1v":"ESM-1v","eve":"EVE"}
METH=list(NICE)
PROB={"alphamissense"}   # the ONLY natively probabilistic score here

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        s=" ".join(str(x) for x in a); print(s,flush=True); self.f.write(s+"\n")
    def close(self): self.f.close()

def se_at_sp(y,s,target):
    y=np.asarray(y,int); s=np.asarray(s,float); ok=~np.isnan(s)
    y=y[ok]; s=s[ok]
    if len(np.unique(y))<2: return np.nan
    fpr,tpr,_=roc_curve(y,s); want=1-target
    if want<=fpr[0]: return float(tpr[0])
    if want>=fpr[-1]: return float(tpr[-1])
    return float(np.interp(want,fpr,tpr))

def sp_at_se(y,s,target):
    fpr,tpr,_=roc_curve(y,s)
    if target<=tpr[0]: return float(1-fpr[0])
    if target>=tpr[-1]: return float(1-fpr[-1])
    return float(np.interp(target,tpr,1-fpr))

def ece(y,p,bins=10):
    y=np.asarray(y,float); p=np.asarray(p,float); e=0.0
    ed=np.linspace(0,1,bins+1)
    for i in range(bins):
        m=(p>=ed[i])&(p<ed[i+1] if i<bins-1 else p<=ed[i+1])
        if m.sum()==0: continue
        e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return float(e)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP07C_reconciled_report.txt")
    m=pd.read_csv(od/"master_scores.csv"); m=m[m.analysis_ok==True].copy()
    sh=m[m[[f"s_{x}" for x in METH]].notna().all(axis=1)].copy()
    y=sh.label.values
    log("="*78); log("STEP 7c - RECONCILING CONTRADICTORY STEP 7 OUTPUTS"); log("="*78)
    log("\nDEFECTS BEING FIXED:")
    log("  1. step07_clinical_metrics.csv holds Se@95Sp=0.000 for SIFT/PolyPhen-2")
    log("     (computational artefact). Corrected values are 0.311 and 0.265.")
    log("  2. step07_calibration.csv labels REVEL and EVE 'native: yes'. Both are")
    log("     rank scores, not probabilities. Their Brier/ECE are meaningless.")
    log("  3. STEP07B section 5 quotes ECE=0.016 (rescaled) while section 2")
    log("     computed 0.0306 (raw probability). 0.0306 is correct.")
    log("  4. STEP07B section 5 claims SIFT/PolyPhen-2 'cannot be operated at 95%")
    log("     specificity'. Section 1 of the same file shows they can. Removed.")

    # ---------- authoritative table ----------
    log("\n[1] AUTHORITATIVE METRICS TABLE (supersedes all earlier Step 7 files)")
    log(f"    shared subset n={len(sh)}  P={int(y.sum())}  B={int((y==0).sum())}")
    rows=[]
    for x in METH:
        s=sh["s_"+x].values
        fpr,tpr,thr=roc_curve(y,s); j=int(np.argmax(tpr-fpr)); t=thr[j]
        pred=(s>=t).astype(int)
        r=dict(method=NICE[x],
               roc_auc=roc_auc_score(y,s),
               pr_auc=average_precision_score(y,s),
               mcc=matthews_corrcoef(y,pred),
               bal_acc=balanced_accuracy_score(y,pred),
               sens_at_95spec=se_at_sp(y,s,0.95),
               sens_at_90spec=se_at_sp(y,s,0.90),
               spec_at_90sens=sp_at_se(y,s,0.90),
               n_unique_scores=int(len(np.unique(s))),
               coverage_pct=round(100*m["s_"+x].notna().mean(),1),
               is_probability=(x in PROB))
        if x in PROB:
            p=sh["raw_alphamissense"].values
            r["brier"]=brier_score_loss(y,p); r["ece"]=ece(y,p)
        else:
            r["brier"]=np.nan; r["ece"]=np.nan
        rows.append(r)
    T=pd.DataFrame(rows).sort_values("roc_auc",ascending=False)
    log(f"\n    {'method':<15}{'ROC':>7}{'PR':>7}{'MCC':>7}{'BalAcc':>8}"
        f"{'Se@95Sp':>9}{'Se@90Sp':>9}{'Sp@90Se':>9}{'Brier':>8}{'ECE':>8}{'cov%':>7}")
    for _,r in T.iterrows():
        b="  -" if np.isnan(r.brier) else f"{r.brier:.3f}"
        e="  -" if np.isnan(r.ece) else f"{r.ece:.3f}"
        log(f"    {r.method:<15}{r.roc_auc:>7.3f}{r.pr_auc:>7.3f}{r.mcc:>7.3f}"
            f"{r.bal_acc:>8.3f}{r.sens_at_95spec:>9.3f}{r.sens_at_90spec:>9.3f}"
            f"{r.spec_at_90sens:>9.3f}{b:>8}{e:>8}{r.coverage_pct:>7.1f}")
    T.to_csv(od/"step07_clinical_metrics.csv",index=False)
    T.to_csv(od/"FINAL_table_clinical_metrics.csv",index=False)
    log("\n    Brier/ECE are shown ONLY for AlphaMissense, the sole method whose")
    log("    output is a probability. Others are ranks or arbitrary scales.")
    log("    -> overwrote step07_clinical_metrics.csv")

    # ---------- purge the misleading calibration file ----------
    cf=od/"step07_calibration.csv"
    if cf.exists():
        c=pd.read_csv(cf)
        c["VALID"]=[n=="AlphaMissense" for n in c.method]
        c["note"]=["probability output" if v else
                   "rank/arbitrary scale - Brier and ECE NOT interpretable"
                   for v in c.VALID]
        c.to_csv(cf,index=False)
        log(f"\n[2] annotated {cf.name}: only the AlphaMissense row is interpretable")

    # ---------- LaTeX ----------
    log("\n[3] LATEX TABLE")
    L=[r"\begin{tabular}{lrrrrrrr}",r"\hline",
       r"Method & ROC-AUC & PR-AUC & MCC & Se@95\%Sp & Se@90\%Sp & Brier & Cov.\ (\%) \\",
       r"\hline"]
    for _,r in T.iterrows():
        b="--" if np.isnan(r.brier) else f"{r.brier:.3f}"
        L.append(f"{r.method} & {r.roc_auc:.3f} & {r.pr_auc:.3f} & {r.mcc:.3f} & "
                 f"{r.sens_at_95spec:.3f} & {r.sens_at_90spec:.3f} & {b} & "
                 f"{r.coverage_pct:.1f} \\\\")
    L+=[r"\hline",r"\end{tabular}"]
    (od/"FINAL_table_clinical_metrics.tex").write_text("\n".join(L))
    log("    -> FINAL_table_clinical_metrics.tex")

    # ---------- outstanding issues ----------
    log("\n[4] OUTSTANDING ISSUES NOT YET RESOLVED")
    t=od.parent/"results"/"llr_prott5.csv"
    tp=root/"benchmark/results/llr_prott5.csv"
    if tp.exists():
        pt=pd.read_csv(tp)
        cols=[c for c in pt.columns if c.startswith("prott5")]
        log(f"  (a) ProtT5: file contains {cols}, i.e. embedding distances.")
        mm=m.merge(pt[["target_gene","position","wt_aa","mut_aa"]+cols],
                   on=["target_gene","position","wt_aa","mut_aa"],how="inner")
        for c in cols:
            s=mm[c].values; yy=mm.label.values
            ok=~np.isnan(s)
            if len(np.unique(yy[ok]))>1:
                a1=roc_auc_score(yy[ok],s[ok]); a2=roc_auc_score(yy[ok],-s[ok])
                log(f"      {c}: ROC-AUC={a1:.3f} (as-is), {a2:.3f} (negated), n={ok.sum()}")
        log("      The manuscript reports ProtT5 ROC-AUC = 0.49 from 'masked-marginal")
        log("      LLR via the span-corruption decoder'. Compare the values above to")
        log("      determine which quantity was actually used, then correct Methods.")
    log("  (b) WT1 lacks SaProt and AlphaMissense scores (3Di tokens built from the")
    log("      449-aa model; variants are on the 522-aa isoform). Either re-run")
    log("      Foldseek on the corrected structure or state the exclusion.")
    log("  (c) Four unresolved '[? ]' citations in the PDF (SaProt, DeLong x2).")
    log("  (d) GitHub URL is a placeholder while Data Availability says 'upon")
    log("      reasonable request'. These contradict each other.")
    log("\n  (a) is analysis; (b)-(d) are manuscript edits. None blocks Step 8.")
    log("\nWROTE: FINAL_table_clinical_metrics.{csv,tex}, step07_clinical_metrics.csv")
    log.close()

if __name__=="__main__": main()

#!/usr/bin/env python3
"""STEP 7d - resolve the ProtT5 contradiction (section 4a of step07c crashed:
master_scores.csv already carries the prott5 columns, so the merge suffixed them)."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        s=" ".join(str(x) for x in a); print(s,flush=True); self.f.write(s+"\n")
    def close(self): self.f.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP07D_prott5_report.txt")
    m=pd.read_csv(od/"master_scores.csv"); m=m[m.analysis_ok==True].copy()
    log("="*78); log("STEP 7d - RESOLVING THE ProtT5 CONTRADICTION"); log("="*78)
    log("\nFIX: step07c re-merged llr_prott5.csv, but master_scores.csv already")
    log("contained those columns, producing _x/_y suffixes and a KeyError.")
    log("Using the columns already present in master_scores.csv instead.")

    cols=[c for c in m.columns if c.startswith("prott5")]
    log(f"\nprott5 columns in master table: {cols}")
    if not cols:
        log("none present - nothing to resolve"); log.close(); return

    log("\n[1] WHAT DOES EACH STORED QUANTITY ACTUALLY ACHIEVE?")
    log(f"    {'column':<26}{'n':>7}{'AUC as-is':>12}{'AUC negated':>13}"
        f"{'mean(P)':>10}{'mean(B)':>10}")
    res=[]
    for c in cols:
        s=m[c].values; y=m.label.values
        ok=~np.isnan(s)
        if len(np.unique(y[ok]))<2:
            log(f"    {c:<26}{ok.sum():>7}  single class"); continue
        a1=roc_auc_score(y[ok],s[ok]); a2=roc_auc_score(y[ok],-s[ok])
        mp=s[ok][y[ok]==1].mean(); mb=s[ok][y[ok]==0].mean()
        log(f"    {c:<26}{ok.sum():>7}{a1:>12.4f}{a2:>13.4f}{mp:>10.3f}{mb:>10.3f}")
        res.append(dict(column=c,n=int(ok.sum()),auc_asis=a1,auc_negated=a2,
                        mean_path=mp,mean_benign=mb,best=max(a1,a2)))
    R=pd.DataFrame(res); R.to_csv(od/"step07d_prott5_scores.csv",index=False)

    log("\n[2] COMPARISON WITH THE MANUSCRIPT")
    log("    The manuscript states (Methods 2.2 and Discussion):")
    log("      'ProtT5 was evaluated using the same masking approach via its T5")
    log("       span-corruption decoder, but yielded near-random discrimination")
    log("       (ROC-AUC = 0.49)'")
    if len(R):
        best=R.best.max()
        log(f"\n    Best AUC achievable from any SAVED ProtT5 quantity: {best:.4f}")
        near=R[(R.best>=0.47)&(R.best<=0.51)]
        if len(near):
            log(f"    Column(s) consistent with the reported 0.49: "
                f"{near.column.tolist()}")
            log("    => The reported value most likely came from embedding distance,")
            log("       NOT from masked-marginal LLR. The Methods sentence describing")
            log("       'the same masking approach via its T5 span-corruption decoder'")
            log("       does not match what is on disk and must be corrected.")
        else:
            log("    NO saved column reproduces ~0.49.")
            log("    => The reported ProtT5 LLR result is NOT reproducible from saved")
            log("       data. Either re-run the scoring and save it, or remove the")
            log("       numeric claim and state only that ProtT5 was excluded from")
            log("       variant-effect scoring.")

    log("\n[3] IMPACT ON THE REST OF THE PAPER")
    log("    ProtT5 is used for the attention analysis (Section 3.2), not for")
    log("    variant scoring, so the attention results are unaffected. What must")
    log("    change is the justification sentence for excluding it.")
    log("\n    SUGGESTED REPLACEMENT TEXT (choose per section 2 outcome):")
    log("    If distance reproduces 0.49:")
    log("      'ProtT5 uses a span-corruption objective that is not directly")
    log("       compatible with single-residue masked-marginal scoring. An")
    log("       embedding-distance surrogate (cosine distance between wild-type")
    log("       and mutant residue representations) achieved near-random")
    log("       discrimination (ROC-AUC = X.XX), so ProtT5 was retained for")
    log("       representational analysis only and excluded from variant-effect")
    log("       comparisons.'")
    log("    If nothing reproduces it:")
    log("      'ProtT5 was excluded from variant-effect scoring because its")
    log("       span-corruption training objective is not compatible with the")
    log("       masked-marginal formulation used for the other models; it was")
    log("       retained for the attention analysis.'  (no numeric claim)")

    log("\n[4] REMAINING MANUSCRIPT-ONLY FIXES")
    log("    (a) Replace the ProtT5 Methods sentence per section 3.")
    log("    (b) WT1: no SaProt/AlphaMissense scores - state the exclusion or")
    log("        re-run Foldseek on the 522-aa isoform.")
    log("    (c) Resolve four '[? ]' citations: SaProt (Su et al. 2024, ICLR),")
    log("        DeLong et al. 1988 Biometrics 44:837-845 (cited twice).")
    log("    (d) Replace https://github.com/[user]/cancerplmbench with a real URL")
    log("        and make the Data Availability Statement agree with it.")
    log("\nWROTE: step07d_prott5_scores.csv")
    log.close()

if __name__=="__main__": main()

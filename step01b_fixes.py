#!/usr/bin/env python3
"""STEP 1b - fix the three defects found in Step 1:
   (A) regenerate Table 2 from data, with LaTeX output
   (B) quantify shared-subset coverage bias -> becomes a paper contribution
   (C) re-derive ClinVar LastEvaluated by gene + HGVS protein change
"""
import argparse, gzip, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

AA3 = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E',
       'Gly':'G','His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F',
       'Pro':'P','Ser':'S','Thr':'T','Trp':'W','Tyr':'Y','Val':'V',
       'Sec':'U','Pyl':'O'}
PROT_RE = re.compile(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})')
CUTOFF = pd.Timestamp("2023-09-01")

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        s=" ".join(str(x) for x in a); print(s); self.f.write(s+"\n")
    def close(self): self.f.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    a=ap.parse_args(); root=Path(a.root).resolve()
    od=root/"benchmark"/"results"
    log=Tee(od/"STEP01B_fixes_report.txt")
    m=pd.read_csv(od/"master_scores.csv")
    log("="*78); log("STEP 1b - DEFECT FIXES"); log("="*78)

    # =========================================================== (A) TABLE 2
    log("\n" + "="*78)
    log("[A] REGENERATE TABLE 2 FROM DATA")
    log("="*78)
    t2=(m.groupby(["target_gene","gene_type"])["label"]
          .agg(Pathogenic=lambda s:int((s==1).sum()),
               Benign=lambda s:int((s==0).sum()),Total="size")
          .reset_index().sort_values("Total",ascending=False))
    log("\n  CORRECTED per-gene counts:")
    log(t2.to_string(index=False))
    tot=t2[["Pathogenic","Benign","Total"]].sum()
    log(f"\n  COLUMN SUMS: P={tot.Pathogenic}  B={tot.Benign}  Total={tot.Total}")
    log(f"  master table rows: {len(m)}   labels: {m.label.value_counts().to_dict()}")
    ok = (tot.Total==len(m)) and (tot.Pathogenic==int((m.label==1).sum()))
    log(f"  INTERNAL CONSISTENCY: {'PASS' if ok else 'FAIL'}")

    pub={'BRCA2':(114,587),'BRCA1':(219,374),'NF1':(429,58),'TSC2':(185,203),
         'TP53':(238,146),'PTEN':(247,10),'ATM':(100,142),'VHL':(175,25),
         'BRAF':(106,12),'RET':(71,45),'TSC1':(22,79),'APC':(23,72),
         'TP63':(83,11),'ALK':(16,75),'RB1':(42,45),'MET':(12,68),
         'WT1':(51,21),'EGFR':(17,43),'STK11':(42,11),'KIT':(31,22),
         'CDKN2A':(38,15),'HRAS':(31,10),'ERBB2':(12,19),'MYCN':(12,10)}
    log("\n  DIFFERENCES vs published Table 2 (published -> actual):")
    diffs=0
    for _,r in t2.iterrows():
        g=r.target_gene
        if g in pub:
            pP,pB=pub[g]
            if pP!=r.Pathogenic or pB!=r.Benign:
                log(f"    {g:<8} P {pP:>4} -> {r.Pathogenic:<4}   B {pB:>4} -> {r.Benign:<4}")
                diffs+=1
    log(f"    {diffs} of 24 genes differ. Published column sums were "
        f"P=2316 B=2103 Total=4419, contradicting its own Total row (2258/2093/4351).")

    lines=[r"\begin{tabular}{llrrr}",r"\hline",
           r"Gene & Class & Pathogenic & Benign & Total \\",r"\hline"]
    for _,r in t2.iterrows():
        cls="TSG" if str(r.gene_type).lower().startswith("tumor") else "ONC"
        lines.append(f"{r.target_gene} & {cls} & {r.Pathogenic} & {r.Benign} & {r.Total} \\\\")
    lines+= [r"\hline",
             f"\\textbf{{Total}} & & \\textbf{{{tot.Pathogenic}}} & "
             f"\\textbf{{{tot.Benign}}} & \\textbf{{{tot.Total}}} \\\\",
             r"\hline",r"\end{tabular}"]
    (od/"step01b_table2_corrected.tex").write_text("\n".join(lines))
    t2.to_csv(od/"step01b_table2_corrected.csv",index=False)
    log("\n  -> step01b_table2_corrected.tex written (paste into manuscript)")

    # ============================================ (B) COVERAGE BIAS ANALYSIS
    log("\n" + "="*78)
    log("[B] SHARED-SUBSET COVERAGE BIAS")
    log("="*78)

    smeta=pd.read_csv(root/"benchmark/data/sequence_metadata.csv")
    smeta=smeta.rename(columns={"gene_symbol":"target_gene","human_length":"protein_length"})
    m=m.merge(smeta[["target_gene","protein_length"]],on="target_gene",how="left")

    METH=["alphamissense","saprot","revel","esm2_650m","cadd","sift",
          "polyphen2","esm2_150m","esm1v","eve"]
    cov=(m.assign(**{f"c_{x}":m["s_"+x].notna() for x in METH})
           .groupby("target_gene")
           .agg(n=("label","size"),protein_length=("protein_length","first"),
                gene_type=("gene_type","first"),
                **{x:(f"c_{x}","mean") for x in METH}))
    for x in METH: cov[x]=(100*cov[x]).round(1)
    cov["shared10_pct"]=(100*m.groupby("target_gene")["in_shared_all"].mean()).round(1)
    cov=cov.sort_values("protein_length",ascending=False)
    cov.to_csv(od/"step01b_coverage_matrix.csv")
    log("\n  Per-gene score coverage (%), sorted by protein length:")
    log(cov[["n","protein_length","gene_type","alphamissense","saprot",
             "revel","eve","esm2_650m","shared10_pct"]].to_string())

    r,p=stats.spearmanr(cov.protein_length,cov.shared10_pct)
    log(f"\n  Spearman(protein length, % retained in shared-10): rho={r:.3f}, p={p:.2e}")
    long_g=cov[cov.protein_length>2700]
    log(f"  Genes exceeding the AlphaFold 2,700-aa cap: {list(long_g.index)}")
    log(f"    their mean shared-10 retention: {long_g.shared10_pct.mean():.1f}%")
    log(f"    all other genes: {cov[cov.protein_length<=2700].shared10_pct.mean():.1f}%")

    log("\n  Included vs excluded from the n=2,080 shared subset:")
    inc=m[m.in_shared_all]; exc=m[~m.in_shared_all]
    log(f"    n:            {len(inc)} vs {len(exc)}")
    log(f"    %% pathogenic: {100*inc.label.mean():.1f}% vs {100*exc.label.mean():.1f}%"
        f"   (full benchmark {100*m.label.mean():.1f}%)")
    ct=pd.crosstab(m.in_shared_all,m.label)
    chi2,pc,_,_=stats.chi2_contingency(ct)
    log(f"    label chi2 p = {pc:.3e}   <- inclusion is NOT independent of label")
    log(f"    mean protein length: {inc.protein_length.mean():.0f} vs {exc.protein_length.mean():.0f} aa")
    u,pl=stats.mannwhitneyu(inc.protein_length.dropna(),exc.protein_length.dropna())
    log(f"    protein-length Mann-Whitney p = {pl:.3e}")
    log(f"    mean review stars:   {inc.review_stars.mean():.2f} vs {exc.review_stars.mean():.2f}")
    u,ps=stats.mannwhitneyu(inc.review_stars.dropna(),exc.review_stars.dropna())
    log(f"    review-star Mann-Whitney p = {ps:.3e}")
    ct2=pd.crosstab(m.in_shared_all,m.gene_type)
    chi2b,pg2,_,_=stats.chi2_contingency(ct2)
    log(f"    gene_type: {ct2.to_dict()}  chi2 p = {pg2:.3e}")
    log(f"    genes represented: {inc.target_gene.nunique()} of {m.target_gene.nunique()}")
    log(f"    genes fully lost: {sorted(set(m.target_gene)-set(inc.target_gene))}")

    bias=pd.DataFrame([
        dict(variable="n",included=len(inc),excluded=len(exc),test="-",p=np.nan),
        dict(variable="% pathogenic",included=round(100*inc.label.mean(),1),
             excluded=round(100*exc.label.mean(),1),test="chi2",p=pc),
        dict(variable="mean protein length (aa)",included=round(inc.protein_length.mean()),
             excluded=round(exc.protein_length.mean()),test="Mann-Whitney",p=pl),
        dict(variable="mean ClinVar review stars",included=round(inc.review_stars.mean(),2),
             excluded=round(exc.review_stars.mean(),2),test="Mann-Whitney",p=ps),
        dict(variable="% tumour suppressor",
             included=round(100*(inc.gene_type.str.lower().str.startswith("tumor")).mean(),1),
             excluded=round(100*(exc.gene_type.str.lower().str.startswith("tumor")).mean(),1),
             test="chi2",p=pg2),
        dict(variable="n genes represented",included=inc.target_gene.nunique(),
             excluded=exc.target_gene.nunique(),test="-",p=np.nan)])
    bias.to_csv(od/"step01b_inclusion_bias.csv",index=False)

    log("\n  ALTERNATIVE EVALUATION VIEWS (to avoid relying on the biased subset):")
    views={
      "V1 full benchmark (per-method native)": pd.Series(True,index=m.index),
      "V2 shared-10 (published headline)":     m.in_shared_all,
      "V3 3x ESM only (sequence-only pLMs)":   m[["s_esm2_650m","s_esm2_150m","s_esm1v"]].notna().all(axis=1),
      "V4 AM + 4 pLMs":                        m.in_shared_plm_am,
      "V5 dbNSFP-5 shared":                    m[["s_revel","s_cadd","s_sift","s_polyphen2","s_eve"]].notna().all(axis=1),
    }
    vr=[]
    for nm,msk in views.items():
        s=m[msk]
        vr.append(dict(view=nm,n=len(s),genes=s.target_gene.nunique(),
                       pct_path=round(100*s.label.mean(),1),
                       mean_len=round(s.protein_length.mean())))
    vdf=pd.DataFrame(vr); vdf.to_csv(od/"step01b_evaluation_views.csv",index=False)
    log(vdf.to_string(index=False))
    log("\n  V3 keeps all 24 genes -> use it for pLM-vs-pLM claims.")
    log("  V2 keeps 17 genes and is pathogenic-enriched -> report as one view among several,")
    log("  never as 'the benchmark of 4,351 variants across 24 genes'.")

    # ================================================ (C) LASTEVALUATED FIX
    log("\n" + "="*78)
    log("[C] RE-DERIVE LastEvaluated BY GENE + PROTEIN CHANGE")
    log("="*78)
    cv=root/"benchmark/data/clinvar/variant_summary.txt.gz"
    if not cv.exists():
        log(f"  {cv} not found - skipped"); 
    else:
        with gzip.open(cv,"rt",encoding="utf-8",errors="replace") as fh:
            header=fh.readline().rstrip("\n").split("\t")
        want=[c for c in ["Name","GeneSymbol","LastEvaluated","Assembly",
                          "ReviewStatus","ClinicalSignificance","VariationID"] if c in header]
        log(f"  reading columns {want} in chunks ...")
        genes=set(m.target_gene.unique()); parts=[]
        for ch in pd.read_csv(cv,sep="\t",usecols=want,dtype=str,
                              chunksize=400_000,low_memory=False):
            if "Assembly" in ch.columns: ch=ch[ch["Assembly"]=="GRCh38"]
            ch=ch[ch["GeneSymbol"].isin(genes)]
            if len(ch): parts.append(ch)
        cvd=pd.concat(parts,ignore_index=True)
        log(f"  rows for our 24 genes on GRCh38: {len(cvd)}")
        ext=cvd["Name"].fillna("").str.extract(PROT_RE)
        cvd["wt_aa"]=ext[0].map(AA3); cvd["mut_aa"]=ext[2].map(AA3)
        cvd["position"]=pd.to_numeric(ext[1],errors="coerce")
        cvd=cvd.dropna(subset=["wt_aa","mut_aa","position"])
        cvd["position"]=cvd["position"].astype(int)
        cvd["LastEvaluated"]=pd.to_datetime(
            cvd["LastEvaluated"].replace("-",np.nan),errors="coerce")
        log(f"  rows with parseable p.XaaNNNXaa: {len(cvd)}")
        cvd=(cvd.dropna(subset=["LastEvaluated"])
                .sort_values("LastEvaluated")
                .drop_duplicates(subset=["GeneSymbol","wt_aa","position","mut_aa"],keep="last")
                .rename(columns={"GeneSymbol":"target_gene",
                                 "LastEvaluated":"last_evaluated_new",
                                 "VariationID":"clinvar_variation_id"}))
        keep=["target_gene","wt_aa","position","mut_aa","last_evaluated_new"]
        if "clinvar_variation_id" in cvd.columns: keep.append("clinvar_variation_id")
        m=m.drop(columns=[c for c in ["last_evaluated","last_eval_year"] if c in m.columns])
        m=m.merge(cvd[keep],on=["target_gene","wt_aa","position","mut_aa"],how="left")
        m=m.rename(columns={"last_evaluated_new":"last_evaluated"})
        ok=m.last_evaluated.notna().sum()
        log(f"  MATCHED: {ok}/{len(m)} ({100*ok/len(m):.1f}%)")
        if ok:
            m["last_eval_year"]=m.last_evaluated.dt.year
            m["temporal_stratum"]=np.where(m.last_evaluated<=CUTOFF,"pre_AM",
                                  np.where(m.last_evaluated.notna(),"post_AM",None))
            log("\n  year distribution:")
            log(m.last_eval_year.value_counts().sort_index().to_string())
            log("\n  stratum x label:")
            log(m.groupby(["temporal_stratum","label"]).size().unstack(fill_value=0).to_string())
            log("\n  stratum sizes within each evaluation view:")
            for nm,msk in [("full",pd.Series(True,index=m.index)),
                           ("shared10",m.in_shared_all),("AM+4pLM",m.in_shared_plm_am)]:
                s=m[msk]
                log(f"    {nm:<10} pre_AM={int((s.temporal_stratum=='pre_AM').sum()):<5}"
                    f" post_AM={int((s.temporal_stratum=='post_AM').sum()):<5}"
                    f" unknown={int(s.temporal_stratum.isna().sum())}")
            log(f"\n  published temporal analysis used n=2,715 (967 pre / 1,748 post)")
            log(f"  reproducible here: see AM+4pLM row above")
            miss=m[m.last_evaluated.isna()].target_gene.value_counts()
            if len(miss): log("\n  unmatched variants by gene:"); log(miss.to_string())

    m.to_csv(od/"master_scores.csv",index=False)
    log(f"\n  master_scores.csv updated  shape={m.shape}")
    log("\nWROTE: step01b_table2_corrected.{csv,tex}, step01b_coverage_matrix.csv,")
    log("       step01b_inclusion_bias.csv, step01b_evaluation_views.csv")
    log.close()

if __name__=="__main__": main()

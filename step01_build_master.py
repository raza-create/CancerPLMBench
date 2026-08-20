#!/usr/bin/env python3
"""STEP 1 - build the single master variant/score table for CancerPLMBench."""
import argparse, gzip, sys
from pathlib import Path
import numpy as np, pandas as pd

KEY = ["target_gene","variant_id","position","wt_aa","mut_aa"]
LLR_SOURCES = {
 "esm2_650m": ("benchmark/results/llr_esm2_650M.csv","llr_esm2_650M",True),
 "esm2_150m": ("benchmark/results/llr_esm2_150M.csv","llr_esm2_150M",True),
 "esm1v":     ("benchmark/results/llr_esm1v.csv","llr_esm1v",True),
 "saprot":    ("benchmark/results/llr_saprot.csv","llr_saprot",True),
}
DBNSFP_COLS = {"sift":"sift","polyphen2":"polyphen2","revel":"revel","eve":"eve","cadd":"cadd_phred"}
REVIEW_STARS = {
 "practice guideline":4,"reviewed by expert panel":3,
 "criteria provided, multiple submitters, no conflicts":2,
 "criteria provided, multiple submitters":2,
 "criteria provided, single submitter":1,
 "criteria provided, conflicting classifications":1,
 "criteria provided, conflicting interpretations":1,
 "no assertion criteria provided":0,"no classification provided":0,
 "no assertion provided":0,"no classifications from unflagged records":0,
 "no interpretation for the single variant":0,"flagged submission":0,
}

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        line=" ".join(str(x) for x in a); print(line); self.f.write(line+"\n")
    def close(self): self.f.close()

def stars(rs):
    return REVIEW_STARS.get(rs.strip().lower(), np.nan) if isinstance(rs,str) else np.nan

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    ap.add_argument("--skip-clinvar",action="store_true")
    args=ap.parse_args()
    root=Path(args.root).resolve()
    outdir=root/"benchmark"/"results"; outdir.mkdir(parents=True,exist_ok=True)
    log=Tee(outdir/"STEP01_master_report.txt")
    log("="*78); log("STEP 1 - BUILD MASTER SCORE TABLE"); log("="*78)

    # ---- spine
    sp=root/"benchmark/data/clinvar/benchmark_variants.csv"
    m=pd.read_csv(sp)
    log(f"\n[1] spine: {sp.name}  shape={m.shape}")
    log(f"    label counts: {m['label'].value_counts().to_dict()}")
    for c in KEY:
        if c not in m.columns: log(f"    FATAL: missing key {c!r}"); sys.exit(1)
    d=m.duplicated(subset=KEY).sum(); log(f"    duplicate key rows: {d}")
    if d: m=m.drop_duplicates(subset=KEY); log(f"    -> now {len(m)}")

    # ---- gene metadata
    gp=root/"benchmark/data/cancer_genes_curated.csv"; g=pd.read_csv(gp)
    log(f"\n[2] gene metadata: {gp.name}  shape={g.shape}")
    log(f"    gene_type values: {g['gene_type'].value_counts().to_dict()}")
    g=g.rename(columns={"gene_symbol":"target_gene"})
    m=m.merge(g[["target_gene","uniprot_id","gene_type","family"]],on="target_gene",how="left")
    miss=m["gene_type"].isna().sum(); log(f"    variants with no gene_type: {miss}")
    if miss: log(f"    genes lacking metadata: {sorted(m.loc[m['gene_type'].isna(),'target_gene'].unique())}")
    log("    per-gene-class counts:")
    log(m.groupby(["gene_type","label"]).size().unstack(fill_value=0).to_string())

    # ---- review stars
    log("\n[3] ClinVar review status -> stars")
    m["review_stars"]=m["ReviewStatus"].map(stars)
    un=m.loc[m["review_stars"].isna(),"ReviewStatus"].value_counts()
    if len(un):
        log("    UNMAPPED ReviewStatus strings:")
        for k,v in un.items(): log(f"      {k!r}: {v}")
    log("    star distribution:")
    log(m["review_stars"].value_counts(dropna=False).sort_index().to_string())
    log("    stars x label:")
    log(m.groupby(["review_stars","label"]).size().unstack(fill_value=0).to_string())

    # ---- LastEvaluated
    m["last_evaluated"]=pd.NaT
    cv=root/"benchmark/data/clinvar/variant_summary.txt.gz"
    if args.skip_clinvar: log("\n[4] LastEvaluated join SKIPPED")
    elif not cv.exists(): log(f"\n[4] {cv} not found")
    else:
        log(f"\n[4] re-joining LastEvaluated from {cv.name} (1-3 min)")
        with gzip.open(cv,"rt",encoding="utf-8",errors="replace") as fh:
            header=fh.readline().rstrip("\n").split("\t")
        log(f"    file has {len(header)} columns")
        ids=[c for c in header if c.replace("#","").lower() in ("variationid","alleleid")]
        log(f"    id column candidates: {ids}")
        want=[c for c in header if c in ids+["LastEvaluated","ReviewStatus","Assembly"]]
        parts=[]
        for ch in pd.read_csv(cv,sep="\t",usecols=want,dtype=str,chunksize=500_000,low_memory=False):
            if "Assembly" in ch.columns: ch=ch[ch["Assembly"]=="GRCh38"]
            parts.append(ch)
        cvd=pd.concat(parts,ignore_index=True); log(f"    GRCh38 rows: {len(cvd)}")
        vid=m["variant_id"].astype(str); best,bn=None,-1
        for c in ids:
            n=vid.isin(set(cvd[c].astype(str))).sum()
            log(f"    variant_id matches {c!r}: {n}/{len(m)} ({100*n/len(m):.1f}%)")
            if n>bn: best,bn=c,n
        if bn<0.5*len(m):
            log("    WARNING: poor match. variant_id may be a custom id.")
            log(f"    sample variant_id: {m['variant_id'].head(5).tolist()}")
            log(f"    sample {best}: {cvd[best].head(5).tolist()}")
        else:
            sub=(cvd[[best,"LastEvaluated"]].assign(**{best:lambda z:z[best].astype(str)})
                 .replace({"LastEvaluated":{"-":np.nan}}).dropna(subset=["LastEvaluated"])
                 .drop_duplicates(subset=[best],keep="last"))
            lut=dict(zip(sub[best],sub["LastEvaluated"]))
            m["last_evaluated"]=pd.to_datetime(vid.map(lut),errors="coerce")
            ok=m["last_evaluated"].notna().sum()
            log(f"    LastEvaluated populated {ok}/{len(m)} ({100*ok/len(m):.1f}%)")
            if ok:
                m["last_eval_year"]=m["last_evaluated"].dt.year
                log("    year distribution:")
                log(m["last_eval_year"].value_counts().sort_index().to_string())
                pre=(m["last_evaluated"]<="2023-09-01").sum()
                log(f"    pre-AM: {pre}   post-AM: {ok-pre}")

    # ---- AlphaMissense
    log("\n[5] AlphaMissense scores")
    amd=root/"benchmark/data/alphamissense_scores"; fr=[]
    for f in sorted(amd.glob("*_AM.csv")):
        up=f.name.replace("_AM.csv",""); dd=pd.read_csv(f)
        if "protein_variant" not in dd.columns: continue
        pv=dd["protein_variant"].astype(str)
        dd=dd.assign(uniprot_id=up,wt_aa=pv.str[0],
                     position=pd.to_numeric(pv.str[1:-1],errors="coerce"),mut_aa=pv.str[-1])
        fr.append(dd[["uniprot_id","wt_aa","position","mut_aa","am_pathogenicity","am_class"]])
    if fr:
        am=pd.concat(fr,ignore_index=True).dropna(subset=["position"])
        am["position"]=am["position"].astype(int)
        log(f"    {len(fr)} AM files, {len(am)} rows, {am['uniprot_id'].nunique()} proteins")
        m=m.merge(am,on=["uniprot_id","wt_aa","position","mut_aa"],how="left")
        log(f"    AM coverage: {m['am_pathogenicity'].notna().sum()}/{len(m)} ({100*m['am_pathogenicity'].notna().mean():.1f}%)")
        nc=(m.groupby("target_gene")["am_pathogenicity"].apply(lambda s:s.isna().mean()*100).round(1))
        bad=nc[nc>0].sort_values(ascending=False)
        if len(bad): log("    genes with missing AM (%):"); log(bad.to_string())
    else:
        log("    !! no AM files found"); m["am_pathogenicity"]=np.nan

    # ---- pLMs
    log("\n[6] pLM LLR scores")
    for name,(rel,col,flip) in LLR_SOURCES.items():
        p=root/rel
        if not p.exists(): log(f"    {name:<11} MISSING {rel}"); m[f"raw_{name}"]=np.nan; continue
        dd=pd.read_csv(p)
        if col not in dd.columns:
            log(f"    {name:<11} col {col!r} not in {p.name}: {list(dd.columns)}")
            m[f"raw_{name}"]=np.nan; continue
        dd=dd[KEY+[col]].drop_duplicates(subset=KEY)
        m=m.merge(dd.rename(columns={col:f"raw_{name}"}),on=KEY,how="left")
        n=m[f"raw_{name}"].notna().sum()
        log(f"    {name:<11} {n}/{len(m)} ({100*n/len(m):.1f}%)  src={p.name} rows={len(dd)}")

    # ---- dbNSFP
    log("\n[7] dbNSFP baselines")
    bp=root/"benchmark/results/baselines_dbnsfp.csv"
    if bp.exists():
        b=pd.read_csv(bp); keep=[c for c in DBNSFP_COLS.values() if c in b.columns]
        b=b[KEY+keep].drop_duplicates(subset=KEY); m=m.merge(b,on=KEY,how="left")
        for name,col in DBNSFP_COLS.items():
            if col in m.columns:
                n=m[col].notna().sum(); log(f"    {name:<11} {n}/{len(m)} ({100*n/len(m):.1f}%)")
                m[f"raw_{name}"]=m[col]
            else:
                m[f"raw_{name}"]=np.nan; log(f"    {name:<11} MISSING")
    else:
        log(f"    !! {bp} not found")
        for name in DBNSFP_COLS: m[f"raw_{name}"]=np.nan
    m["raw_alphamissense"]=m["am_pathogenicity"]

    # ---- ProtT5
    log("\n[8] ProtT5 (embedding distances - NOT a masked-marginal LLR)")
    tp=root/"benchmark/results/llr_prott5.csv"
    if tp.exists():
        t=pd.read_csv(tp); cols=[c for c in t.columns if c.startswith("prott5")]
        log(f"    columns present: {cols}")
        log("    NOTE: Methods text says ProtT5 was scored by masked-marginal LLR via the")
        log("          span-corruption decoder, but this file stores embedding distance.")
        log("          Methods must be corrected to match whatever produced ROC-AUC ~0.49.")
        t=t[KEY+cols].drop_duplicates(subset=KEY); m=m.merge(t,on=KEY,how="left")
    else: log("    file not found")

    # ---- orientation
    log("\n[9] SCORE ORIENTATION CHECK (higher must = more pathogenic)")
    methods=["alphamissense","saprot","esm2_650m","esm2_150m","esm1v","revel","cadd","sift","polyphen2","eve"]
    fm={k:v[2] for k,v in LLR_SOURCES.items()}
    log(f"    {'method':<14}{'n':>6}{'mean(P)':>11}{'mean(B)':>11}{'flip?':>7}{'oriented dP-dB':>16}{'verdict':>12}")
    bad=[]
    for meth in methods:
        raw=f"raw_{meth}"
        if raw not in m.columns: continue
        s=m[raw]; flip=fm.get(meth,False); m[f"s_{meth}"]=-s if flip else s
        n=s.notna().sum()
        if n==0: log(f"    {meth:<14}{0:>6}{'-':>11}{'-':>11}{str(flip):>7}{'-':>16}{'NO DATA':>12}"); continue
        mp=m.loc[m.label==1,f"s_{meth}"].mean(); mb=m.loc[m.label==0,f"s_{meth}"].mean()
        v="OK" if mp>mb else "*** FLIPPED"
        if v!="OK": bad.append(meth)
        log(f"    {meth:<14}{n:>6}{s[m.label==1].mean():>11.3f}{s[m.label==0].mean():>11.3f}{str(flip):>7}{mp-mb:>16.3f}{v:>12}")
    if bad:
        log("    !!! ORIENTATION PROBLEM in: "+", ".join(bad))
        log("        Do not proceed to Step 2 until resolved.")

    # ---- subsets
    log("\n[10] EVALUATION SUBSETS")
    sc=[f"s_{x}" for x in methods if f"s_{x}" in m.columns]
    m["n_methods_scored"]=m[sc].notna().sum(axis=1)
    m["in_shared_all"]=m[sc].notna().all(axis=1)
    pl=[f"s_{x}" for x in ["saprot","esm2_650m","esm2_150m","esm1v"] if f"s_{x}" in m.columns]
    m["in_shared_plm_am"]=m[pl+["s_alphamissense"]].notna().all(axis=1)
    log(f"    full benchmark            n = {len(m)}")
    log(f"    all {len(sc)} methods scored     n = {int(m['in_shared_all'].sum())}")
    log(f"    4 pLMs + AlphaMissense    n = {int(m['in_shared_plm_am'].sum())}")
    log("    #methods scored per variant:")
    log(m["n_methods_scored"].value_counts().sort_index().to_string())
    log("\n    label balance per subset:")
    for nm,mask in [("full",pd.Series(True,index=m.index)),("shared_all",m["in_shared_all"]),("shared_plm_am",m["in_shared_plm_am"])]:
        s2=m[mask]
        log(f"      {nm:<14} n={len(s2):<6} P={int((s2.label==1).sum()):<5} B={int((s2.label==0).sum()):<5} genes={s2.target_gene.nunique()}")
    log("\n    per-gene retention in fully-shared subset:")
    ret=(m.groupby("target_gene").agg(n_full=("label","size"),n_shared=("in_shared_all","sum"))
         .assign(pct=lambda z:(100*z.n_shared/z.n_full).round(1)).sort_values("n_full",ascending=False))
    log(ret.to_string())
    dr=ret[ret.n_shared<20]
    if len(dr): log(f"\n    genes with <20 variants left in shared subset: {dr.index.tolist()}")

    out=outdir/"master_scores.csv"; m.to_csv(out,index=False)
    log(f"\n[11] WROTE {out}  shape={m.shape}")
    log(f"     columns: {list(m.columns)}")
    log("\nDone. Paste STEP01_master_report.txt back before Step 2.")
    log.close()

if __name__=="__main__": main()

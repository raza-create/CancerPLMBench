#!/usr/bin/env python3
"""STEP 6 - structural environment analysis from AlphaFold PDBs.
Computes per-residue pLDDT and relative solvent accessibility, then asks
whether the structure-aware advantage (SaProt - ESM-2 650M) depends on
burial and order. Gene-clustered inference + multiple-testing correction."""
import argparse, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

# Tien et al. 2013 theoretical max ASA (A^2)
MAXASA={'A':129,'R':274,'N':195,'D':193,'C':167,'Q':225,'E':223,'G':104,'H':224,
        'I':197,'L':201,'K':236,'M':224,'F':240,'P':159,'S':155,'T':172,'W':285,
        'Y':263,'V':174}
THREE2ONE={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
           'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
           'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
NICE={"alphamissense":"AlphaMissense","saprot":"SaProt","revel":"REVEL",
      "esm2_650m":"ESM-2 650M","cadd":"CADD","sift":"SIFT","polyphen2":"PolyPhen-2",
      "esm2_150m":"ESM-2 150M","esm1v":"ESM-1v","eve":"EVE"}
RNG=np.random.default_rng(31)

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        s=" ".join(str(x) for x in a); print(s,flush=True); self.f.write(s+"\n")
    def close(self): self.f.close()

def auc(y,s):
    y=np.asarray(y,float); s=np.asarray(s,float); ok=~np.isnan(s)
    y=y[ok]; s=s[ok]
    return roc_auc_score(y,s) if len(np.unique(y))>1 else np.nan

def bh(p):
    p=np.asarray(p,float); n=len(p); o=np.argsort(p); adj=np.empty(n); prev=1.0
    for rank,i in enumerate(o[::-1]):
        prev=min(prev,p[i]*n/(n-rank)); adj[i]=prev
    return adj

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP06_structure_report.txt")
    m=pd.read_csv(od/"master_scores.csv"); m=m[m.analysis_ok==True].copy()
    log("="*78); log("STEP 6 - STRUCTURAL ENVIRONMENT"); log("="*78)
    log("\nHYPOTHESIS: SaProt's advantage over sequence-only ESM-2 650M is")
    log("concentrated in BURIED, WELL-ORDERED residues and absent in exposed or")
    log("disordered ones. If true, it explains why the pooled delta of +0.016 was")
    log("too small to survive correction.")

    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.SASA import ShrakeRupley
    except Exception as e:
        log(f"\nFATAL: Biopython with SASA needed ({e}).")
        log("Install: pip install --user --upgrade biopython"); log.close(); return

    # ---------- [1] find structures ----------
    log("\n[1] LOCATING ALPHAFOLD STRUCTURES")
    upmap=dict(zip(m.target_gene,m.uniprot_id))
    pdbs=[p for p in root.rglob("*.pdb")]
    log(f"    {len(pdbs)} .pdb files on disk")
    gene_pdb={}
    for g,up in upmap.items():
        hits=[p for p in pdbs if up in p.name]
        if not hits:
            hits=[p for p in pdbs if re.search(rf"(^|[_-]){g}([_-]|\.)",p.name,re.I)
                  and "mouse" not in p.name.lower() and "zebra" not in p.name.lower()]
        if hits:
            gene_pdb[g]=max(hits,key=lambda p:p.stat().st_size)
    log(f"    matched {len(gene_pdb)}/{len(upmap)} genes")
    miss=sorted(set(upmap)-set(gene_pdb))
    log(f"    no structure: {miss}")

    # ---------- [2] per-residue features ----------
    log("\n[2] COMPUTING pLDDT AND RELATIVE SOLVENT ACCESSIBILITY")
    parser=PDBParser(QUIET=True); sr=ShrakeRupley()
    feats=[]
    for g,p in sorted(gene_pdb.items()):
        try:
            st=parser.get_structure(g,str(p)); model=st[0]
            sr.compute(model,level="R")
            n=0
            for chain in model:
                for res in chain:
                    rn=res.get_resname()
                    if rn not in THREE2ONE: continue
                    aa=THREE2ONE[rn]; pos=res.get_id()[1]
                    bs=[at.get_bfactor() for at in res]
                    plddt=float(np.mean(bs)) if bs else np.nan
                    sasa=float(getattr(res,"sasa",np.nan))
                    rsa=sasa/MAXASA.get(aa,np.nan) if MAXASA.get(aa) else np.nan
                    feats.append(dict(target_gene=g,position=pos,pdb_aa=aa,
                                      plddt=plddt,sasa=sasa,rsa=rsa))
                    n+=1
                break
            log(f"    {g:<8} {p.name[:44]:<46} {n} residues")
        except Exception as e:
            log(f"    {g:<8} FAILED: {type(e).__name__}: {e}")
    F=pd.DataFrame(feats)
    if F.empty:
        log("    no features computed - stopping"); log.close(); return
    F.to_csv(od/"step06_residue_features.csv",index=False)
    log(f"    total residues: {len(F)}")

    # ---------- [3] merge + validate ----------
    log("\n[3] MERGING TO VARIANTS")
    d=m.merge(F,on=["target_gene","position"],how="left")
    ok=d.rsa.notna().sum()
    log(f"    variants with structural features: {ok}/{len(d)} ({100*ok/len(d):.1f}%)")
    agree=(d.pdb_aa==d.wt_aa)
    log(f"    PDB residue matches wild-type: {int(agree.sum())}/{ok} "
        f"({100*agree.sum()/max(ok,1):.1f}%)")
    if agree.sum()/max(ok,1) < 0.95:
        log("    WARNING: poor agreement - structures may use different numbering.")
    d=d[agree | d.rsa.isna()].copy()

    d["burial"]=pd.cut(d.rsa,[-.01,.25,.50,10],labels=["buried","intermediate","exposed"])
    d["order"]=pd.cut(d.plddt,[0,70,90,100],labels=["disordered","flexible","ordered"])
    log("\n    strata (n / %pathogenic):")
    for col in ["burial","order"]:
        log(f"      {col}:")
        for k,gd in d.dropna(subset=[col]).groupby(col,observed=True):
            log(f"        {str(k):<14} n={len(gd):<5} %P={100*gd.label.mean():5.1f}  "
                f"genes={gd.target_gene.nunique()}")
    log("\n    NOTE: pathogenic variants are enriched in buried/ordered positions.")
    log("    AUC is computed WITHIN each stratum, so this enrichment does not")
    log("    itself create a difference between methods.")

    # ---------- [4] per-stratum performance ----------
    log("\n[4] METHOD PERFORMANCE BY STRUCTURAL ENVIRONMENT")
    METH=["alphamissense","saprot","esm2_650m","esm2_150m","esm1v","revel","cadd","eve"]
    pair=d[["s_saprot","s_esm2_650m"]].notna().all(axis=1)
    rows=[]
    for col in ["burial","order"]:
        log(f"\n  by {col} (SaProt/ESM-2 shared subset, n={int(pair.sum())}):")
        hdr=f"    {'stratum':<14}{'n':>6}"+"".join(f"{NICE[x][:11]:>12}" for x in METH)
        log(hdr)
        for k,gd in d[pair].dropna(subset=[col]).groupby(col,observed=True):
            line=f"    {str(k):<14}{len(gd):>6}"
            for x in METH:
                v=auc(gd.label.values,gd["s_"+x].values) if "s_"+x in gd else np.nan
                line+=f"{v:>12.3f}" if not np.isnan(v) else f"{'-':>12}"
                rows.append(dict(strat=col,stratum=str(k),method=NICE[x],n=len(gd),auc=v))
            log(line)
    pd.DataFrame(rows).to_csv(od/"step06_auc_by_stratum.csv",index=False)

    # ---------- [5] THE KEY TEST ----------
    log("\n[5] KEY TEST: DOES THE STRUCTURE-AWARE ADVANTAGE DEPEND ON ENVIRONMENT?")
    log("    delta = AUC(SaProt) - AUC(ESM-2 650M), computed within each stratum.")
    log("    CI from GENE-cluster bootstrap (1000 iters), so gene structure respected.")
    res=[]
    for col in ["burial","order"]:
        log(f"\n  {col}")
        log(f"    {'stratum':<14}{'n':>6}{'saprot':>9}{'esm2':>9}{'delta':>9}"
            f"{'CI_lo':>9}{'CI_hi':>9}{'p_gene':>9}")
        for k,gd in d[pair].dropna(subset=[col]).groupby(col,observed=True):
            genes=np.array(sorted(gd.target_gene.unique()))
            idx={g:gd.index[gd.target_gene==g].values for g in genes}
            boot=[]
            for _ in range(1000):
                pick=RNG.choice(genes,size=len(genes),replace=True)
                s=gd.loc[np.concatenate([idx[g] for g in pick])]
                if s.label.nunique()<2: continue
                boot.append(auc(s.label.values,s.s_saprot.values)
                            -auc(s.label.values,s.s_esm2_650m.values))
            boot=np.array(boot,float); boot=boot[~np.isnan(boot)]
            asp=auc(gd.label.values,gd.s_saprot.values)
            aes=auc(gd.label.values,gd.s_esm2_650m.values)
            lo,hi=np.percentile(boot,2.5),np.percentile(boot,97.5)
            pg_=2*min((boot<=0).mean(),(boot>=0).mean()); pg_=max(pg_,1/len(boot))
            log(f"    {str(k):<14}{len(gd):>6}{asp:>9.3f}{aes:>9.3f}{asp-aes:>+9.3f}"
                f"{lo:>+9.3f}{hi:>+9.3f}{pg_:>9.4f}")
            res.append(dict(strat=col,stratum=str(k),n=len(gd),n_genes=len(genes),
                            saprot=asp,esm2=aes,delta=asp-aes,lo=lo,hi=hi,p=pg_))
    R=pd.DataFrame(res)
    R["bh_q"]=bh(R.p.values); R["sig_bh10"]=R.bh_q<0.10
    R.to_csv(od/"step06_structure_advantage.csv",index=False)
    log("\n  after Benjamini-Hochberg correction:")
    log(R[["strat","stratum","n","delta","lo","hi","p","bh_q","sig_bh10"]]
          .round(4).to_string(index=False))

    # ---------- [6] per-gene paired ----------
    log("\n[6] PER-GENE PAIRED TEST WITHIN EACH STRATUM (gene = unit)")
    pr=[]
    for col in ["burial","order"]:
        for k,gd in d[pair].dropna(subset=[col]).groupby(col,observed=True):
            per=[]
            for g,gg in gd.groupby("target_gene"):
                if (gg.label==1).sum()<5 or (gg.label==0).sum()<5: continue
                a1=auc(gg.label.values,gg.s_saprot.values)
                a2=auc(gg.label.values,gg.s_esm2_650m.values)
                if not (np.isnan(a1) or np.isnan(a2)): per.append((g,a1-a2))
            if len(per)<4:
                log(f"    {col}/{k}: only {len(per)} eligible genes - skipped"); continue
            dl=np.array([x[1] for x in per])
            try: _,pv=stats.wilcoxon(dl)
            except Exception: pv=np.nan
            log(f"    {col}/{str(k):<14} genes={len(per):<3} median delta={np.median(dl):+.4f}  "
                f"wins={int((dl>0).sum())}/{len(dl)}  Wilcoxon p={pv:.4f}")
            pr.append(dict(strat=col,stratum=str(k),n_genes=len(per),
                           median_delta=float(np.median(dl)),
                           wins=int((dl>0).sum()),p=pv))
    if pr: pd.DataFrame(pr).to_csv(od/"step06_per_gene_by_stratum.csv",index=False)

    # ---------- [7] verdict ----------
    log("\n[7] VERDICT")
    b=R[R.strat=="burial"].set_index("stratum")
    if {"buried","exposed"} <= set(b.index):
        db,de=b.loc["buried","delta"],b.loc["exposed","delta"]
        log(f"    buried delta={db:+.4f}   exposed delta={de:+.4f}   "
            f"difference={db-de:+.4f}")
        if db>de and b.loc["buried","lo"]>0:
            log("    => SUPPORTED: structural input helps specifically at buried")
            log("       positions. This is a mechanistic finding and explains why")
            log("       the pooled comparison was underpowered.")
        elif db>de:
            log("    => Directionally consistent but the buried CI includes zero.")
            log("       Report as a trend, not a confirmed effect.")
        else:
            log("    => NOT SUPPORTED. The advantage is not concentrated in buried")
            log("       positions. Report this as a tested-and-rejected explanation.")
    log("\nWROTE: step06_residue_features.csv, step06_auc_by_stratum.csv,")
    log("       step06_structure_advantage.csv, step06_per_gene_by_stratum.csv")
    log.close()

if __name__=="__main__": main()

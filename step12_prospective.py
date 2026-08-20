#!/usr/bin/env python3
"""STEP 12 - true prospective validation (critique item 4).
Downloads an ARCHIVED ClinVar release and identifies variants absent from it,
giving a genuinely held-out test of training-data overlap for supervised methods."""
import argparse, gzip, re, urllib.request, shutil, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

AA3={'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G',
     'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S',
     'Thr':'T','Trp':'W','Tyr':'Y','Val':'V','Sec':'U','Pyl':'O'}
PROT=re.compile(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})')
BASE="https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/"
RNG=np.random.default_rng(233)

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
    for r,i in enumerate(o[::-1]):
        prev=min(prev,p[i]*n/(n-r)); adj[i]=prev
    return adj

def download(url,dest,log):
    if dest.exists() and dest.stat().st_size>1e7:
        log(f"    already present: {dest.name} ({dest.stat().st_size/1e6:.0f} MB)")
        return True
    log(f"    downloading {url}")
    try:
        req=urllib.request.Request(url,headers={"User-Agent":"CancerPLMBench/1.0"})
        with urllib.request.urlopen(req,timeout=600) as r, open(dest,"wb") as f:
            shutil.copyfileobj(r,f)
        log(f"    saved {dest.name} ({dest.stat().st_size/1e6:.0f} MB)")
        return True
    except Exception as e:
        log(f"    FAILED: {type(e).__name__}: {e}")
        if dest.exists(): dest.unlink()
        return False

def parse_release(path, genes, log):
    """return set of (gene,wt,pos,mut) present in this archived release"""
    keep=set(); rows=0
    with gzip.open(path,"rt",encoding="utf-8",errors="replace") as fh:
        header=fh.readline().rstrip("\n").split("\t")
    cols=[c for c in ["Name","GeneSymbol","Assembly","ClinicalSignificance"] if c in header]
    for ch in pd.read_csv(path,sep="\t",usecols=cols,dtype=str,
                          chunksize=400_000,low_memory=False):
        if "Assembly" in ch.columns: ch=ch[ch["Assembly"]=="GRCh38"]
        ch=ch[ch["GeneSymbol"].isin(genes)]
        if not len(ch): continue
        ext=ch["Name"].fillna("").str.extract(PROT)
        ok=ext[0].notna()
        if ok.sum():
            g=ch.loc[ok,"GeneSymbol"].values
            wt=ext.loc[ok,0].map(AA3).values
            po=pd.to_numeric(ext.loc[ok,1],errors="coerce").values
            mu=ext.loc[ok,2].map(AA3).values
            for i in range(len(g)):
                if wt[i] and mu[i] and not np.isnan(po[i]):
                    keep.add((g[i],wt[i],int(po[i]),mu[i]))
        rows+=len(ch)
    log(f"    parsed {rows} gene-matched rows -> {len(keep)} distinct protein changes")
    return keep

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    ap.add_argument("--archive",default="variant_summary_2022-12.txt.gz",
                    help="archived release filename on the ClinVar FTP")
    ap.add_argument("--year",default="2022")
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    dd=root/"benchmark/data/clinvar"; dd.mkdir(parents=True,exist_ok=True)
    log=Tee(od/"STEP12_prospective_report.txt")
    m=pd.read_csv(od/"master_scores.csv")
    if "analysis_ok" in m.columns: m=m[m.analysis_ok==True].copy()
    log("="*78); log("STEP 12 - TRUE PROSPECTIVE VALIDATION (item 4)"); log("="*78)
    log("\nThe LastEvaluated analysis in the submitted manuscript tests robustness")
    log("to RE-CURATION, not training-set exclusion. This identifies variants")
    log("ABSENT from an archived ClinVar release, giving a genuinely held-out set.")
    log("\nRelevant cutoffs: AlphaMissense (Sep 2023), MetaRNN (2021),")
    log("PHACTboost (2024), ClinPred (2018), VARITY (2021), BayesDel (2016).")
    log("An archive from Dec 2022 predates AlphaMissense and PHACTboost.")

    genes=set(m.target_gene.unique())
    dest=dd/a.archive
    log(f"\n[1] ARCHIVED RELEASE: {a.archive}")
    if not download(BASE+a.year+"/"+a.archive,dest,log):
        log("\n    Could not download. Try listing available archives:")
        log(f"      curl -s {BASE} | grep -o 'variant_summary_[0-9-]*\\.txt\\.gz' | sort -u | tail -30")
        log("    then re-run with --archive <filename>")
        log.close(); return

    log("\n[2] PARSING ARCHIVED RELEASE")
    old=parse_release(dest,genes,log)

    log("\n[3] IDENTIFYING PROSPECTIVE VARIANTS")
    key=list(zip(m.target_gene,m.wt_aa,m.position.astype(int),m.mut_aa))
    m["in_old"]=[k in old for k in key]
    m["prospective"]=~m["in_old"]
    log(f"    present in {a.archive}: {int(m.in_old.sum())}")
    log(f"    ABSENT (prospective):   {int(m.prospective.sum())}")
    log(f"    prospective label balance: "
        f"P={int(m[m.prospective].label.sum())} "
        f"B={int((m[m.prospective].label==0).sum())} "
        f"({100*m[m.prospective].label.mean():.1f}% pathogenic)")
    log(f"    retrospective balance:     "
        f"P={int(m[m.in_old].label.sum())} "
        f"B={int((m[m.in_old].label==0).sum())} "
        f"({100*m[m.in_old].label.mean():.1f}% pathogenic)")
    log(f"    genes in prospective set: {m[m.prospective].target_gene.nunique()}")
    log("\n    per-gene prospective counts:")
    log(m[m.prospective].target_gene.value_counts().to_string())

    if m.prospective.sum()<200:
        log("\n    TOO FEW prospective variants for reliable evaluation.")
        log("    Try an older archive (e.g. variant_summary_2020-12.txt.gz).")
        log.close(); return

    # ---------- [4] performance ----------
    log("\n[4] PERFORMANCE: RETROSPECTIVE vs PROSPECTIVE")
    OWN={"s_alphamissense":"AlphaMissense","s_saprot":"SaProt",
         "s_esm2_650m":"ESM-2 650M","s_esm2_150m":"ESM-2 150M","s_esm1v":"ESM-1v",
         "s_revel":"REVEL","s_cadd":"CADD","s_sift":"SIFT",
         "s_polyphen2":"PolyPhen-2","s_eve":"EVE"}
    NEW={c:c.replace("s_new_","") for c in m.columns if c.startswith("s_new_")}
    CONS={"phylop_100way_vertebrate","phylop_470way_mammalian","phylop_17way_primate",
          "phastcons_100way_vertebrate","phastcons_470way_mammalian",
          "gerp_91_mammals","siphy_29way_logodds","gerppp_rs",
          "alphamissense","revel","eve","sift","polyphen2_hdiv"}
    NEW={k:v for k,v in NEW.items() if v not in CONS}
    ALL={**OWN,**NEW}
    SUPERVISED={"AlphaMissense","REVEL","CADD","PolyPhen-2","metarnn","phactboost",
                "clinpred","varity_r","varity_er","bayesdel_add_af","bayesdel_no_af",
                "vest4","gmvp","mvp","m_cap","metalr","metasvm","mutformer",
                "primateai","deogen2","mutpred","mpc","list_s2","mutationassessor"}
    UNSUP={"SaProt","ESM-2 650M","ESM-2 150M","ESM-1v","EVE","SIFT","esm1b",
           "provean","sift4g","fathmm"}

    ret=m[m.in_old]; pro=m[m.prospective]
    log(f"    {'method':<24}{'family':<14}{'retro':>8}{'prosp':>8}{'drop':>9}"
        f"{'n_pro':>7}")
    rows=[]
    for c,nm in ALL.items():
        if c not in m.columns: continue
        r_=ret.dropna(subset=[c]); p_=pro.dropna(subset=[c])
        if len(p_)<100 or p_.label.nunique()<2: continue
        ar=auc(r_.label.values,r_[c].values); ap_=auc(p_.label.values,p_[c].values)
        fam=("supervised" if nm in SUPERVISED else
             "unsupervised" if nm in UNSUP else "other")
        log(f"    {nm:<24}{fam:<14}{ar:>8.3f}{ap_:>8.3f}{ar-ap_:>+9.3f}{len(p_):>7}")
        rows.append(dict(method=nm,family=fam,retro=ar,prospective=ap_,
                         drop=ar-ap_,n_pro=len(p_)))
    D=pd.DataFrame(rows)
    D.to_csv(od/"step12_prospective_performance.csv",index=False)

    log("\n[5] DO SUPERVISED METHODS DROP MORE ON UNSEEN VARIANTS?")
    for f,g in D.groupby("family"):
        log(f"    {f:<14} n={len(g):<3} median drop={g['drop'].median():+.4f}  "
            f"range [{g['drop'].min():+.3f}, {g['drop'].max():+.3f}]")
    s=D[D.family=="supervised"]["drop"]; u=D[D.family=="unsupervised"]["drop"]
    if len(s)>=3 and len(u)>=3:
        _,p=stats.mannwhitneyu(s,u,alternative="greater")
        log(f"    Mann-Whitney (supervised drop > unsupervised): p={p:.4f}")
        log("    A significant result means supervised methods were partly")
        log("    memorising ClinVar labels, and their apparent lead is inflated.")

    log("\n[6] LEADING-TIER STABILITY ON PROSPECTIVE VARIANTS")
    top=D.sort_values("prospective",ascending=False).head(12)
    log(f"    {'rank':>5}  {'method':<24}{'prospective AUC':>18}{'retro rank':>12}")
    rr=D.sort_values("retro",ascending=False).reset_index(drop=True)
    rmap={r.method:i+1 for i,r in rr.iterrows()}
    for i,(_,r) in enumerate(top.iterrows()):
        log(f"    {i+1:>5}  {r.method:<24}{r.prospective:>18.4f}"
            f"{rmap.get(r.method,'-'):>12}")
    rho,pr=stats.spearmanr(D.retro,D.prospective)
    log(f"\n    rank correlation retro vs prospective: rho={rho:.3f} p={pr:.2e}")

    log("\n[7] AlphaMissense vs THE NEW LEADERS, PROSPECTIVE ONLY")
    for other in ["phactboost","metarnn"]:
        c1="s_alphamissense"; c2="s_new_"+other
        if c2 not in m.columns: continue
        both=pro.dropna(subset=[c1,c2])
        if len(both)<100: continue
        a1=auc(both.label.values,both[c1].values)
        a2=auc(both.label.values,both[c2].values)
        log(f"    n={len(both)}  AlphaMissense={a1:.4f}  {other}={a2:.4f}  "
            f"delta={a1-a2:+.4f}")
    log("    If PHACTboost/MetaRNN fall behind here but not overall, their")
    log("    near-parity in Step 11 reflected training overlap.")

    m.to_csv(od/"master_scores.csv",index=False)
    log("\n    master_scores.csv updated with in_old / prospective flags")
    log("\nWROTE: step12_prospective_performance.csv")
    log.close()

if __name__=="__main__": main()

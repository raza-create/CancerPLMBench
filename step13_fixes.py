#!/usr/bin/env python3
"""STEP 13 - fix outstanding defects and extend the disorder finding to the
full 36-predictor field, testing whether degradation tracks SUPERVISION or
EVOLUTIONARY DEPENDENCE."""
import argparse, gzip, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

AA3={'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G',
     'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S',
     'Thr':'T','Trp':'W','Tyr':'Y','Val':'V','Sec':'U','Pyl':'O'}
PROT=re.compile(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})')
RNG=np.random.default_rng(311)

# family assignment: what signal does each method actually use?
SUPERVISED={"AlphaMissense","REVEL","CADD","PolyPhen-2","metarnn","phactboost",
  "clinpred","varity_r","varity_er","bayesdel_add_af","bayesdel_no_af","vest4",
  "gmvp","mvp","m_cap","metalr","metasvm","mutformer","primateai","deogen2",
  "mutpred","mpc","list_s2","mutationassessor"}
UNSUP_PLM={"SaProt","ESM-2 650M","ESM-2 150M","ESM-1v","esm1b"}
UNSUP_MSA={"SIFT","sift4g","provean","fathmm","EVE"}
PURE_CONS={"phylop_100way_vertebrate","phylop_470way_mammalian","phylop_17way_primate",
  "phastcons_100way_vertebrate","phastcons_470way_mammalian","gerp_91_mammals",
  "siphy_29way_logodds"}

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

def fam_of(nm):
    if nm in SUPERVISED: return "supervised"
    if nm in UNSUP_PLM: return "pLM"
    if nm in UNSUP_MSA: return "MSA-based"
    if nm in PURE_CONS: return "pure conservation"
    return "other"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    dd=root/"benchmark/data/clinvar"
    log=Tee(od/"STEP13_fixes_report.txt")
    m=pd.read_csv(od/"master_scores.csv")
    if "analysis_ok" in m.columns: m=m[m.analysis_ok==True].copy()
    log("="*78); log("STEP 13 - DEFECT FIXES + DISORDER ACROSS 36 PREDICTORS"); log("="*78)

    # ---------- FIX 1: restore both prospective flags ----------
    log("\n[FIX 1] BOTH PROSPECTIVE FLAGS (2020 run overwrote 2022)")
    genes=set(m.target_gene.unique())
    for yr,fn in [("2022","variant_summary_2022-12.txt.gz"),
                  ("2020","variant_summary_2020-12.txt.gz")]:
        p=dd/fn
        if not p.exists(): log(f"    {fn} missing - skipped"); continue
        keep=set()
        with gzip.open(p,"rt",encoding="utf-8",errors="replace") as fh:
            hdr=fh.readline().rstrip("\n").split("\t")
        cols=[c for c in ["Name","GeneSymbol","Assembly"] if c in hdr]
        for ch in pd.read_csv(p,sep="\t",usecols=cols,dtype=str,
                              chunksize=400_000,low_memory=False):
            if "Assembly" in ch.columns: ch=ch[ch["Assembly"]=="GRCh38"]
            ch=ch[ch["GeneSymbol"].isin(genes)]
            if not len(ch): continue
            ext=ch["Name"].fillna("").str.extract(PROT); ok=ext[0].notna()
            if ok.sum():
                g=ch.loc[ok,"GeneSymbol"].values
                wt=ext.loc[ok,0].map(AA3).values
                po=pd.to_numeric(ext.loc[ok,1],errors="coerce").values
                mu=ext.loc[ok,2].map(AA3).values
                for i in range(len(g)):
                    if wt[i] and mu[i] and not np.isnan(po[i]):
                        keep.add((g[i],wt[i],int(po[i]),mu[i]))
        key=list(zip(m.target_gene,m.wt_aa,m.position.astype(int),m.mut_aa))
        m[f"prospective_{yr}"]=[k not in keep for k in key]
        log(f"    prospective_{yr}: {int(m[f'prospective_{yr}'].sum())} variants absent")
    if "prospective" in m.columns: m=m.drop(columns=["prospective","in_old"],errors="ignore")

    # ---------- FIX 2: annotate the uncorrected p-value ----------
    log("\n[FIX 2] ANNOTATING step12b_within_gene.csv")
    f=od/"step12b_within_gene.csv"
    if f.exists():
        w=pd.read_csv(f); w["bh_q"]=bh(w.p.values); w["sig_bh"]=w.bh_q<0.05
        w["WARNING"]="per-method tests all non-significant; family-level p=0.029 uncorrected"
        w.to_csv(f,index=False)
        log(f"    smallest per-method p = {w.p.min():.4f}; "
            f"{int(w.sig_bh.sum())}/{len(w)} significant after BH")
        log("    => the family-level p=0.0294 is not reportable")

    # ---------- MAIN: disorder across all predictors ----------
    log("\n[MAIN] DISORDER DEGRADATION ACROSS THE FULL PREDICTOR FIELD")
    log("    Does degradation track SUPERVISION (labels) or EVOLUTIONARY")
    log("    DEPENDENCE (reliance on alignment/sequence conservation)?")
    log("    Pure conservation scores (phyloP, phastCons, GERP, SiPhy) are the")
    log("    decisive test: they have no supervision at all.")
    sp=od/"step06e_structural_variants.csv"
    if not sp.exists():
        log("    step06e file missing - cannot proceed"); m.to_csv(od/"master_scores.csv",index=False); log.close(); return
    struct=pd.read_csv(sp)[["target_gene","position","plddt","rsa","pdb_aa"]]
    d=m.merge(struct,on=["target_gene","position"],how="inner")
    d=d[d.pdb_aa==d.wt_aa].copy()
    d["order"]=pd.cut(d.plddt,[0,70,90,100],labels=["disordered","flexible","ordered"])
    log(f"    structurally annotated: n={len(d)}  genes={d.target_gene.nunique()}")

    OWN={"s_alphamissense":"AlphaMissense","s_saprot":"SaProt","s_esm2_650m":"ESM-2 650M",
         "s_esm2_150m":"ESM-2 150M","s_esm1v":"ESM-1v","s_revel":"REVEL","s_cadd":"CADD",
         "s_sift":"SIFT","s_polyphen2":"PolyPhen-2","s_eve":"EVE"}
    DUP={"alphamissense","revel","eve","sift","polyphen2_hdiv","gerppp_rs"}
    NEW={c:c.replace("s_new_","") for c in d.columns
         if c.startswith("s_new_") and c.replace("s_new_","") not in DUP}
    ALL={**OWN,**NEW}

    pairs=[]
    for g,gd in d.groupby("target_gene"):
        for lab in [0,1]:
            o=gd[(gd.order=="ordered")&(gd.label==lab)]
            u=gd[(gd.order=="disordered")&(gd.label==lab)]
            n=min(len(o),len(u))
            if n>=3: pairs.append((g,lab,n,o,u))
    log(f"    matched pair sets: {len(pairs)}  total pairs: {sum(p[2] for p in pairs)}")

    log(f"\n    {'method':<26}{'family':<20}{'drop':>9}{'CI_lo':>9}{'CI_hi':>9}{'p':>9}")
    B={}; rows=[]
    for c,nm in ALL.items():
        if c not in d.columns or d[c].notna().sum()<300: continue
        v=[]
        for _ in range(600):
            O=pd.concat([o.sample(n,random_state=int(RNG.integers(1e9)))
                         for _,_,n,o,_ in pairs])
            U=pd.concat([u.sample(n,random_state=int(RNG.integers(1e9)))
                         for _,_,n,_,u in pairs])
            ao=auc(O.label.values,O[c].values); au_=auc(U.label.values,U[c].values)
            if not (np.isnan(ao) or np.isnan(au_)): v.append(ao-au_)
        if len(v)<100: continue
        v=np.array(v); B[nm]=v
        lo,hi=np.percentile(v,2.5),np.percentile(v,97.5)
        p=max(2*min((v<=0).mean(),(v>=0).mean()),1/len(v))
        fam=fam_of(nm)
        log(f"    {nm:<26}{fam:<20}{v.mean():>+9.4f}{lo:>+9.4f}{hi:>+9.4f}{p:>9.4f}")
        rows.append(dict(method=nm,family=fam,drop=v.mean(),lo=lo,hi=hi,p=p))
    R=pd.DataFrame(rows)
    if len(R):
        R["bh_q"]=bh(R.p.values); R["sig"]=R.bh_q<0.05
        R.to_csv(od/"step13_disorder_all_methods.csv",index=False)
        log(f"\n    {int(R.sig.sum())}/{len(R)} methods with significant drop after BH")

        log("\n    BY FAMILY:")
        log(f"    {'family':<20}{'n':>4}{'median drop':>14}{'range':>26}")
        for f_,g in R.groupby("family"):
            log(f"    {f_:<20}{len(g):>4}{g['drop'].median():>+14.4f}"
                f"   [{g['drop'].min():+.4f}, {g['drop'].max():+.4f}]")

        log("\n    KEY CONTRASTS:")
        def cmp(f1,f2):
            x=R[R.family==f1]["drop"]; y=R[R.family==f2]["drop"]
            if len(x)<3 or len(y)<3: return
            _,p=stats.mannwhitneyu(x,y)
            log(f"      {f1:<20} vs {f2:<20} medians {x.median():+.4f} / "
                f"{y.median():+.4f}   p={p:.4f}")
        cmp("pLM","supervised"); cmp("MSA-based","supervised")
        cmp("pure conservation","supervised"); cmp("pLM","MSA-based")
        cmp("pLM","pure conservation")

        log("\n    INTERPRETATION KEY:")
        log("      If pure conservation AND pLMs both degrade more than supervised")
        log("      methods -> the mechanism is EVOLUTIONARY SIGNAL, not supervision.")
        log("      If only pLMs degrade -> the mechanism is specific to pLM")
        log("      representations.")
        log("      If nothing separates -> report per-method drops only.")

        pc=R[R.family=="pure conservation"]["drop"]; su=R[R.family=="supervised"]["drop"]
        pl=R[R.family=="pLM"]["drop"]
        if len(pc)>=3 and len(su)>=3 and len(pl)>=3:
            _,p1=stats.mannwhitneyu(pc,su); _,p2=stats.mannwhitneyu(pl,su)
            if p1<0.05 and p2<0.05 and pc.median()>su.median() and pl.median()>su.median():
                log("\n      => EVOLUTIONARY-SIGNAL MECHANISM SUPPORTED.")
                log("         Methods relying on evolutionary constraint - whether raw")
                log("         conservation or learned by a language model - lose more")
                log("         accuracy in disordered regions than methods trained on")
                log("         labelled variants. This explains why pLM-vs-supervised")
                log("         comparisons depend on benchmark structural composition.")
            elif p2<0.05 and pl.median()>su.median():
                log("\n      => pLM-SPECIFIC effect; conservation scores do not follow.")
            else:
                log("\n      => No clean family separation; report per-method only.")

    m.to_csv(od/"master_scores.csv",index=False)
    log(f"\n[FIX 3] master_scores.csv rewritten with both prospective flags: {m.shape}")

    log("\n[FIX 4] MANUSCRIPT EDITS STILL REQUIRED (not code)")
    log("    (a) ProtT5 Methods sentence -> use the Step 7f wording")
    log("    (b) WT1 lacks SaProt/AlphaMissense - state it")
    log("    (c) four '[? ]' citations: SaProt = Su et al. ICLR 2024;")
    log("        DeLong et al. 1988 Biometrics 44:837-845 (x2)")
    log("    (d) real GitHub URL + matching Data Availability Statement")
    log("\nWROTE: step13_disorder_all_methods.csv")
    log.close()

if __name__=="__main__": main()

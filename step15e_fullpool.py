#!/usr/bin/env python3
"""STEP 15e - rebuild the attention control pool from the full dbNSFP cache.
step15d could only draw controls from positions carrying a benchmark variant,
capping the analysis at 7 proteins. The Step 10 cache has phyloP for every
dbNSFP-annotated position in these genes, giving a proper control pool."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats

RNG=np.random.default_rng(1031); WINDOW=1022

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        s=" ".join(str(x) for x in a); print(s,flush=True); self.f.write(s+"\n")
    def close(self): self.f.close()

def parse_fasta(t): return "".join(l.strip() for l in t.splitlines() if not l.startswith(">"))

def find_seq(root,gene,up):
    c=list((root/"benchmark/data/sequences_fixed").glob(f"{gene}_*.fasta")) \
        if (root/"benchmark/data/sequences_fixed").exists() else []
    c+=[p for p in root.rglob(f"*{up}*human*.fasta")]+[p for p in root.rglob(f"*{gene}*human*.fasta")]
    for p in c:
        n=p.name.lower()
        if "3di" in n or "mouse" in n or "zebra" in n: continue
        s=parse_fasta(p.read_text())
        if len(set(s.upper()))>=15: return s
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    ap.add_argument("--tol",type=float,default=0.25)
    ap.add_argument("--max-len",type=int,default=1400)
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP15E_fullpool_report.txt")
    m=pd.read_csv(od/"master_scores.csv")
    if "analysis_ok" in m.columns: m=m[m.analysis_ok==True].copy()
    log("="*78); log("STEP 15e - FULL CONTROL POOL FROM dbNSFP CACHE"); log("="*78)
    log("\nPROBLEM: step15d drew controls only from positions carrying a benchmark")
    log("variant. PTEN had 114 pathogenic positions but a pool of 6, so only 7")
    log("proteins could be analysed.")
    log("FIX: the Step 10 dbNSFP cache covers every annotated position in these")
    log("genes, giving control pools two orders of magnitude larger.")

    # ---------- build the pool ----------
    cf=od/"step10_dbnsfp_cache.json"
    if not cf.exists():
        log("step10_dbnsfp_cache.json not found - run step10 first"); log.close(); return
    cache=json.loads(cf.read_text())
    rows=[]
    for g,recs in cache.items():
        for r in recs:
            v=None
            for k in ("phylop_100way_vertebrate","phylop.100way_vertebrate",
                      "phylop_100way_vertebrate_score"):
                if k in r and r[k] is not None: v=r[k]; break
            if v is None: continue
            rows.append((g,r.get("position"),v))
    P=pd.DataFrame(rows,columns=["target_gene","position","phylop"])
    P["position"]=pd.to_numeric(P.position,errors="coerce")
    P["phylop"]=pd.to_numeric(P.phylop,errors="coerce")
    P=P.dropna().astype({"position":int})
    P=P.groupby(["target_gene","position"],as_index=False).phylop.median()
    log(f"\npositions with phyloP from cache: {len(P)} "
        f"across {P.target_gene.nunique()} genes")
    log(f"    per-gene: " +
        ", ".join(f"{g}={n}" for g,n in P.target_gene.value_counts().head(8).items()))

    path=(m[m.label==1].groupby(["target_gene","position"],as_index=False)
            .agg(uniprot=("uniprot_id","first")))
    path["is_path"]=True
    ben=set(zip(m[m.label==1].target_gene,m[m.label==1].position))
    P["is_path"]=[(g,p) in ben for g,p in zip(P.target_gene,P.position)]
    log(f"    pathogenic positions in pool: {int(P.is_path.sum())}")
    log(f"    control candidates: {int((~P.is_path).sum())}")

    import esm as esm_lib
    dev="cuda" if torch.cuda.is_available() else "cpu"
    model,alph=getattr(esm_lib.pretrained,"esm2_t33_650M_UR50D")()
    model=model.eval().to(dev); bc=alph.get_batch_converter(); NL=model.num_layers
    upmap=dict(zip(m.target_gene,m.uniprot_id))

    log(f"\n[1] STRICT MATCHING (tolerance {a.tol} phyloP units, no replacement)")
    log(f"    {'gene':<9}{'nPpos':>7}{'pool':>8}{'matched':>9}"
        f"{'phyloP P':>10}{'phyloP C':>10}{'diff':>8}")
    per={l:[] for l in range(NL)}; meta=[]
    for g,gd in P.groupby("target_gene"):
        Pp=gd[gd.is_path]; pool=gd[~gd.is_path]
        if len(Pp)<5 or len(pool)<20: continue
        seq=find_seq(root,g,upmap.get(g,""))
        if seq is None or len(seq)>a.max_len: continue
        L0=min(len(seq),WINDOW)
        Pp=Pp[Pp.position<=L0]; pool=pool[pool.position<=L0]
        if len(Pp)<5 or len(pool)<20: continue
        avail=pool.copy(); mp=[]
        for _,r in Pp.sort_values("phylop",ascending=False).iterrows():
            if not len(avail): break
            dst=(avail.phylop-r.phylop).abs(); j=dst.idxmin()
            if dst.loc[j]>a.tol: continue
            mp.append((int(r.position),int(avail.loc[j,"position"]),
                       float(r.phylop),float(avail.loc[j,"phylop"])))
            avail=avail.drop(j)
        if len(mp)<8:
            log(f"    {g:<9}{len(Pp):>7}{len(pool):>8}{len(mp):>9}   too few"); continue
        store={}; hooks=[]
        def mk(li):
            def fn(mod,inp,outp):
                at=outp[1] if isinstance(outp,tuple) and len(outp)>1 else None
                if at is None or at.dim()!=4: return
                store[li]=at.detach().sum(dim=-2)[:,0,:].float().cpu().numpy()
            return fn
        for li,lm in enumerate(model.layers): hooks.append(lm.register_forward_hook(mk(li)))
        _,_,toks=bc([("x",seq[:WINDOW])]); toks=toks.to(dev)
        with torch.no_grad(): model(toks,need_head_weights=True,return_contacts=False)
        for h in hooks: h.remove()
        if len(store)<NL: continue
        recv=np.stack([store[k] for k in sorted(store)],axis=0)[:,:,1:-1]
        Lr=recv.shape[2]; store.clear()
        if dev=="cuda": torch.cuda.empty_cache()
        mp=[x for x in mp if x[0]<=Lr and x[1]<=Lr]
        if len(mp)<8: continue
        pv=np.array([x[2] for x in mp]); cv=np.array([x[3] for x in mp])
        log(f"    {g:<9}{len(Pp):>7}{len(pool):>8}{len(mp):>9}"
            f"{pv.mean():>10.2f}{cv.mean():>10.2f}{pv.mean()-cv.mean():>+8.3f}")
        meta.append(dict(gene=g,n_path=len(Pp),pool=len(pool),n_matched=len(mp),
                         diff=pv.mean()-cv.mean()))
        for l in range(NL):
            layer=recv[l].mean(axis=0)
            per[l].append((g,np.mean([layer[x[0]-1] for x in mp]),
                              np.mean([layer[x[1]-1] for x in mp])))
    del model
    if dev=="cuda": torch.cuda.empty_cache()

    M=pd.DataFrame(meta); M.to_csv(od/"step15e_matching.csv",index=False)
    n=len(M)
    log(f"\n    proteins analysed: {n}")
    if n:
        log(f"    total matched pairs: {int(M.n_matched.sum())}")
        log(f"    mean phyloP residual: {M['diff'].mean():+.4f}  "
            f"max |diff|: {M['diff'].abs().max():.3f}")
        try:
            _,pm=stats.wilcoxon(M['diff'])
            log(f"    residual imbalance test: p={pm:.4f}")
        except Exception: pass
    if n<8:
        log("\n    still fewer than 8 proteins - raise --tol or --max-len")

    log("\n[2] LAYER-RESOLVED ENRICHMENT")
    floor=2**-(n-1) if n>1 else 1.0
    log(f"    Wilcoxon floor at n={n}: p={floor:.2e}; "
        f"Bonferroni alpha={0.05/NL:.2e}  "
        f"({'reachable' if floor<0.05/NL else 'NOT reachable'})")
    rows=[]
    for l in range(NL):
        v=per[l]
        if len(v)<6: continue
        ap_=np.array([x[1] for x in v]); ac_=np.array([x[2] for x in v])
        try: _,p=stats.wilcoxon(ap_,ac_)
        except Exception: p=np.nan
        rows.append(dict(layer=l,n=len(v),n_up=int((ap_>ac_).sum()),
            median_enrichment=float(np.median(ap_/np.where(ac_==0,np.nan,ac_))),
            wilcoxon_p=p))
    R=pd.DataFrame(rows)
    if len(R):
        R["sig_bonf"]=R.wilcoxon_p<0.05/len(R)
        o=np.argsort(R.wilcoxon_p.values); adj=np.empty(len(R)); prev=1.0
        for r_,i in enumerate(o[::-1]):
            prev=min(prev,R.wilcoxon_p.values[i]*len(R)/(len(R)-r_)); adj[i]=prev
        R["bh_q"]=adj
        R.to_csv(od/"step15e_layers.csv",index=False)
        log(f"    {'layer':>6}{'up/n':>8}{'enrichment':>12}{'p':>11}{'BH q':>9}{'Bonf':>6}")
        for _,r in R.iterrows():
            log(f"    {int(r.layer):>6}{str(int(r.n_up))+'/'+str(int(r.n)):>8}"
                f"{r.median_enrichment:>12.3f}{r.wilcoxon_p:>11.2e}"
                f"{r.bh_q:>9.4f}{'*' if r.sig_bonf else '':>6}")
        log(f"\n    Bonferroni-significant: {int(R.sig_bonf.sum())}/{len(R)}")
        log(f"    BH-significant: {int((R.bh_q<0.05).sum())}/{len(R)}")
        log(f"    all-proteins-enriched layers: {int((R.n_up==R.n).sum())}/{len(R)}")
        log(f"    median enrichment {R.median_enrichment.median():.3f}, "
            f"max {R.median_enrichment.max():.3f} at layer "
            f"{int(R.loc[R.median_enrichment.idxmax(),'layer'])}")

    log("\n[3] COMPARISON ACROSS MATCHING SCHEMES")
    log("    scheme                          n_prot  median enrich  max")
    for nm,f in [("loose, benign controls","step15b_layers.csv"),
                 ("strict, benign-only","step15c_layers.csv"),
                 ("strict, variant positions","step15d_layers.csv"),
                 ("strict, full dbNSFP pool","step15e_layers.csv")]:
        p=od/f
        if not p.exists(): continue
        T=pd.read_csv(p)
        col="median_enrichment"
        if col in T.columns:
            log(f"    {nm:<32}{int(T.n.iloc[0]):>6}{T[col].median():>15.3f}"
                f"{T[col].max():>7.3f}")
    log("\n    Stability of the effect size across schemes is the primary evidence;")
    log("    p-values at small n are floor-limited and should not be emphasised.")
    log("\nWROTE: step15e_matching.csv, step15e_layers.csv")
    log.close()

if __name__=="__main__": main()

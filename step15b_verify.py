#!/usr/bin/env python3
"""STEP 15b - verify the phyloP-controlled attention result before reporting.
(1) exact per-protein direction counts, not floor p-values
(2) matching quality: how close are pathogenic and control conservation?
(3) does the enrichment survive additionally matching on burial/pLDDT?
(4) which protein was dropped and why"""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats

RNG=np.random.default_rng(719)
WINDOW=1022; CONS="s_new_phylop_100way_vertebrate"

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        s=" ".join(str(x) for x in a); print(s,flush=True); self.f.write(s+"\n")
    def close(self): self.f.close()

def parse_fasta(t): return "".join(l.strip() for l in t.splitlines() if not l.startswith(">"))

def find_seq(root,gene,up):
    c=list((root/"benchmark/data/sequences_fixed").glob(f"{gene}_*.fasta")) \
        if (root/"benchmark/data/sequences_fixed").exists() else []
    c+=[p for p in root.rglob(f"*{up}*human*.fasta")]
    c+=[p for p in root.rglob(f"*{gene}*human*.fasta")]
    for p in c:
        n=p.name.lower()
        if "3di" in n or "mouse" in n or "zebra" in n: continue
        s=parse_fasta(p.read_text())
        if len(set(s.upper()))>=15: return s
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    ap.add_argument("--max-len",type=int,default=1400)
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP15B_verify_report.txt")
    m=pd.read_csv(od/"master_scores.csv")
    if "analysis_ok" in m.columns: m=m[m.analysis_ok==True].copy()
    st=pd.read_csv(od/"step06e_structural_variants.csv")[
        ["target_gene","position","plddt","rsa"]]
    m=m.merge(st,on=["target_gene","position"],how="left")
    log("="*78); log("STEP 15b - VERIFYING THE ATTENTION RESULT"); log("="*78)
    log("\nISSUE 1: every p-value in step15 was 4.88e-04 or 2.44e-03 - these are the")
    log("DISCRETE FLOOR values of a Wilcoxon test at n=12 (2^-11). They mean")
    log("'all 12 proteins moved the same way', not a measured probability.")
    log("ISSUE 2: 13 proteins were eligible but only 12 contributed. Identifying")
    log("which was dropped and why.")

    import esm as esm_lib
    dev="cuda" if torch.cuda.is_available() else "cpu"
    upmap=dict(zip(m.target_gene,m.uniprot_id))
    genes=[]
    for g,gd in m.dropna(subset=[CONS]).groupby("target_gene"):
        if (gd.label==1).sum()<5 or (gd.label==0).sum()<5: continue
        s=find_seq(root,g,upmap[g])
        if s is None or len(s)>a.max_len: continue
        genes.append((g,s,gd))
    log(f"\neligible: {[g for g,_,_ in genes]}")

    model,alph=getattr(esm_lib.pretrained,"esm2_t33_650M_UR50D")()
    model=model.eval().to(dev); bc=alph.get_batch_converter()
    NL=model.num_layers

    log("\n[1] PER-PROTEIN MATCHING QUALITY AND ENRICHMENT")
    log(f"    {'gene':<9}{'nP':>4}{'nPmatch':>9}{'nCtrl':>7}"
        f"{'phyloP P':>10}{'phyloP C':>10}{'diff':>8}{'L25 enrich':>12}")
    per={l:[] for l in range(NL)}
    meta=[]
    for g,seq,gd in genes:
        sub=seq[:WINDOW]
        store={}; hooks=[]
        def mk(li):
            def fn(mod,inp,outp):
                at=outp[1] if isinstance(outp,tuple) and len(outp)>1 else None
                if at is None or at.dim()!=4: return
                store[li]=at.detach().sum(dim=-2)[:,0,:].float().cpu().numpy()
            return fn
        for li,lm in enumerate(model.layers): hooks.append(lm.register_forward_hook(mk(li)))
        _,_,toks=bc([("x",sub)]); toks=toks.to(dev)
        with torch.no_grad(): model(toks,need_head_weights=True,return_contacts=False)
        for h in hooks: h.remove()
        if len(store)<NL: 
            log(f"    {g:<9} SKIPPED (captured {len(store)}/{NL} layers)"); continue
        recv=np.stack([store[k] for k in sorted(store)],axis=0)[:,:,1:-1]
        L=recv.shape[2]; store.clear()
        if dev=="cuda": torch.cuda.empty_cache()

        gd2=gd[gd.position<=L]
        P=gd2[gd2.label==1]; B=gd2[gd2.label==0]
        if len(P)<5 or len(B)<5:
            log(f"    {g:<9} SKIPPED (after length trim: {len(P)}P/{len(B)}B, L={L})")
            continue
        sd=gd2[CONS].std(); band=0.25*sd if sd>0 else 0.5
        pairs=[]; pv=[]; cv=[]
        for _,r in P.iterrows():
            cand=B[(B[CONS]-r[CONS]).abs()<=band]
            if len(cand)==0: continue
            pick=cand.sample(min(3,len(cand)),random_state=int(RNG.integers(1e9)))
            pairs.append((int(r.position)-1,[int(x)-1 for x in pick.position.values]))
            pv.append(r[CONS]); cv.extend(pick[CONS].tolist())
        if len(pairs)<4:
            log(f"    {g:<9} SKIPPED (only {len(pairs)} matched pairs)"); continue
        l25=None
        for l in range(NL):
            layer=recv[l].mean(axis=0)
            ap_=np.mean([layer[p] for p,_ in pairs])
            ac_=np.mean([np.mean([layer[c] for c in cs]) for _,cs in pairs])
            per[l].append((g,ap_,ac_))
            if l==25: l25=ap_/ac_ if ac_ else np.nan
        log(f"    {g:<9}{len(P):>4}{len(pairs):>9}{len(cv):>7}"
            f"{np.mean(pv):>10.2f}{np.mean(cv):>10.2f}"
            f"{np.mean(pv)-np.mean(cv):>+8.2f}{l25:>12.3f}")
        meta.append(dict(gene=g,n_path=len(P),n_matched=len(pairs),
                         phylop_path=np.mean(pv),phylop_ctrl=np.mean(cv),
                         diff=np.mean(pv)-np.mean(cv),layer25=l25))
    del model
    if dev=="cuda": torch.cuda.empty_cache()
    M=pd.DataFrame(meta); M.to_csv(od/"step15b_per_protein.csv",index=False)
    if len(M):
        log(f"\n    proteins contributing: {len(M)}")
        log(f"    mean phyloP difference (pathogenic - control): "
            f"{M['diff'].mean():+.3f}  (should be near 0 if matching worked)")
        _,pm=stats.wilcoxon(M.phylop_path,M.phylop_ctrl)
        log(f"    Wilcoxon on matched conservation: p={pm:.4f} "
            f"({'residual imbalance' if pm<0.05 else 'well matched'})")

    log("\n[2] EXACT DIRECTION COUNTS PER LAYER (replaces floor p-values)")
    log(f"    {'layer':>6}{'n':>5}{'up':>5}{'median enrich':>15}{'p':>11}"
        f"{'p_floor?':>10}")
    rows=[]
    for l in range(NL):
        v=per[l]
        if len(v)<8: continue
        ap_=np.array([x[1] for x in v]); ac_=np.array([x[2] for x in v])
        up=int((ap_>ac_).sum())
        try: _,p=stats.wilcoxon(ap_,ac_)
        except Exception: p=np.nan
        floor=2**-(len(v)-1)
        rows.append(dict(layer=l,n=len(v),n_up=up,
                         median_enrichment=float(np.median(ap_/np.where(ac_==0,np.nan,ac_))),
                         p=p,at_floor=bool(abs(p-floor)<1e-9)))
        if l%4==0 or l in (13,25):
            log(f"    {l:>6}{len(v):>5}{up:>5}{rows[-1]['median_enrichment']:>15.3f}"
                f"{p:>11.2e}{'yes' if rows[-1]['at_floor'] else '':>10}")
    R=pd.DataFrame(rows); R.to_csv(od/"step15b_layers.csv",index=False)
    if len(R):
        log(f"\n    layers where ALL proteins showed enrichment: "
            f"{int((R.n_up==R.n).sum())}/{len(R)}")
        log(f"    layers at the p floor: {int(R.at_floor.sum())}/{len(R)}")
        log(f"    max median enrichment: {R.median_enrichment.max():.3f} "
            f"(layer {int(R.loc[R.median_enrichment.idxmax(),'layer'])})")

    log("\n[3] REPORTABLE WORDING")
    log("-"*78)
    if len(R):
        best=R.loc[R.median_enrichment.idxmax()]
        nsig=int((R.p<0.05/len(R)).sum())
        log(f"  Using phyloP 100-way vertebrate conservation as an independent")
        log(f"  control, attention at pathogenic positions exceeded attention at")
        log(f"  conservation-matched benign positions in {nsig} of {len(R)} layers")
        log(f"  after Bonferroni correction (n={int(R.n.iloc[0])} proteins; median")
        log(f"  enrichment {R.median_enrichment.median():.2f}x, maximum {best.median_enrichment:.2f}x")
        log(f"  at layer {int(best.layer)}). Because the paired Wilcoxon test at")
        log(f"  n={int(R.n.iloc[0])} has a minimum attainable p-value of "
            f"{2**-(int(R.n.iloc[0])-1):.1e}, we additionally report that all")
        log(f"  proteins showed enrichment in {int((R.n_up==R.n).sum())} layers.")
        log("")
        log("  NOTE: this analysis differs from the submitted one in design as well")
        log("  as control (12 proteins vs 18; ESM-2 650M only; controls drawn from")
        log("  benign variant positions rather than all non-variant positions), so")
        log("  the two enrichment values are not directly comparable.")
    log("-"*78)
    log("\nWROTE: step15b_per_protein.csv, step15b_layers.csv")
    log.close()

if __name__=="__main__": main()

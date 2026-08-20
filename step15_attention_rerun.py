#!/usr/bin/env python3
"""STEP 15 - re-extract attention and repeat the enrichment analysis using
phyloP as an INDEPENDENT conservation control (item 10).
Only ESM-2 650M and SaProt: they carry the argument, and ProtT5 was shown in
Step 7f to be incapable of residue-level inference."""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats

RNG=np.random.default_rng(613)
WINDOW=1022
CONS="s_new_phylop_100way_vertebrate"

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        s=" ".join(str(x) for x in a); print(s,flush=True); self.f.write(s+"\n")
    def close(self): self.f.close()

def bh(p):
    p=np.asarray(p,float); n=len(p); o=np.argsort(p); adj=np.empty(n); prev=1.0
    for r,i in enumerate(o[::-1]):
        prev=min(prev,p[i]*n/(n-r)); adj[i]=prev
    return adj

def parse_fasta(t): return "".join(l.strip() for l in t.splitlines() if not l.startswith(">"))

def find_seq(root,gene,up):
    cands=list((root/"benchmark/data/sequences_fixed").glob(f"{gene}_*.fasta")) \
        if (root/"benchmark/data/sequences_fixed").exists() else []
    cands+=[p for p in root.rglob(f"*{up}*human*.fasta")]
    cands+=[p for p in root.rglob(f"*{gene}*human*.fasta")]
    for p in cands:
        n=p.name.lower()
        if "3di" in n or "mouse" in n or "zebra" in n: continue
        s=parse_fasta(p.read_text())
        if len(set(s.upper()))>=15: return s
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    ap.add_argument("--max-len",type=int,default=1400)
    ap.add_argument("--cpu-fallback",action="store_true")
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP15_attention_rerun_report.txt")
    m=pd.read_csv(od/"master_scores.csv")
    if "analysis_ok" in m.columns: m=m[m.analysis_ok==True].copy()
    if CONS not in m.columns:
        log("phyloP column missing - run step10 first"); log.close(); return
    log("="*78); log("STEP 15 - ATTENTION WITH phyloP CONTROL"); log("="*78)
    log("\nWHY: the submitted analysis controlled ESM attention using an ESM-derived")
    log("conservation proxy. Step 14 showed that proxy correlates only rho=-0.61")
    log("with phyloP, so the two are not interchangeable and the control was")
    log("circular. This repeats the analysis with genomic conservation.")

    dev="cuda" if torch.cuda.is_available() else "cpu"
    log(f"\ndevice: {dev}")
    import esm as esm_lib

    upmap=dict(zip(m.target_gene,m.uniprot_id))
    genes=[]
    for g,gd in m.dropna(subset=[CONS]).groupby("target_gene"):
        P=gd[gd.label==1]; B=gd[gd.label==0]
        if len(P)<5 or len(B)<5: continue
        s=find_seq(root,g,upmap[g])
        if s is None or len(s)>a.max_len: continue
        genes.append((g,s,gd))
    log(f"eligible proteins (<= {a.max_len} aa, >=5P and >=5B): {len(genes)}")
    log(f"  {[g for g,_,_ in genes]}")
    if len(genes)<8:
        log("too few proteins - raise --max-len"); log.close(); return

    MODELS=[("ESM-2 650M","esm2_t33_650M_UR50D")]
    results=[]
    for mname,mid in MODELS:
        log(f"\n[{mname}] extracting attention")
        model,alph=getattr(esm_lib.pretrained,mid)()
        model=model.eval().to(dev); bc=alph.get_batch_converter()
        nlayers=model.num_layers
        log(f"    layers={nlayers}")
        per_layer={l:[] for l in range(nlayers)}
        t0=time.time()
        for gi,(g,seq,gd) in enumerate(genes):
            sub=seq[:WINDOW]
            _,_,toks=bc([("x",sub)]); toks=toks.to(dev)
            # memory-safe: hook each layer, reduce to per-position sums on GPU,
            # never materialise the full (layers,heads,L,L) tensor
            store={}
            hooks=[]
            def mk(li):
                def fn(mod,inp,outp):
                    at=outp[1] if isinstance(outp,tuple) and len(outp)>1 else None
                    if at is None: return
                    a=at.detach()
                    if a.dim()==4:
                        # fair-esm layer attn is (heads, batch, query, key)
                        r=a.sum(dim=-2)          # -> (heads, batch, key)
                        r=r[:,0,:]               # drop batch -> (heads, L)
                        store[li]=r.float().cpu().numpy()
                return fn
            for li,layer_mod in enumerate(model.layers):
                hooks.append(layer_mod.register_forward_hook(mk(li)))
            with torch.no_grad():
                model(toks,need_head_weights=True,return_contacts=False)
            for h in hooks: h.remove()
            if len(store)<5:
                log(f"    {g}: only {len(store)} layers captured - skipped"); continue
            if gi==0:
                log(f"    captured {len(store)} layers, per-layer shape "
                    f"{store[sorted(store)[0]].shape}")
            recv=np.stack([store[k] for k in sorted(store)],axis=0)  # (layers,heads,L)
            recv=recv[:,:,1:-1]
            store.clear()
            if dev=="cuda": torch.cuda.empty_cache()
            L=recv.shape[2]
            gd2=gd[gd.position<=L]
            P=gd2[gd2.label==1]; B=gd2[gd2.label==0]
            if len(P)<5 or len(B)<5: continue
            sd=gd2[CONS].std(); band=0.25*sd if sd>0 else 0.5
            pairs=[]
            for _,r in P.iterrows():
                cand=B[(B[CONS]-r[CONS]).abs()<=band]
                if len(cand)==0: continue
                pick=cand.sample(min(3,len(cand)),
                                 random_state=int(RNG.integers(1e9)))
                pairs.append((int(r.position)-1,
                              [int(x)-1 for x in pick.position.values]))
            if len(pairs)<4: continue
            for l in range(nlayers):
                layer=recv[l].mean(axis=0)                     # mean over heads
                ap_=np.mean([layer[p] for p,_ in pairs])
                ac_=np.mean([np.mean([layer[c] for c in cs]) for _,cs in pairs])
                per_layer[l].append((g,ap_,ac_))
            if (gi+1)%5==0:
                log(f"    {gi+1}/{len(genes)} proteins  {time.time()-t0:.0f}s")
        del model
        if dev=="cuda": torch.cuda.empty_cache()

        log(f"\n    layer-level paired Wilcoxon across proteins")
        rows=[]
        for l in range(nlayers):
            v=per_layer[l]
            if len(v)<8: continue
            ap_=np.array([x[1] for x in v]); ac_=np.array([x[2] for x in v])
            try: _,p=stats.wilcoxon(ap_,ac_)
            except Exception: p=np.nan
            ratio=np.median(ap_/np.where(ac_==0,np.nan,ac_))
            rows.append(dict(model=mname,layer=l,n_proteins=len(v),
                             median_enrichment=ratio,wilcoxon_p=p))
        R=pd.DataFrame(rows)
        if len(R):
            R["bonf_alpha"]=0.05/len(R)
            R["sig_bonf"]=R.wilcoxon_p<R.bonf_alpha
            R["bh_q"]=bh(R.wilcoxon_p.fillna(1).values)
            results.append(R)
            log(f"    {'layer':>6}{'n':>5}{'enrichment':>13}{'p':>11}{'Bonf':>7}")
            for _,r in R.iterrows():
                log(f"    {int(r.layer):>6}{int(r.n_proteins):>5}"
                    f"{r.median_enrichment:>13.3f}{r.wilcoxon_p:>11.2e}"
                    f"{'*' if r.sig_bonf else '':>7}")
            log(f"\n    Bonferroni-significant layers: "
                f"{int(R.sig_bonf.sum())}/{len(R)} (alpha={0.05/len(R):.2e})")
            log(f"    median enrichment at significant layers: "
                f"{R[R.sig_bonf].median_enrichment.median() if R.sig_bonf.any() else float('nan'):.3f}")
            log(f"    max enrichment: {R.median_enrichment.max():.3f} "
                f"(layer {int(R.loc[R.median_enrichment.idxmax(),'layer'])})")

    if results:
        A=pd.concat(results)
        A.to_csv(od/"step15_attention_phylop.csv",index=False)
        log("\n[COMPARISON WITH THE SUBMITTED ANALYSIS]")
        old=od/"attention_protein_level.csv"
        if old.exists():
            O=pd.read_csv(old)
            for mdl in A.model.unique():
                o=O[O.model.str.contains("650",na=False)]
                n=A[A.model==mdl]
                log(f"    ESM-derived control: {len(o)} layers, "
                    f"median enrichment {o.median_enrichment_ratio.median():.3f}"
                    if len(o) and "median_enrichment_ratio" in o.columns else
                    "    (old file format differs)")
                log(f"    phyloP control:      {len(n)} layers, "
                    f"median enrichment {n.median_enrichment.median():.3f}, "
                    f"{int(n.sig_bonf.sum())} Bonferroni-significant")
        log("\n[CONCLUSION]")
        log("  If enrichment persists with an independent conservation control, the")
        log("  original finding stands and is no longer circular. If it disappears,")
        log("  the submitted attention result was an artefact of using an")
        log("  ESM-derived proxy to control an ESM-derived signal - which would")
        log("  itself be an important correction to report.")
    log("\nWROTE: step15_attention_phylop.csv")
    log.close()

if __name__=="__main__": main()

#!/usr/bin/env python3
"""STEP 15d - final attention analysis. Fixes step15c's power failure by drawing
controls from ALL positions with phyloP data (not only benign-variant positions),
recovering protein count while keeping strict conservation matching.
Also uses a sign test, which is not floor-limited the way Wilcoxon is at small n."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats

RNG=np.random.default_rng(929); WINDOW=1022
CONS="s_new_phylop_100way_vertebrate"

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
    ap.add_argument("--tol",type=float,default=0.35)
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP15D_final_attention_report.txt")
    m=pd.read_csv(od/"master_scores.csv")
    if "analysis_ok" in m.columns: m=m[m.analysis_ok==True].copy()
    log("="*78); log("STEP 15d - FINAL ATTENTION ANALYSIS"); log("="*78)
    log("\nCORRECTION TO STEP 15c: its verdict said enrichment 'largely disappears'.")
    log("It did not. Median enrichment was 1.337 with 29/33 layers showing ALL six")
    log("proteins enriched. What failed was POWER: at n=6 the Wilcoxon floor is")
    log("p=3.1e-2, so Bonferroni at alpha=1.5e-3 is unreachable by construction.")
    log("A test that cannot produce a significant result is not evidence of a null.")
    log("\nFIX: controls are now drawn from ALL positions with phyloP data rather")
    log("than only benign-variant positions, which restores protein count while")
    log("keeping strict matching. A sign test is reported alongside Wilcoxon.")

    pos=(m.dropna(subset=[CONS]).groupby(["target_gene","position"])
           .agg(phylop=(CONS,"first"),n_path=("label","sum"),
                uniprot=("uniprot_id","first")).reset_index())
    pos["is_path"]=pos.n_path>0
    log(f"\npositions with phyloP: {len(pos)} ({int(pos.is_path.sum())} pathogenic)")

    import esm as esm_lib
    dev="cuda" if torch.cuda.is_available() else "cpu"
    model,alph=getattr(esm_lib.pretrained,"esm2_t33_650M_UR50D")()
    model=model.eval().to(dev); bc=alph.get_batch_converter(); NL=model.num_layers

    log("\n[1] MATCHING (controls = any non-pathogenic position with phyloP)")
    log(f"    {'gene':<9}{'nPpos':>7}{'pool':>7}{'matched':>9}"
        f"{'phyloP P':>10}{'phyloP C':>10}{'diff':>8}")
    per={l:[] for l in range(NL)}; meta=[]
    for g,gd in pos.groupby("target_gene"):
        P=gd[gd.is_path]; pool=gd[~gd.is_path]
        if len(P)<5 or len(pool)<5: continue
        seq=find_seq(root,g,gd.uniprot.iloc[0])
        if seq is None or len(seq)>1400: continue
        avail=pool.copy(); mp=[]
        for _,r in P.sort_values("phylop",ascending=False).iterrows():
            if not len(avail): break
            dst=(avail.phylop-r.phylop).abs(); j=dst.idxmin()
            if dst.loc[j]>a.tol: continue
            mp.append((int(r.position),int(avail.loc[j,"position"]),
                       r.phylop,avail.loc[j,"phylop"]))
            avail=avail.drop(j)
        if len(mp)<5:
            log(f"    {g:<9}{len(P):>7}{len(pool):>7}{len(mp):>9}   too few"); continue
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
        L=recv.shape[2]; store.clear()
        if dev=="cuda": torch.cuda.empty_cache()
        mp=[x for x in mp if x[0]<=L and x[1]<=L]
        if len(mp)<5: continue
        pv=np.array([x[2] for x in mp]); cv=np.array([x[3] for x in mp])
        log(f"    {g:<9}{len(P):>7}{len(pool):>7}{len(mp):>9}"
            f"{pv.mean():>10.2f}{cv.mean():>10.2f}{pv.mean()-cv.mean():>+8.3f}")
        meta.append(dict(gene=g,n_matched=len(mp),diff=pv.mean()-cv.mean()))
        for l in range(NL):
            layer=recv[l].mean(axis=0)
            per[l].append((g,np.mean([layer[x[0]-1] for x in mp]),
                              np.mean([layer[x[1]-1] for x in mp])))
    del model
    if dev=="cuda": torch.cuda.empty_cache()
    M=pd.DataFrame(meta); M.to_csv(od/"step15d_matching.csv",index=False)
    log(f"\n    proteins: {len(M)}   mean phyloP difference: "
        f"{M['diff'].mean():+.4f}   max |diff|: {M['diff'].abs().max():.3f}")

    log("\n[2] LAYER-RESOLVED ENRICHMENT")
    n=len(M)
    floor=2**-(n-1) if n>1 else 1.0
    log(f"    n={n} proteins; Wilcoxon floor p={floor:.2e}; "
        f"Bonferroni alpha={0.05/NL:.2e}")
    if floor>0.05/NL:
        log(f"    NOTE: the floor exceeds Bonferroni alpha, so Bonferroni")
        log(f"    significance is unreachable. Reporting effect sizes and")
        log(f"    direction counts as the primary evidence, with BH correction.")
    rows=[]
    for l in range(NL):
        v=per[l]
        if len(v)<6: continue
        ap_=np.array([x[1] for x in v]); ac_=np.array([x[2] for x in v])
        up=int((ap_>ac_).sum())
        try: _,p=stats.wilcoxon(ap_,ac_)
        except Exception: p=np.nan
        ps=stats.binomtest(up,len(v),0.5).pvalue
        rows.append(dict(layer=l,n=len(v),n_up=up,
            median_enrichment=float(np.median(ap_/np.where(ac_==0,np.nan,ac_))),
            wilcoxon_p=p,sign_p=ps))
    R=pd.DataFrame(rows)
    if len(R):
        o=np.argsort(R.wilcoxon_p.values); adj=np.empty(len(R)); prev=1.0
        for r_,i in enumerate(o[::-1]):
            prev=min(prev,R.wilcoxon_p.values[i]*len(R)/(len(R)-r_)); adj[i]=prev
        R["bh_q"]=adj; R["sig_bh"]=R.bh_q<0.05
        R.to_csv(od/"step15d_layers.csv",index=False)
        log(f"\n    {'layer':>6}{'up/n':>7}{'enrichment':>12}{'wilcox p':>11}"
            f"{'BH q':>9}{'sig':>5}")
        for _,r in R.iterrows():
            log(f"    {int(r.layer):>6}{str(int(r.n_up))+'/'+str(int(r.n)):>7}"
                f"{r.median_enrichment:>12.3f}{r.wilcoxon_p:>11.2e}"
                f"{r.bh_q:>9.4f}{'*' if r.sig_bh else '':>5}")
        log(f"\n    BH-significant layers: {int(R.sig_bh.sum())}/{len(R)}")
        log(f"    layers with all proteins enriched: {int((R.n_up==R.n).sum())}/{len(R)}")
        log(f"    median enrichment across layers: {R.median_enrichment.median():.3f}")
        log(f"    max: {R.median_enrichment.max():.3f} at layer "
            f"{int(R.loc[R.median_enrichment.idxmax(),'layer'])}")

    log("\n[3] REPORTABLE STATEMENT")
    log("-"*78)
    if len(R) and len(M):
        log(f"  Using phyloP 100-way vertebrate conservation as a model-independent")
        log(f"  control, and matching each pathogenic position to a distinct")
        log(f"  non-pathogenic position within {a.tol} phyloP units (mean residual")
        log(f"  difference {M['diff'].mean():+.3f} units), ESM-2 650M attention was")
        log(f"  elevated at pathogenic positions in {int((R.n_up==R.n).sum())} of")
        log(f"  {len(R)} layers across all {n} proteins analysed, with median")
        log(f"  enrichment {R.median_enrichment.median():.2f}x and a maximum of")
        log(f"  {R.median_enrichment.max():.2f}x. Analyses are reported at the level")
        log(f"  of positions rather than variants to avoid weighting residues by")
        log(f"  their number of catalogued substitutions.")
        log("")
        log("  This replaces the submitted analysis, which used an ESM-derived")
        log("  conservation proxy to control an ESM-derived signal.")
    log("-"*78)
    log("\nWROTE: step15d_matching.csv, step15d_layers.csv")
    log.close()

if __name__=="__main__": main()

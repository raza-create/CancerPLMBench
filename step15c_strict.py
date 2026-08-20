#!/usr/bin/env python3
"""STEP 15c - strict conservation matching for the attention analysis.
Fixes two defects in step15b: (a) residual conservation imbalance (+0.608,
p=0.002), (b) positions counted once per variant rather than once per residue."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats

RNG=np.random.default_rng(827); WINDOW=1022
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
    ap.add_argument("--tol",type=float,default=0.30,
                    help="max allowed phyloP difference for a match")
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP15C_strict_report.txt")
    m=pd.read_csv(od/"master_scores.csv")
    if "analysis_ok" in m.columns: m=m[m.analysis_ok==True].copy()
    log("="*78); log("STEP 15c - STRICT CONSERVATION MATCHING"); log("="*78)
    log("\nDEFECTS FIXED:")
    log("  (a) step15b left pathogenic positions +0.608 phyloP units more conserved")
    log("      than controls (Wilcoxon p=0.002). Enrichment could be conservation")
    log("      leaking through - the same flaw as the original ESM-proxy control.")
    log("  (b) positions were counted once per VARIANT, so a residue with 10")
    log("      pathogenic substitutions was weighted 10x. Now one entry per POSITION.")
    log(f"  Matching now uses a hard tolerance of {a.tol} phyloP units with greedy")
    log("  nearest-neighbour assignment WITHOUT replacement, and unmatched")
    log("  pathogenic positions are DISCARDED rather than loosely matched.")

    # collapse to positions
    pos=(m.dropna(subset=[CONS])
           .groupby(["target_gene","position"])
           .agg(phylop=(CONS,"first"),
                n_path=("label","sum"),
                n_var=("label","size"),
                uniprot=("uniprot_id","first")).reset_index())
    pos["is_path"]=pos.n_path>0
    pos["is_benign"]=pos.n_path==0
    log(f"\ncollapsed to {len(pos)} unique positions "
        f"({int(pos.is_path.sum())} pathogenic, {int(pos.is_benign.sum())} benign-only)")

    import esm as esm_lib
    dev="cuda" if torch.cuda.is_available() else "cpu"
    model,alph=getattr(esm_lib.pretrained,"esm2_t33_650M_UR50D")()
    model=model.eval().to(dev); bc=alph.get_batch_converter(); NL=model.num_layers

    log("\n[1] STRICT MATCHING PER PROTEIN")
    log(f"    {'gene':<9}{'nPpos':>7}{'nBpos':>7}{'matched':>9}"
        f"{'phyloP P':>10}{'phyloP C':>10}{'diff':>8}")
    per={l:[] for l in range(NL)}; meta=[]
    for g,gd in pos.groupby("target_gene"):
        P=gd[gd.is_path]; B=gd[gd.is_benign]
        if len(P)<5 or len(B)<5: continue
        seq=find_seq(root,g,gd.uniprot.iloc[0])
        if seq is None or len(seq)>1400: continue
        sub=seq[:WINDOW]
        # greedy nearest-neighbour without replacement
        avail=B.copy(); mp=[]
        for _,r in P.sort_values("phylop",ascending=False).iterrows():
            if not len(avail): break
            dsts=(avail.phylop-r.phylop).abs()
            j=dsts.idxmin()
            if dsts.loc[j]>a.tol: continue
            mp.append((int(r.position),int(avail.loc[j,"position"]),
                       r.phylop,avail.loc[j,"phylop"]))
            avail=avail.drop(j)
        if len(mp)<5:
            log(f"    {g:<9}{len(P):>7}{len(B):>7}{len(mp):>9}   too few matches")
            continue
        pv=np.array([x[2] for x in mp]); cv=np.array([x[3] for x in mp])
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
        if len(store)<NL: continue
        recv=np.stack([store[k] for k in sorted(store)],axis=0)[:,:,1:-1]
        L=recv.shape[2]; store.clear()
        if dev=="cuda": torch.cuda.empty_cache()
        mp=[x for x in mp if x[0]<=L and x[1]<=L]
        if len(mp)<5: continue
        pv=np.array([x[2] for x in mp]); cv=np.array([x[3] for x in mp])
        log(f"    {g:<9}{len(P):>7}{len(B):>7}{len(mp):>9}"
            f"{pv.mean():>10.2f}{cv.mean():>10.2f}{pv.mean()-cv.mean():>+8.3f}")
        meta.append(dict(gene=g,n_matched=len(mp),phylop_p=pv.mean(),
                         phylop_c=cv.mean(),diff=pv.mean()-cv.mean()))
        for l in range(NL):
            layer=recv[l].mean(axis=0)
            per[l].append((g,np.mean([layer[x[0]-1] for x in mp]),
                              np.mean([layer[x[1]-1] for x in mp])))
    del model
    if dev=="cuda": torch.cuda.empty_cache()

    M=pd.DataFrame(meta); M.to_csv(od/"step15c_matching.csv",index=False)
    log(f"\n    proteins: {len(M)}")
    if len(M):
        log(f"    mean phyloP difference: {M['diff'].mean():+.4f}")
        try:
            _,pm=stats.wilcoxon(M.phylop_p,M.phylop_c)
            log(f"    Wilcoxon on conservation: p={pm:.4f} "
                f"({'STILL IMBALANCED' if pm<0.05 else 'well matched'})")
        except Exception as e: log(f"    matching test failed: {e}")

    log("\n[2] LAYER-RESOLVED ENRICHMENT (strictly matched, position-level)")
    rows=[]
    for l in range(NL):
        v=per[l]
        if len(v)<6: continue
        ap_=np.array([x[1] for x in v]); ac_=np.array([x[2] for x in v])
        try: _,p=stats.wilcoxon(ap_,ac_)
        except Exception: p=np.nan
        rows.append(dict(layer=l,n=len(v),n_up=int((ap_>ac_).sum()),
                         median_enrichment=float(np.median(ap_/np.where(ac_==0,np.nan,ac_))),
                         p=p))
    R=pd.DataFrame(rows)
    if len(R):
        alpha=0.05/len(R); R["sig"]=R.p<alpha
        R.to_csv(od/"step15c_layers.csv",index=False)
        log(f"    {'layer':>6}{'n':>4}{'up':>4}{'enrichment':>12}{'p':>11}{'sig':>5}")
        for _,r in R.iterrows():
            log(f"    {int(r.layer):>6}{int(r.n):>4}{int(r.n_up):>4}"
                f"{r.median_enrichment:>12.3f}{r.p:>11.2e}{'*' if r.sig else '':>5}")
        log(f"\n    Bonferroni-significant: {int(R.sig.sum())}/{len(R)} "
            f"(alpha={alpha:.2e}, floor p={2**-(int(R.n.iloc[0])-1):.1e})")
        log(f"    median enrichment: {R.median_enrichment.median():.3f}  "
            f"max {R.median_enrichment.max():.3f} at layer "
            f"{int(R.loc[R.median_enrichment.idxmax(),'layer'])}")
        log(f"    layers with all proteins up: {int((R.n_up==R.n).sum())}/{len(R)}")

    log("\n[3] VERDICT")
    if len(M) and len(R):
        ok = abs(M['diff'].mean())<0.15
        log(f"    conservation balanced: {ok} (mean diff {M['diff'].mean():+.3f})")
        if ok and R.sig.sum()>len(R)*0.3:
            log("    => Enrichment SURVIVES strict, position-level, independently")
            log("       controlled matching. The finding is defensible.")
        elif not ok:
            log("    => Matching still imbalanced. Enrichment cannot be separated")
            log("       from residual conservation; report as a limitation.")
        else:
            log("    => Enrichment largely disappears under strict matching. The")
            log("       original result was driven by conservation imbalance - an")
            log("       important correction to report.")
    log("\nWROTE: step15c_matching.csv, step15c_layers.csv")
    log.close()

if __name__=="__main__": main()

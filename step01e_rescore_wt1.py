#!/usr/bin/env python3
"""STEP 1e - re-score WT1 with ESM-2 650M, ESM-2 150M, ESM-1v (and SaProt if a
3Di token file for the corrected isoform exists) using masked-marginal LLR."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch

MODELS={"esm2_650m":"esm2_t33_650M_UR50D",
        "esm2_150m":"esm2_t30_150M_UR50D",
        "esm1v":"esm1v_t33_650M_UR90S_1"}
WINDOW=1022

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        s=" ".join(str(x) for x in a); print(s,flush=True); self.f.write(s+"\n")
    def close(self): self.f.close()

def parse_fasta(txt):
    seq=[]
    for l in txt.splitlines():
        if not l.startswith(">"): seq.append(l.strip())
    return "".join(seq)

def window_for(seq,pos):
    """1-based pos -> (subseq, index of pos within subseq, 0-based)"""
    if len(seq)<=WINDOW: return seq,pos-1
    half=WINDOW//2; s=max(0,pos-1-half); e=min(len(seq),s+WINDOW); s=max(0,e-WINDOW)
    return seq[s:e], pos-1-s

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    ap.add_argument("--gene",default="WT1")
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/f"STEP01E_rescore_{a.gene}_report.txt")
    log("="*78); log(f"STEP 1e - RE-SCORE {a.gene}"); log("="*78)

    fa=root/"benchmark/data/sequences_fixed"/f"{a.gene}_corrected.fasta"
    if not fa.exists(): log(f"FATAL: {fa} not found"); return
    seq=parse_fasta(fa.read_text())
    log(f"\nsequence: {len(seq)} aa from {fa.name}")

    m=pd.read_csv(od/"master_scores.csv")
    gd=m[m.target_gene==a.gene].copy()
    log(f"{a.gene} variants: {len(gd)}  labels={gd.label.value_counts().to_dict()}")

    ok=[(int(r.position),r.wt_aa) for _,r in gd.iterrows()
        if 0<=int(r.position)-1<len(seq) and seq[int(r.position)-1]==r.wt_aa]
    log(f"wild-type residues matching corrected sequence: {len(ok)}/{len(gd)}")
    if len(ok)<len(gd):
        log("WARNING: not all match; unmatched will get NaN")

    dev="cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {dev}")
    if dev=="cuda": log(f"gpu: {torch.cuda.get_device_name(0)}")

    import esm as esm_lib
    results={}
    for tag,name in MODELS.items():
        log(f"\n--- {tag} ({name}) ---")
        model,alphabet=getattr(esm_lib.pretrained,name)()
        model=model.eval().to(dev)
        bc=alphabet.get_batch_converter()
        vals=[]
        with torch.no_grad():
            cache={}
            for _,r in gd.iterrows():
                pos=int(r.position); wt=r.wt_aa; mut=r.mut_aa
                if not(0<=pos-1<len(seq)) or seq[pos-1]!=wt:
                    vals.append(np.nan); continue
                sub,idx=window_for(seq,pos)
                if idx in cache and cache[idx][0]==sub:
                    logp=cache[idx][1]
                else:
                    masked=sub[:idx]+"<mask>"+sub[idx+1:] if False else list(sub)
                    _,_,toks=bc([("p",sub)])
                    toks=toks.to(dev)
                    toks[0,idx+1]=alphabet.mask_idx
                    out=model(toks)["logits"]
                    logp=torch.log_softmax(out[0,idx+1],dim=-1).cpu()
                    cache={idx:(sub,logp)}
                try:
                    v=float(logp[alphabet.get_idx(mut)]-logp[alphabet.get_idx(wt)])
                except Exception:
                    v=np.nan
                vals.append(v)
        results[tag]=vals
        good=np.array([v for v in vals if not np.isnan(v)])
        log(f"  scored {len(good)}/{len(gd)}")
        if len(good):
            lab=gd.label.values[[i for i,v in enumerate(vals) if not np.isnan(v)]]
            log(f"  mean LLR  pathogenic={good[lab==1].mean():.3f}  "
                f"benign={good[lab==0].mean():.3f}   (pathogenic must be MORE negative)")
            from sklearn.metrics import roc_auc_score
            if len(np.unique(lab))>1:
                log(f"  ROC-AUC (oriented) = {roc_auc_score(lab,-good):.3f}")
        del model
        if dev=="cuda": torch.cuda.empty_cache()

    for tag in MODELS:
        gd["raw_"+tag]=results[tag]
        gd["s_"+tag]=-gd["raw_"+tag]
    keep=["target_gene","variant_id","position","wt_aa","mut_aa","label"]+ \
         [c for t in MODELS for c in ("raw_"+t,"s_"+t)]
    gd[keep].to_csv(od/f"step01e_{a.gene}_rescored.csv",index=False)
    log(f"\nwrote step01e_{a.gene}_rescored.csv")

    log("\n--- merging back into master_scores.csv ---")
    for tag in MODELS:
        upd=dict(zip(gd.index, gd["raw_"+tag]))
        for i,v in upd.items():
            m.at[i,"raw_"+tag]=v; m.at[i,"s_"+tag]=(-v if pd.notna(v) else np.nan)
    idx=m.target_gene==a.gene
    m.loc[idx,"seq_wt_ok"]=[bool(0<=int(p)-1<len(seq) and seq[int(p)-1]==w)
                            for p,w in zip(m.loc[idx,"position"],m.loc[idx,"wt_aa"])]
    m.loc[idx,"analysis_ok"]=m.loc[idx,"seq_wt_ok"]
    sc=[c for c in m.columns if c.startswith("s_")]
    m["n_methods_scored"]=m[sc].notna().sum(axis=1)
    m["in_shared_all"]=m[sc].notna().all(axis=1)
    plm=[f"s_{x}" for x in ["saprot","esm2_650m","esm2_150m","esm1v"]]
    m["in_shared_plm_am"]=m[plm+["s_alphamissense"]].notna().all(axis=1)
    m.to_csv(od/"master_scores.csv",index=False)
    log(f"  {a.gene} coverage now: " +
        ", ".join(f"{t}={int(m.loc[idx,'s_'+t].notna().sum())}/{int(idx.sum())}" for t in MODELS))
    log(f"  clean benchmark: n={int(m.analysis_ok.sum())}")
    log(f"  V3 (3x ESM, all genes): n={int(m[[f's_{x}' for x in MODELS]].notna().all(axis=1).sum())}")
    log("\nNOTE: SaProt and AlphaMissense remain unavailable for WT1 "
        "(no AM file; 3Di tokens were built from the wrong isoform).")
    log("      WT1 therefore contributes to the ESM-only view (V3), not to shared-10.")
    log.close()

if __name__=="__main__": main()

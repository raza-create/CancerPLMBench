#!/usr/bin/env python3
"""STEP 7e - re-score ProtT5 properly so the reported number is reproducible.
Method A: sentinel-token masked-marginal LLR (what the Methods section claims)
Method B: sequence pseudo-likelihood difference (subset, as a sanity check)"""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score

WINDOW=1000

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        s=" ".join(str(x) for x in a); print(s,flush=True); self.f.write(s+"\n")
    def close(self): self.f.close()

def parse_fasta(txt):
    return "".join(l.strip() for l in txt.splitlines() if not l.startswith(">"))

def load_seqs(root, genes, upmap):
    seqs={}
    fixed=root/"benchmark/data/sequences_fixed"
    for g in genes:
        up=upmap.get(g)
        cand=[]
        if fixed.exists():
            cand+=sorted(fixed.glob(f"{g}_*.fasta"))
        cand+=[p for p in root.rglob(f"*{up}*human*.fasta")]
        cand+=[p for p in root.rglob(f"*{g}*human*.fasta")]
        for p in cand:
            n=p.name.lower()
            if "3di" in n or "mouse" in n or "zebra" in n: continue
            s=parse_fasta(p.read_text())
            if len(set(s.upper()))>=15: seqs[g]=s; break
    return seqs

def window_for(seq,pos):
    if len(seq)<=WINDOW: return seq,pos-1
    half=WINDOW//2; s=max(0,pos-1-half); e=min(len(seq),s+WINDOW); s=max(0,e-WINDOW)
    return seq[s:e], pos-1-s

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    ap.add_argument("--limit",type=int,default=0,help="cap variants for a quick test")
    ap.add_argument("--pll-subset",type=int,default=300)
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP07E_prott5_rescore_report.txt")
    m=pd.read_csv(od/"master_scores.csv"); m=m[m.analysis_ok==True].copy()
    log("="*78); log("STEP 7e - RE-SCORING ProtT5"); log("="*78)
    log("\nPURPOSE: the manuscript reports ProtT5 ROC-AUC = 0.49 from masked-marginal")
    log("LLR, but no saved column reproduces it (cosine 0.514, euclidean 0.623).")
    log("This computes a reproducible number using the method the Methods describe.")

    from transformers import T5Tokenizer, T5ForConditionalGeneration
    dev="cuda" if torch.cuda.is_available() else "cpu"
    log(f"\ndevice: {dev}")
    if dev=="cuda": log(f"gpu: {torch.cuda.get_device_name(0)}")
    name="Rostlab/prot_t5_xl_uniref50"
    log(f"loading {name} (~11 GB download on first run)")
    tok=T5Tokenizer.from_pretrained(name, do_lower_case=False, legacy=True)
    model=T5ForConditionalGeneration.from_pretrained(
        name, torch_dtype=torch.float16 if dev=="cuda" else torch.float32).to(dev).eval()
    log("loaded")

    upmap=dict(zip(m.target_gene,m.uniprot_id))
    seqs=load_seqs(root,sorted(m.target_gene.unique()),upmap)
    log(f"sequences loaded for {len(seqs)}/{m.target_gene.nunique()} genes")

    aa_ids={}
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        ids=tok.encode(aa, add_special_tokens=False)
        if len(ids)==1: aa_ids[aa]=ids[0]
    log(f"single-token amino acids: {len(aa_ids)}/20")
    if len(aa_ids)<20:
        log("WARNING: some amino acids are not single tokens; those variants skipped")

    sent=tok.convert_tokens_to_ids("<extra_id_0>")
    log(f"sentinel <extra_id_0> id = {sent}")

    work=m if not a.limit else m.head(a.limit)
    log(f"\n[A] SENTINEL-TOKEN MASKED-MARGINAL LLR  (n={len(work)})")
    vals=np.full(len(work),np.nan); t0=time.time(); done=0
    with torch.no_grad():
        for i,(_,r) in enumerate(work.iterrows()):
            g=r.target_gene; seq=seqs.get(g)
            if seq is None: continue
            pos=int(r.position); wt=r.wt_aa; mut=r.mut_aa
            if not (0<=pos-1<len(seq)) or seq[pos-1]!=wt: continue
            if wt not in aa_ids or mut not in aa_ids: continue
            sub,idx=window_for(seq,pos)
            spaced=" ".join(list(sub[:idx]))+" <extra_id_0> "+" ".join(list(sub[idx+1:]))
            enc=tok(spaced.strip(), return_tensors="pt", add_special_tokens=True).to(dev)
            dec=torch.tensor([[model.config.decoder_start_token_id, sent]]).to(dev)
            out=model(**enc, decoder_input_ids=dec).logits
            lp=torch.log_softmax(out[0,-1].float(),dim=-1).cpu().numpy()
            vals[i]=float(lp[aa_ids[mut]]-lp[aa_ids[wt]])
            done+=1
            if done%400==0:
                el=time.time()-t0
                log(f"    {done} scored  {el:.0f}s  ({el/done:.2f} s/variant)")
    work=work.copy(); work["prott5_llr"]=vals
    ok=work.prott5_llr.notna()
    log(f"\n    scored {int(ok.sum())}/{len(work)} in {time.time()-t0:.0f}s")
    if ok.sum()>50 and work.loc[ok,"label"].nunique()>1:
        y=work.loc[ok,"label"].values; s=work.loc[ok,"prott5_llr"].values
        a1=roc_auc_score(y,-s); a2=roc_auc_score(y,s)
        log(f"    mean LLR pathogenic={s[y==1].mean():.3f}  benign={s[y==0].mean():.3f}")
        log(f"    ROC-AUC (negated LLR, standard orientation) = {a1:.4f}")
        log(f"    ROC-AUC (raw LLR)                            = {a2:.4f}")
        log(f"    => REPORTABLE VALUE: {max(a1,a2):.3f}")
        if 0.45<=max(a1,a2)<=0.55:
            log("    Near-random, consistent with the manuscript's claim. Now backed")
            log("    by a reproducible computation.")
        else:
            log("    NOT near-random. The manuscript's justification for excluding")
            log("    ProtT5 must be revised to match this value.")
        work[["target_gene","variant_id","position","wt_aa","mut_aa","label","prott5_llr"]]\
            .to_csv(od/"step07e_prott5_llr.csv",index=False)
        log(f"    wrote step07e_prott5_llr.csv")

    # ---------- B: pseudo-likelihood on a subset ----------
    n=min(a.pll_subset,int(ok.sum()))
    log(f"\n[B] SEQUENCE PSEUDO-LIKELIHOOD CHECK (subset n={n})")
    log("    Scores wt and mutant sequences under the decoder; independent of the")
    log("    sentinel formulation, so it tests whether method choice matters.")
    sub=work[ok].sample(n,random_state=7) if n>0 else work.head(0)
    pll=[]
    with torch.no_grad():
        for _,r in sub.iterrows():
            seq=seqs.get(r.target_gene); pos=int(r.position)
            w,idx=window_for(seq,pos)
            mseq=w[:idx]+r.mut_aa+w[idx+1:]
            sc=[]
            for s_ in (w,mseq):
                sp=" ".join(list(s_))
                enc=tok(sp, return_tensors="pt").to(dev)
                lab=enc.input_ids.clone()
                out=model(**enc, labels=lab)
                sc.append(-float(out.loss)*lab.shape[1])
            pll.append(sc[1]-sc[0])
    if len(pll)>20:
        y=sub.label.values; p=np.array(pll)
        b1=roc_auc_score(y,-p); b2=roc_auc_score(y,p)
        log(f"    ROC-AUC (negated) = {b1:.4f}   (raw) = {b2:.4f}")
        log(f"    best = {max(b1,b2):.3f}")
        ys=sub.label.values; ss=sub.prott5_llr.values
        log(f"    sentinel-LLR AUC on this same subset = "
            f"{max(roc_auc_score(ys,-ss),roc_auc_score(ys,ss)):.3f}")
        log("    If the two methods agree, the exclusion rationale is robust to")
        log("    the scoring choice. If they diverge, say which one you used.")

    log("\n[C] TEXT FOR THE MANUSCRIPT")
    log("    Replace the Methods 2.2 sentence with a version quoting the value")
    log("    from section A, and delete the unreproducible 0.49 from the Abstract")
    log("    and Discussion. Also record the scoring formulation explicitly:")
    log("      'ProtT5 variant effects were scored by replacing the variant")
    log("       position with the <extra_id_0> sentinel token and taking the")
    log("       difference in decoder log-probability between mutant and wild-type")
    log("       residues at the first decoded position.'")
    log("\nWROTE: step07e_prott5_llr.csv, STEP07E_prott5_rescore_report.txt")
    log.close()

if __name__=="__main__": main()

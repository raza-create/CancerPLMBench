#!/usr/bin/env python3
"""STEP 7f - sanity-check the ProtT5 sentinel decoding before a full run.
If the setup is correct, the model should reconstruct the WILD-TYPE residue
with high probability at masked positions. If it cannot, the scoring is
meaningless and the exclusion is justified on mechanistic grounds."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        s=" ".join(str(x) for x in a); print(s,flush=True); self.f.write(s+"\n")
    def close(self): self.f.close()

def parse_fasta(t): return "".join(l.strip() for l in t.splitlines() if not l.startswith(">"))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP07F_prott5_sanity_report.txt")
    log("="*78); log("STEP 7f - ProtT5 SENTINEL DECODING SANITY CHECK"); log("="*78)
    log("\nWHY: step07e gave mean LLR pathogenic=-0.010 vs benign=-1.547, i.e.")
    log("ProtT5 scores pathogenic substitutions as MORE likely. That inverted")
    log("direction means either the decoding setup is wrong, or the model cannot")
    log("do residue-level inference at all. This distinguishes the two.")

    from transformers import T5Tokenizer, T5ForConditionalGeneration
    dev="cuda" if torch.cuda.is_available() else "cpu"
    tok=T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50",
                                    do_lower_case=False, legacy=True)
    model=T5ForConditionalGeneration.from_pretrained(
        "Rostlab/prot_t5_xl_uniref50",
        dtype=torch.float16 if dev=="cuda" else torch.float32).to(dev).eval()
    log(f"\ndevice={dev}  loaded")

    p=root/"benchmark/data/sequences"
    fa=None
    for c in root.rglob("*TP53*human*.fasta"):
        if "3di" not in c.name.lower() and "mouse" not in c.name.lower(): fa=c; break
    if fa is None:
        log("TP53 fasta not found"); log.close(); return
    seq=parse_fasta(fa.read_text())
    log(f"test protein: TP53 from {fa.name}, {len(seq)} aa")

    aa_ids={aa:tok.encode(aa,add_special_tokens=False)[0] for aa in "ACDEFGHIKLMNPQRSTVWY"}
    sent=tok.convert_tokens_to_ids("<extra_id_0>")
    start=model.config.decoder_start_token_id
    log(f"sentinel id={sent}  decoder_start={start}")

    log("\n[1] CAN THE MODEL RECONSTRUCT KNOWN RESIDUES?")
    log("    Mask 30 random positions; check whether the true residue ranks highly.")
    rng=np.random.default_rng(3)
    positions=rng.choice(np.arange(50,min(len(seq),350)),size=30,replace=False)

    variants={
        "A: dec=[start,sent], read last": lambda o: o[0,-1],
        "B: dec=[start], read last":      None,
        "C: dec=[start,sent], read idx1": lambda o: o[0,1] if o.shape[1]>1 else o[0,-1],
    }
    for tag in ["A: dec=[start,sent], read last","B: dec=[start], read last"]:
        ranks=[]; top1=0; probs=[]
        with torch.no_grad():
            for pos in positions:
                wt=seq[pos]
                if wt not in aa_ids: continue
                spaced=" ".join(list(seq[:pos]))+" <extra_id_0> "+" ".join(list(seq[pos+1:]))
                enc=tok(spaced.strip(),return_tensors="pt").to(dev)
                dec=torch.tensor([[start,sent]] if tag.startswith("A") else [[start]]).to(dev)
                out=model(**enc,decoder_input_ids=dec).logits
                lp=torch.log_softmax(out[0,-1].float(),dim=-1).cpu().numpy()
                aa_lp={k:lp[v] for k,v in aa_ids.items()}
                order=sorted(aa_lp,key=lambda k:-aa_lp[k])
                r=order.index(wt)+1
                ranks.append(r); top1+= (r==1)
                probs.append(float(np.exp(aa_lp[wt])/sum(np.exp(list(aa_lp.values())))))
        if ranks:
            log(f"\n    {tag}")
            log(f"      median rank of true residue: {np.median(ranks):.1f} / 20")
            log(f"      top-1 accuracy: {top1}/{len(ranks)} ({100*top1/len(ranks):.0f}%)")
            log(f"      mean P(true residue): {np.mean(probs):.3f} (random = 0.05)")
            log(f"      rank distribution: {sorted(ranks)}")
    log("\n    A working masked-LM recovers the true residue with median rank 1-3")
    log("    and top-1 accuracy well above 5%. If median rank is near 10, the")
    log("    model is not performing residue-level inference in this setup.")

    log("\n[2] COMPARISON: ESM-2 650M ON THE SAME POSITIONS")
    try:
        import esm as esm_lib
        em,alph=esm_lib.pretrained.esm2_t33_650M_UR50D()
        em=em.eval().to(dev); bc=alph.get_batch_converter()
        ranks=[]; top1=0
        with torch.no_grad():
            for pos in positions:
                wt=seq[pos]
                sub=seq[:1022] if len(seq)>1022 else seq
                if pos>=len(sub): continue
                _,_,toks=bc([("p",sub)]); toks=toks.to(dev)
                toks[0,pos+1]=alph.mask_idx
                out=em(toks)["logits"]
                lp=torch.log_softmax(out[0,pos+1].float(),dim=-1).cpu().numpy()
                aa_lp={aa:lp[alph.get_idx(aa)] for aa in aa_ids}
                order=sorted(aa_lp,key=lambda k:-aa_lp[k])
                r=order.index(wt)+1; ranks.append(r); top1+=(r==1)
        log(f"    ESM-2 650M median rank: {np.median(ranks):.1f} / 20")
        log(f"    top-1 accuracy: {top1}/{len(ranks)} ({100*top1/len(ranks):.0f}%)")
        log("    This is the reference for what correct residue-level inference")
        log("    looks like on the same positions.")
        del em
        if dev=="cuda": torch.cuda.empty_cache()
    except Exception as e:
        log(f"    ESM comparison unavailable: {e}")

    log("\n[3] VERDICT")
    log("    If ProtT5 cannot reconstruct wild-type residues while ESM-2 can, then")
    log("    ProtT5's inverted variant scores reflect a genuine incompatibility")
    log("    between span-corruption pretraining and single-residue scoring - which")
    log("    is exactly the manuscript's claim, now with evidence.")
    log("    In that case report:")
    log("      'ProtT5 recovered the wild-type residue at masked positions with")
    log("       median rank X of 20 (versus rank Y for ESM-2 650M), confirming that")
    log("       its span-corruption objective does not support single-residue")
    log("       masked-marginal inference. ProtT5 was therefore excluded from")
    log("       variant-effect scoring and retained only for attention analysis.'")
    log("    Do NOT report an ROC-AUC obtained by choosing whichever sign exceeded")
    log("    0.5; that is orientation fitting, not measurement.")
    log.close()

if __name__=="__main__": main()

#!/usr/bin/env python3
"""STEP 1c - sequence integrity: verify every variant's wild-type residue
matches the scored protein sequence; diagnose and repair isoform mismatches."""
import argparse, re, urllib.request
from pathlib import Path
import numpy as np, pandas as pd

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        s=" ".join(str(x) for x in a); print(s); self.f.write(s+"\n")
    def close(self): self.f.close()

def read_fasta_text(txt):
    """return {header: seq}"""
    out={}; h=None; buf=[]
    for line in txt.splitlines():
        if line.startswith(">"):
            if h is not None: out[h]="".join(buf)
            h=line[1:].strip(); buf=[]
        else: buf.append(line.strip())
    if h is not None: out[h]="".join(buf)
    return out

def find_fasta(root, uniprot, gene, hint):
    cands=[]
    if isinstance(hint,str) and hint:
        for p in [Path(hint), root/hint, root/"benchmark"/hint, root/"data"/hint]:
            if p.exists(): cands.append(p)
    for pat in (f"*{uniprot}*.fasta", f"*{gene}*human*.fasta", f"*{gene}*.fasta"):
        cands += sorted(root.rglob(pat))
    seen=set(); out=[]
    for c in cands:
        if c not in seen: seen.add(c); out.append(c)
    return out

def match_frac(seq, sub, shift=0):
    ok=tot=0
    for pos,wt in sub:
        i=pos-1+shift
        if 0<=i<len(seq):
            tot+=1; ok+= (seq[i]==wt)
    return (ok/tot if tot else 0.0), tot

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    ap.add_argument("--no-fetch",action="store_true",help="skip UniProt isoform download")
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP01C_integrity_report.txt")
    m=pd.read_csv(od/"master_scores.csv")
    meta=pd.read_csv(root/"benchmark/data/sequence_metadata.csv").rename(
        columns={"gene_symbol":"target_gene"})
    log("="*78); log("STEP 1c - SEQUENCE INTEGRITY CHECK"); log("="*78)

    hint={r.target_gene: r.get("human_file",None) for _,r in meta.iterrows()}
    upid=dict(zip(m.target_gene,m.uniprot_id))

    # which genes even have an AlphaMissense file?
    amdir=root/"benchmark/data/alphamissense_scores"
    have_am={f.name.replace("_AM.csv","") for f in amdir.glob("*_AM.csv")}
    log(f"\n[0] AlphaMissense files present for {len(have_am)} UniProt accessions")
    no_am=sorted({g for g in m.target_gene.unique() if upid[g] not in have_am})
    log(f"    genes with NO AM file at all: {no_am}")
    log("    -> for these, 0% AM coverage is a missing-file issue, not a mismatch.")

    # ---------------- per-gene check ----------------
    log("\n[1] WILD-TYPE RESIDUE CHECK AGAINST LOCAL FASTA")
    log(f"    {'gene':<8}{'uniprot':<9}{'len':>6}{'maxpos':>8}{'n':>6}"
        f"{'%match':>9}{'bestshift':>10}{'%@shift':>9}  file")
    rows=[]; seqs={}
    for gene, gd in m.groupby("target_gene"):
        up=upid[gene]
        sub=list(zip(gd.position.astype(int), gd.wt_aa.astype(str)))
        files=find_fasta(root, up, gene, hint.get(gene))
        if not files:
            log(f"    {gene:<8}{up:<9}{'-':>6}{gd.position.max():>8}{len(gd):>6}"
                f"{'NO FASTA FOUND':>28}")
            rows.append(dict(gene=gene,uniprot=up,seq_len=np.nan,max_pos=int(gd.position.max()),
                             n=len(gd),pct_match=np.nan,best_shift=np.nan,pct_at_shift=np.nan,
                             fasta="NONE")); continue
        best=None
        for f in files[:6]:
            recs=read_fasta_text(f.read_text())
            for hdr,s in recs.items():
                if len(s)<50: continue
                frac,_=match_frac(s,sub)
                if best is None or frac>best[0]: best=(frac,s,f,hdr)
        frac,s,f,hdr=best
        seqs[gene]=s
        bs,bf=0,frac
        if frac<0.99:
            for sh in range(-60,61):
                fr,_=match_frac(s,sub,sh)
                if fr>bf: bf,bs=fr,sh
        log(f"    {gene:<8}{up:<9}{len(s):>6}{gd.position.max():>8}{len(gd):>6}"
            f"{100*frac:>8.1f}%{bs:>10}{100*bf:>8.1f}%  {f.name}")
        rows.append(dict(gene=gene,uniprot=up,seq_len=len(s),max_pos=int(gd.position.max()),
                         n=len(gd),pct_match=round(100*frac,2),best_shift=bs,
                         pct_at_shift=round(100*bf,2),fasta=f.name))
    chk=pd.DataFrame(rows)
    bad=chk[(chk.pct_match<99)|(chk.pct_match.isna())]
    log(f"\n    genes below 99% wild-type match: {bad.gene.tolist() if len(bad) else 'NONE'}")

    # ---------------- isoform rescue ----------------
    if len(bad) and not a.no_fetch:
        log("\n[2] UNIPROT ISOFORM RESCUE for mismatched genes")
        for _,r in bad.iterrows():
            gene=r.gene; up=r.uniprot
            gd=m[m.target_gene==gene]
            sub=list(zip(gd.position.astype(int), gd.wt_aa.astype(str)))
            url=f"https://rest.uniprot.org/uniprotkb/{up}.fasta?includeIsoform=true"
            try:
                txt=urllib.request.urlopen(url,timeout=60).read().decode()
            except Exception as e:
                log(f"    {gene}: fetch failed ({type(e).__name__}) - check network"); continue
            recs=read_fasta_text(txt)
            log(f"    {gene} ({up}): {len(recs)} isoform(s) from UniProt")
            best=None
            for hdr,s in recs.items():
                frac,tot=match_frac(s,sub)
                acc=hdr.split("|")[1] if "|" in hdr else hdr[:20]
                log(f"        {acc:<14} len={len(s):<6} match={100*frac:6.1f}%")
                if best is None or frac>best[0]: best=(frac,s,acc)
            if best and best[0]>=0.99:
                log(f"      -> FIX: isoform {best[2]} matches {100*best[0]:.1f}%. "
                    f"Re-score {gene} against this sequence.")
                outp=root/"benchmark/data/sequences_fixed"; outp.mkdir(parents=True,exist_ok=True)
                (outp/f"{gene}_{best[2]}.fasta").write_text(f">{best[2]}\n{best[1]}\n")
                seqs[gene]=best[1]
            else:
                log(f"      -> no isoform reaches 99%; best {100*best[0]:.1f}%. "
                    f"Variants for {gene} must be re-mapped or excluded.")

    # ---------------- per-variant flag ----------------
    log("\n[3] PER-VARIANT FLAGGING")
    def ok_row(r):
        s=seqs.get(r.target_gene)
        if s is None: return np.nan
        i=int(r.position)-1
        return bool(0<=i<len(s) and s[i]==r.wt_aa)
    m["seq_wt_ok"]=m.apply(ok_row,axis=1)
    n_ok=int((m.seq_wt_ok==True).sum()); n_bad=int((m.seq_wt_ok==False).sum())
    n_unk=int(m.seq_wt_ok.isna().sum())
    log(f"    verified OK: {n_ok}   MISMATCH: {n_bad}   unverifiable: {n_unk}")
    if n_bad:
        log("\n    mismatches by gene:")
        log(m[m.seq_wt_ok==False].target_gene.value_counts().to_string())
        log("\n    example mismatches (gene, pos, claimed wt, actual residue):")
        for _,r in m[m.seq_wt_ok==False].head(15).iterrows():
            s=seqs.get(r.target_gene,""); i=int(r.position)-1
            act=s[i] if 0<=i<len(s) else "?"
            log(f"      {r.target_gene:<8} {int(r.position):>5} claimed={r.wt_aa} actual={act}")

    m["analysis_ok"]=(m.seq_wt_ok!=False)
    log(f"\n    analysis_ok (excludes confirmed mismatches): "
        f"{int(m.analysis_ok.sum())}/{len(m)}")
    log("    label balance among analysis_ok:")
    log(m[m.analysis_ok].label.value_counts().to_string())

    chk.to_csv(od/"step01c_sequence_check.csv",index=False)
    m.to_csv(od/"master_scores.csv",index=False)
    log(f"\n    master_scores.csv updated with seq_wt_ok + analysis_ok  shape={m.shape}")
    log("\nWROTE: step01c_sequence_check.csv, STEP01C_integrity_report.txt")
    log("       (any rescued isoforms in benchmark/data/sequences_fixed/)")
    log.close()

if __name__=="__main__": main()

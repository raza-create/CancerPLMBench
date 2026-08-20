#!/usr/bin/env python3
"""STEP 1d - resolve the WT1 sequence error, re-verify every gene against the
sequence ACTUALLY used for scoring, and finalise the exclusion list."""
import argparse, urllib.request, json
from pathlib import Path
import numpy as np, pandas as pd

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        s=" ".join(str(x) for x in a); print(s); self.f.write(s+"\n")
    def close(self): self.f.close()

VALID_AA=set("ACDEFGHIKLMNPQRSTVWYUOX")

def parse_fasta(txt):
    out={}; h=None; buf=[]
    for line in txt.splitlines():
        if line.startswith(">"):
            if h is not None: out[h]="".join(buf)
            h=line[1:].strip(); buf=[]
        else: buf.append(line.strip())
    if h is not None: out[h]="".join(buf)
    return out

def looks_like_protein(s):
    """3Di strings are lowercase-ish / heavily skewed; real proteins are diverse."""
    if not s: return False
    up=s.upper()
    if not set(up) <= VALID_AA: return False
    # 3Di sequences over-use a few letters; real proteins have >=15 distinct AAs
    return len(set(up))>=15

def match_frac(seq, sub, shift=0):
    ok=tot=0
    for pos,wt in sub:
        i=pos-1+shift
        if 0<=i<len(seq):
            tot+=1; ok+=(seq[i].upper()==wt)
    return (ok/tot if tot else 0.0), tot

def fetch(url, timeout=60):
    req=urllib.request.Request(url, headers={"User-Agent":"CancerPLMBench/1.0"})
    return urllib.request.urlopen(req, timeout=timeout).read().decode()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    ap.add_argument("--no-fetch",action="store_true")
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP01D_wt1_fix_report.txt")
    m=pd.read_csv(od/"master_scores.csv")
    meta=pd.read_csv(root/"benchmark/data/sequence_metadata.csv").rename(
        columns={"gene_symbol":"target_gene"})
    log("="*78); log("STEP 1d - WT1 SEQUENCE FIX + STRICT RE-VERIFICATION"); log("="*78)

    # ---- [1] resolve the exact file each gene was scored against ----
    log("\n[1] RESOLVING human_file FROM sequence_metadata.csv (the scored sequence)")
    hint=dict(zip(meta.target_gene, meta.human_file))
    upid=dict(zip(m.target_gene, m.uniprot_id))
    resolved={}
    for gene in sorted(m.target_gene.unique()):
        h=hint.get(gene)
        cand=[]
        if isinstance(h,str) and h:
            for p in [Path(h), root/h, root/"benchmark"/h, root/"benchmark/data"/h]:
                if p.exists(): cand.append(p); break
            if not cand:
                nm=Path(h).name
                cand=sorted(root.rglob(nm))
        status = "OK" if cand else "UNRESOLVED"
        resolved[gene]=cand[0] if cand else None
        if not cand:
            log(f"    {gene:<8} human_file={h!r}  -> {status}")
    log(f"    resolved {sum(v is not None for v in resolved.values())}/{len(resolved)} genes")

    # ---- [2] strict check, rejecting non-protein files ----
    log("\n[2] STRICT VERIFICATION (3Di / non-protein files rejected)")
    log(f"    {'gene':<8}{'file':<34}{'len':>6}{'maxpos':>8}{'%match':>9}  flag")
    seqs={}; rows=[]
    for gene, gd in m.groupby("target_gene"):
        sub=list(zip(gd.position.astype(int), gd.wt_aa.astype(str)))
        f=resolved.get(gene); seq=None; flag=""
        if f is not None:
            recs=parse_fasta(f.read_text())
            best=None
            for hdr,s in recs.items():
                if not looks_like_protein(s):
                    flag="NON-PROTEIN FILE"; continue
                fr,_=match_frac(s,sub)
                if best is None or fr>best[0]: best=(fr,s)
            if best: seq=best[1]
        if seq is None:
            log(f"    {gene:<8}{(f.name if f else 'NONE'):<34}{'-':>6}"
                f"{gd.position.max():>8}{'-':>9}  {flag or 'NO USABLE SEQUENCE'}")
            rows.append(dict(gene=gene,file=(f.name if f else None),seq_len=np.nan,
                             max_pos=int(gd.position.max()),pct_match=np.nan,flag=flag or "NO SEQ"))
            continue
        fr,_=match_frac(seq,sub); seqs[gene]=seq
        flag = "OK" if fr>=0.99 else ("TRUNCATED" if gd.position.max()>len(seq) else "MISMATCH")
        log(f"    {gene:<8}{f.name:<34}{len(seq):>6}{gd.position.max():>8}"
            f"{100*fr:>8.1f}%  {flag}")
        rows.append(dict(gene=gene,file=f.name,seq_len=len(seq),
                         max_pos=int(gd.position.max()),pct_match=round(100*fr,2),flag=flag))
    chk=pd.DataFrame(rows)

    # ---- [3] WT1 / any broken gene: hunt the correct sequence ----
    broken=chk[(chk.pct_match.isna())|(chk.pct_match<99)].gene.tolist()
    log(f"\n[3] GENES NEEDING A CORRECT SEQUENCE: {broken}")
    fixdir=root/"benchmark/data/sequences_fixed"; fixdir.mkdir(parents=True,exist_ok=True)
    fixed={}
    for gene in broken:
        gd=m[m.target_gene==gene]
        sub=list(zip(gd.position.astype(int), gd.wt_aa.astype(str)))
        up=upid[gene]
        log(f"\n    --- {gene} ({up})  maxpos={gd.position.max()}  n={len(gd)} ---")
        cands={}
        # local protein fastas mentioning the gene
        for p in sorted(root.rglob(f"*{gene}*.fasta"))+sorted(root.rglob(f"*{up}*.fasta")):
            if "3di" in p.name.lower() or "mouse" in p.name.lower() or "zebrafish" in p.name.lower():
                continue
            for hdr,s in parse_fasta(p.read_text()).items():
                if looks_like_protein(s): cands[f"local:{p.name}"]=s
        if not a.no_fetch:
            urls={
              f"uniprot:{up}+isoforms":
                f"https://rest.uniprot.org/uniprotkb/search?query=accession:{up}&format=fasta&includeIsoform=true",
              f"uniprot:gene {gene}":
                f"https://rest.uniprot.org/uniprotkb/search?query=gene_exact:{gene}+AND+organism_id:9606+AND+reviewed:true&format=fasta&includeIsoform=true",
            }
            for tag,u in urls.items():
                try:
                    for hdr,s in parse_fasta(fetch(u)).items():
                        if looks_like_protein(s):
                            acc=hdr.split("|")[1] if "|" in hdr else hdr[:24]
                            cands[f"{tag.split(':')[0]}:{acc}"]=s
                except Exception as e:
                    log(f"        fetch failed {tag}: {type(e).__name__}")
        best=None
        for tag,s in cands.items():
            fr,tot=match_frac(s,sub)
            log(f"        {tag:<40} len={len(s):<6} match={100*fr:6.1f}%  (n_checked={tot})")
            if best is None or fr>best[0]: best=(fr,s,tag)
        if best and best[0]>=0.99:
            log(f"      => CORRECT SEQUENCE FOUND: {best[2]} ({100*best[0]:.1f}%)")
            outp=fixdir/f"{gene}_corrected.fasta"
            outp.write_text(f">{gene}|corrected|{best[2]}\n{best[1]}\n")
            log(f"      => written {outp}")
            log(f"      => {gene} MUST BE RE-SCORED by all pLMs against this sequence")
            seqs[gene]=best[1]; fixed[gene]=best[2]
        else:
            bf=best[0] if best else 0
            log(f"      => no sequence reaches 99% (best {100*bf:.1f}%). "
                f"Affected variants will be excluded individually.")
            if best: seqs[gene]=best[1]

    # ---- [4] final per-variant flags ----
    log("\n[4] FINAL PER-VARIANT INTEGRITY FLAGS")
    def ok_row(r):
        s=seqs.get(r.target_gene)
        if s is None: return False
        i=int(r.position)-1
        return bool(0<=i<len(s) and s[i].upper()==r.wt_aa)
    m["seq_wt_ok"]=m.apply(ok_row,axis=1)
    m["analysis_ok"]=m["seq_wt_ok"]
    nb=int((~m.seq_wt_ok).sum())
    log(f"    verified OK: {int(m.seq_wt_ok.sum())}   excluded: {nb}")
    if nb:
        log("\n    exclusions by gene:")
        log(m[~m.seq_wt_ok].target_gene.value_counts().to_string())
        log("\n    exclusions by label:")
        log(m[~m.seq_wt_ok].label.value_counts().to_string())
    log(f"\n    CLEAN BENCHMARK: n={int(m.analysis_ok.sum())}  "
        f"P={int(m[m.analysis_ok].label.sum())}  "
        f"B={int((m[m.analysis_ok].label==0).sum())}  "
        f"genes={m[m.analysis_ok].target_gene.nunique()}")

    log("\n    per-gene clean counts (use these for Table 2):")
    t2=(m[m.analysis_ok].groupby(["target_gene","gene_type"])["label"]
          .agg(P=lambda s:int((s==1).sum()),B=lambda s:int((s==0).sum()),Total="size")
          .reset_index().sort_values("Total",ascending=False))
    log(t2.to_string(index=False))
    log(f"    SUMS: P={t2.P.sum()} B={t2.B.sum()} Total={t2.Total.sum()}")
    t2.to_csv(od/"step01d_table2_clean.csv",index=False)

    lines=[r"\begin{tabular}{llrrr}",r"\hline",
           r"Gene & Class & Pathogenic & Benign & Total \\",r"\hline"]
    for _,r in t2.iterrows():
        cls="TSG" if str(r.gene_type).lower().startswith("tumor") else "ONC"
        lines.append(f"{r.target_gene} & {cls} & {r.P} & {r.B} & {r.Total} \\\\")
    lines+=[r"\hline",
            f"\\textbf{{Total}} & & \\textbf{{{t2.P.sum()}}} & "
            f"\\textbf{{{t2.B.sum()}}} & \\textbf{{{t2.Total.sum()}}} \\\\",
            r"\hline",r"\end{tabular}"]
    (od/"step01d_table2_clean.tex").write_text("\n".join(lines))

    if fixed:
        log(f"\n[5] ACTION REQUIRED: re-score these genes -> {list(fixed)}")
        log("    corrected FASTAs are in benchmark/data/sequences_fixed/")
        log("    Until re-scored, their existing scores are INVALID and excluded.")
        for g in fixed: m.loc[m.target_gene==g,"analysis_ok"]=False
        log(f"    benchmark with re-score-pending genes held out: "
            f"n={int(m.analysis_ok.sum())}")
    chk.to_csv(od/"step01d_sequence_check.csv",index=False)
    m.to_csv(od/"master_scores.csv",index=False)
    log(f"\n    master_scores.csv updated  shape={m.shape}")
    log.close()

if __name__=="__main__": main()

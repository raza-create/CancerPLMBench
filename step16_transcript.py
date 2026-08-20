#!/usr/bin/env python3
"""STEP 16 - transcript consistency (critique item 6).
The Methods take the MAXIMUM dbNSFP score across transcripts, which inflates
baselines relative to pLMs scored on one canonical sequence. This quantifies
the inflation and rebuilds the comparison on consistent transcripts."""
import argparse, json, time, urllib.request, urllib.parse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

URL="https://myvariant.info/v1/query"

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        s=" ".join(str(x) for x in a); print(s,flush=True); self.f.write(s+"\n")
    def close(self): self.f.close()

def auc(y,s):
    y=np.asarray(y,float); s=np.asarray(s,float); ok=~np.isnan(s)
    y=y[ok]; s=s[ok]
    return roc_auc_score(y,s) if len(np.unique(y))>1 else np.nan

def get(q,fields,frm=0,size=1000,fetch_all=False,tries=3):
    params={"q":q,"fields":",".join(fields),"size":size}
    if fetch_all: params["fetch_all"]="true"
    else: params["from"]=frm
    u=f"{URL}?{urllib.parse.urlencode(params)}"
    for t in range(tries):
        try:
            r=urllib.request.Request(u,headers={"User-Agent":"CancerPLMBench/1.0"})
            return json.loads(urllib.request.urlopen(r,timeout=120).read().decode())
        except Exception:
            if t==tries-1: return {}
            time.sleep(4*(t+1))
    return {}

def scroll(sid,tries=3):
    u=f"{URL}?{urllib.parse.urlencode({'scroll_id':sid})}"
    for t in range(tries):
        try:
            r=urllib.request.Request(u,headers={"User-Agent":"CancerPLMBench/1.0"})
            return json.loads(urllib.request.urlopen(r,timeout=120).read().decode())
        except Exception:
            if t==tries-1: return {}
            time.sleep(4*(t+1))
    return {}

def aslist(x):
    return x if isinstance(x,list) else ([x] if x is not None else [])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP16_transcript_report.txt")
    m=pd.read_csv(od/"master_scores.csv")
    if "analysis_ok" in m.columns: m=m[m.analysis_ok==True].copy()
    log("="*78); log("STEP 16 - TRANSCRIPT CONSISTENCY (item 6)"); log("="*78)
    log("\nYour Methods state: 'For variants annotated against multiple transcripts,")
    log("the maximum rank score across transcripts was retained.' Taking the")
    log("maximum selects the most damaging annotation, inflating baselines, while")
    log("the pLMs are scored on a single canonical sequence. This quantifies that")
    log("inflation by retrieving PER-TRANSCRIPT scores.")

    FIELDS=["dbnsfp.genename","dbnsfp.aa.pos","dbnsfp.aa.ref","dbnsfp.aa.alt",
            "dbnsfp.ensembl.transcriptid","dbnsfp.ensembl.proteinid",
            "dbnsfp.mutationassessor.rankscore","dbnsfp.provean.converted_rankscore",
            "dbnsfp.sift.converted_rankscore","dbnsfp.polyphen2.hdiv.rankscore",
            "dbnsfp.vest4.rankscore","dbnsfp.revel.rankscore",
            "dbnsfp.metarnn.rankscore","dbnsfp.hgvsp"]
    cache_p=od/"step16_transcript_cache.json"
    cache=json.loads(cache_p.read_text()) if cache_p.exists() else {}
    genes=sorted(m.target_gene.unique())
    log(f"\n[1] RETRIEVING PER-TRANSCRIPT ANNOTATIONS ({len(genes)} genes)")
    recs=[]
    for g in genes:
        if g in cache:
            recs.extend(cache[g]); log(f"    {g:<9} cached ({len(cache[g])})"); continue
        got=[]; frm=0; sid=None
        while True:
            if sid: r=scroll(sid)
            elif frm>=9000:
                r=get(f"dbnsfp.genename:{g}",FIELDS,fetch_all=True); sid=r.get("_scroll_id")
            else: r=get(f"dbnsfp.genename:{g}",FIELDS,frm=frm)
            hits=r.get("hits",[])
            if not hits: break
            for h in hits:
                d=h.get("dbnsfp",{})
                if not d: continue
                aa=d.get("aa",{})
                if isinstance(aa,list): aa=aa[0] if aa else {}
                tid=aslist((d.get("ensembl") or {}).get("transcriptid")
                           if isinstance(d.get("ensembl"),dict) else None)
                got.append(dict(gene=g,pos=aa.get("pos"),ref=aa.get("ref"),
                                alt=aa.get("alt"),n_tx=len(tid),
                                ma=(d.get("mutationassessor") or {}).get("rankscore"),
                                pv=(d.get("provean") or {}).get("converted_rankscore"),
                                sf=(d.get("sift") or {}).get("converted_rankscore"),
                                pp=((d.get("polyphen2") or {}).get("hdiv") or {}).get("rankscore"),
                                vs=(d.get("vest4") or {}).get("rankscore"),
                                rv=(d.get("revel") or {}).get("rankscore"),
                                mr=(d.get("metarnn") or {}).get("rankscore")))
            frm+=len(hits)
            if sid is None and len(hits)<1000: break
            if sid: sid=r.get("_scroll_id",sid)
            if frm>120000: break
            time.sleep(0.3)
        cache[g]=got; recs.extend(got)
        log(f"    {g:<9} {len(got)} rows")
        cache_p.write_text(json.dumps(cache))
    if not recs:
        log("    no data - check network"); log.close(); return

    D=pd.DataFrame(recs)
    log(f"\n    total rows: {len(D)}")

    # ---------- [2] multi-transcript prevalence ----------
    log("\n[2] HOW MANY VARIANTS HAVE MULTIPLE TRANSCRIPTS?")
    D["pos"]=pd.to_numeric(D["pos"],errors="coerce")
    D=D.dropna(subset=["pos"]); D["pos"]=D["pos"].astype(int)
    grp=D.groupby(["gene","pos","ref","alt"])
    sizes=grp.size()
    log(f"    unique (gene,pos,ref,alt): {len(sizes)}")
    log(f"    with >1 dbNSFP record: {int((sizes>1).sum())} "
        f"({100*(sizes>1).mean():.1f}%)")
    if "n_tx" in D.columns:
        log(f"    median Ensembl transcripts per record: {D.n_tx.median():.0f}")
        log(f"    records annotated on >1 transcript: "
            f"{int((D.n_tx>1).sum())} ({100*(D.n_tx>1).mean():.1f}%)")

    # ---------- [3] max vs median across transcripts ----------
    log("\n[3] MAXIMUM vs MEDIAN ACROSS TRANSCRIPT RECORDS")
    log("    For variants with multiple records, how much does taking the maximum")
    log("    raise the score compared with the median?")
    SC={"ma":"MutationAssessor","pv":"PROVEAN","sf":"SIFT","pp":"PolyPhen-2",
        "vs":"VEST4","rv":"REVEL","mr":"MetaRNN"}
    for c in SC: D[c]=pd.to_numeric(D[c],errors="coerce")
    agg=grp.agg(**{f"{c}_max":(c,"max") for c in SC},
                **{f"{c}_med":(c,"median") for c in SC},
                n_rec=("pos","size")).reset_index()
    multi=agg[agg.n_rec>1]
    log(f"    variants with multiple records: {len(multi)}")
    log(f"    {'score':<18}{'mean max-med':>14}{'% inflated':>12}")
    for c,nm in SC.items():
        d=multi.dropna(subset=[f"{c}_max",f"{c}_med"])
        if len(d)<30: continue
        diff=d[f"{c}_max"]-d[f"{c}_med"]
        log(f"    {nm:<18}{diff.mean():>14.4f}{100*(diff>0.001).mean():>11.1f}%")

    # ---------- [4] effect on AUC ----------
    log("\n[4] EFFECT ON BENCHMARK PERFORMANCE")
    key=agg.rename(columns={"gene":"target_gene","pos":"position",
                            "ref":"wt_aa","alt":"mut_aa"})
    mm=m.merge(key,on=["target_gene","position","wt_aa","mut_aa"],how="left")
    log(f"    merged: {int(mm.n_rec.notna().sum())}/{len(mm)} benchmark variants")
    log(f"    {'method':<18}{'n':>7}{'AUC(max)':>11}{'AUC(median)':>13}{'delta':>9}")
    rows=[]
    for c,nm in SC.items():
        d=mm.dropna(subset=[f"{c}_max",f"{c}_med"])
        if len(d)<200: continue
        am=auc(d.label.values,d[f"{c}_max"].values)
        ad=auc(d.label.values,d[f"{c}_med"].values)
        log(f"    {nm:<18}{len(d):>7}{am:>11.4f}{ad:>13.4f}{am-ad:>+9.4f}")
        rows.append(dict(method=nm,n=len(d),auc_max=am,auc_median=ad,delta=am-ad))
    T=pd.DataFrame(rows)
    if len(T):
        T.to_csv(od/"step16_max_vs_median.csv",index=False)
        log(f"\n    mean inflation from taking the maximum: {T.delta.mean():+.4f}")
        log(f"    {int((T.delta>0).sum())}/{len(T)} methods inflated")
        try:
            _,p=stats.wilcoxon(T.delta.values)
            log(f"    Wilcoxon across methods: p={p:.4f}")
        except Exception: pass
        log("\n    A small delta means the maximum-across-transcripts rule had")
        log("    little practical effect and the comparison remains fair; a large")
        log("    delta means published baseline AUCs are inflated and the")
        log("    single-transcript values should be reported instead.")

    # ---------- [5] recommendation ----------
    log("\n[5] WHAT TO REPORT")
    log("-"*78)
    if len(T):
        big=T[T.delta.abs()>0.01]
        if len(big)==0:
            log("  The maximum-across-transcripts rule changed baseline ROC-AUC by")
            log(f"  less than 0.01 for every method (mean {T.delta.mean():+.4f}).")
            log("  Report this as a sensitivity analysis: transcript selection does")
            log("  not materially affect the benchmark, and the Methods sentence")
            log("  should state that both rules were evaluated.")
        else:
            log(f"  {len(big)} methods changed by more than 0.01 AUC:")
            for _,r in big.iterrows():
                log(f"    {r.method:<18}{r.auc_max:.4f} (max) vs "
                    f"{r.auc_median:.4f} (median), delta {r.delta:+.4f}")
            log("  Report the single-transcript (median) values as primary and the")
            log("  maximum-across-transcripts values as a sensitivity analysis.")
    log("-"*78)
    log("\nWROTE: step16_max_vs_median.csv, step16_transcript_cache.json")
    log.close()

if __name__=="__main__": main()

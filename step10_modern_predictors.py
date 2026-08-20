#!/usr/bin/env python3
"""STEP 10 - add modern predictors from dbNSFP via MyVariant.info (item 7).
Also retrieves phyloP/phastCons/GERP, which serve as independent conservation
measures for the attention analysis (item 10)."""
import argparse, json, time, urllib.request, urllib.error
from pathlib import Path
import numpy as np, pandas as pd

FIELDS = [
 # modern supervised / ensemble
 "dbnsfp.metarnn.rankscore","dbnsfp.bayesdel.add_af.rankscore",
 "dbnsfp.bayesdel.no_af.rankscore","dbnsfp.clinpred.rankscore",
 "dbnsfp.vest4.rankscore","dbnsfp.varity.er.rankscore",
 "dbnsfp.varity.r.rankscore","dbnsfp.metalr.rankscore",
 "dbnsfp.metasvm.rankscore","dbnsfp.mvp.rankscore",
 "dbnsfp.gmvp.rankscore","dbnsfp.m-cap.rankscore",
 "dbnsfp.mutformer.rankscore","dbnsfp.phactboost.rankscore",
 # deep / pLM
 "dbnsfp.primateai.rankscore","dbnsfp.esm1b.rankscore",
 # mechanistic / regional
 "dbnsfp.mutpred.rankscore","dbnsfp.mpc.rankscore",
 "dbnsfp.deogen2.rankscore","dbnsfp.list-s2.rankscore",
 "dbnsfp.mutationassessor.rankscore",
 # alignment-based
 "dbnsfp.provean.converted_rankscore","dbnsfp.fathmm.converted_rankscore",
 "dbnsfp.sift4g.converted_rankscore",
 # independent conservation (item 10)
 "dbnsfp.phylop.100way_vertebrate.score","dbnsfp.phylop.470way_mammalian.score",
 "dbnsfp.phylop.17way_primate.score",
 "dbnsfp.phastcons.100way_vertebrate.score","dbnsfp.phastcons.470way_mammalian.score",
 "dbnsfp.gerp++.rs","dbnsfp.gerp.91_mammals.score",
 "dbnsfp.siphy_29way.logodds_score",
 # dbNSFP versions of scores we already have - cross-check
 "dbnsfp.alphamissense.rankscore","dbnsfp.revel.rankscore",
 "dbnsfp.eve.rankscore","dbnsfp.sift.converted_rankscore",
 "dbnsfp.polyphen2.hdiv.rankscore",
 "dbnsfp.genename","dbnsfp.aa.pos","dbnsfp.aa.ref","dbnsfp.aa.alt",
]
URL="https://myvariant.info/v1/query"

class Tee:
    def __init__(self,p): self.f=open(p,"w",encoding="utf-8")
    def __call__(self,*a):
        s=" ".join(str(x) for x in a); print(s,flush=True); self.f.write(s+"\n")
    def close(self): self.f.close()

def post(genes_query, fields, tries=3):
    data=json.dumps({"q":genes_query,"fields":",".join(fields),
                     "size":1000}).encode()
    for t in range(tries):
        try:
            req=urllib.request.Request(URL,data=data,
                headers={"Content-Type":"application/json",
                         "User-Agent":"CancerPLMBench/1.0"})
            return json.loads(urllib.request.urlopen(req,timeout=120).read().decode())
        except Exception as e:
            if t==tries-1: raise
            time.sleep(3*(t+1))

def scroll(scroll_id, tries=3):
    q=urllib.parse.urlencode({"scroll_id":scroll_id})
    for t in range(tries):
        try:
            req=urllib.request.Request(f"{URL}?{q}",
                headers={"User-Agent":"CancerPLMBench/1.0"})
            return json.loads(urllib.request.urlopen(req,timeout=120).read().decode())
        except Exception:
            if t==tries-1: return {}
            time.sleep(5*(t+1))
    return {}

def get_query(gene, fields, frm=0, size=1000, tries=3, fetch_all=False):
    params={"q":f'dbnsfp.genename:{gene}',"fields":",".join(fields),"size":size}
    if fetch_all: params["fetch_all"]="true"
    else: params["from"]=frm
    q=urllib.parse.urlencode(params)
    for t in range(tries):
        try:
            req=urllib.request.Request(f"{URL}?{q}",
                headers={"User-Agent":"CancerPLMBench/1.0"})
            return json.loads(urllib.request.urlopen(req,timeout=120).read().decode())
        except urllib.error.HTTPError as e:
            if t==tries-1: raise
            time.sleep(5*(t+1))
        except Exception:
            if t==tries-1: raise
            time.sleep(5*(t+1))

def dig(d,path):
    cur=d
    for k in path.split("."):
        if isinstance(cur,list): cur=cur[0] if cur else None
        if not isinstance(cur,dict): return None
        cur=cur.get(k)
        if cur is None: return None
    if isinstance(cur,list): cur=cur[0] if cur else None
    return cur

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/raza/p53_family_analysis")
    ap.add_argument("--genes",default="")
    a=ap.parse_args(); root=Path(a.root).resolve(); od=root/"benchmark"/"results"
    log=Tee(od/"STEP10_modern_predictors_report.txt")
    m=pd.read_csv(od/"master_scores.csv"); m=m[m.analysis_ok==True].copy()
    log("="*78); log("STEP 10 - MODERN PREDICTORS FROM dbNSFP (item 7)"); log("="*78)
    log("\nRetrieving MetaRNN, BayesDel, ClinPred, VEST4, PrimateAI, ESM1b, MutPred,")
    log("MPC, PROVEAN, FATHMM, MutationAssessor, DEOGEN2, LIST-S2, M-CAP, plus")
    log("GERP++/phyloP/phastCons conservation scores.")

    genes=sorted(m.target_gene.unique()) if not a.genes else a.genes.split(",")
    cache_p=od/"step10_dbnsfp_cache.json"
    cache=json.loads(cache_p.read_text()) if cache_p.exists() else {}
    log(f"\n[1] FETCHING ({len(genes)} genes; cached: {len(cache)})")

    recs=[]
    for gi,g in enumerate(genes):
        if g in cache:
            recs.extend(cache[g]); log(f"    {g:<8} cached ({len(cache[g])} rows)")
            continue
        got=[]; frm=0; scroll_id=None
        try:
            while True:
                if scroll_id:
                    r=scroll(scroll_id)
                elif frm>=9000:
                    r=get_query(g,FIELDS,frm=0,size=1000,fetch_all=True)
                    scroll_id=r.get("_scroll_id")
                else:
                    r=get_query(g,FIELDS,frm=frm,size=1000)
                hits=r.get("hits",[])
                if not hits: break
                for h in hits:
                    d=h.get("dbnsfp",{})
                    if not d: continue
                    pos=dig(h,"dbnsfp.aa.pos"); ref=dig(h,"dbnsfp.aa.ref")
                    alt=dig(h,"dbnsfp.aa.alt")
                    if pos is None or ref is None or alt is None: continue
                    row={"target_gene":g,"position":pos,"wt_aa":ref,"mut_aa":alt}
                    for f in FIELDS:
                        if f.endswith(("genename","pos","ref","alt")): continue
                        key=(f.split("dbnsfp.",1)[1]
                             .replace(".converted_rankscore","")
                             .replace(".rankscore","").replace(".score","")
                             .replace(".","_").replace("++","pp").replace("-","_"))
                        row[key]=dig(h,f)
                    got.append(row)
                frm+=len(hits)
                if scroll_id is None and len(hits)<1000: break
                if scroll_id is not None and not r.get("_scroll_id"): break
                if scroll_id: scroll_id=r.get("_scroll_id",scroll_id)
                if frm>120000: break
                time.sleep(0.4)
            cache[g]=got; recs.extend(got)
            log(f"    {g:<8} fetched {len(got)} rows")
            cache_p.write_text(json.dumps(cache))
        except Exception as e:
            log(f"    {g:<8} FAILED: {type(e).__name__}: {e}")
        time.sleep(0.5)

    if not recs:
        log("\n  no data retrieved - check network access to myvariant.info")
        log.close(); return
    D=pd.DataFrame(recs)
    D["position"]=pd.to_numeric(D["position"],errors="coerce")
    D=D.dropna(subset=["position"]); D["position"]=D["position"].astype(int)
    D=D.drop_duplicates(subset=["target_gene","position","wt_aa","mut_aa"])
    log(f"\n    total unique (gene,pos,wt,mut) rows: {len(D)}")

    # ---------- merge ----------
    log("\n[2] MERGING TO BENCHMARK")
    newcols=[c for c in D.columns
             if c not in ("target_gene","position","wt_aa","mut_aa")]
    print(f"    columns retrieved: {newcols}")
    before=len(m)
    m2=m.merge(D,on=["target_gene","position","wt_aa","mut_aa"],how="left")
    assert len(m2)==before
    log(f"    {'predictor':<28}{'n':>7}{'coverage':>11}{'mean(P)':>10}{'mean(B)':>10}{'dir':>6}")
    keep=[]
    for c in newcols:
        if c not in m2.columns:
            log(f"    {c:<28} MISSING after merge - skipped"); continue
        v=pd.to_numeric(m2[c],errors="coerce")
        m2[c]=v; n=int(v.notna().sum())
        if n<200:
            log(f"    {c:<28}{n:>7}{'':>11}  (too sparse - dropped)")
            continue
        mp=v[m2.label==1].mean(); mb=v[m2.label==0].mean()
        d="+" if mp>mb else "-"
        log(f"    {c:<28}{n:>7}{100*n/len(m2):>10.1f}%{mp:>10.3f}{mb:>10.3f}{d:>6}")
        keep.append(c)
    log(f"\n    retained {len(keep)} predictors with >=200 scored variants")
    log("    NOTE: all dbNSFP rankscores are oriented so higher = more damaging.")
    log("    Any predictor showing '-' has an unexpected direction and is flagged.")
    bad=[c for c in keep if m2.loc[m2.label==1,c].mean() <= m2.loc[m2.label==0,c].mean()]
    if bad: log(f"    DIRECTION WARNING: {bad}")

    for c in keep:
        m2["s_new_"+c]=m2[c]
    m2.to_csv(od/"master_scores.csv",index=False)
    log(f"    master_scores.csv updated: {m2.shape}")

    # ---------- quick performance ----------
    log("\n[3] PERFORMANCE OF NEW PREDICTORS (native coverage)")
    from sklearn.metrics import roc_auc_score
    def auc(y,s):
        y=np.asarray(y,float); s=np.asarray(s,float); ok=~np.isnan(s)
        y=y[ok]; s=s[ok]
        return roc_auc_score(y,s) if len(np.unique(y))>1 else np.nan
    rows=[]
    OLD={"alphamissense":"AlphaMissense","saprot":"SaProt","revel":"REVEL",
         "esm2_650m":"ESM-2 650M","cadd":"CADD","sift":"SIFT",
         "polyphen2":"PolyPhen-2","esm2_150m":"ESM-2 150M","esm1v":"ESM-1v","eve":"EVE"}
    for x,nm in OLD.items():
        c="s_"+x; d=m2.dropna(subset=[c])
        rows.append(dict(method=nm,source="existing",n=len(d),
                         auc=auc(d.label.values,d[c].values),
                         coverage=round(100*m2[c].notna().mean(),1)))
    for c in keep:
        d=m2.dropna(subset=[c])
        rows.append(dict(method=c,source="new",n=len(d),
                         auc=auc(d.label.values,d[c].values),
                         coverage=round(100*m2[c].notna().mean(),1)))
    R=pd.DataFrame(rows).sort_values("auc",ascending=False)
    R.to_csv(od/"step10_all_predictors_auc.csv",index=False)
    log(f"    {'method':<28}{'source':<10}{'n':>7}{'ROC-AUC':>10}{'cov%':>8}")
    for _,r in R.iterrows():
        log(f"    {r.method:<28}{r.source:<10}{r.n:>7}{r.auc:>10.3f}{r.coverage:>8.1f}")

    log("\n[4] DOES ANY NEW METHOD BEAT AlphaMissense?")
    am=R[R.method=="AlphaMissense"].auc.iloc[0]
    better=R[(R.auc>am)]
    log(f"    AlphaMissense native AUC = {am:.3f}")
    log(f"    methods scoring higher: {better.method.tolist() if len(better) else 'NONE'}")
    log("    (unpaired comparison on differing coverage - a paired DeLong test on")
    log("     a shared subset is required before claiming any difference)")

    log("\n[5] CONSERVATION SCORES FOR THE ATTENTION ANALYSIS (item 10)")
    cons=[c for c in keep if any(k in c for k in ("gerp","phylop","phastcons","siphy"))]
    log(f"    retrieved: {cons if cons else 'NONE'}")
    if cons:
        log("    These provide the independent conservation measures the critique")
        log("    requests, replacing ESM-2 masked probability as the sole proxy.")
        for c in cons:
            d=m2.dropna(subset=[c])
            log(f"      {c:<32} n={len(d)}  AUC={auc(d.label.values,d[c].values):.3f}")

    log("\nWROTE: step10_all_predictors_auc.csv, step10_dbnsfp_cache.json")
    log("       master_scores.csv updated with s_new_* columns")
    log.close()

if __name__=="__main__": main()

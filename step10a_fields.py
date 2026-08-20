#!/usr/bin/env python3
"""STEP 10a - discover the ACTUAL dbNSFP field names in MyVariant.info before
the full fetch. Six of my guessed names returned zero rows, including all
conservation scores needed for item 10."""
import json, urllib.request, urllib.parse, re, sys

URL="https://myvariant.info/v1/query"

def get(q,**kw):
    params={"q":q,"size":kw.pop("size",1)}
    params.update(kw)
    u=f"{URL}?{urllib.parse.urlencode(params)}"
    req=urllib.request.Request(u,headers={"User-Agent":"CancerPLMBench/1.0"})
    return json.loads(urllib.request.urlopen(req,timeout=90).read().decode())

print("="*78)
print("STEP 10a - DISCOVERING REAL dbNSFP FIELD NAMES")
print("="*78)

print("\n[1] Pulling one fully-populated TP53 variant with ALL dbnsfp fields")
r=get('dbnsfp.genename:TP53',fields="dbnsfp",size=3)
hits=r.get("hits",[])
print(f"    hits: {len(hits)}")
if not hits:
    print("    no hits - check network"); sys.exit(1)

best=max(hits,key=lambda h: len(json.dumps(h)))
d=best.get("dbnsfp",{})

def walk(o,pre=""):
    out=[]
    if isinstance(o,dict):
        for k,v in o.items():
            out+=walk(v,f"{pre}.{k}" if pre else k)
    elif isinstance(o,list):
        if o: out+=walk(o[0],pre)
    else:
        out.append((pre,o))
    return out

flat=walk(d)
print(f"    {len(flat)} leaf fields present\n")

print("[2] ALL AVAILABLE RANKSCORE FIELDS")
rs=[(k,v) for k,v in flat if "rankscore" in k.lower()]
for k,v in sorted(rs):
    print(f"    dbnsfp.{k:<52} = {v}")

print(f"\n    total rankscore fields: {len(rs)}")

print("\n[3] CONSERVATION FIELDS (needed for item 10)")
pat=re.compile(r"gerp|phylop|phastcons|siphy|conserv",re.I)
cons=[(k,v) for k,v in flat if pat.search(k)]
if cons:
    for k,v in sorted(cons):
        print(f"    dbnsfp.{k:<52} = {v}")
else:
    print("    NONE found under dbnsfp - checking top level of the record")
    top=walk(best)
    cons2=[(k,v) for k,v in top if pat.search(k)]
    for k,v in sorted(cons2)[:30]:
        print(f"    {k:<60} = {v}")

print("\n[4] CHECKING MY FAILED FIELD NAMES")
failed=["bayesdel_addaf","bayesdel_noaf","gerp++_rs",
        "phylop100way_vertebrate","phastcons100way_vertebrate",
        "provean","fathmm"]
keys=[k.lower() for k,_ in flat]
for f in failed:
    stem=f.split("_")[0].replace("++","").replace("-","")
    matches=[k for k in keys if stem in k]
    print(f"    {f:<32} -> {matches[:4] if matches else 'NOT PRESENT'}")

print("\n[5] SUGGESTED FIELD LIST FOR THE FULL RUN")
want=re.compile(r"metarnn|bayesdel|clinpred|vest4|primateai|esm1b|mutpred|"
                r"^mpc|provean|fathmm|mutationassessor|deogen2|list|mcap|m_cap|"
                r"gerp|phylop|phastcons|revel|alphamissense|eve|sift|polyphen|cadd",re.I)
sel=sorted({k for k,_ in rs if want.search(k)})
extra=sorted({k for k,_ in cons if "rankscore" not in k.lower()})
print("    FIELDS = [")
for k in sel+extra:
    print(f'      "dbnsfp.{k}",')
print('      "dbnsfp.genename","dbnsfp.aa.pos","dbnsfp.aa.ref","dbnsfp.aa.alt",')
print("    ]")
print("\n    Copy this list into step10 and re-run for all 24 genes.")

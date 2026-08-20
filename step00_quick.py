import os, sys, pandas as pd
root = "/home/raza/p53_family_analysis"
rows = []
for dp, dn, fn in os.walk(root):
    dn[:] = [d for d in dn if d not in {'.git','__pycache__','.ipynb_checkpoints','venv','.venv'}]
    for f in fn:
        p = os.path.join(dp, f)
        try: rows.append((os.path.getsize(p), os.path.relpath(p, root)))
        except OSError: pass
rows.sort(reverse=True)
print("=== TOP 60 FILES BY SIZE ===")
for s, r in rows[:60]:
    print(f"{s/1e6:10.2f} MB  {r}")
print("\n=== EXTENSION COUNTS ===")
from collections import Counter
print(Counter(os.path.splitext(r)[1] for _, r in rows).most_common(25))
print("\n=== TABULAR FILES ===")
for s, r in rows:
    if not r.lower().endswith(('.csv','.tsv','.csv.gz','.tsv.gz')) or s < 200:
        continue
    p = os.path.join(root, r)
    sep = '\t' if '.tsv' in r else ','
    try:
        df = pd.read_csv(p, sep=sep, low_memory=False) if s < 120e6 else pd.read_csv(p, sep=sep, nrows=500, low_memory=False)
    except Exception as e:
        print(f"\n--- {r}  !! {e}"); continue
    print(f"\n--- {r}  ({s/1e6:.2f} MB)  shape={df.shape}")
    print("    cols:", list(df.columns))
    g = next((c for c in df.columns if c.lower() in ('gene','symbol','gene_symbol')), None)
    l = next((c for c in df.columns if c.lower() in ('label','y','target','is_pathogenic')), None)
    if g: print("    genes:", df[g].nunique(), sorted(df[g].dropna().unique())[:30])
    if l: print("    labels:", df[l].value_counts().to_dict())
    if g and l:
        t = df.groupby(g)[l].agg(['size','sum'])
        print("    per-gene total/P:", {k: (int(v['size']), int(v['sum'])) for k, v in t.iterrows()})
    print("    nonnull%:", {c: round(100*df[c].notna().mean(),1) for c in df.columns[:40]})

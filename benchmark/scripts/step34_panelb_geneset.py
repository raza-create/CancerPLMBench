import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy import stats

R = 'results/'
d = pd.read_csv(R + 'master_scores.csv')
M = {'AlphaMissense':'s_alphamissense','SaProt':'s_saprot','REVEL':'s_revel',
     'ESM-2 650M':'s_esm2_650m','CADD':'s_cadd','SIFT':'s_sift','PolyPhen-2':'s_polyphen2',
     'ESM-2 150M':'s_esm2_150m','ESM-1v':'s_esm1v','EVE':'s_eve'}
cands = ['APC','ATM','BRAF','BRCA1','BRCA2','HRAS','PTEN','TP53','VHL']

def auc(df, col):
    x = df[[col,'label']].dropna()
    return roc_auc_score(x.label, x[col]) if x.label.nunique()==2 and len(x) > 5 else np.nan

print('='*74); print('STEP 34  RESOLVING THE PANEL (b) GENE SET'); print('='*74)

print('\n--- per-gene, per-threshold class counts (P/B) among the 9 candidates ---')
for g in cands:
    row = []
    for t in [0,1,2,3]:
        s = d[(d.target_gene==g)&(d.review_stars>=t)]
        row.append(f">={t}:{int(s.label.sum())}P/{int((s.label==0).sum())}B")
    print(f'  {g:8s} ' + '  '.join(row))

claimed_b = {
 'AlphaMissense': (0.981,0.982,0.986,0.986), 'SaProt': (0.963,0.965,0.974,0.969),
 'REVEL': (0.935,0.936,0.933,0.927), 'ESM-2 650M': (0.929,0.932,0.950,0.949),
 'CADD': (0.885,0.888,0.889,0.860), 'PolyPhen-2': (0.846,0.851,0.868,0.841),
 'ESM-2 150M': (0.862,0.866,0.882,0.896), 'SIFT': (0.832,0.835,0.839,0.813),
 'ESM-1v': (0.829,0.832,0.831,0.862), 'EVE': (0.803,0.807,0.843,0.828),
}
claimed_vec = np.array([v for m in M for v in claimed_b[m]])

print('\n--- brute force: drop each of the 9 genes in turn, compare to claimed values ---')
best = None
for dropped in cands:
    subset = [g for g in cands if g != dropped]
    sub = d[d.target_gene.isin(subset)]
    computed_vec = []
    for m, c in M.items():
        for t in [0,1,2,3]:
            s = sub[sub.review_stars >= t]
            computed_vec.append(auc(s, c))
    computed_vec = np.array(computed_vec)
    sse = np.nansum((computed_vec - claimed_vec)**2)
    maxerr = np.nanmax(np.abs(computed_vec - claimed_vec))
    print(f'  drop {dropped:8s}  SSE={sse:.5f}  max|err|={maxerr:.4f}')
    if best is None or sse < best[1]:
        best = (dropped, sse, computed_vec)

print(f'\nBEST MATCH: dropping {best[0]}  (SSE={best[1]:.5f})')

dropped, _, computed_vec = best
subset = [g for g in cands if g != dropped]
print(f'\n--- full comparison, 8-gene subset dropping {dropped} ---')
idx = 0
for m in M:
    comp = tuple(round(computed_vec[idx+k], 3) for k in range(4))
    idx += 4
    flag = '' if comp == claimed_b[m] else '  <<< still mismatched'
    print(f'{m:<14}{str(comp):<40}{str(claimed_b[m]):<40}{flag}')

sub = d[d.target_gene.isin(subset)]
p0 = {m: auc(sub[sub.review_stars>=0], c) for m,c in M.items()}
p2 = {m: auc(sub[sub.review_stars>=2], c) for m,c in M.items()}
deltas = [p2[m]-p0[m] for m in M if not np.isnan(p2[m]) and not np.isnan(p0[m])]
print(f'\nsummary test on this 8-gene subset: median delta={np.median(deltas):+.4f}  '
      f'Wilcoxon P={stats.wilcoxon(deltas).pvalue:.4f}   [ms: +0.005, P=0.16]')
print('='*74)

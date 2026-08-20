import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy import stats

R = 'results/'
d = pd.read_csv(R + 'master_scores.csv')

M = {'AlphaMissense':'s_alphamissense','SaProt':'s_saprot','REVEL':'s_revel',
     'ESM-2 650M':'s_esm2_650m','CADD':'s_cadd','SIFT':'s_sift','PolyPhen-2':'s_polyphen2',
     'ESM-2 150M':'s_esm2_150m','ESM-1v':'s_esm1v','EVE':'s_eve'}

def auc(df, col):
    x = df[[col,'label']].dropna()
    return roc_auc_score(x.label, x[col]) if x.label.nunique()==2 and len(x) > 5 else np.nan

print('='*74); print('STEP 33  TABLE 8 / S6 REVIEW-STAR VERIFICATION'); print('='*74)
print('review_stars distribution:')
print(d.review_stars.value_counts(dropna=False).sort_index().to_string())

thresholds = [0,1,2,3]
claimed_a = {
 'AlphaMissense': (0.972,0.974,0.980,0.986), 'SaProt': (0.939,0.943,0.959,0.968),
 'REVEL': (0.926,0.929,0.928,0.928), 'ESM-2 650M': (0.916,0.919,0.938,0.950),
 'CADD': (0.883,0.884,0.878,0.861), 'SIFT': (0.859,0.863,0.852,0.813),
 'PolyPhen-2': (0.856,0.858,0.856,0.840), 'EVE': (0.833,0.837,0.857,0.825),
 'ESM-2 150M': (0.817,0.819,0.831,0.897), 'ESM-1v': (0.778,0.776,0.770,0.863),
}

print('\n--- PANEL (a): all eligible variants ---')
panel_a = {}
for t in thresholds:
    sub = d[d.review_stars >= t]
    print(f'\n>= {t} stars: n={len(sub)} variants, {sub.target_gene.nunique()} genes  '
          f'[ms: n={[4316,4163,2196,689][t]}, genes={[24,24,24,9][t]}]')
    panel_a[t] = {m: auc(sub, c) for m, c in M.items()}

print(f'\n{"method":<14}{"computed (>=0,1,2,3)":<40}{"claimed":<40}')
for m in M:
    comp = tuple(round(panel_a[t].get(m, np.nan), 3) for t in thresholds)
    flag = '' if comp == claimed_a[m] else '  <<< MISMATCH'
    print(f'{m:<14}{str(comp):<40}{str(claimed_a[m]):<40}{flag}')

print('\n--- PANEL (b): gene selection check ---')
genes_present_3 = set(d[d.review_stars >= 3].target_gene.unique())
print(f'genes with >=1 variant at >=3 stars: {len(genes_present_3)}  {sorted(genes_present_3)}  [ms panel a: 9]')

usable = []
for g in genes_present_3:
    s = d[(d.target_gene == g) & (d.review_stars >= 3)]
    if s.label.nunique() == 2:
        usable.append(g)
print(f'of these, genes with BOTH classes at >=3 stars: {len(usable)}  {sorted(usable)}  [ms panel b: 8]')
dropped = genes_present_3 - set(usable)
print(f'dropped (present but single-class at >=3 stars): {dropped}')

claimed_b = {
 'AlphaMissense': (0.981,0.982,0.986,0.986), 'SaProt': (0.963,0.965,0.974,0.969),
 'REVEL': (0.935,0.936,0.933,0.927), 'ESM-2 650M': (0.929,0.932,0.950,0.949),
 'CADD': (0.885,0.888,0.889,0.860), 'PolyPhen-2': (0.846,0.851,0.868,0.841),
 'ESM-2 150M': (0.862,0.866,0.882,0.896), 'SIFT': (0.832,0.835,0.839,0.813),
 'ESM-1v': (0.829,0.832,0.831,0.862), 'EVE': (0.803,0.807,0.843,0.828),
}
sub8 = d[d.target_gene.isin(usable)]
panel_b = {}
for t in thresholds:
    s = sub8[sub8.review_stars >= t]
    panel_b[t] = {m: auc(s, c) for m, c in M.items()}
    print(f'\n>= {t} stars, 8-gene subset: n={len(s)}')

print(f'\n{"method":<14}{"computed (>=0,1,2,3)":<40}{"claimed":<40}')
for m in M:
    comp = tuple(round(panel_b[t].get(m, np.nan), 3) for t in thresholds)
    flag = '' if comp == claimed_b[m] else '  <<< MISMATCH'
    print(f'{m:<14}{str(comp):<40}{str(claimed_b[m]):<40}{flag}')

print('\n--- SUMMARY TEST: change >=0 -> >=2 stars across methods, panel (b) ---')
deltas = [panel_b[2].get(m, np.nan) - panel_b[0].get(m, np.nan) for m in M]
deltas = [x for x in deltas if not np.isnan(x)]
print('deltas:', [round(x, 4) for x in deltas])
print(f'median delta = {np.median(deltas):+.4f}   [ms: +0.005]')
w = stats.wilcoxon(deltas)
print(f'Wilcoxon P = {w.pvalue:.4f}   [ms: P=0.16]')
print('='*74)

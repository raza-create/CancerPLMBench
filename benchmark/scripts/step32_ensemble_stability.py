import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

R = 'results/'
ens = pd.read_csv(R + 'step31_esm1v_ensemble5.csv')
d = pd.read_csv(R + 'master_scores.csv')
d['key'] = d.target_gene + '_' + d.variant_id
ens['key'] = ens.target_gene + '_' + ens.variant_id
shared = d[d.in_shared_all == True][['key']]
sh = ens.merge(shared, on='key')

print('='*72); print('STEP 32  DOES THE ENSEMBLE NARROW GENE-CLUSTERED UNCERTAINTY?'); print('='*72)
print(f'shared-subset n={len(sh)}, {sh.target_gene.nunique()} genes '
      f'[manuscript ESM-1v row: n=2080, 17 genes]')

def auc(df, col):
    x = df[[col, 'label']].dropna()
    return roc_auc_score(x.label, x[col]) if x.label.nunique() == 2 and len(x) > 5 else np.nan

pt = auc(sh, 's_esm1v_ensemble5')
rng = np.random.default_rng(0)
genes = sh.target_gene.unique()
boot = []
for _ in range(1000):
    gs = rng.choice(genes, len(genes), replace=True)
    s = pd.concat([sh[sh.target_gene == g] for g in gs])
    a = auc(s, 's_esm1v_ensemble5')
    if not np.isnan(a):
        boot.append(a)
lo, hi = np.percentile(boot, [2.5, 97.5])
width = hi - lo

elig = [g for g, gd in sh.groupby('target_gene')
        if gd.label.sum() >= 10 and (1 - gd.label).sum() >= 10]
per_gene = {g: auc(sh[sh.target_gene == g], 's_esm1v_ensemble5') for g in elig}
per_gene = {g: v for g, v in per_gene.items() if not np.isnan(v)}

print(f'\npoint estimate:            {pt:.3f}')
print(f'gene-clustered 95% CI:     [{lo:.3f}, {hi:.3f}]   width={width:.3f}')
print(f'[manuscript member 1 row:  [0.622, 0.903]   width=0.282]')
print(f'\nper-gene AUC ({len(per_gene)} eligible genes):')
for g, v in sorted(per_gene.items(), key=lambda x: x[1]):
    print(f'  {g:8s} {v:.3f}')
vals = list(per_gene.values())
print(f'\nrange: {max(vals)-min(vals):.3f}  (min {min(vals):.3f} on '
      f'{min(per_gene,key=per_gene.get)}, max {max(vals):.3f} on {max(per_gene,key=per_gene.get)})')
print(f'[manuscript member 1: range=0.524, min 0.474 on VHL, max 0.998 on EGFR]')

macro = np.mean(vals)
print(f'\nmacro-AUC (equal gene weight): {macro:.3f}   micro-AUC (pooled): {pt:.3f}')
print('='*72)

import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy import stats
R='results/'
d=pd.read_csv(R+'master_scores.csv')
rk=pd.read_csv(R+'step11_ranking.csv')
print('='*74); print('STEP 28  NATIVE-COVERAGE RANKING vs 34-WAY INTERSECTION'); print('='*74)

cols={r.method:r.col for _,r in rk.iterrows() if r.col in d.columns}
print(f'{len(cols)} predictors with score columns')

def auc(x,c):
    x=x[[c,'label']].dropna()
    return roc_auc_score(x.label,x[c]) if x.label.nunique()==2 and len(x)>10 else np.nan

shared34=d[[c for c in cols.values()]].notna().all(axis=1)
print(f'34-way intersection: {int(shared34.sum())} variants, '
      f'{d[shared34].target_gene.nunique()} genes   [ms: 1299, 14]')

rows=[]
for m,c in cols.items():
    nat=d[d[c].notna()]
    per=[auc(nat[nat.target_gene==g],c) for g in nat.target_gene.unique()
         if nat[nat.target_gene==g].label.nunique()==2
         and (nat[nat.target_gene==g].label==1).sum()>=10
         and (nat[nat.target_gene==g].label==0).sum()>=10]
    rows.append(dict(method=m,
        n_native=len(nat), cov=100*len(nat)/len(d),
        genes_native=nat.target_gene.nunique(),
        micro_native=auc(nat,c),
        macro_native=np.nanmean(per) if per else np.nan,
        n_genes_macro=len(per),
        micro_shared34=auc(d[shared34],c)))
t=pd.DataFrame(rows)
t['rank_shared']=t.micro_shared34.rank(ascending=False)
t['rank_native_micro']=t.micro_native.rank(ascending=False)
t['rank_native_macro']=t.macro_native.rank(ascending=False)
t=t.sort_values('rank_shared')
print('\n--- RANKING UNDER THREE EVALUATION VIEWS ---')
print(t[['method','cov','genes_native','micro_shared34','micro_native','macro_native',
         'rank_shared','rank_native_micro','rank_native_macro']].round(3).to_string(index=False))

for a,b,la,lb in [('rank_shared','rank_native_micro','shared34','native micro'),
                  ('rank_shared','rank_native_macro','shared34','native macro'),
                  ('rank_native_micro','rank_native_macro','native micro','native macro')]:
    x=t[[a,b]].dropna()
    r,p=stats.spearmanr(x[a],x[b])
    print(f'\nSpearman {la:<14} vs {lb:<14} rho={r:.3f}  P={p:.3g}  (n={len(x)})')
    mv=(x[a]-x[b]).abs()
    print(f'  max rank shift: {mv.max():.0f}   methods shifting >3 places: {int((mv>3).sum())}')

print('\n--- LARGEST RANK MOVERS (shared34 -> native macro) ---')
t['rshift']=t.rank_shared-t.rank_native_macro
print(t.reindex(t['rshift'].abs().sort_values(ascending=False).index)
       [['method','cov','rank_shared','rank_native_macro','rshift']].head(10).round(1).to_string(index=False))

print('\n--- PARETO FRONT (coverage vs native macro-AUC) ---')
p=t[['method','cov','macro_native']].dropna().values
front=[m for m,cv,a in p if not any((cv2>=cv and a2>=a and (cv2>cv or a2>a))
                                     for m2,cv2,a2 in p if m2!=m)]
print('non-dominated methods:',front)
t.to_csv(R+'step28_native_ranking.csv',index=False)
print('\n[saved] results/step28_native_ranking.csv'); print('='*74)

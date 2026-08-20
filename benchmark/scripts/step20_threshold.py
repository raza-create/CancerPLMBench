import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
R='results/'
d=pd.read_csv(R+'master_scores.csv')
AF=2700
print('='*72); print('STEP 20  THRESHOLD vs GRADIENT'); print('='*72)

d['q']=pd.qcut(d.protein_length,5,labels=False,duplicates='drop')
q=d.groupby('q').agg(n=('label','size'),lo=('protein_length','min'),
                     hi=('protein_length','max'),ret=('in_shared_all','mean'))
q['ret']=(100*q.ret).round(1)
print('\n--- variant-level quintiles (manuscript claims 83.7/69.8/68.1/0.0/0.0) ---')
print(q.to_string())

mod=d[d.protein_length<=AF]; unm=d[d.protein_length>AF]
print(f'\n--- threshold split ---')
print(f'modelled (<={AF}aa):   {len(mod)} variants, {mod.target_gene.nunique()} genes, '
      f'retained {mod.in_shared_all.sum()} ({100*mod.in_shared_all.mean():.1f}%)')
print(f'unmodelled (>{AF}aa):  {len(unm)} variants, {unm.target_gene.nunique()} genes, '
      f'retained {unm.in_shared_all.sum()} ({100*unm.in_shared_all.mean():.1f}%)')
print('unmodelled genes:',sorted(unm.target_gene.unique()))

pgm=mod.groupby('target_gene').agg(ret=('in_shared_all','mean'),
                                   length=('protein_length','first'),n=('label','size'))
rho,pv=stats.spearmanr(pgm.length,pgm.ret)
print(f'\n--- among MODELLED proteins only ({len(pgm)} genes) ---')
print(f'Spearman length vs retention: rho={rho:.3f}  P={pv:.3f}')
print(pgm.assign(ret=(100*pgm.ret).round(1)).sort_values('length').to_string())

pgall=d.groupby('target_gene').agg(ret=('in_shared_all','mean'),length=('protein_length','first'))
r_all,p_all=stats.spearmanr(pgall.length,pgall.ret)
print(f'\nall 24 genes: rho={r_all:.3f} P={p_all:.4f}   |   '
      f'20 modelled genes: rho={rho:.3f} P={pv:.3f}')

print('\n--- class composition, decomposed ---')
for nm,s in [('all',d),('modelled only',mod)]:
    i=s[s.in_shared_all==True]; e=s[s.in_shared_all==False]
    if len(e)==0: print(f'{nm:14s} no excluded variants'); continue
    chi=stats.chi2_contingency([[i.label.sum(),len(i)-i.label.sum()],
                                [e.label.sum(),len(e)-e.label.sum()]])[1]
    print(f'{nm:14s} retained {100*i.label.mean():.1f}% path (n={len(i)})   '
          f'excluded {100*e.label.mean():.1f}% path (n={len(e)})   chi2 P={chi:.3g}')
print('='*72)

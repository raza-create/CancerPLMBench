import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy import stats
R='results/'
d=pd.read_csv(R+'step13d_final_disorder.csv')

POP={'AlphaMissense','primateai'}
EVOL={'ESM-2 650M','ESM-2 150M','ESM-1v','esm1b','SaProt','EVE','SIFT','sift4g',
      'provean','mutationassessor','fathmm'}
def tier(m):
    if m in POP: return 'population'
    if m in EVOL: return 'evolutionary'
    return 'clinical'
d['tier']=d.method.map(tier)

print('='*72); print('STEP 27  PREDICTOR TAXONOMY SENSITIVITY'); print('='*72)
print('tier counts:',d.groupby('tier').size().to_dict())
print('\ncross-check against original family assignment:')
print(pd.crosstab(d.family,d.tier).to_string())
mism=d[((d.tier=='evolutionary')&(d.family!='evolutionary'))|
       ((d.tier=='clinical')&(d.family!='trained'))]
print('\nreassignments:',mism[['method','family','tier']].to_string(index=False) if len(mism) else 'none beyond population tier')

def mw(a,b,la,lb):
    if len(a)<2 or len(b)<2: return None
    p=stats.mannwhitneyu(a,b).pvalue
    print(f'  {la:<26}n={len(a):>2}  median={np.median(a):+.4f}')
    print(f'  {lb:<26}n={len(b):>2}  median={np.median(b):+.4f}')
    print(f'  two-sided Mann-Whitney P = {p:.4f}   {"SIG" if p<0.05 else "ns"}')
    return p

print('\n'+'-'*72); print('[A] AS PUBLISHED (evolutionary vs trained)'); print('-'*72)
mw(d[d.family=='evolutionary']['drop'],d[d.family=='trained']['drop'],
   'evolutionary','trained')

print('\n'+'-'*72); print('[B] AlphaMissense EXCLUDED'); print('-'*72)
x=d[d.method!='AlphaMissense']
mw(x[x.family=='evolutionary']['drop'],x[x.family=='trained']['drop'],
   'evolutionary','trained')

print('\n'+'-'*72); print('[C] AlphaMissense + PrimateAI EXCLUDED'); print('-'*72)
x=d[~d.method.isin(POP)]
mw(x[x.family=='evolutionary']['drop'],x[x.family=='trained']['drop'],
   'evolutionary','trained')

print('\n'+'-'*72); print('[D] THREE-TIER: evolutionary vs clinically supervised'); print('-'*72)
mw(d[d.tier=='evolutionary']['drop'],d[d.tier=='clinical']['drop'],
   'evolutionary','clinically supervised')
print('\n  population-supervised tier (n=2), for reference:')
print(d[d.tier=='population'][['method','drop']].to_string(index=False))

print('\n'+'-'*72); print('[E] KRUSKAL-WALLIS ACROSS ALL THREE TIERS'); print('-'*72)
gs=[d[d.tier==t]['drop'].values for t in ['evolutionary','population','clinical']]
print(f'  H-test P = {stats.kruskal(*gs).pvalue:.4f}')
for t in ['evolutionary','population','clinical']:
    v=d[d.tier==t]['drop']
    print(f'  {t:<14} n={len(v):>2}  median={v.median():+.4f}  IQR=[{v.quantile(.25):+.4f}, {v.quantile(.75):+.4f}]')

d.to_csv(R+'step27_taxonomy.csv',index=False)
print('\n[saved] results/step27_taxonomy.csv'); print('='*72)

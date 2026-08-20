import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score
from scipy import stats
R='results/'
d=pd.read_csv(R+'master_scores.csv')
f=pd.read_csv(R+'step06_residue_features.csv')
d=d.merge(f[['target_gene','position','plddt','rsa']],on=['target_gene','position'],how='left')
s=d[d.plddt.notna()].copy()
M={'AlphaMissense':'s_alphamissense','SaProt':'s_saprot','REVEL':'s_revel',
   'ESM-2 650M':'s_esm2_650m','CADD':'s_cadd','SIFT':'s_sift','PolyPhen-2':'s_polyphen2',
   'ESM-2 150M':'s_esm2_150m','ESM-1v':'s_esm1v','EVE':'s_eve'}
FAM={'AlphaMissense':'trained','REVEL':'trained','CADD':'trained','PolyPhen-2':'trained',
     'SaProt':'evol','ESM-2 650M':'evol','ESM-2 150M':'evol','ESM-1v':'evol',
     'SIFT':'evol','EVE':'evol'}
print('='*74); print('STEP 23  PREVALENCE-FREE CONTINUOUS pLDDT'); print('='*74)
print(f'{len(s)} variants, {s.target_gene.nunique()} genes')

# ---- per-variant placement score (prevalence-free) ----
rows=[]
for m,c in M.items():
    for g,sg in s.groupby('target_gene'):
        x=sg[[c,'label','plddt','variant_id']].dropna()
        P=x[x.label==1][c].values; B=x[x.label==0][c].values
        if len(P)<5 or len(B)<5: continue
        for _,r in x.iterrows():
            if r.label==1: pl=(np.sum(B<r[c])+0.5*np.sum(B==r[c]))/len(B)
            else:          pl=(np.sum(P>r[c])+0.5*np.sum(P==r[c]))/len(P)
            rows.append(dict(method=m,family=FAM[m],gene=g,label=int(r.label),
                             plddt=r.plddt,place=pl))
pl=pd.DataFrame(rows)
print(f'placement scores: {len(pl)} rows, {pl.gene.nunique()} genes, {pl.method.nunique()} methods')
chk=pl.groupby(['method','gene']).place.mean().groupby('method').mean()
print('\nsanity — mean placement (should track per-gene macro-AUC):')
print(chk.round(3).to_string())

print('\n--- per-method slope of placement on pLDDT (gene-clustered OLS) ---')
out=[]
for m in M:
    x=pl[pl.method==m].copy(); x['p100']=x.plddt/100
    try:
        r=smf.ols('place ~ p100 + C(label)',data=x).fit(
            cov_type='cluster',cov_kwds={'groups':x.gene.astype('category').cat.codes})
        out.append(dict(method=m,family=FAM[m],slope=r.params['p100'],
                        se=r.bse['p100'],p=r.pvalues['p100'],n=len(x)))
    except Exception as e: print(m,'failed',e)
o=pd.DataFrame(out).sort_values('slope',ascending=False)
o['lo']=o.slope-1.96*o.se; o['hi']=o.slope+1.96*o.se
print(o[['method','family','slope','lo','hi','p','n']].round(4).to_string(index=False))
ev=o[o.family=='evol'].slope; tr=o[o.family=='trained'].slope
print(f'\nmedian slope  evolutionary={ev.median():.4f} (n={len(ev)})   trained={tr.median():.4f} (n={len(tr)})')
print(f'Mann-Whitney P = {stats.mannwhitneyu(ev,tr).pvalue:.4f}')

print('\n--- interaction test: does slope differ by family? ---')
pl['p100']=pl.plddt/100
r=smf.ols('place ~ p100*family + C(label)',data=pl).fit(
    cov_type='cluster',cov_kwds={'groups':pl.gene.astype('category').cat.codes})
for t in r.params.index:
    if 'p100' in t or 'family' in t:
        print(f'  {t:<34}{r.params[t]:>9.4f}  SE={r.bse[t]:.4f}  P={r.pvalues[t]:.4g}')

# ---- prevalence-matched binned AUC (cross-check) ----
print('\n--- prevalence-matched AUC by pLDDT tercile (50/50, 200 resamples) ---')
s['ter']=pd.qcut(s.plddt,3,labels=['low','mid','high'])
rng=np.random.default_rng(0); res=[]
for m,c in M.items():
    for t in ['low','mid','high']:
        x=s[(s.ter==t)][[c,'label']].dropna()
        P=x[x.label==1]; B=x[x.label==0]
        k=min(len(P),len(B))
        if k<30: res.append(dict(method=m,ter=t,auc=np.nan,k=k)); continue
        a=[]
        for _ in range(200):
            ss=pd.concat([P.sample(k,replace=True,random_state=int(rng.integers(1e6))),
                          B.sample(k,replace=True,random_state=int(rng.integers(1e6)))])
            a.append(roc_auc_score(ss.label,ss[c]))
        res.append(dict(method=m,ter=t,auc=np.mean(a),k=k))
rr=pd.DataFrame(res).pivot(index='method',columns='ter',values='auc')[['low','mid','high']]
rr['high_minus_low']=rr.high-rr.low
rr['family']=[FAM[i] for i in rr.index]
print(rr.round(3).to_string())
print('\nmedian high-low  evol=%.3f  trained=%.3f'%(
    rr[rr.family=='evol'].high_minus_low.median(),
    rr[rr.family=='trained'].high_minus_low.median()))
g=rr.dropna(subset=['high_minus_low'])
print('Mann-Whitney P = %.4f'%stats.mannwhitneyu(
    g[g.family=='evol'].high_minus_low,g[g.family=='trained'].high_minus_low).pvalue)
pl.to_csv(R+'step23_placement.csv',index=False); rr.to_csv(R+'step23_matched_tercile.csv')
print('\n[saved] step23_placement.csv, step23_matched_tercile.csv'); print('='*74)

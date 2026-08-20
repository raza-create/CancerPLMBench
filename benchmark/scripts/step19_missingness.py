import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

R='results/'
d=pd.read_csv(R+'master_scores.csv')
print('='*72)
print('STEP 19  MULTIVARIABLE MODEL OF SHARED-SUBSET RETENTION')
print('='*72)
print(f'{len(d)} variants, {d.target_gene.nunique()} genes')

# ---------- VERIFICATION BLOCK ----------
inc=d[d.in_shared_all==True]; exc=d[d.in_shared_all==False]
print('\n--- verification of Table 6 / Section 3.1 ---')
print(f'retained {len(inc)} ({100*len(inc)/len(d):.1f}%)  excluded {len(exc)}   [ms: 2080 / 48.2% / 2236]')
print(f'genes retained {inc.target_gene.nunique()}  [ms: 17]')
print(f'pathogenic%% retained {100*inc.label.mean():.1f}  excluded {100*exc.label.mean():.1f}   [ms: 59.0 / 45.3]')
print(f'median length retained {inc.protein_length.median():.0f}  excluded {exc.protein_length.median():.0f}   [ms: 976 / 2839]')
print(f'mean stars retained {inc.review_stars.mean():.2f}  excluded {exc.review_stars.mean():.2f}   [ms: 1.68 / 1.59]')
pg=d.groupby('target_gene').agg(ret=('in_shared_all','mean'),length=('protein_length','first'))
rho,pv=stats.spearmanr(pg.length,pg.ret)
print(f'gene-level Spearman rho={rho:.3f} P={pv:.4f}   [ms: -0.47, P=0.020]')

# ---------- MODEL ----------
m=pd.DataFrame({
 'retained':d.in_shared_all.astype(int),
 'log_len':np.log10(d.protein_length),
 'pathogenic':d.label.astype(int),
 'stars':d.review_stars.fillna(d.review_stars.median()),
 'tsg':(d.gene_type=='Tumor suppressor').astype(int),
 'af_model':(d.protein_length<=2700).astype(int),
 'eval_year':d.last_eval_year.fillna(d.last_eval_year.median()),
 'gene':d.target_gene})
m['eval_year']=m.eval_year-m.eval_year.min()
print(f'\nmodelling n={len(m)}   retained={m.retained.sum()}')

def fit(formula,data,label):
    print('\n'+'-'*72); print(label); print('-'*72)
    grp=data.gene.astype('category').cat.codes.values
    try:
        res=smf.logit(formula,data=data).fit(disp=0,maxiter=200,
             cov_type='cluster',cov_kwds={'groups':grp})
    except Exception as e:
        print('  cluster fit failed (%s); falling back to HC1'%e)
        try:
            res=smf.logit(formula,data=data).fit(disp=0,maxiter=200,cov_type='HC1')
        except Exception as e2:
            print('  FAILED:',e2); return None
    print(f'{"term":<28}{"OR":>9}{"95% CI":>22}{"P":>11}')
    for nm in res.params.index:
        if nm=='Intercept': continue
        b=res.params[nm]; se=res.bse[nm]; p=res.pvalues[nm]
        lo,hi=np.exp(b-1.96*se),np.exp(b+1.96*se)
        print(f'{nm:<28}{np.exp(b):>9.3f}{f"[{lo:.3f}, {hi:.3f}]":>22}{p:>11.3g}')
    try: print(f'  pseudo-R2 = {res.prsquared:.3f}   n = {int(res.nobs)}   clusters = {len(set(grp))}')
    except Exception: print(f'  n = {int(res.nobs)}   clusters = {len(set(grp))}')
    return res

f1='retained ~ log_len + pathogenic + stars + tsg + eval_year'
fit(f1,m,'MODEL 1: full sample, length continuous (gene-clustered robust SE)')

f2='retained ~ log_len + pathogenic + stars + tsg + eval_year + af_model'
fit(f2,m,'MODEL 2: adding AlphaFold-model availability')

sub=m[m.af_model==1].copy()
print(f'\n[MODEL 3 restricted to proteins with an AlphaFold model: n={len(sub)}, '
      f'{sub.gene.nunique()} genes, retained={sub.retained.sum()}]')
fit(f1,sub,'MODEL 3: AF-modelled proteins only — is retention still non-random?')

# ---------- marginal effect of length ----------
print('\n'+'-'*72); print('RETENTION BY LENGTH DECILE (observed)'); print('-'*72)
m['dec']=pd.qcut(m.log_len,10,labels=False,duplicates='drop')
t=m.groupby('dec').agg(n=('retained','size'),ret=('retained','mean'),
                       lo=('log_len','min'),hi=('log_len','max'))
t['len_lo']=(10**t.lo).round(0); t['len_hi']=(10**t.hi).round(0); t['ret']=(100*t.ret).round(1)
print(t[['n','len_lo','len_hi','ret']].to_string())

m.drop(columns=['dec']).to_csv(R+'step19_missingness_design.csv',index=False)
print('\n[saved] results/step19_missingness_design.csv')
print('='*72)

import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score
R='results/'
d=pd.read_csv(R+'master_scores.csv')
print('='*74); print('STEP 29  SMALL-CLUSTER INFERENCE'); print('='*74)

m=pd.DataFrame({
 'retained':d.in_shared_all.astype(int),
 'log_len':np.log10(d.protein_length),
 'pathogenic':d.label.astype(int),
 'stars':d.review_stars.fillna(d.review_stars.median()),
 'tsg':(d.gene_type=='Tumor suppressor').astype(int),
 'eval_year':d.last_eval_year.fillna(d.last_eval_year.median()),
 'gene':d.target_gene})
m['eval_year']-=m.eval_year.min()
sub=m[d.protein_length<=2700].copy()
print(f'modelled subsample: n={len(sub)}  clusters={sub.gene.nunique()}')

F='retained ~ log_len + pathogenic + stars + tsg + eval_year'
grp=sub.gene.astype('category').cat.codes.values
base=smf.logit(F,data=sub).fit(disp=0,cov_type='cluster',cov_kwds={'groups':grp})
terms=[t for t in base.params.index if t!='Intercept']
print('\n--- analytic cluster-robust (as reported) ---')
for t in terms:
    print(f'  {t:<14} OR={np.exp(base.params[t]):>7.3f}  P={base.pvalues[t]:.3f}')

print('\n--- wild-cluster bootstrap-t, Rademacher, B=999, null imposed ---')
rng=np.random.default_rng(0)
B=999
clusters=np.unique(grp)
for t in terms:
    r=[c for c in terms if c!=t]
    f0='retained ~ '+' + '.join(r) if r else 'retained ~ 1'
    try:
        m0=smf.logit(f0,data=sub).fit(disp=0)
    except Exception as e:
        print(f'  {t:<14} restricted fit failed'); continue
    p0=m0.predict(sub)
    w_obs=abs(base.params[t]/base.bse[t])
    cnt=0; ok=0
    for b in range(B):
        s=rng.choice([-1,1],size=len(clusters))
        wmap=dict(zip(clusters,s))
        wv=np.array([wmap[g] for g in grp])
        res=(sub.retained.values-p0.values)
        ystar=(p0.values+wv*res)
        ystar=(rng.random(len(ystar))<np.clip(ystar,0.001,0.999)).astype(int)
        if ystar.sum() in (0,len(ystar)): continue
        tmp=sub.copy(); tmp['retained']=ystar
        try:
            mb=smf.logit(F,data=tmp).fit(disp=0,cov_type='cluster',
                                          cov_kwds={'groups':grp})
            wb=abs((mb.params[t]-0)/mb.bse[t])
            ok+=1; cnt+= (wb>=w_obs)
        except Exception: continue
    pw=(cnt+1)/(ok+1) if ok else np.nan
    print(f'  {t:<14} analytic P={base.pvalues[t]:.3f}   wild-cluster P={pw:.3f}   (B_eff={ok})')

print('\n--- gene-level permutation for the disorder family contrast ---')
dis=pd.read_csv(R+'step13d_final_disorder.csv')
ev=dis[dis.family=='evolutionary']['drop'].values
tr=dis[dis.family=='trained']['drop'].values
obs=np.median(ev)-np.median(tr)
allv=np.concatenate([ev,tr]); n=len(ev)
rng=np.random.default_rng(1)
perm=[]
for _ in range(20000):
    p=rng.permutation(allv)
    perm.append(np.median(p[:n])-np.median(p[n:]))
perm=np.array(perm)
pp=(np.sum(np.abs(perm)>=abs(obs))+1)/(len(perm)+1)
print(f'  observed median difference = {obs:+.4f}')
print(f'  permutation P (20,000 draws) = {pp:.4f}   [Mann-Whitney gave 0.025]')
print('='*74)

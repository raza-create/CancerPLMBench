import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy import stats
R='results/'; F='figures/'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
BLUE='#2c6fbb'; DBLUE='#16437e'; ORANGE='#e08214'; PALE='#bcd7f0'; RED='#c0392b'

d=pd.read_csv(R+'master_scores.csv'); sh=d[d.in_shared_all==True].copy()
M={'AlphaMissense':'s_alphamissense','SaProt':'s_saprot','REVEL':'s_revel',
   'ESM-2 650M':'s_esm2_650m','CADD':'s_cadd','SIFT':'s_sift',
   'PolyPhen-2':'s_polyphen2','ESM-2 150M':'s_esm2_150m','ESM-1v':'s_esm1v','EVE':'s_eve'}
print('='*72); print('STEP 21  FIG 4'); print('='*72)

def auc(df,c):
    x=df[[c,'label']].dropna()
    return roc_auc_score(x.label,x[c]) if x.label.nunique()==2 and len(x)>5 else np.nan

elig=[g for g,s in sh.groupby('target_gene') if s.label.sum()>=10 and (1-s.label).sum()>=10]
print('eligible genes:',len(elig),elig)

rows=[]
rng=np.random.default_rng(0)
genes=np.array(elig)
for m,c in M.items():
    pt=auc(sh,c)
    vb=[]
    x=sh[[c,'label']].dropna()
    for _ in range(1000):
        i=rng.integers(0,len(x),len(x)); s=x.iloc[i]
        if s.label.nunique()==2: vb.append(roc_auc_score(s.label,s[c]))
    gb=[]
    for _ in range(1000):
        gs=rng.choice(genes,len(genes),replace=True)
        s=pd.concat([sh[sh.target_gene==g] for g in gs])[[c,'label']].dropna()
        if s.label.nunique()==2: gb.append(roc_auc_score(s.label,s[c]))
    per=[auc(sh[sh.target_gene==g],c) for g in elig]
    rows.append(dict(method=m,auc=pt,v_lo=np.percentile(vb,2.5),v_hi=np.percentile(vb,97.5),
      g_lo=np.percentile(gb,2.5),g_hi=np.percentile(gb,97.5),
      macro=np.nanmean(per)))
t=pd.DataFrame(rows)
t['v_w']=t.v_hi-t.v_lo; t['g_w']=t.g_hi-t.g_lo; t['ratio']=t.g_w/t.v_w
t['macro_micro']=t.macro-t.auc
t=t.sort_values('auc',ascending=False)
print('\n--- FIG 4A / TABLE 8 ---')
print(t[['method','auc','g_lo','g_hi','g_w','ratio','macro','macro_micro']].round(3).to_string(index=False))
print(f'\ninterval width ratio range: {t.ratio.min():.1f}x to {t.ratio.max():.1f}x  [ms: 2-20x]')
t.to_csv(R+'step21_fig4_intervals.csv',index=False)

dl=[]
for g in elig:
    s=sh[sh.target_gene==g]
    dl.append(dict(gene=g,d=auc(s,'s_saprot')-auc(s,'s_esm2_650m'),n=len(s)))
dl=pd.DataFrame(dl).sort_values('d',ascending=False)
pooled=auc(sh,'s_saprot')-auc(sh,'s_esm2_650m')
w=stats.wilcoxon(dl.d)
print('\n--- FIG 4C SaProt - ESM-2 650M ---')
print(dl.round(4).to_string(index=False))
print(f'pooled {pooled:+.4f} | median {dl.d.median():+.4f} | SaProt higher in '
      f'{(dl.d>0).sum()}/{len(dl)} | Wilcoxon P={w.pvalue:.3f}   [ms: -0.006, 4/10, P=0.85]')

rk=pd.read_csv(R+'step11_ranking.csv')
plm=['saprot','esm2_650m','esm2_150m','esm1v','esm1b','alphamissense']
rk['is_plm']=rk.col.str.contains('saprot|esm',case=False)&~rk.col.str.contains('alphamissense')
print('\n--- FIG 4B top 12 of 34 ---')
print(rk.head(12)[['rank','method','auc']].round(4).to_string(index=False))
print('pLM ranks:',rk[rk.is_plm][['rank','method']].values.tolist())

fig=plt.figure(figsize=(7.2,6.2))
gs=fig.add_gridspec(2,2,hspace=0.42,wspace=0.30)

ax=fig.add_subplot(gs[0,0]); y=np.arange(len(t))[::-1]
ax.barh(y,t.v_hi-t.v_lo,left=t.v_lo,height=0.30,color=PALE,label='Variant-level 95% CI')
ax.barh(y,t.g_hi-t.g_lo,left=t.g_lo,height=0.13,color=DBLUE,label='Gene-clustered 95% CI')
ax.scatter(t.auc,y,s=13,c='black',zorder=5,label='Point estimate')
ax.set_yticks(y); ax.set_yticklabels(t.method,fontsize=6.4)
ax.set_xlabel('ROC-AUC',fontsize=7.5); ax.legend(fontsize=5.5,frameon=False,loc='lower left')
ax.set_title(f'A   Gene-clustered intervals are {t.ratio.min():.0f}–{t.ratio.max():.0f}$\\times$ wider',
             fontsize=8,fontweight='bold',loc='left'); ax.tick_params(labelsize=6.4)

ax=fig.add_subplot(gs[0,1])
cols=[BLUE if p else ORANGE for p in rk.is_plm]
ax.bar(range(len(rk)),rk.auc,color=cols,width=0.82)
for _,r in rk[rk.is_plm].iterrows():
    ax.text(r['rank']-1,r.auc+0.004,r.method,rotation=90,fontsize=4.8,ha='center',va='bottom')
ax.set_ylim(0.78,1.005); ax.set_xlabel('34 predictors, ranked',fontsize=7.5)
ax.set_ylabel('ROC-AUC',fontsize=7.5); ax.set_xticks([])
best=rk[rk.is_plm]['rank'].min()
ax.set_title(f'B   Best pLM ranks {best}th of 34',fontsize=8,fontweight='bold',loc='left')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=ORANGE,label='Trained on labelled variants'),
                   Patch(color=BLUE,label='Protein language model')],fontsize=5.5,frameon=False,loc='lower left')
ax.tick_params(labelsize=6.4)

ax=fig.add_subplot(gs[1,0])
c=[ORANGE if v>0 else BLUE for v in dl.d]
ax.barh(range(len(dl)),dl.d,color=c,height=0.66)
ax.set_yticks(range(len(dl))); ax.set_yticklabels(dl.gene,fontsize=6.2); ax.invert_yaxis()
ax.axvline(0,c='black',lw=0.8); ax.axvline(pooled,c=RED,ls='--',lw=1.0)
ax.text(pooled,len(dl)-0.3,f'pooled {pooled:+.3f}',fontsize=5.8,color=RED,ha='center')
ax.set_xlabel('$\\Delta$ROC-AUC (SaProt $-$ ESM-2 650M)',fontsize=7.5)
ax.set_title(f'C   SaProt higher in {(dl.d>0).sum()} of {len(dl)} genes',
             fontsize=8,fontweight='bold',loc='left'); ax.tick_params(labelsize=6.4)

ax=fig.add_subplot(gs[1,1])
ax.scatter(t.auc,t.macro,s=30,c=ORANGE,ec='white',lw=0.6,zorder=3)
lim=[min(t.auc.min(),t.macro.min())-0.02,1.005]
ax.plot(lim,lim,ls='--',c='grey',lw=0.8)
for _,r in t.iterrows():
    ax.annotate(r.method,(r.auc,r.macro),fontsize=5.2,xytext=(3,3),textcoords='offset points')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Micro-AUC (variants pooled)',fontsize=7.5)
ax.set_ylabel('Macro-AUC (genes equally weighted)',fontsize=7.5)
ax.set_title('D   Pooling favours methods differently',fontsize=8,fontweight='bold',loc='left')
ax.tick_params(labelsize=6.4)
fig.savefig(F+'Fig4_gene_aware.pdf'); fig.savefig(F+'Fig4_gene_aware.png',dpi=400)
print('\n[saved] figures/Fig4_gene_aware.pdf/.png'); print('='*72)

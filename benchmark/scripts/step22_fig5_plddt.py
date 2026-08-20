import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy import stats
R='results/'; F='figures/'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
GREEN='#4a9b5e'; RED='#c0392b'; GREY='#8c8c8c'

d=pd.read_csv(R+'master_scores.csv')
f=pd.read_csv(R+'step06_residue_features.csv')
d=d.merge(f[['target_gene','position','plddt','rsa']],on=['target_gene','position'],how='left')
s=d[d.plddt.notna()].copy()
print('='*72); print('STEP 22  DISORDER: VERIFY + CONTINUOUS pLDDT'); print('='*72)
print(f'{len(s)} variants with structural annotation, {s.target_gene.nunique()} genes '
      f'  [ms: 2836 variants, 20 genes]')
print('genes:',sorted(s.target_gene.unique()))
print('RSA null:',int(s.rsa.isna().sum()),' pLDDT null:',int(s.plddt.isna().sum()))

s['stratum']=pd.cut(s.plddt,[0,70,90,101],labels=['disordered','flexible','ordered'])
print('\n--- FIG 5A prevalence by pLDDT stratum ---')
g=s.groupby('stratum').agg(n=('label','size'),path=('label','mean'))
g['path']=(100*g.path).round(1); print(g.to_string())
print('[ms: 13.8 disordered / 55.6 flexible / 82.1 ordered]')
ct=pd.crosstab(s.stratum,s.label)
print('chi2 P =',f'{stats.chi2_contingency(ct)[1]:.3g}')

print('\n--- RSA strata ---')
s['burial']=pd.cut(s.rsa,[-.01,.09,.36,1.01],labels=['buried','intermediate','exposed'])
b=s.groupby('burial').agg(n=('label','size'),path=('label','mean'))
b['path']=(100*b.path).round(1); print(b.to_string()); print('[ms: 85.6 buried / 16.3 exposed]')

dis=pd.read_csv(R+'step13d_final_disorder.csv')
print(f'\n--- TABLE 13 ({len(dis)} methods) ---')
print(dis.sort_values('drop',ascending=False)[['method','family','drop','lo','hi','bh_q']].round(3).to_string(index=False))
med=dis.groupby('family')['drop'].agg(['median','size'])
print('\nfamily medians:'); print(med.round(3).to_string())
print('[ms: evolutionary +0.069 (n=11), trained +0.012 (n=23)]')
ev=dis[dis.family=='evolutionary']['drop']; tr=dis[dis.family=='trained']['drop']
mw=stats.mannwhitneyu(ev,tr)
print(f'Mann-Whitney U P = {mw.pvalue:.4f}   [ms: P=0.012]')

# ---------- CONTINUOUS pLDDT (new) ----------
print('\n'+'='*72); print('NEW: pLDDT AS A CONTINUOUS VARIABLE'); print('='*72)
M={'AlphaMissense':'s_alphamissense','SaProt':'s_saprot','REVEL':'s_revel',
   'ESM-2 650M':'s_esm2_650m','CADD':'s_cadd','SIFT':'s_sift','PolyPhen-2':'s_polyphen2',
   'ESM-2 150M':'s_esm2_150m','ESM-1v':'s_esm1v','EVE':'s_eve'}
FAM={'AlphaMissense':'trained','REVEL':'trained','CADD':'trained','PolyPhen-2':'trained',
     'SaProt':'evolutionary','ESM-2 650M':'evolutionary','ESM-2 150M':'evolutionary',
     'ESM-1v':'evolutionary','SIFT':'evolutionary','EVE':'evolutionary'}
s['bin']=pd.qcut(s.plddt,6,labels=False,duplicates='drop')
bi=s.groupby('bin').plddt.agg(['min','max','median','size'])
print('\npLDDT sextiles:'); print(bi.round(1).to_string())
rows=[]
for m,c in M.items():
    for b in sorted(s.bin.dropna().unique()):
        x=s[(s.bin==b)][[c,'label']].dropna()
        if x.label.nunique()==2 and len(x)>=25:
            rows.append(dict(method=m,family=FAM[m],bin=b,
                plddt=bi.loc[b,'median'],n=len(x),auc=roc_auc_score(x.label,x[c])))
cb=pd.DataFrame(rows)
pv=cb.pivot_table(index='method',columns='bin',values='auc')
print('\nROC-AUC by pLDDT sextile (blank = <25 variants or single class):')
print(pv.round(3).to_string())
print('\nSpearman AUC vs pLDDT within each method:')
out=[]
for m in M:
    x=cb[cb.method==m]
    if len(x)>=4:
        r,p=stats.spearmanr(x.plddt,x.auc)
        out.append(dict(method=m,family=FAM[m],rho=r,p=p,n_bins=len(x)))
o=pd.DataFrame(out).sort_values('rho',ascending=False)
print(o.round(3).to_string(index=False))
ev=o[o.family=='evolutionary'].rho; tr=o[o.family=='trained'].rho
print(f'\nmedian rho evolutionary={ev.median():.3f} (n={len(ev)})  '
      f'trained={tr.median():.3f} (n={len(tr)})')
if len(ev)>1 and len(tr)>1:
    print(f'Mann-Whitney P = {stats.mannwhitneyu(ev,tr).pvalue:.4f}')
cb.to_csv(R+'step22_plddt_continuous.csv',index=False)

fig,axes=plt.subplots(1,3,figsize=(7.2,2.6))
ax=axes[0]
gg=s.groupby('stratum').agg(n=('label','size'),p=('label','mean'))
ax.bar(range(3),100*gg.p,color=GREEN,width=0.62)
for i,(n,p) in enumerate(zip(gg.n,gg.p)): ax.text(i,100*p+2,f'{100*p:.1f}%\n(n={n})',ha='center',fontsize=6)
ax.set_xticks(range(3)); ax.set_xticklabels(['Disordered\n(<70)','Flexible\n(70–90)','Ordered\n(>90)'],fontsize=6.4)
ax.set_ylabel('Pathogenic (%)',fontsize=7.5); ax.set_ylim(0,100)
ax.set_title('A   Pathogenic variants by pLDDT',fontsize=8,fontweight='bold',loc='left')

ax=axes[1]
dd=dis.sort_values('drop')
c=[GREEN if x=='evolutionary' else RED for x in dd.family]
ax.barh(range(len(dd)),dd['drop'].values,xerr=[(dd['drop']-dd.lo).values,(dd.hi-dd['drop']).values],
        color=c,height=0.7,error_kw=dict(lw=0.5,ecolor='#555'))
ax.set_yticks(range(len(dd))); ax.set_yticklabels(dd.method,fontsize=4.4)
ax.axvline(0,c='black',lw=0.8); ax.set_xlabel('AUC loss, ordered $\\to$ disordered',fontsize=7)
ax.set_title('B   Accuracy loss by method',fontsize=8,fontweight='bold',loc='left')
ax.tick_params(labelsize=6)

ax=axes[2]
for m in M:
    x=cb[cb.method==m].sort_values('plddt')
    if len(x)>=4:
        ax.plot(x.plddt,x.auc,'-o',ms=2.5,lw=0.9,
                color=GREEN if FAM[m]=='evolutionary' else RED,alpha=0.75)
ax.set_xlabel('pLDDT (sextile median)',fontsize=7.5); ax.set_ylabel('ROC-AUC',fontsize=7.5)
ax.set_title('C   Continuous pLDDT trend',fontsize=8,fontweight='bold',loc='left')
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],color=GREEN,label='Evolutionary'),
                   Line2D([],[],color=RED,label='Trained')],fontsize=5.5,frameon=False,loc='lower right')
ax.tick_params(labelsize=6.4)
plt.tight_layout()
fig.savefig(F+'Fig5_disorder.pdf'); fig.savefig(F+'Fig5_disorder.png',dpi=400)
print('\n[saved] figures/Fig5_disorder.pdf/.png'); print('='*72)

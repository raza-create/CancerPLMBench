import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
R='results/'; F='figures/'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
GREEN='#4a9b5e'; RED='#c0392b'; GREY='#8c8c8c'

s=pd.read_csv(R+'step06e_structural_variants.csv')
dis=pd.read_csv(R+'step13d_final_disorder.csv')
print('='*70); print('STEP 26  FIG 5 REBUILD from step06e'); print('='*70)
print(f'n={len(s)}  genes={s.target_gene.nunique()}   [text: 2836, 20 genes]')
print('WT1 present:', 'WT1' in set(s.target_gene))

s['st']=pd.cut(s.plddt,[0,70,90,101],labels=['dis','flex','ord'])
g=s.groupby('st',observed=True).agg(n=('label','size'),p=('label','mean'))
print('\n--- PANEL A ---')
print(g.assign(p=(100*g.p).round(1)).to_string())
print('[text: 13.8 / 55.6 / 82.1]')

M={'AlphaMissense':'s_alphamissense','SaProt':'s_saprot','REVEL':'s_revel',
   'ESM-2 650M':'s_esm2_650m','CADD':'s_cadd','SIFT':'s_sift','PolyPhen-2':'s_polyphen2',
   'ESM-2 150M':'s_esm2_150m','ESM-1v':'s_esm1v','EVE':'s_eve'}
FAM={'AlphaMissense':'trained','REVEL':'trained','CADD':'trained','PolyPhen-2':'trained',
     'SaProt':'evol','ESM-2 650M':'evol','ESM-2 150M':'evol','ESM-1v':'evol',
     'SIFT':'evol','EVE':'evol'}
s['bin']=pd.qcut(s.plddt,6,labels=False,duplicates='drop')
bi=s.groupby('bin').plddt.agg(['median','size'])
rows=[]
for m,c in M.items():
    for b in sorted(s.bin.dropna().unique()):
        x=s[s.bin==b][[c,'label']].dropna()
        if x.label.nunique()==2 and len(x)>=25:
            rows.append(dict(method=m,family=FAM[m],plddt=bi.loc[b,'median'],
                             n=len(x),auc=roc_auc_score(x.label,x[c])))
cb=pd.DataFrame(rows)
print('\n--- PANEL C sextiles ---'); print(bi.round(1).to_string())
print(cb.pivot_table(index='method',columns='plddt',values='auc').round(3).to_string())
cb.to_csv(R+'step26_plddt_sextiles.csv',index=False)

fig,axes=plt.subplots(1,3,figsize=(7.2,2.6))
ax=axes[0]
ax.bar(range(3),100*g.p,color=GREEN,width=0.62)
for i,(n,p) in enumerate(zip(g.n,g.p)):
    ax.text(i,100*p+2.5,f'{100*p:.1f}%\n(n={n})',ha='center',fontsize=6)
ax.set_xticks(range(3))
ax.set_xticklabels(['Disordered\n(<70)','Flexible\n(70–90)','Ordered\n(>90)'],fontsize=6.4)
ax.set_ylabel('Pathogenic (%)',fontsize=7.5); ax.set_ylim(0,100)
ax.set_title('A   Pathogenic variants by pLDDT',fontsize=8,fontweight='bold',loc='left')
ax.tick_params(labelsize=6.4)

ax=axes[1]
dd=dis.sort_values('drop')
c=[GREEN if x=='evolutionary' else RED for x in dd.family]
ax.barh(range(len(dd)),dd['drop'].values,
        xerr=[(dd['drop']-dd.lo).values,(dd.hi-dd['drop']).values],
        color=c,height=0.72,error_kw=dict(lw=0.4,ecolor='#888'))
ax.set_yticks(range(len(dd)))
ax.set_yticklabels(dd.method,fontsize=3.6)
ax.axvline(0,c='black',lw=0.8)
ax.set_xlabel('AUC loss, ordered $\\to$ disordered',fontsize=7)
ax.set_title('B   Accuracy loss by method',fontsize=8,fontweight='bold',loc='left')
ax.tick_params(axis='x',labelsize=6)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=GREEN,label='Evolutionary'),
                   Patch(color=RED,label='Trained')],fontsize=5.2,frameon=False,loc='lower right')

ax=axes[2]
for m in M:
    x=cb[cb.method==m].sort_values('plddt')
    if len(x)>=4:
        ax.plot(x.plddt,x.auc,'-o',ms=2.5,lw=0.9,
                color=GREEN if FAM[m]=='evol' else RED,alpha=0.75)
ax.set_xlabel('pLDDT (sextile median)',fontsize=7.5)
ax.set_ylabel('ROC-AUC',fontsize=7.5)
ax.set_title('C   Continuous pLDDT trend',fontsize=8,fontweight='bold',loc='left')
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],color=GREEN,label='Evolutionary'),
                   Line2D([],[],color=RED,label='Trained')],
          fontsize=5.5,frameon=False,loc='lower right')
ax.tick_params(labelsize=6.4)
plt.tight_layout()
fig.savefig(F+'Fig5_disorder.pdf'); fig.savefig(F+'Fig5_disorder.png',dpi=400)
print('\n[saved] figures/Fig5_disorder.pdf/.png (overwritten)'); print('='*70)

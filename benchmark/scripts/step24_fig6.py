import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
R='results/'; F='figures/'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
DBLUE='#16437e'; RED='#c0392b'; GREY='#8c8c8c'

L=pd.read_csv(R+'step15e_layers.csv'); Mt=pd.read_csv(R+'step15e_matching.csv')
sch=[('Loose','step15b_layers.csv'),('Strict,\nbenign only','step15c_layers.csv'),
     ('Strict,\nvariant pos.','step15d_layers.csv'),('Strict,\nfull pool','step15e_layers.csv')]
vals=[]
for nm,fn in sch:
    x=pd.read_csv(R+fn); vals.append((nm,x.median_enrichment.median(),int(x.n.iloc[0])))
print('='*70); print('STEP 24  FIG 6'); print('='*70)
print(f'layers {len(L)}  proteins {L.n.iloc[0]}  pairs {int(Mt.n_matched.sum())}')
print(f'median {L.median_enrichment.median():.3f}  max {L.median_enrichment.max():.3f} '
      f'at layer {L.loc[L.median_enrichment.idxmax(),"layer"]}')
print(f'bonf {int(L.sig_bonf.sum())}/33   bh {int((L.bh_q<0.05).sum())}/33')
print('matching schemes:'); [print(f'  {n:22s} {v:.3f}') for n,v,_ in vals]

fig,axes=plt.subplots(1,3,figsize=(7.2,2.5),gridspec_kw={'width_ratios':[1.5,1,1]})
ax=axes[0]
sig=L[L.sig_bonf]; ns=L[~L.sig_bonf]
ax.plot(L.layer,L.median_enrichment,'-',c=GREY,lw=1.0,zorder=1)
ax.scatter(sig.layer,sig.median_enrichment,s=26,c=DBLUE,zorder=3,label='Bonferroni significant')
ax.scatter(ns.layer,ns.median_enrichment,s=26,facecolors='none',ec=RED,lw=0.9,zorder=3,label='Not significant')
ax.axhline(1.0,ls='--',c='black',lw=0.8)
i=L.median_enrichment.idxmax()
ax.annotate(f'Layer {int(L.layer[i])}\n{L.median_enrichment[i]:.2f}$\\times$',
    (L.layer[i],L.median_enrichment[i]),xytext=(-4,10),textcoords='offset points',
    fontsize=6.2,ha='center')
ax.set_xlabel('Transformer layer',fontsize=7.5); ax.set_ylabel('Median enrichment',fontsize=7.5)
ax.legend(fontsize=5.5,frameon=False,loc='upper left')
ax.set_title('A   ESM-2 650M attention at pathogenic positions',fontsize=7.6,fontweight='bold',loc='left')
ax.tick_params(labelsize=6.4)

ax=axes[1]
m=Mt.sort_values('diff')
c=[RED if v<0 else DBLUE for v in m['diff']]
ax.barh(range(len(m)),m['diff'],color=c,height=0.68)
ax.set_yticks(range(len(m))); ax.set_yticklabels(m.gene,fontsize=5.2)
ax.axvline(0,c='black',lw=0.8)
ax.text(0.97,0.03,f'mean |diff|\n{m["diff"].abs().mean():.4f}',transform=ax.transAxes,
        ha='right',va='bottom',fontsize=5.8)
ax.set_xlabel('Residual phyloP difference',fontsize=7)
ax.set_title('B   Conservation matching quality',fontsize=7.6,fontweight='bold',loc='left')
ax.tick_params(labelsize=6)

ax=axes[2]
names=[v[0] for v in vals]; ys=[v[1] for v in vals]
ax.plot(range(4),ys,'-o',c=GREY,ms=7,lw=1.2,zorder=2)
for i,(n,y,np_) in enumerate(vals):
    ax.scatter([i],[y],s=70,c=plt.cm.RdYlGn_r((y-1.15)/0.45),ec='black',lw=0.6,zorder=3)
    ax.text(i,y+0.028,f'{y:.2f}$\\times$',ha='center',fontsize=6)
ax.axhline(1.0,ls='--',c='black',lw=0.8)
ax.set_xticks(range(4)); ax.set_xticklabels(names,fontsize=5.4)
ax.set_ylabel('Median enrichment',fontsize=7.5)
ax.set_title('C   Effect declines as matching tightens',fontsize=7.6,fontweight='bold',loc='left')
ax.tick_params(labelsize=6.4)
plt.tight_layout()
fig.savefig(F+'Fig6_attention.pdf'); fig.savefig(F+'Fig6_attention.png',dpi=400)
print('\n[saved] figures/Fig6_attention.pdf/.png'); print('='*70)

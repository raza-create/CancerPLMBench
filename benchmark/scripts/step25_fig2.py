import warnings; warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
F='figures/'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
TEAL='#4bb3a8'; SALMON='#e8a887'; PURPLE='#7b4fa8'

A=[2,3,3,4,4,4,5,6,6,7,7,8,8,9,12,14,14,14,15,15,15,16,17,18,19,19,19,19,20,20]
B=[2,3,4,4,5,5,6,6,7,8,8,8,10,10,12,13,13,13,14,14,14,16,17,18,19,19,19,19,20,20]
ESM_MED=1.5; ESM_TOP1=50.0
print('='*66); print('STEP 25  FIG 2  (ProtT5 diagnostic)'); print('='*66)
print(f'A (sentinel)  n={len(A)}  median={np.median(A):.1f}  top1=0/30 (0%)   [ms: 13.0, 0%]')
print(f'B (no sentinel) n={len(B)} median={np.median(B):.1f}  top1=0/30 (0%)   [ms: 12.5, 0%]')
print(f'ESM-2 650M    median={ESM_MED}  top1=15/30 ({ESM_TOP1:.0f}%)          [ms: 1.5, 50%]')
print(f'chance: median rank 10.5, top-1 5%')

fig,axes=plt.subplots(1,2,figsize=(7.2,2.9),gridspec_kw={'width_ratios':[1.7,1]})

ax=axes[0]
bins=np.arange(0.5,21.5,1)
ax.hist([A,B],bins=bins,color=[TEAL,SALMON],
        label=['ProtT5 (sentinel decoder)','ProtT5 (no sentinel)'],rwidth=0.9)
ax.axvline(ESM_MED,color=PURPLE,lw=2.0,zorder=5)
ax.text(ESM_MED+0.35,ax.get_ylim()[1]*0.93,'ESM-2 650M\nmedian',fontsize=5.8,
        color=PURPLE,va='top')
ax.axvline(10.5,ls='--',color='black',lw=1.0)
ax.text(10.8,ax.get_ylim()[1]*0.93,'expected\nby chance',fontsize=5.8,va='top')
ax.set_xticks(range(1,21)); ax.tick_params(labelsize=6)
ax.set_xlabel('Rank of true wild-type residue (of 20 amino acids)',fontsize=7.5)
ax.set_ylabel('Number of masked positions',fontsize=7.5)
ax.legend(fontsize=5.8,frameon=False,loc='upper left')
ax.set_title('A   Residue reconstruction at 30 masked positions in TP53',
             fontsize=8,fontweight='bold',loc='left')

ax=axes[1]
vals=[0.0,0.0,ESM_TOP1]
labs=['ProtT5\n(sentinel)','ProtT5\n(no sentinel)','ESM-2\n650M']
cols=[TEAL,SALMON,PURPLE]
ax.bar(range(3),vals,color=cols,width=0.62)
for i,v in enumerate(vals):
    ax.text(i,v+1.6,f'{v:.0f}%',ha='center',fontsize=8.5,
            fontweight='bold' if v>0 else 'normal')
ax.axhline(5,ls='--',color='black',lw=1.0)
ax.text(2.45,6.2,'chance (5%)',fontsize=5.8,ha='right')
ax.set_xticks(range(3)); ax.set_xticklabels(labs,fontsize=6.2)
ax.set_ylim(0,58); ax.set_ylabel('Top-1 accuracy (%)',fontsize=7.5)
ax.tick_params(labelsize=6.3)
ax.set_title('B   Correct residue ranked first',fontsize=8,fontweight='bold',loc='left')

plt.tight_layout()
fig.savefig(F+'Fig2_prott5.pdf'); fig.savefig(F+'Fig2_prott5.png',dpi=400)
print('\n[saved] figures/Fig2_prott5.pdf/.png'); print('='*66)

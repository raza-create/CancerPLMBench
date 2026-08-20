import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy import stats

R='results/'; F='figures/'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.8,
    'xtick.major.width':0.8,'ytick.major.width':0.8,'savefig.bbox':'tight'})

BLUE='#2c6fbb'; DBLUE='#16437e'; RED='#c0392b'; ORANGE='#e08214'; GREY='#8c8c8c'
AF_LIMIT=2700

d=pd.read_csv(R+'master_scores.csv')
print('='*70); print('MASTER:',d.shape[0],'variants,',d.target_gene.nunique(),'genes')

# ---------- numbers for Fig 1C ----------
n_full=len(d); g_full=d.target_gene.nunique()
sh=d[d.in_shared_all==True]; n_sh=len(sh); g_sh=sh.target_gene.nunique()
pa=d[d.in_shared_plm_am==True]; n_pa=len(pa); g_pa=pa.target_gene.nunique()
esm=d[d.s_esm2_650m.notna()]; n_esm=len(esm); g_esm=esm.target_gene.nunique()
print('\n--- FIG 1C EVALUATION VIEWS ---')
print(f'Full benchmark        : {n_full} variants, {g_full} genes')
print(f'ESM-2 650M coverage   : {n_esm} variants, {g_esm} genes')
print(f'pLM + AlphaMissense   : {n_pa} variants, {g_pa} genes')
print(f'All 10 methods shared : {n_sh} variants, {g_sh} genes  ({100*n_sh/n_full:.1f}%)')
print('genes dropped by shared10:',sorted(set(d.target_gene)-set(sh.target_gene)))

# ---------- per-gene retention ----------
pg=d.groupby('target_gene').agg(n=('label','size'),ret=('in_shared_all','sum'),
    length=('protein_length','first'),path=('label','sum')).reset_index()
pg['ret_pct']=100*pg.ret/pg.n
pg=pg.sort_values('length',ascending=False)
rho,pv=stats.spearmanr(pg.length,pg.ret_pct)
print('\n--- FIG 3B RETENTION vs LENGTH (gene level, n=%d) ---'%len(pg))
print(f'Spearman rho = {rho:.3f}   P = {pv:.4f}    [manuscript: -0.47, P=0.020]')

# ---------- quintiles ----------
d['q']=pd.qcut(d.protein_length,5,labels=False,duplicates='drop')
qt=d.groupby('q').agg(ret=('in_shared_all','mean'),lo=('protein_length','min'),
    hi=('protein_length','max'),med=('protein_length','median'),n=('label','size'))
qt['ret']*=100
print('\n--- FIG 3C RETENTION BY LENGTH QUINTILE (variant level) ---')
print(qt.to_string())
print('[manuscript: 83.7, 69.8, 68.1, 0.0, 0.0]')

# ---------- retained vs excluded ----------
inc=d[d.in_shared_all==True]; exc=d[d.in_shared_all==False]
comp={'n':(len(inc),len(exc)),
 'genes':(inc.target_gene.nunique(),exc.target_gene.nunique()),
 'pathogenic_pct':(100*inc.label.mean(),100*exc.label.mean()),
 'median_length':(inc.protein_length.median(),exc.protein_length.median()),
 'mean_stars':(inc.review_stars.mean(),exc.review_stars.mean()),
 'tsg_pct':(100*(inc.gene_type=='Tumor suppressor').mean(),
            100*(exc.gene_type=='Tumor suppressor').mean())}
print('\n--- FIG 3D / TABLE 6 RETAINED vs EXCLUDED ---')
for k,(a,b) in comp.items(): print(f'{k:16s} retained={a:>10.4g}   excluded={b:>10.4g}')
print('[manuscript: 2080/2236, 59.0/45.3 path, 976/2839 len, 1.68/1.59 stars, 83.1/89.0 tsg]')

# ================= FIGURE 1 =================
fig=plt.figure(figsize=(7.2,3.3))
gs=fig.add_gridspec(1,3,width_ratios=[1,1,1.05],wspace=0.42)

axA=fig.add_subplot(gs[0]); axA.axis('off'); axA.set_xlim(0,10); axA.set_ylim(0,10)
axA.text(0,9.75,'A',fontsize=11,fontweight='bold',va='top')
axA.text(0.9,9.75,'Benchmark construction',fontsize=8.5,fontweight='bold',va='top')
boxes=[(f'40 curated cancer genes',None,'#dbe8f6'),
       (f'73,258 raw ClinVar missense',None,'#dbe8f6'),
       (f'4,351 labelled variants\n24 genes','P/B filters, $\\geq$10 each','#dbe8f6'),
       ('Sequence-integrity check','WT1 isoform corrected','#fadbd8'),
       (f'{n_full:,} verified variants\n{g_full} genes','35 removed','#fadbd8')]
y=8.6
for i,(t,sub,c) in enumerate(boxes):
    h=1.05 if sub is None else 1.35
    axA.add_patch(FancyBboxPatch((0.6,y-h),8.6,h,boxstyle='round,pad=0.06,rounding_size=0.15',
        fc=c,ec=DBLUE if c=='#dbe8f6' else RED,lw=0.9))
    axA.text(4.9,y-h/2+(0.18 if sub else 0),t,ha='center',va='center',fontsize=6.9,fontweight='bold')
    if sub: axA.text(4.9,y-h+0.3,sub,ha='center',va='center',fontsize=6.1,style='italic')
    if i<len(boxes)-1: axA.annotate('',xy=(4.9,y-h-0.42),xytext=(4.9,y-h),
        arrowprops=dict(arrowstyle='-|>',color='#555',lw=0.9))
    y-=h+0.42

axB=fig.add_subplot(gs[1]); axB.axis('off'); axB.set_xlim(0,10); axB.set_ylim(0,10)
axB.text(0,9.75,'B',fontsize=11,fontweight='bold',va='top')
axB.text(0.9,9.75,'Design choices varied',fontsize=8.5,fontweight='bold',va='top')
ch=[('Which\nvariants are\nretained','shared subset\nvs\nnative coverage'),
    ('Which\ncomparators\nare included','10 methods\nvs\n34 methods'),
    ('Unit of\nanalysis','variant-level\nvs\ngene-clustered')]
y=8.5
for t,sub in ch:
    axB.add_patch(FancyBboxPatch((0.4,y-2.2),9.2,2.2,boxstyle='round,pad=0.06,rounding_size=0.25',
        fc='white',ec='#444',lw=1.0))
    axB.text(2.9,y-1.1,t,ha='center',va='center',fontsize=6.9,fontweight='bold')
    axB.text(7.0,y-1.1,sub,ha='center',va='center',fontsize=6.4,style='italic')
    y-=2.75

axC=fig.add_subplot(gs[2])
labs=[f'Full\nbenchmark',f'pLM +\nAlphaMissense',f'All 10\nmethods shared']
vals=[n_full,n_pa,n_sh]; gns=[g_full,g_pa,g_sh]; cols=[DBLUE,BLUE,RED]
b=axC.barh(range(3),vals,color=cols,height=0.62)
for i,(v,g) in enumerate(zip(vals,gns)):
    axC.text(v*0.97,i,f'{v:,}\n({g} genes)',ha='right',va='center',fontsize=6.6,
             color='white',fontweight='bold')
axC.set_yticks(range(3)); axC.set_yticklabels(labs,fontsize=6.8)
axC.invert_yaxis(); axC.set_xlabel('Variants evaluated',fontsize=7.5)
axC.set_title('C  Evaluation views',fontsize=8.5,fontweight='bold',loc='left',x=-0.34)
axC.tick_params(labelsize=6.8)
fig.savefig(F+'Fig1_study_design.pdf'); fig.savefig(F+'Fig1_study_design.png',dpi=400)
plt.close(fig); print('\n[saved] figures/Fig1_study_design.pdf/.png')

# ================= FIGURE 3 =================
cm=pd.read_csv(R+'step09_coverage_matrix.csv').sort_values('len',ascending=False)
meths=['AlphaMissense','SaProt','REVEL','ESM-2 650M','CADD','SIFT','PolyPhen-2',
       'ESM-2 150M','ESM-1v','EVE']
fig=plt.figure(figsize=(7.2,6.4))
gs=fig.add_gridspec(2,3,height_ratios=[1.35,1],width_ratios=[1.5,1,1],hspace=0.42,wspace=0.38)

ax=fig.add_subplot(gs[0,:2])
M=cm[meths].values
im=ax.imshow(M,cmap='RdYlGn',vmin=0,vmax=100,aspect='auto')
ax.set_xticks(range(len(meths))); ax.set_xticklabels(meths,rotation=45,ha='left',fontsize=6.2)
ax.xaxis.set_ticks_position('top')
ax.set_yticks(range(len(cm)))
ax.set_yticklabels([f'{g} ({l})' for g,l in zip(cm.target_gene,cm.len)],fontsize=5.8)
for i in range(len(cm)):
    for j in range(len(meths)):
        v=M[i,j]
        ax.text(j,i,f'{v:.0f}',ha='center',va='center',fontsize=4.6,
                color='white' if v<28 or v>88 else 'black')
    if cm.len.iloc[i]>AF_LIMIT:
        ax.add_patch(Rectangle((-0.5,i-0.5),len(meths),1,fill=False,ec='black',lw=1.6))
cb=fig.colorbar(im,ax=ax,fraction=0.028,pad=0.02); cb.set_label('Variants scored (%)',fontsize=6.5)
cb.ax.tick_params(labelsize=6)
ax.set_title('A   Per-gene score coverage (genes ordered by length)',
             fontsize=8,fontweight='bold',loc='left',pad=26)

ax=fig.add_subplot(gs[0,2])
over=pg.length>AF_LIMIT
ax.scatter(pg.length[~over],pg.ret_pct[~over],s=26,c=BLUE,ec='white',lw=0.5,label='Within AF limit',zorder=3)
ax.scatter(pg.length[over],pg.ret_pct[over],s=32,facecolors='none',ec=RED,lw=1.3,label='Exceeds AF limit',zorder=3)
ax.axvline(AF_LIMIT,ls='--',c=GREY,lw=0.9)
ax.text(0.97,0.97,f'$\\rho$ = {rho:.2f}\nP = {pv:.3f}',transform=ax.transAxes,
        ha='right',va='top',fontsize=6.8)
ax.set_xlabel('Protein length (aa)',fontsize=7); ax.set_ylabel('Retained (%)',fontsize=7)
ax.legend(fontsize=5.6,frameon=False,loc='lower left')
ax.tick_params(labelsize=6.3)
ax.set_title('B   Retention vs length',fontsize=8,fontweight='bold',loc='left')

ax=fig.add_subplot(gs[1,0])
qc=[RED if v<5 else BLUE for v in qt.ret]
ax.bar(range(len(qt)),qt.ret,color=qc,width=0.68)
for i,v in enumerate(qt.ret): ax.text(i,v+2,f'{v:.1f}%',ha='center',fontsize=6.3)
ax.set_xticks(range(len(qt)))
ax.set_xticklabels([f'{int(a)}–\n{int(b)}' for a,b in zip(qt.lo,qt.hi)],fontsize=5.8)
ax.set_xlabel('Protein length quintile (aa)',fontsize=7)
ax.set_ylabel('Retained (%)',fontsize=7); ax.set_ylim(0,105); ax.tick_params(labelsize=6.3)
ax.set_title('C   Retention by length quintile',fontsize=8,fontweight='bold',loc='left')

for k,(key,lab,fmt) in enumerate([('pathogenic_pct','Pathogenic (%)','{:.1f}'),
                                  ('median_length','Median length (aa)','{:.0f}'),
                                  ('mean_stars','Mean review stars','{:.2f}')]):
    ax=fig.add_subplot(gs[1,1+k]) if k<2 else None
    if ax is None: break
for k,(key,lab,fmt) in enumerate([('pathogenic_pct','Pathogenic (%)','{:.1f}'),
                                  ('median_length','Median length (aa)','{:.0f}')]):
    ax=fig.add_subplot(gs[1,1+k])
    a,bv=comp[key]
    ax.bar([0,1],[a,bv],color=[BLUE,ORANGE],width=0.55)
    for x,v in zip([0,1],[a,bv]): ax.text(x,v*1.02,fmt.format(v),ha='center',fontsize=6.5)
    ax.set_xticks([0,1]); ax.set_xticklabels([f'Retained\n(n={comp["n"][0]:,})',
        f'Excluded\n(n={comp["n"][1]:,})'],fontsize=6)
    ax.set_ylabel(lab,fontsize=7); ax.tick_params(labelsize=6.3)
    ax.set_ylim(0,max(a,bv)*1.18)
    if k==0: ax.set_title('D   Retained vs excluded differ',fontsize=8,fontweight='bold',loc='left')
fig.savefig(F+'Fig3_coverage_bias.pdf'); fig.savefig(F+'Fig3_coverage_bias.png',dpi=400)
plt.close(fig); print('[saved] figures/Fig3_coverage_bias.pdf/.png')
print('='*70)

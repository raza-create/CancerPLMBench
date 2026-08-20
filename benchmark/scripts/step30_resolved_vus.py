import warnings; warnings.filterwarnings('ignore')
import gzip, re
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

R = Path('results')
AA3 = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G',
       'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S',
       'Thr':'T','Trp':'W','Tyr':'Y','Val':'V','Sec':'U','Pyl':'O'}
PROT = re.compile(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})')

d = pd.read_csv(R/'master_scores.csv')
genes = set(d.target_gene.unique())
print('='*72); print('STEP 30  FUTURE-RESOLVED VUS'); print('='*72)
print(f'current benchmark: {len(d)} variants, {len(genes)} genes')

def parse_release_with_sig(path, genes):
    rows = []
    with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as fh:
        header = fh.readline().rstrip('\n').split('\t')
    cols = [c for c in ['Name','GeneSymbol','Assembly','ClinicalSignificance'] if c in header]
    for ch in pd.read_csv(path, sep='\t', usecols=cols, dtype=str,
                          chunksize=400_000, low_memory=False):
        if 'Assembly' in ch.columns:
            ch = ch[ch['Assembly']=='GRCh38']
        ch = ch[ch['GeneSymbol'].isin(genes)]
        if not len(ch): continue
        ext = ch['Name'].fillna('').str.extract(PROT)
        ok = ext[0].notna()
        if ok.sum():
            sub = ch.loc[ok].copy()
            sub['wt']  = ext.loc[ok,0].map(AA3)
            sub['pos'] = pd.to_numeric(ext.loc[ok,1], errors='coerce')
            sub['mut'] = ext.loc[ok,2].map(AA3)
            sub = sub.dropna(subset=['wt','mut','pos'])
            rows.append(sub[['GeneSymbol','wt','pos','mut','ClinicalSignificance']])
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    print(f'  parsed {len(out)} rows with resolvable protein notation')
    return out

VUS_TERMS = ['uncertain significance', 'conflicting']
M = {'AlphaMissense':'s_alphamissense','SaProt':'s_saprot','REVEL':'s_revel',
     'ESM-2 650M':'s_esm2_650m','CADD':'s_cadd','SIFT':'s_sift','PolyPhen-2':'s_polyphen2',
     'ESM-2 150M':'s_esm2_150m','ESM-1v':'s_esm1v','EVE':'s_eve',
     'PHACTboost':'s_new_phactboost','MetaRNN':'s_new_metarnn'}

for label, fname in [('2020','variant_summary_2020-12.txt.gz'),
                      ('2022','variant_summary_2022-12.txt.gz')]:
    path = Path('data/clinvar')/fname
    print(f'\n--- archive {label} ---')
    if not path.exists():
        print(f'  NOT FOUND at {path}, skipping'); continue

    arch = parse_release_with_sig(path, genes)
    if not len(arch):
        print('  no parsable rows for these genes'); continue
    arch['pos'] = arch['pos'].astype(int)

    was_amb = arch['ClinicalSignificance'].str.lower().apply(
        lambda s: any(t in s for t in VUS_TERMS) if isinstance(s, str) else False)
    amb = arch[was_amb][['GeneSymbol','wt','pos','mut']].drop_duplicates()
    amb.columns = ['target_gene','wt_aa','position','mut_aa']
    print(f'  ambiguous (VUS/conflicting) at archive time: {len(amb)} distinct protein changes')

    merged = d.merge(amb, on=['target_gene','wt_aa','position','mut_aa'], how='inner')
    print(f'  resolved into current P/B benchmark: {len(merged)} variants '
          f'({int(merged.label.sum())} now pathogenic, {int((merged.label==0).sum())} now benign)')
    merged.to_csv(R/f'step30_resolved_vus_{label}.csv', index=False)

    if len(merged) < 15 or merged.label.nunique() < 2:
        print('  too few variants / single class -> cannot compute AUC yet')
        continue

    print(f'\n  {"method":<16}{"n_scored":>9}{"AUC_resolved":>14}{"AUC_full":>10}{"delta":>9}')
    for m, c in M.items():
        if c not in merged.columns: continue
        x = merged[[c,'label']].dropna()
        if x.label.nunique() < 2 or len(x) < 10:
            print(f'  {m:<16}{"--":>9}{"--":>14}{"--":>10}{"--":>9}')
            continue
        auc_r = roc_auc_score(x.label, x[c])
        xf = d[[c,'label']].dropna()
        auc_f = roc_auc_score(xf.label, xf[c]) if xf.label.nunique() == 2 else np.nan
        print(f'  {m:<16}{len(x):>9}{auc_r:>14.3f}{auc_f:>10.3f}{auc_r-auc_f:>+9.3f}')

print('\n' + '='*72)

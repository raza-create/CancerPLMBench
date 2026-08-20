import warnings; warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.metrics import roc_auc_score

R = 'results/'
d = pd.read_csv(R + 'master_scores.csv')
d['key'] = d.target_gene + '_' + d.variant_id
assert d.key.is_unique, "master_scores.csv has duplicate (gene, variant_id) -- stop, investigate"

m1 = d.set_index('key')['s_esm1v']
frames = {1: m1}
for member in [2, 3, 4, 5]:
    mdf = pd.read_csv(R + f'step31_esm1v_member{member}.csv')
    mdf['key'] = mdf.target_gene + '_' + mdf.variant_id
    assert mdf.key.is_unique, f"member {member} file has duplicate keys -- stop, investigate"
    frames[member] = mdf.set_index('key')['score']

all_members = pd.DataFrame(frames)
common = all_members.dropna()
labels = d.set_index('key').loc[common.index, 'label']
print('='*72); print('STEP 31b  ENSEMBLE FIX-UP (corrected composite key)'); print('='*72)
print(f'variants scored by all 5 members: {len(common)} of {len(d)}')

print('\n--- per-member AUC ---')
for k in [1,2,3,4,5]:
    print(f'  member {k}: AUC={roc_auc_score(labels, common[k]):.4f}')

ensemble_mean = common.mean(axis=1)
auc_ens = roc_auc_score(labels, ensemble_mean)
auc_m1_samen = roc_auc_score(labels, common[1])
print(f'\n  5-member ensemble (mean score): AUC={auc_ens:.4f}')
print(f'  member 1 alone, same n:         AUC={auc_m1_samen:.4f}')
print(f'  delta (ensemble - member 1):     {auc_ens-auc_m1_samen:+.4f}')

# compare against the full-benchmark member-1 AUC already reported in the paper (n=4316)
full_m1 = roc_auc_score(d.label, d.s_esm1v)
print(f'\n  member 1, full benchmark (n={len(d)}, as reported in manuscript): AUC={full_m1:.4f}')

out = d.set_index('key').loc[common.index, ['target_gene','variant_id','position','label']].reset_index(drop=True)
out['s_esm1v_ensemble5'] = ensemble_mean.values
out.to_csv(R + 'step31_esm1v_ensemble5.csv', index=False)
print('\n[saved] results/step31_esm1v_ensemble5.csv')
print('='*72)

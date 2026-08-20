"""
STEP 31 - complete ESM-1v ensemble (members 2-5).
Reuses the masked-marginal windowing logic from
benchmark/scripts/16_compute_llr_esm1v.py (member 1), scored on the final
verified benchmark (master_scores.csv) rather than the older
benchmark_variants.csv, so it merges cleanly with everything else.
"""
import warnings; warnings.filterwarnings('ignore')
import torch, esm
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT = Path.home() / "p53_family_analysis" / "benchmark"
DATA = ROOT / "data"
RES  = ROOT / "results"
WINDOW = 1022

print("="*72); print("STEP 31  ESM-1v ENSEMBLE COMPLETION (members 2-5)"); print("="*72)

d = pd.read_csv(RES / "master_scores.csv")
print(f"benchmark: {len(d)} variants, {d.target_gene.nunique()} genes")

def find_fasta(gene):
    fixed = DATA / "sequences_fixed" / f"{gene}_corrected.fasta"
    if fixed.exists():
        return fixed, "sequences_fixed (corrected isoform)"
    hits = list((DATA / "sequences").glob(f"{gene}_*_human.fasta")) if (DATA/"sequences").exists() else []
    if not hits:
        hits = list(DATA.glob(f"**/{gene}_*_human.fasta"))
    return (hits[0], "sequences (canonical)") if hits else (None, None)

def read_fasta(path):
    return "".join(l for l in path.read_text().splitlines() if not l.startswith(">")).strip()

seq_map = {}
print("\n--- sequence resolution + wild-type verification (same check that caught WT1) ---")
bad_genes = []
for gene, gd in d.groupby("target_gene"):
    fpath, src = find_fasta(gene)
    if fpath is None:
        print(f"  {gene:8s} NO FASTA FOUND -- will be skipped entirely")
        continue
    seq = read_fasta(fpath)
    seq_map[gene] = seq
    n = len(gd)
    ok = sum(1 for _, r in gd.iterrows()
             if 1 <= int(r.position) <= len(seq) and seq[int(r.position)-1] == r.wt_aa)
    pct = 100*ok/n
    flag = "" if pct >= 98.0 else "  <<< BELOW TOLERANCE"
    print(f"  {gene:8s} len={len(seq):5d}  src={src:28s}  wt-match={pct:5.1f}% ({ok}/{n}){flag}")
    if pct < 98.0:
        bad_genes.append(gene)

if bad_genes:
    print(f"\n  *** {bad_genes} failed wild-type check. STOPPING before GPU work. ***")
    raise SystemExit(1)

print(f"\nAll {len(seq_map)} genes verified. Proceeding to scoring.")

def compute_llr_windowed(model, alphabet, device, sequence, position, wt_aa, mut_aa):
    L = len(sequence)
    if position < 1 or position > L or sequence[position-1] != wt_aa:
        return None
    if L <= WINDOW:
        window_seq, local_pos = sequence, position
    else:
        half = WINDOW // 2
        start = max(0, position - 1 - half)
        end = start + WINDOW
        if end > L:
            end = L; start = end - WINDOW
        window_seq = sequence[start:end]
        local_pos = (position - 1) - start + 1
        if window_seq[local_pos-1] != wt_aa:
            return None
    masked = window_seq[:local_pos-1] + "<mask>" + window_seq[local_pos:]
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("p", masked)])
    tokens = tokens.to(device)
    with torch.no_grad():
        logits = model(tokens)["logits"]
    log_probs = torch.log_softmax(logits[0, local_pos], dim=-1)
    wt_idx = alphabet.get_idx(wt_aa)
    mut_idx = alphabet.get_idx(mut_aa)
    return (log_probs[mut_idx] - log_probs[wt_idx]).item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\ndevice: {device}")

member_scores = {}
for member in [2, 3, 4, 5]:
    print(f"\n--- ESM-1v member {member} ---")
    loader = getattr(esm.pretrained, f"esm1v_t33_650M_UR90S_{member}")
    model, alphabet = loader()
    model.eval().to(device)

    rows = []
    for _, r in d.iterrows():
        gene = r.target_gene
        if gene not in seq_map:
            continue
        try:
            llr = compute_llr_windowed(model, alphabet, device, seq_map[gene],
                                        int(r.position), r.wt_aa, r.mut_aa)
        except Exception:
            llr = None
        if llr is not None:
            rows.append({"variant_id": r.variant_id, "target_gene": gene,
                         "position": r.position, "label": r.label, "llr": llr})

    mdf = pd.DataFrame(rows)
    auc_raw = roc_auc_score(mdf.label, mdf.llr)
    auc_neg = roc_auc_score(mdf.label, -mdf.llr)
    sign = -1 if auc_neg > auc_raw else 1
    mdf["score"] = sign * mdf.llr
    print(f"  scored {len(mdf)}/{len(d)}  sign={'*-1' if sign==-1 else '*+1'}  "
          f"ROC-AUC={max(auc_raw, auc_neg):.4f}")
    mdf.to_csv(RES / f"step31_esm1v_member{member}.csv", index=False)
    member_scores[member] = mdf.set_index("variant_id")["score"]
    del model; torch.cuda.empty_cache()

m1 = d.set_index("variant_id")["s_esm1v"]
all_members = pd.DataFrame({1: m1, **member_scores})
common = all_members.dropna()
labels = d.set_index("variant_id").loc[common.index, "label"]

print("\n--- per-member AUC, variants scored by all 5 (n=%d) ---" % len(common))
for k in [1,2,3,4,5]:
    print(f"  member {k}: AUC={roc_auc_score(labels, common[k]):.4f}")

ensemble_mean = common.mean(axis=1)
print(f"\n  5-member ensemble (mean score): AUC={roc_auc_score(labels, ensemble_mean):.4f}")
print(f"  member 1 alone, same n:         AUC={roc_auc_score(labels, common[1]):.4f}")

out = pd.DataFrame({"variant_id": common.index, "label": labels.values,
                    "s_esm1v_ensemble5": ensemble_mean.values})
out.to_csv(RES / "step31_esm1v_ensemble5.csv", index=False)
print("\n[saved] step31_esm1v_member{2,3,4,5}.csv, step31_esm1v_ensemble5.csv")
print("="*72)

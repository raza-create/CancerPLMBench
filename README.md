# CancerPLMBench

Benchmark-design study of protein language models for cancer missense
variant interpretation. Accompanies:

**Benchmark Design Determines Conclusions About Protein Language Models in
Cancer Variant Interpretation**
Syed Raza Abbas, Zeeshan Abbas, Arifa Zahir, Mobeen Ur Rehman, Seung Won Lee.

## Contents

- `step*.py` (repository root) — benchmark construction, sequence-integrity
  verification, WT1 isoform correction, gene-aware statistical framework,
  calibration and clinical metrics, prospective validation, structural
  disorder analysis.
- `benchmark/scripts/step18-34*.py` — retention threshold modelling,
  predictor taxonomy sensitivity, native-coverage ranking, wild-cluster
  bootstrap inference, future-resolved VUS analysis, ESM-1v ensemble
  completion, review-star and clinical-metrics verification.
- `benchmark/results/*.csv` — every intermediate and final result table
  referenced in the manuscript and supplement, including the fixed random
  seed used for all bootstrap intervals.
- `data/sequences/`, `data/sequences_fixed/` — canonical and corrected
  (WT1 isoform P19544-7) sequences used for scoring.
- `manuscript/` — main text, supplementary material, and bibliography.

## Not included

Raw ClinVar releases (`variant_summary*.txt.gz`, ~620MB combined) and the
dbNSFP score cache (~250MB) are excluded for size; both are re-derivable
from public sources (NCBI ClinVar FTP; dbNSFP via MyVariant.info) using the
download scripts, and `benchmark/results/master_scores.csv` already
contains every score used in the paper.

## License

MIT. See `LICENSE`.

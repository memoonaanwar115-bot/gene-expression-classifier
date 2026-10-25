# Can machine learning find patterns in gene expression?

A leakage-aware classifier that separates **colorectal tumor from normal colon tissue** using
gene-expression microarray data, comparing Logistic Regression, Random Forest and SVM, and
investigating which genes drive the predictions.

> **Disclaimer:** This is a research classifier trained on a single public dataset.
> It is not a clinically validated diagnostic tool.

## Dataset
- **GEO accession:** [GSE20916](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE20916), human colon, Affymetrix HG-U133 Plus 2.0 (GPL570)
- Source study: Skrzypczak M et al., *PLoS One* 2010 (PMID 20957034)
- **Samples used: 60** (36 adenocarcinoma = "tumor", 24 normal colon = "normal")
- Only the macro-dissected samples were used, to avoid mixing two tissue-handling procedures
  (a batch effect). Adenomas and micro-dissected samples were excluded.
- Probes were mapped to gene symbols with the GPL570 annotation; ambiguous probes were removed
  and duplicate genes averaged. Values were already on a log2 scale.
- 688 genes with missing values were removed, leaving **12,921 genes**.

## Methods
1. Stratified 80/20 train/test split (48 train / 12 test) **before** any modelling.
2. Pipeline per model: StandardScaler -> SelectKBest (ANOVA F, k tuned) -> classifier.
   Scaling and gene selection are refit inside every CV fold, which prevents leakage.
3. Nested cross-validation on the training set (5-fold inner for tuning, 5-fold outer for
   estimation, 1 repeat), scored by ROC-AUC.
4. Permutation test with shuffled labels as a leakage check.
5. Best model refit on the training set; the test set was used once.
6. Feature importance: SHAP (see `results/top_genes.csv`).

## Results
| Model | Nested-CV AUC (mean +/- SD) |
|---|---|
| Logistic Regression | 1.000 +/- 0.000 |
| Random Forest | 1.000 +/- 0.000 |
| SVM | 1.000 +/- 0.000 |

- Best model by CV: Logistic Regression. **Held-out test AUC = 1.000** (12 samples).
- Shuffled-label control AUC = **0.475** (about 0.5 as expected), consistent with no leakage.
- Figures: `results/figures/` (PCA, CV boxplot, ROC, feature importance).

**Top genes** (importance, highest first): CDH3, MMP7, FOXQ1, CEMIP, AJUBA, MMP1, MMP3, CLDN1,
INHBA, CKS2, ACTL6A and others. Some of these have been linked to colorectal cancer in the
literature: FOXQ1 was reported as highly expressed in colorectal cancer cells [2], and serum MMP7
levels were associated with tumor stage and outcome in colorectal cancer patients [1]. This is
consistent with the model's findings, but it does not validate these genes as biomarkers.

## Limitations
- **Perfect scores reflect an easy problem** (tumor vs normal tissue is a strong signal) and a
  **very small test set (n = 12)**. They should not be read as real-world accuracy.
- Single cohort; no external validation on an independent dataset.
- Patient IDs are not given, so repeated samples from the same patient cannot be ruled out.
- Importance scores show what the model relies on, not causal biology, and correlated genes
  share credit. Importance rankings were not tested for stability.
- Not a clinically validated diagnostic tool.

## Reproduce
```bash
pip install -r requirements.txt
python src/prepare_data.py
python src/run_pipeline.py --csv data/expr.csv --label-col label
```
Python 3.10+; random seed 42. `--demo` runs on synthetic data to test the code only.


## References
1. Zhou Y, Wang L, Zhou F. Clinical Significance of MMP7 Levels in Colorectal Cancer Patients
   Receiving FOLFOX4 Chemotherapy Treatment. Int J Gen Med. 2023;16:2671-2678.
   doi:10.2147/IJGM.S416363. PMID: 37398512.
2. Yang G, Chen H, Ma X, Wang F, Ma G, Qi H. FOXQ1 Suppressing Apoptosis in Colorectal Cancer
   Cells by P53 Deacetylation. J Coll Physicians Surg Pak. 2025;35(6):722-729.
   doi:10.29271/jcpsp.2025.06.722. PMID: 40491105.
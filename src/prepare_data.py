"""Download GSE20916 (colon, Affymetrix GPL570) and build data/expr.csv for run_pipeline.py.

Design choices (document these in your README):
- Use only the 105 MACRO-dissected samples (titles without "(micro)"), so the two tissue-handling
  procedures are not mixed (a batch confound).
- Tumor = "adenocarcinoma_*"  -> label "tumor"
  Control = "normal_colon_*"   -> label "normal"
  Adenomas are DROPPED (benign, ambiguous class for a disease-vs-control question).
- Probes -> gene symbols via the GPL570 annotation; ambiguous probes ("///") dropped;
  duplicate genes averaged.
Run:  python src/prepare_data.py
"""
import numpy as np, pandas as pd, GEOparse

gse = GEOparse.get_GEO(geo="GSE20916", destdir="data/raw")
pheno = gse.phenotype_data
titles = pheno["title"].str.lower()

keep = ~titles.str.contains(r"\(micro\)")
lab = pd.Series(np.nan, index=pheno.index, dtype=object)
lab[titles.str.startswith("normal_colon")] = "normal"
lab[titles.str.startswith("adenocarcinoma")] = "tumor"
lab = lab[keep].dropna()
print("Label counts (macro only, adenomas dropped):\n", lab.value_counts())
assert lab.nunique() == 2, "Check titles: expected both 'normal' and 'tumor' groups"

expr = gse.pivot_samples("VALUE")[lab.index]          # rows = probes, cols = samples
print("Value range:", float(expr.min().min()), "to", float(expr.max().max()))
if expr.max().max() > 50:                              # looks like linear scale -> log2
    expr = np.log2(expr.clip(lower=0) + 1)
    print("Applied log2(x+1)")

gpl = list(gse.gpls.values())[0].table
sym_col = [c for c in gpl.columns if c.lower().replace(" ", "_") in ("gene_symbol", "symbol")][0]
mapping = gpl.set_index("ID")[sym_col].dropna().astype(str)
mapping = mapping[(mapping != "") & ~mapping.str.contains("///")]
expr = expr.loc[expr.index.intersection(mapping.index)]
expr.index = mapping.loc[expr.index].values
expr = expr.groupby(level=0).mean()

df = expr.T
df["label"] = lab.loc[df.index].values
df.to_csv("data/expr.csv")
print("Saved data/expr.csv:", df.shape, "(samples x [genes + label])")

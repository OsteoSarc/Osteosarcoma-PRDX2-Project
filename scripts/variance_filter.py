import pandas as pd
from sklearn.feature_selection import VarianceThreshold

print("\nRunning variance filtering...\n")

df = pd.read_csv("data/processed/pooled_corrected.csv", index_col = 0)

meta_cols = ["label", "cohort"]
gene_cols = [c for c in df.columns if c not in meta_cols]

expr = df[gene_cols]

# Inspect distribution before picking a threshold
print(expr.var().describe())

selector = VarianceThreshold(threshold = 0.05)
selector.fit(expr)
mask = selector.get_support()

n_before = expr.shape[1]
expr_filtered = expr.loc[:, mask].copy()

# Reattach metadata
expr_filtered["label"] = df["label"]
expr_filtered["cohort"] = df["cohort"]

expr_filtered.to_csv("data/processed/pooled_filtered.csv")

print(f"\n{expr_filtered.shape[1] - len(meta_cols)} genes passed the variance filter from {n_before} total genes")
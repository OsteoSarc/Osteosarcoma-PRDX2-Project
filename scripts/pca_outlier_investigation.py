import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

raw = pd.read_csv("data/processed/pooled_cohorts.csv", index_col=0)
corrected = pd.read_csv("data/processed/pooled_corrected.csv", index_col=0)

meta_cols = ["label", "cohort"]
gene_cols = [c for c in corrected.columns if c not in meta_cols]

# Recompute PCA coordinates the same way as the sanity check
X = StandardScaler().fit_transform(corrected[gene_cols])

pcs = PCA(n_components = 2).fit_transform(X)

corrected["PC1"] = pcs[:, 0]
corrected["PC2"] = pcs[:, 1]

# Flag TARGET-OS outliers by PC1 threshold, adjust threshold after eyeballing your plot
outliers = corrected[
    (corrected["cohort"] == "TARGET-OS") &
    ((corrected["PC1"] > 100) | (corrected["PC1"] < -100))
]

print(f"Flagged {len(outliers)} outlier samples:")
print(outliers.index.tolist())

# Compare these samples' raw expression stats to the rest of TARGET-OS
target_os_raw = raw[raw["cohort"] == "TARGET-OS"]
for sample_id in outliers.index:
    sample_vals = target_os_raw.loc[sample_id, gene_cols]
    cohort_mean = target_os_raw[gene_cols].mean(axis = 1).mean()
    print(f"{sample_id}: mean expr = {sample_vals.mean():.3f}, "
          f"cohort mean = {cohort_mean:.3f}, "
          f"label = {target_os_raw.loc[sample_id, 'label']}")
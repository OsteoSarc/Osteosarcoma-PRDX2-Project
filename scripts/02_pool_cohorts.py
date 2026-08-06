import pandas as pd

print("\nPooling cohorts together...")

# Read in preprocessed csv files
target = pd.read_csv("data/processed/target_os_processed.csv", index_col = 0)
gse874 = pd.read_csv("data/processed/gse87437_processed.csv", index_col = 0)
gse333 = pd.read_csv("data/processed/gse33382_processed.csv", index_col = 0)

cohorts = {"TARGET-OS": target, "GSE87437": gse874, "GSE33382": gse333}

# Intersect gene symbols (exclude non-gene columns like label)
gene_cols = [set(df.columns) - {"label"} for df in cohorts.values()]
common_genes = sorted(set.intersection(*gene_cols))

print(f"Common genes: {len(common_genes)}")  # sanity check before proceeding

# Kept only shared genes, added a cohort label column, then combined all cohorts into one table
pooled = []
for name, df in cohorts.items():
    sub = df[common_genes + ["label"]].copy()
    sub["cohort"] = name
    pooled.append(sub)

pooled_df = pd.concat(pooled, axis = 0)

# Additional sanity checks
print(f"\nPooled matrix shape (samples, columns): {pooled_df.shape}")
print(f"\nSamples per cohort: {pooled_df["cohort"].value_counts()}")
print(f"\nLabel Distribution: {pooled_df["label"].value_counts()}")
assert pooled_df.isna().sum().sum() == 0, "unexpected NaNs after pooling"

pooled_df.to_csv("data/processed/pooled_cohorts.csv")


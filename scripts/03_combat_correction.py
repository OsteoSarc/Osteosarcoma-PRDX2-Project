import pandas as pd
from combat.pycombat import pycombat

print("\nRunning ComBat batch correction...")

# Load pooled matrix from pool_cohorts.py
df = pd.read_csv("data/processed/pooled_cohorts.csv", index_col = 0)

meta_cols = ["label", "cohort"]
gene_cols = [c for c in df.columns if c not in meta_cols]

# Transpose gene_cols since pycombat expects genes x samples
expr = df[gene_cols].T

batch = df["cohort"]
covariate = df[["label"]]  # Protect label so biological signal isn't removed

corrected = pycombat(expr, batch, mod = covariate.values)

# Transpose back to samples x genes and reattach metadata
corrected = corrected.T
corrected["label"] = df["label"]
corrected["cohort"] = df["cohort"]

corrected.to_csv("data/processed/pooled_corrected.csv")

# Additional sanity checks
print(f"Batch corrected matrix shape: {corrected.shape}")
num_nan = corrected[gene_cols].isna().sum().sum()
print(f"NaNs after correction: {num_nan}")
print(f"Before batch correction: {df[gene_cols].values.mean()} (mean), {df[gene_cols].values.std()} (std dev)")
print(f"After batch correction: {corrected[gene_cols].values.mean()} (mean), {corrected[gene_cols].values.std()} (std dev)")
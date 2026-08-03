import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("\nPerforming PCA...")

df = pd.read_csv("data/processed/pooled_corrected.csv", index_col = 0)

meta_cols = ["label", "cohort"]
gene_cols = [c for c in df.columns if c not in meta_cols]

# Standardize before PCA since genes are on different scales
X = StandardScaler().fit_transform(df[gene_cols])

pca = PCA(n_components = 2)
pcs = pca.fit_transform(X)

df["PC1"] = pcs[:, 0]
df["PC2"] = pcs[:, 1]

print("Explained variance ratio:", pca.explained_variance_ratio_)

# Visualization for PCA
fig, axes = plt.subplots(1, 2, figsize = (12, 5))

for cohort in df["cohort"].unique():
    subset = df[df["cohort"] == cohort]
    axes[0].scatter(subset["PC1"], subset["PC2"], label = cohort, alpha = 0.7)

axes[0].set_title("Colored by cohort")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].legend()


for label in df["label"].unique():
    subset = df[df["label"] == label]
    axes[1].scatter(subset["PC1"], subset["PC2"], label=f"label={label}", alpha = 0.7)

axes[1].set_title("Colored by label")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].legend()

plt.tight_layout()
plt.savefig("data/processed/pca_visualization.png", dpi = 300)
print("Saved plot to pca_visualization.png")
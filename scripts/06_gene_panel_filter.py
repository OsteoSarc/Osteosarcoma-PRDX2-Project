import pandas as pd
import yaml

print("\nFiltering to osteosarcoma gene panel...\n")

df = pd.read_csv("data/processed/pooled_filtered.csv", index_col = 0)

with open("config.yaml") as f:
    config = yaml.safe_load(f)

panels = config["gene_panels"]
panel_genes = sorted(set(g for panel in panels.values() for g in panel["genes"]))
print(f"Panel: {len(panel_genes)} unique genes across {len(panels)} sources")

meta_cols = ["label", "cohort"]
gene_cols = [c for c in df.columns if c not in meta_cols]

missing = [g for g in panel_genes if g not in gene_cols]
if missing:
    # ABCC2, DHFR, ABCC6, ABCC11, ABCG2 were found to be dropped by preprocess_gse33382()'s
    # per-cohort variance filter before pooling and is determined not recoverable as real signal
    print(f"WARNING: {len(missing)} panel genes not present in pooled_filtered.csv, skipping: {missing}")

kept_genes = [g for g in panel_genes if g in gene_cols]
panel_filtered = df[kept_genes + meta_cols].copy()

panel_filtered.to_csv("data/processed/panel_filtered.csv")

print(f"\n{len(kept_genes)} of {len(panel_genes)} panel genes retained -> panel_filtered.csv")
print(f"Shape (samples, columns): {panel_filtered.shape}")

"""
Preprocess TARGET-OS, GSE87437, GSE33382 into (samples x gene_symbol) matrices with a binary 
response label (1 = good responder, 0 = poor responder). Run from project root. Input files 
expected in data/raw/ and output written to data/processed/.
"""
import gzip
import re
import numpy as np
import pandas as pd
import mygene
from io import StringIO
import GEOparse

RAW = "data/raw"
OUT = "data/processed"

# Prints a blank line before custom messages to stand out from GEOparses's logging
def log(msg):
    print(f"\n{msg}")


# Parse series matrix file into an expression dataframe (probes x sample) and metadata dataframe (sample x characteristic)
def parse_series_matrix(path):
    meta_rows = {}
    table_lines = []
    in_table = False
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if in_table:
                table_lines.append(line)
            elif line.startswith("!Sample_geo_accession"):
                meta_rows["geo_accession"] = [x.strip('"\n') for x in line.split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                vals = [x.strip('"\n') for x in line.split("\t")[1:]]
                # Each ch1 line is "field: value". Use the field name (from first non-empty entry) as column
                field = next((v.split(":")[0].strip() for v in vals if ":" in v), f"ch1_{len(meta_rows)}")
                meta_rows[field] = [v.split(":", 1)[1].strip() if ":" in v else "" for v in vals]
 
    expr_df = pd.read_csv(StringIO("".join(table_lines)), sep="\t", index_col = 0)
    expr_df.columns = [c.strip('"') for c in expr_df.columns]
    expr_df.index = [str(i).strip('"') for i in expr_df.index]
 
    meta_df = pd.DataFrame(meta_rows)
    meta_df.index = meta_df["geo_accession"]
    
    return expr_df, meta_df
 

# Maps probe/Ensembl index to gene symbol and drops unmapped duplicates
def collapse_to_gene_symbol(expr_df, id_to_symbol, min_coverage = 0.5):
    coverage = expr_df.index.isin(id_to_symbol).mean()
    log(f"Gene mapping coverage: {coverage:.1%} of {len(expr_df.index)} IDs resolved")
    
    if coverage < min_coverage:
        raise RuntimeError(
            log(f"Only {coverage:.1%} of IDs mapped to a gene symbol (threshold of 50%). Check the mapping source (mygene scope, or GPL annotation table) before trusting this dataset.")
        )
    
    expr_df = expr_df.copy()
    expr_df["gene_symbol"] = expr_df.index.map(id_to_symbol)
    expr_df = expr_df.dropna(subset=["gene_symbol"])
    collapsed = expr_df.groupby("gene_symbol").mean(numeric_only = True)
    
    return collapsed.T  # samples x genes
 

# Filters out any genes with low variance
def filter_low_variance(expr, var_threshold = 0.01):
    n_before = expr.shape[1]
    variances = expr.var()
    expr = expr.loc[:, variances > var_threshold]
    
    log(f"{expr.shape[1]} genes passed the variance filter from {n_before} total genes")
    return expr
 
 
def preprocess_gse87437():
    log("Running preprocessing on GSE87437 dataset...")

    expr, meta = parse_series_matrix(f"{RAW}/GSE87437_series_matrix.txt.gz")
 
    # values are MAS5.0 linear scale - log2 transform to make them comparable to the other cohorts
    expr = expr.apply(pd.to_numeric, errors = "coerce")
    expr_log = np.log2(expr + 1)
 
    label_col = [c for c in meta.columns if "response to preoperative" in c][0]
    labels = meta[label_col].map({"GOOD": 1, "POOR": 0})
 
    mg = mygene.MyGeneInfo()
    probe_ids = list(expr_log.index)
    query_results = mg.querymany(probe_ids, scopes = "reporter", fields = "symbol", species = "human")
    id_to_symbol = {r["query"]: r["symbol"] for r in query_results if "symbol" in r}
 
    genes = collapse_to_gene_symbol(expr_log, id_to_symbol)
    genes = filter_low_variance(genes)
    out = genes.join(labels.rename("label"), how = "inner")
    out = out.dropna(subset = ["label"])
    out.to_csv(f"{OUT}/gse87437_processed.csv")
    
    log(f"GSE87437: {out.shape[0]} samples, {out.shape[1] - 1} genes, "
          f"{int((out['label'] == 1).sum())} good / {int((out['label'] == 0).sum())} poor")
    
    return out
 
 
def preprocess_gse33382():
    log("Running preprocessing on GSE33382 dataset...")

    expr, meta = parse_series_matrix(f"{RAW}/GSE33382_series_matrix.txt.gz")
    expr = expr.apply(pd.to_numeric, errors="coerce")  # already log2-scale so no transform needed
 
    huvos_col = [c for c in meta.columns if "huvos" in c.lower()][0]
    huvos = meta[huvos_col]
    huvos_numeric = pd.to_numeric(huvos, errors = "coerce") # drop osteoblast controls (blank huvos) and unknown grade
    huvos_numeric = huvos_numeric.dropna()
    labels = huvos_numeric.map(lambda g: 1 if g >= 3 else 0)
    expr = expr[huvos_numeric.index]
 
    # GSE33382 ran on platform GPL10295 but probe IDs aren't standard ILMN_ format so fetching GPL10295's annotation table instead
    gpl = GEOparse.get_GEO(geo = "GPL10295", destdir = "data/raw")
    symbol_col = [c for c in gpl.table.columns if "symbol" in c.lower()]
    if not symbol_col:
        raise RuntimeError("GPL10295 annotation table has no gene symbol column - inspect gpl.table.columns")
    id_to_symbol = gpl.table.set_index(gpl.table.columns[0])[symbol_col[0]].to_dict()
 
    genes = collapse_to_gene_symbol(expr, id_to_symbol)
    genes = filter_low_variance(genes)
    out = genes.join(labels.rename("label"), how = "inner")
    out.to_csv(f"{OUT}/gse33382_processed.csv")
    
    log(f"GSE33382: {out.shape[0]} samples, {out.shape[1]-1} genes, "
          f"{int((out['label'] == 1).sum())} good / {int((out['label'] == 0).sum())} poor")
    
    return out
 
 
def preprocess_target_os():
    log("Running preprocessing on TARGET OS dataset...")

    clin = pd.read_csv(f"{RAW}/TARGET-OS.clinical.tsv.gz", sep = "\t", compression = "gzip", low_memory = False)
    expr = pd.read_csv(f"{RAW}/TARGET-OS.star_counts.tsv.gz", sep = "\t", compression = "gzip", index_col = 0)
 
    necrosis_col = "necrosis_percent.pathology_details.diagnoses"
    clin = clin[clin["sample"].isin(expr.columns)]
    clin = clin[clin[necrosis_col].notna() & (clin[necrosis_col] != "")] # values are already log2-scale so no transform needed
    labels = clin.set_index("sample")[necrosis_col].astype(float).map(lambda n: 1 if n >= 90 else 0)
 
    expr = expr[labels.index]
    expr.index = [re.sub(r"\.\d+$", "", i) for i in expr.index]  # strip Ensembl version suffix
 
    mg = mygene.MyGeneInfo()
    res = mg.querymany(list(expr.index), scopes = "ensembl.gene", fields = "symbol", species = "human")
    id_to_symbol = {r["query"]: r["symbol"] for r in res if "symbol" in r}
 
    genes = collapse_to_gene_symbol(expr, id_to_symbol)
    genes = filter_low_variance(genes)
    out = genes.join(labels.rename("label"), how = "inner")
    out.to_csv(f"{OUT}/target_os_processed.csv")
    
    log(f"TARGET-OS: {out.shape[0]} samples, {out.shape[1]-1} genes, "
          f"{int(out['label'].sum())} good / {int((out['label'] == 0).sum())} poor")
    
    return out
 
 
if __name__ == "__main__":
    preprocess_gse87437()
    preprocess_target_os()
    preprocess_gse33382()
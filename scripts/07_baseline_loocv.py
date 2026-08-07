import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

print("\nRunning elastic net logistic regression baseline (manual LOOCV)...\n")

df = pd.read_csv("data/processed/panel_filtered.csv", index_col = 0)

meta_cols = ["label", "cohort"]
gene_cols = [c for c in df.columns if c not in meta_cols]

X = df[gene_cols].values
y = df["label"].values

y_true, y_prob = [], []
selected_counts = pd.Series(0, index = gene_cols)

for train_idx, test_idx in LeaveOneOut().split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegressionCV(
        penalty = "elasticnet", solver = "saga", l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9],
        class_weight = "balanced", cv = 5, scoring = "roc_auc", random_state = 0, max_iter = 5000
    )
    clf.fit(X_train, y_train)

    y_true.append(y_test[0])
    y_prob.append(clf.predict_proba(X_test)[0, 1])
    selected_counts[np.array(gene_cols)[clf.coef_[0] != 0]] += 1

auc = roc_auc_score(y_true, y_prob)
print(f"LOOCV AUC: {auc:.3f}")

print("\nGene selected (nonzero coefficient) frequency across 112 folds:")
print(selected_counts.sort_values(ascending = False))

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef, confusion_matrix, roc_curve,
                             ConfusionMatrixDisplay)
from xgboost import XGBClassifier

from src.features import featurize_smiles, FEATURE_NAMES
from src.split import scaffold_split, assert_no_scaffold_leak

SEED = 42


def best_threshold(y, p):
    ts = np.linspace(0.2, 0.8, 61)
    return ts[int(np.argmax([matthews_corrcoef(y, p >= t) for t in ts]))]


def evaluate(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {
        "ROC_AUC": roc_auc_score(y, p),
        "PR_AUC": average_precision_score(y, p),
        "F1": f1_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
    }


def bootstrap_ci(y, p, n=1000, seed=SEED):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def tune_xgb(Xtr, ytr, Xva, yva, n_iter=15):
    space = {
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.3, 0.5, 0.8],
        "min_child_weight": [1, 3, 5],
        "reg_lambda": [1, 5, 10],
    }
    best, best_auc = None, -1
    for params in ParameterSampler(space, n_iter=n_iter, random_state=SEED):
        m = XGBClassifier(n_estimators=600, early_stopping_rounds=40, eval_metric="auc",
                          random_state=SEED, n_jobs=-1, tree_method="hist", **params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        if auc > best_auc:
            best, best_auc = {**params, "n_estimators": m.best_iteration + 1}, auc
    print(f"Best validation AUC: {best_auc:.4f} | params: {best}")
    return best


def main():
    df = pd.read_csv("data/herg_clean.csv")
    smiles = df["smiles"].tolist()
    y = df["label"].values.astype(int)

    X, valid = featurize_smiles(smiles)
    assert valid.all()
    print("Feature matrix:", X.shape)

    tr, va, te = scaffold_split(smiles)
    assert_no_scaffold_leak(smiles, tr, va, te)
    print(f"Train/Valid/Test: {len(tr)}/{len(va)}/{len(te)}")
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ytr, yva, yte = y[tr], y[va], y[te]

    results = {}

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    lr.fit(Xtr, ytr)
    results["LogReg"] = evaluate(yte, lr.predict_proba(Xte)[:, 1])

    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEED)
    rf.fit(Xtr, ytr)
    results["RandomForest"] = evaluate(yte, rf.predict_proba(Xte)[:, 1])

    params = tune_xgb(Xtr, ytr, Xva, yva)
    n_est = params.pop("n_estimators")
    xgb = XGBClassifier(n_estimators=n_est, eval_metric="auc", random_state=SEED,
                        n_jobs=-1, tree_method="hist", **params)
    xgb.fit(Xtr, ytr)

    thr = float(best_threshold(yva, xgb.predict_proba(Xva)[:, 1]))
    p_te = xgb.predict_proba(Xte)[:, 1]
    results["XGBoost"] = evaluate(yte, p_te, thr)
    results["XGBoost"]["threshold"] = thr
    lo, hi = bootstrap_ci(yte, p_te)
    results["XGBoost"]["ROC_AUC_95CI"] = [lo, hi]

    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED)
    xgb_r = XGBClassifier(n_estimators=n_est, eval_metric="auc", random_state=SEED,
                          n_jobs=-1, tree_method="hist", **params).fit(Xr_tr, yr_tr)
    results["XGBoost_RANDOM_split"] = evaluate(yr_te, xgb_r.predict_proba(Xr_te)[:, 1])

    print(pd.DataFrame(results).T.round(3))

    fpr, tpr, _ = roc_curve(yte, p_te)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"XGBoost (AUC={results['XGBoost']['ROC_AUC']:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("Scaffold-split test ROC"); plt.legend()
    plt.savefig("reports/figures/roc_curve.png", dpi=200, bbox_inches="tight"); plt.close()

    cm = confusion_matrix(yte, (p_te >= thr).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=["Non-blocker", "Blocker"]).plot(cmap="Blues")
    plt.savefig("reports/figures/confusion_matrix.png", dpi=200, bbox_inches="tight"); plt.close()

    joblib.dump({"model": xgb, "threshold": thr, "feature_names": FEATURE_NAMES},
                "models/cardioguard_xgb.joblib")
    with open("models/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame({"smiles": smiles, "label": y,
                  "split": np.select([np.isin(np.arange(len(y)), tr), np.isin(np.arange(len(y)), va)],
                                     ["train", "valid"], "test")}).to_csv("data/herg_splits.csv", index=False)
    print("Saved model + metrics.")


if __name__ == "__main__":
    main()

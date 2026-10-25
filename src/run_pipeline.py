"""Leakage-safe disease-vs-control classifier for gene-expression data.

  python src/run_pipeline.py --csv data/expr.csv --label-col label   # real data
  python src/run_pipeline.py --demo                                  # SYNTHETIC, code test only

CSV format: rows = samples, columns = genes (log2 scale) + one label column.
"""
import argparse, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, GridSearchCV, StratifiedKFold,
                                     RepeatedStratifiedKFold, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_auc_score, classification_report, RocCurveDisplay,
                             balanced_accuracy_score)

SEED = 42

def load(args):
    if args.demo:
        rng = np.random.default_rng(SEED)
        X = rng.normal(size=(80, 2000)); y = np.repeat([0, 1], 40)
        X[y == 1, :20] += 1.0   # 20 synthetic informative genes
        return pd.DataFrame(X, columns=[f"GENE{i}" for i in range(2000)]), pd.Series(y)
    df = pd.read_csv(args.csv, index_col=0)
    df = df.dropna(axis=1, how="any")   # drop genes with missing values
    y = pd.Series(LabelEncoder().fit_transform(df.pop(args.label_col)), index=df.index)
    return df, y

def make(clf):
    return Pipeline([("scale", StandardScaler()),
                     ("select", SelectKBest(f_classif)),  # fitted inside each CV fold
                     ("clf", clf)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--label-col", default="label")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if not (args.csv or args.demo): ap.error("choose --csv or --demo")

    X, y = load(args)
    X = X.reset_index(drop=True); y = y.reset_index(drop=True)
    assert y.nunique() == 2, "binary label expected"
    print("Samples x genes:", X.shape, "| classes:", y.value_counts().to_dict(),
          "| missing:", int(X.isna().sum().sum()))

    p = PCA(2).fit_transform(StandardScaler().fit_transform(X))  # QC plot only
    plt.figure(figsize=(5, 4)); plt.scatter(p[:, 0], p[:, 1], c=y, cmap="coolwarm", s=20)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA (QC only)")
    plt.savefig("results/figures/pca.png", dpi=200, bbox_inches="tight"); plt.close()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    ks = [k for k in (50, 200, 500) if k < X.shape[1]] or [X.shape[1]]
    models = {
        "LogReg": (make(LogisticRegression(max_iter=5000, class_weight="balanced")),
                   {"select__k": ks, "clf__C": [0.01, 0.1, 1]}),
        "RandomForest": (make(RandomForestClassifier(n_estimators=100, random_state=SEED,
                                                     class_weight="balanced")),
                         {"select__k": ks, "clf__max_depth": [None, 5]}),
        "SVM": (make(SVC(class_weight="balanced")),
                {"select__k": ks, "clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]}),
    }

    inner = StratifiedKFold(5, shuffle=True, random_state=1)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=2)
    cv = {}
    for name, (pipe, grid) in models.items():
        gs = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=-1)
        cv[name] = cross_val_score(gs, X_tr, y_tr, cv=outer, scoring="roc_auc", n_jobs=-1)
        print(f"{name}: nested-CV AUC {cv[name].mean():.3f} +/- {cv[name].std():.3f}")
    plt.figure(figsize=(5, 4)); plt.boxplot(list(cv.values()), tick_labels=list(cv.keys()))
    plt.ylabel("Nested CV AUC")
    plt.savefig("results/figures/cv_boxplot.png", dpi=200, bbox_inches="tight"); plt.close()

    # Leakage check: shuffled labels must give AUC ~ 0.5
    pipe, grid = models["LogReg"]
    perm = np.random.default_rng(0).permutation(y_tr.values)
    perm_auc = cross_val_score(GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=-1),
                               X_tr, perm, cv=StratifiedKFold(5, shuffle=True, random_state=3),
                               scoring="roc_auc").mean()
    print(f"Permuted-label AUC (expect ~0.5): {perm_auc:.3f}")

    best_name = max(cv, key=lambda k: cv[k].mean())
    pipe, grid = models[best_name]
    best = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr).best_estimator_
    score = best.predict_proba(X_te)[:, 1] if hasattr(best, "predict_proba") else best.decision_function(X_te)
    test_auc = roc_auc_score(y_te, score)   # test set used ONCE
    print(f"\nBest: {best_name} | TEST AUC {test_auc:.3f} | balanced acc "
          f"{balanced_accuracy_score(y_te, best.predict(X_te)):.3f}")
    print(classification_report(y_te, best.predict(X_te)))
    RocCurveDisplay.from_predictions(y_te, score)
    plt.savefig("results/figures/roc.png", dpi=200, bbox_inches="tight"); plt.close()

    # Save metrics BEFORE explanation, so results are kept even if SHAP cannot run
    metrics = {"data": "SYNTHETIC DEMO" if args.demo else args.csv, "best_model": best_name,
               "test_auc": test_auc, "permuted_auc": perm_auc,
               "nested_cv_auc": {k: [float(v.mean()), float(v.std())] for k, v in cv.items()}}

    prep = best[:-1]; names = prep.get_feature_names_out()
    Xt = pd.DataFrame(prep.transform(X_tr), columns=names)
    Xe = pd.DataFrame(prep.transform(X_te), columns=names)
    clf = best[-1]
    try:
        import shap
        if isinstance(clf, SVC):
            clf = RandomForestClassifier(100, random_state=SEED).fit(Xt, y_tr)
            print("Note: SHAP computed on an auxiliary RandomForest (SVM too slow to explain).")
        sv = shap.Explainer(clf, Xt)(Xe)
        vals = sv[:, :, 1] if sv.values.ndim == 3 else sv
        shap.plots.beeswarm(vals, show=False, max_display=15)
        plt.savefig("results/figures/shap_beeswarm.png", dpi=200, bbox_inches="tight"); plt.close()
        imp = pd.Series(np.abs(vals.values).mean(0), index=names)
        metrics["explanation"] = "SHAP"
    except Exception as e:   # e.g. blocked DLL on locked-down Windows machines
        print(f"SHAP unavailable ({type(e).__name__}); using permutation importance instead.")
        from sklearn.inspection import permutation_importance
        r = permutation_importance(best[-1], Xe, y_te, scoring="roc_auc", n_repeats=30, random_state=SEED)
        imp = pd.Series(r.importances_mean, index=names)
        top = imp.sort_values(ascending=False).head(15)[::-1]
        plt.figure(figsize=(6, 5)); plt.barh(top.index, top.values)
        plt.xlabel("Permutation importance (drop in AUC)")
        plt.savefig("results/figures/permutation_importance.png", dpi=200, bbox_inches="tight"); plt.close()
        metrics["explanation"] = "permutation importance (SHAP unavailable)"
    imp.sort_values(ascending=False).head(20).to_csv("results/top_genes.csv")
    json.dump(metrics, open("results/metrics.json", "w"), indent=2)

if __name__ == "__main__":
    main()

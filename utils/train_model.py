"""
train_model.py
==============
Train the two surrogate models used by the optimizer.

1. Strength: chained CatBoost regressors (7d -> 28d -> 56d).
     Training  : stage k uses the TRUE strength of stage k-1 as an input feature.
     Inference : stage k uses the PREDICTED strength of stage k-1 (errors propagate).
     One 80/20 split of mixes is shared by all stages and by the chloride model;
     stage-wise (true input) and chained (predicted input) test metrics are both reported.
2. Chloride: CatBoost classifier for the 28-day RCPT result (pass = coulomb < 1200, i.e. Low or better).

Data: data/Concrete_Dataset_SI.xlsx (built by utils/prepare_data.py, all quantities in kg/m3).

Output:
  models/strength_chain.pkl     keys: models, feature_names, unit
  models/chloride_clf.pkl       keys: model, feature_names, unit, limit
  models/metrics.json           hold-out and cross-validation metrics (for the paper)

Usage:  python utils/train_model.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (accuracy_score, mean_absolute_error, r2_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     train_test_split)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_XLSX = os.path.join(ROOT, "data", "Concrete_Dataset_SI.xlsx")
OUT_DIR = os.path.join(ROOT, "models")
SEED = 42

RAW_VARS = ["PC", "FA", "SC", "FAGG", "CAGG", "WATER", "AEA", "WR_HR", "WR", "ACC"]
FEATURES = (RAW_VARS + ["TOTAL_BINDER", "w/b", "b/a", "SCM%", "CAGG%", "FAGG%", "PC%", "FA%", "SC%",
                        "AEA_pct", "WR_HR_pct", "WR_pct", "ACC_pct"])
RCPT_LIMIT = 1200

REG_GRID = {"iterations": [500, 1000], "learning_rate": [0.05, 0.1],
            "depth": [6, 8], "l2_leaf_reg": [3, 5, 10]}
CLF_GRID = {"iterations": [300, 600], "learning_rate": [0.03, 0.05, 0.1],
            "depth": [4, 6], "l2_leaf_reg": [3, 5, 10]}


def train_strength(df: pd.DataFrame, train_idx, test_idx) -> dict:
    """Chained 7d -> 28d -> 56d regressors, one train/test split of mixes shared by all stages."""
    tr, te = df.loc[train_idx], df.loc[test_idx]
    models = {}
    for target, prev in [("7day", None), ("28day", "7day"), ("56day", "28day")]:
        feats = FEATURES + ([prev] if prev else [])
        sub = tr.dropna(subset=[target] + ([prev] if prev else []))
        print(f"\n>>> Tuning [{target}] n_train={len(sub)}")
        gs = GridSearchCV(CatBoostRegressor(random_seed=SEED, verbose=0), REG_GRID,
                          cv=5, scoring="r2", n_jobs=-1)
        gs.fit(sub[feats], sub[target])
        print(f"  best {gs.best_params_}  CV R2={gs.best_score_:.4f}")
        models[target] = gs.best_estimator_

    metrics = {"stagewise": {}, "chained": {}}
    # (a) stage-wise: true previous-stage strength as input
    for target, prev in [("7day", None), ("28day", "7day"), ("56day", "28day")]:
        feats = FEATURES + ([prev] if prev else [])
        t_ = te.dropna(subset=[target] + ([prev] if prev else []))
        p = models[target].predict(t_[feats])
        metrics["stagewise"][target] = {"n_test": len(t_), "r2": round(r2_score(t_[target], p), 4),
                                        "mae": round(mean_absolute_error(t_[target], p), 3)}
        print(f"  stagewise [{target}] {metrics['stagewise'][target]}")
    # (b) chained: predicted previous-stage strength is propagated (deployment setting)
    te = te.copy()
    te["p7"] = models["7day"].predict(te[FEATURES])
    te["p28"] = models["28day"].predict(te[FEATURES].assign(**{"7day": te["p7"]}))
    te["p56"] = models["56day"].predict(te[FEATURES].assign(**{"28day": te["p28"]}))
    for day, col in [("7day", "p7"), ("28day", "p28"), ("56day", "p56")]:
        t_ = te.dropna(subset=[day])
        metrics["chained"][day] = {"n_test": len(t_), "r2": round(r2_score(t_[day], t_[col]), 4),
                                   "mae": round(mean_absolute_error(t_[day], t_[col]), 3)}
        print(f"  chained   [{day}] {metrics['chained'][day]}")

    joblib.dump({"models": models, "feature_names": FEATURES, "unit": "kg/m3"},
                os.path.join(OUT_DIR, "strength_chain.pkl"))
    return metrics


def train_chloride(df: pd.DataFrame, train_idx, test_idx) -> dict:
    """Binary classifier: 28-day RCPT < RCPT_LIMIT coulombs."""
    sub = df.dropna(subset=["RCPT_28d"]).copy()
    sub["y"] = (sub["RCPT_28d"] < RCPT_LIMIT).astype(int)
    X, y = sub[FEATURES], sub["y"]
    print(f"\n>>> Chloride classifier  n={len(sub)}  pass rate={y.mean():.2f}")

    is_tr = sub.index.isin(train_idx)           # same mix split as the strength model
    Xtr, Xte, ytr, yte = X[is_tr], X[~is_tr], y[is_tr], y[~is_tr]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(CatBoostClassifier(random_seed=SEED, verbose=0), CLF_GRID,
                      cv=cv, scoring="roc_auc", n_jobs=-1)
    gs.fit(Xtr, ytr)
    print(f"  best {gs.best_params_}  CV AUC={gs.best_score_:.4f}")
    clf = gs.best_estimator_
    prob = clf.predict_proba(Xte)[:, 1]
    metrics = {"n": len(sub), "n_train": len(Xtr), "n_test": len(Xte),
               "pass_rate": round(float(y.mean()), 3),
               "cv_auc_train": round(gs.best_score_, 4),
               "test_auc": round(roc_auc_score(yte, prob), 4),
               "test_acc": round(accuracy_score(yte, prob >= 0.5), 4)}
    # cross-validated estimate on all rows with the selected hyper-parameters
    p_all = cross_val_predict(CatBoostClassifier(random_seed=SEED, verbose=0, **gs.best_params_),
                              X, y, cv=cv, method="predict_proba")[:, 1]
    metrics["cv_auc_all"] = round(roc_auc_score(y, p_all), 4)
    metrics["cv_acc_all"] = round(accuracy_score(y, p_all >= 0.5), 4)
    print(f"  {metrics}")

    clf.fit(X, y)       # final model on all 28-day mixes
    joblib.dump({"model": clf, "feature_names": FEATURES, "unit": "kg/m3",
                 "limit": RCPT_LIMIT, "age_days": 28},
                os.path.join(OUT_DIR, "chloride_clf.pkl"))
    return metrics


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_excel(DATA_XLSX, sheet_name="mix_level")
    print(f"Mixes: {len(df)}")
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=SEED)
    metrics = {"split": {"n_train_mixes": len(train_idx), "n_test_mixes": len(test_idx), "seed": SEED},
               "strength": train_strength(df, train_idx, test_idx),
               "chloride": train_chloride(df, train_idx, test_idx)}
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved models and metrics to {OUT_DIR}")


if __name__ == "__main__":
    main()

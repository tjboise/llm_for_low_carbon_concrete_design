"""
model_comparison.py
===================
Compare strength surrogates on the SI dataset over repeated random 80/20 splits of mixes.

Models : RandomForest, XGBoost, CatBoost, MLP   (fixed hyper-parameters, same for every split)
Setups : "independent"          each age predicted from the mix features only
         "chain_true_input"     stage k gets the TRUE strength of stage k-1 (stage-wise ability)
         "chain_pred_input"     stage k gets the PREDICTED strength of stage k-1 (deployment setting)

Output: results/model_comparison/{raw.csv, summary.csv}   (mean and std of R2 / MAE over splits)

Usage:  python utils/model_comparison.py [--splits 10]
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.train_model import DATA_XLSX, FEATURES, ROOT  # noqa: E402

AGES = [("7day", None), ("28day", "7day"), ("56day", "28day")]


def make(name, seed):
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)
    if name == "XGBoost":
        return XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.9,
                            random_state=seed, n_jobs=-1)
    if name == "CatBoost":
        return CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                                 random_seed=seed, verbose=0, thread_count=-1)
    if name == "MLP":
        return make_pipeline(StandardScaler(), MLPRegressor((64, 64), max_iter=3000, early_stopping=True,
                                                            random_state=seed))
    raise ValueError(name)


def evaluate(df, name, seed):
    tr_i, te_i = train_test_split(df.index, test_size=0.2, random_state=seed)
    tr, te = df.loc[tr_i], df.loc[te_i].copy()
    rows = []
    # independent
    for age, _ in AGES:
        t = tr.dropna(subset=[age])
        m = make(name, seed).fit(t[FEATURES], t[age])
        e = te.dropna(subset=[age])
        p = m.predict(e[FEATURES])
        rows.append((name, "independent", age, r2_score(e[age], p), mean_absolute_error(e[age], p)))
    # chain, trained with true previous-stage strength
    models = {}
    for age, prev in AGES:
        feats = FEATURES + ([prev] if prev else [])
        t = tr.dropna(subset=[age] + ([prev] if prev else []))
        models[age] = make(name, seed).fit(t[feats], t[age])
    for age, prev in AGES:
        feats = FEATURES + ([prev] if prev else [])
        e = te.dropna(subset=[age] + ([prev] if prev else []))
        p = models[age].predict(e[feats])
        rows.append((name, "chain_true_input", age, r2_score(e[age], p), mean_absolute_error(e[age], p)))
    te["p7"] = models["7day"].predict(te[FEATURES])
    te["p28"] = models["28day"].predict(te[FEATURES].assign(**{"7day": te["p7"]}))
    te["p56"] = models["56day"].predict(te[FEATURES].assign(**{"28day": te["p28"]}))
    for age, col in [("7day", "p7"), ("28day", "p28"), ("56day", "p56")]:
        e = te.dropna(subset=[age])
        rows.append((name, "chain_pred_input", age, r2_score(e[age], e[col]), mean_absolute_error(e[age], e[col])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=10)
    a = ap.parse_args()
    df = pd.read_excel(DATA_XLSX, sheet_name="mix_level")
    out = []
    for name in ["RandomForest", "XGBoost", "CatBoost", "MLP"]:
        for seed in range(a.splits):
            for r in evaluate(df, name, seed):
                out.append({"model": r[0], "setup": r[1], "age": r[2], "split": seed, "r2": r[3], "mae": r[4]})
        print("done", name, flush=True)
    raw = pd.DataFrame(out)
    d = os.path.join(ROOT, "results", "model_comparison")
    os.makedirs(d, exist_ok=True)
    raw.to_csv(os.path.join(d, "raw.csv"), index=False)
    summ = raw.groupby(["model", "setup", "age"])[["r2", "mae"]].agg(["mean", "std"]).round(4)
    summ.to_csv(os.path.join(d, "summary.csv"))
    print(summ.to_string())


if __name__ == "__main__":
    main()

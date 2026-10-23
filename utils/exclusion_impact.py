"""
exclusion_impact.py
===================
Effect of excluding records whose strength decreases with age (reviewer 3, comment 1).

The records are removed in utils/prepare_data.py. Here the chained strength model is trained with and without them.
  A: training set = cleaned training mixes
  B: training set = cleaned training mixes + all non-monotonic mixes (that pass the volume-balance filter)
Both models are tested on the same held-out cleaned mixes (10 random 80/20 splits).
The non-monotonic mixes are also predicted by model A to show how large their errors are.

Output: results/tables/exclusion_impact.csv (per split) and exclusion_impact_summary.csv
Usage : python utils/exclusion_impact.py
"""
import os
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import utils.prepare_data as pd_  # noqa: E402
from utils.train_model import FEATURES  # noqa: E402

ROOT = pd_.ROOT
PARAMS = {"7day": dict(depth=6, iterations=1000, l2_leaf_reg=3, learning_rate=0.05),
          "28day": dict(depth=6, iterations=1000, l2_leaf_reg=10, learning_rate=0.1),
          "56day": dict(depth=6, iterations=500, l2_leaf_reg=10, learning_rate=0.05)}
AGES = [("7day", None), ("28day", "7day"), ("56day", "28day")]


def load_all():
    """Strength mixes in SI units that pass the volume-balance filter, with a flag for non-monotonic strength."""
    mix = pd_.load_strength()
    for v in pd_.BULK_VARS:
        mix[v] = mix[v] * pd_.LB_YD3_TO_KG_M3
    for v in pd_.ADMIX_VARS:
        mix[v] = mix[v] * pd_.OZ_YD3_TO_KG_M3
    mix["Vfinal"] = pd_.vfinal(mix)
    mix = mix[mix["Vfinal"].between(pd_.VFINAL_MIN, pd_.VFINAL_MAX)]
    mix = pd_.add_derived(mix)
    mix["nonmono"] = (mix["7day"] > mix["28day"]) | (mix["28day"] > mix["56day"])
    return mix


def fit_chain(train, seed):
    models = {}
    for age, prev in AGES:
        feats = FEATURES + ([prev] if prev else [])
        t = train.dropna(subset=[age] + ([prev] if prev else []))
        models[age] = CatBoostRegressor(random_seed=seed, verbose=0, **PARAMS[age]).fit(t[feats], t[age])
    return models


def evaluate(models, test):
    out = {}
    for age, prev in AGES:                       # stage-wise: true previous-stage strength as input
        feats = FEATURES + ([prev] if prev else [])
        t = test.dropna(subset=[age] + ([prev] if prev else []))
        p = models[age].predict(t[feats])
        out[f"{age}_stagewise_r2"] = r2_score(t[age], p)
        out[f"{age}_stagewise_mae"] = mean_absolute_error(t[age], p)
    te = test.copy()                              # chained: predicted previous stage is propagated
    te["p7"] = models["7day"].predict(te[FEATURES])
    te["p28"] = models["28day"].predict(te[FEATURES].assign(**{"7day": te["p7"]}))
    for age, col in (("28day", "p28"),):
        t = te.dropna(subset=[age])
        out[f"{age}_chained_r2"] = r2_score(t[age], t[col])
        out[f"{age}_chained_mae"] = mean_absolute_error(t[age], t[col])
    return out


def main():
    df = load_all()
    clean, bad = df[~df["nonmono"]], df[df["nonmono"]]
    print(f"clean mixes: {len(clean)}   non-monotonic mixes: {len(bad)}")
    diff = (bad["28day"] - bad["56day"]).where(bad["28day"] > bad["56day"], bad["7day"] - bad["28day"])
    print("strength decrease of the non-monotonic mixes (MPa): "
          f"min {diff.min():.1f}, median {diff.median():.1f}, max {diff.max():.1f}; "
          f"relative {np.round((diff / bad['28day']).median() * 100, 1)}% of the 28-day strength (median)")
    rows = []
    for seed in range(10):
        tr_i, te_i = train_test_split(clean.index, test_size=0.2, random_state=seed)
        tr, te = clean.loc[tr_i], clean.loc[te_i]
        for name, train in (("A_clean", tr), ("B_with_nonmonotonic", pd.concat([tr, bad]))):
            m = fit_chain(train, seed)
            r = evaluate(m, te)
            r.update({"setup": name, "split": seed})
            if name == "A_clean":   # error of model A on the excluded mixes (28-day, true 7-day input)
                p = m["28day"].predict(bad[FEATURES + ["7day"]])
                r["excluded_28day_mae"] = mean_absolute_error(bad["28day"], p)
                r["excluded_28day_bias"] = float(np.mean(p - bad["28day"]))
            rows.append(r)
        print("split", seed, "done", flush=True)
    res = pd.DataFrame(rows)
    d = os.path.join(ROOT, "results", "tables")
    os.makedirs(d, exist_ok=True)
    res.to_csv(os.path.join(d, "exclusion_impact.csv"), index=False)
    summ = res.groupby("setup").agg(["mean", "std"]).drop(columns="split").round(3)
    summ.to_csv(os.path.join(d, "exclusion_impact_summary.csv"))
    print(summ.T.to_string())
    a = res[res.setup == "A_clean"].set_index("split")
    b = res[res.setup == "B_with_nonmonotonic"].set_index("split")
    print("difference B - A, 28-day stage-wise R2: mean %.4f std %.4f" % ((b["28day_stagewise_r2"] - a["28day_stagewise_r2"]).mean(),
                                                                         (b["28day_stagewise_r2"] - a["28day_stagewise_r2"]).std()))


if __name__ == "__main__":
    main()

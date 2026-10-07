"""
sensitivity.py
==============
Post-hoc sensitivity analysis of the best solutions (GA reference and every LLM run). No LLM calls.

Three perturbations (Monte Carlo, N samples each):
  1. Prediction error   : 28-day strength ~ N(pred, sigma28), sigma28 = std of the chained 28-day
                          residual on the held-out mixes.  -> P(strength >= floor)
  2. Material variability: every ingredient quantity multiplied by N(1, cv), cv in {2%, 5%}, then the surrogate
                          is re-evaluated.                   -> share of samples that still satisfy
                          strength (and chloride if durability is on); all hard constraints too.
  3. GWP factor error   : every GWP factor multiplied by U(1-d, 1+d), d in {10%, 20%}.
                                                             -> mean/std of GWP and of OGR vs the GA reference.

Output: results/sensitivity/{solutions.csv, summary.csv}
Usage : python utils/sensitivity.py [--samples 1000]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import optimizer_core as oc  # noqa: E402

RES = os.path.join(oc.ROOT, "results")
CVS = [0.02, 0.05]
GWP_DELTAS = [0.10, 0.20]


def chained_sigma(spec):
    df = pd.read_excel(oc.DATASET_PATH, sheet_name="mix_level")
    _, te = train_test_split(df.index, test_size=0.2, random_state=42)
    d = df.loc[te].dropna(subset=["28day"])
    res = d["28day"].values - oc.predict_batch(spec, d)["pred_28day"].values
    return float(np.std(res))


def solutions():
    rows = []
    for scn in sorted(os.listdir(os.path.join(RES, "ga"))):
        f = os.path.join(RES, "ga", scn, "reference.json")
        if os.path.exists(f):
            rows.append({"scenario": scn, "method": "GA", "run": 0, "mix": json.load(open(f))["reference"]})
    base = os.path.join(RES, "llm")
    for scn in sorted(os.listdir(base)):
        for m in sorted(os.listdir(os.path.join(base, scn))):
            for run in sorted(os.listdir(os.path.join(base, scn, m))):
                f = os.path.join(base, scn, m, run, "trajectory.csv")
                if os.path.exists(f) and int(run.split("_")[1]) <= 5:
                    t = pd.read_csv(f)
                    rows.append({"scenario": scn, "method": m, "run": int(run.split("_")[1]),
                                 "mix": t.loc[t["gwp"].idxmin()].to_dict()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=1000)
    a = ap.parse_args()
    spec = oc.Spec()
    sigma = chained_sigma(spec)
    print(f"sigma28 (chained residual std) = {sigma:.2f} MPa")
    rng = np.random.default_rng(0)
    sols = solutions()
    ga_gwp = {s["scenario"]: s["mix"]["gwp"] for s in sols if s["method"] == "GA"}
    out = []
    for s in sols:
        smin = int(s["scenario"].split("_")[0][1:])
        dur = s["scenario"].endswith("_dur")
        mix = {v: float(s["mix"][v]) for v in oc.RAW_VARS}
        pred = oc.predict(spec, mix)
        row = {"scenario": s["scenario"], "method": s["method"], "run": s["run"], "gwp": oc.compute_gwp(mix, spec),
               "pred_28day": pred["28day"], "chloride_prob": pred["chloride_prob"],
               "p_strength_ok_pred_error": float(1 - norm.cdf((smin - pred["28day"]) / sigma))}
        base = np.array([mix[v] for v in oc.RAW_VARS])
        for cv in CVS:
            X = np.clip(base * rng.normal(1.0, cv, size=(a.samples, len(base))),
                        [spec.raw_b[v]["min"] for v in oc.RAW_VARS] and 0, None)
            df = pd.DataFrame(X, columns=oc.RAW_VARS)
            pr = oc.predict_batch(spec, df)
            row[f"p_strength_ok_cv{int(cv * 100)}"] = float((pr["pred_28day"] >= smin).mean())
            if dur:
                row[f"p_chloride_ok_cv{int(cv * 100)}"] = float((pr["chloride_prob"] >= spec.min_pass_prob).mean())
        for d in GWP_DELTAS:
            f = np.array([spec.gwp_factors[v] for v in oc.RAW_VARS])
            g = (base * f * rng.uniform(1 - d, 1 + d, size=(a.samples, len(base)))).sum(axis=1)
            row[f"gwp_mean_d{int(d * 100)}"] = float(g.mean())
            row[f"gwp_std_d{int(d * 100)}"] = float(g.std())
        row["ga_gwp"] = ga_gwp.get(s["scenario"], np.nan)
        out.append(row)
    d = os.path.join(RES, "sensitivity")
    os.makedirs(d, exist_ok=True)
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(d, "solutions.csv"), index=False)
    cols = [c for c in df.columns if c.startswith(("p_", "gwp_std"))]
    summ = df.groupby(["scenario", "method"])[cols].mean().round(3)
    summ.to_csv(os.path.join(d, "summary.csv"))
    print(summ.to_string())


if __name__ == "__main__":
    main()

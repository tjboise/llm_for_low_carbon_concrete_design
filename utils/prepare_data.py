"""
prepare_data.py
===============
Single, reproducible pipeline that builds the SI dataset used for every model and experiment.

Inputs
  data/raw/Concrete_Data_raw_imperial.csv      strength database (756 mixes)
        lb/yd^3 (PC, FA, SC, SF, FAGG, CAGG, WATER), oz/yd^3 (AEA, WR_HR, WR, ACC), MPa
  data/classified_chloride_Classification.xlsx RCPT (coulomb) tests, sheet all_data (1314 tests)

Steps
  1. WR / WR_HR labels: the strength file and the chloride file disagree (they are swapped).
     The chloride-file labelling is used (WR_HR = high-range, large dosage; WR = normal).
  2. Unit conversion to kg/m^3:  lb/yd^3 x 0.5933,  oz/yd^3 x 0.03708.
  3. Drop mixes whose strength decreases with age (7d > 28d or 28d > 56d).
     Then Vfinal (Pfeiffer et al. 2024) and filter  Vfinal in [0.95, 1.05].
  4. GWP and derived ratios.
  5. Chloride tests are matched to mixes (all 11 ingredient quantities, tol 0.5), special mixes
     (second cement, latex, lightweight, fibre) are dropped, and the chloride class is
     recomputed from the coulomb value (Port Authority classes).

Outputs
  data/Concrete_Dataset_SI.xlsx    sheets: mix_level, chloride_tests, readme
  data/constraints.json            bounds for generated mixes

Usage:  python utils/prepare_data.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW_PATH = os.path.join(ROOT, "data", "raw", "Concrete_Data_raw_imperial.csv")
CL_PATH = os.path.join(ROOT, "data", "classified_chloride_Classification.xlsx")
OUT_XLSX = os.path.join(ROOT, "data", "Concrete_Dataset_SI.xlsx")
OUT_JSON = os.path.join(ROOT, "data", "constraints.json")

LB_YD3_TO_KG_M3 = 0.45359237 / 0.764555    # 0.5933
OZ_YD3_TO_KG_M3 = 0.028349523 / 0.764555   # 0.03708

BULK_VARS = ["PC", "FA", "SC", "SF", "FAGG", "CAGG", "WATER"]
ADMIX_VARS = ["AEA", "WR_HR", "WR", "ACC"]
MIX_VARS = BULK_VARS + ADMIX_VARS
RAW_VARS = ["PC", "FA", "SC", "FAGG", "CAGG", "WATER", "AEA", "WR_HR", "WR", "ACC"]

GWP_FACTORS = {"PC": 1.048, "FA": 0.328, "SC": 0.264, "FAGG": 0.0026, "CAGG": 0.0037}
DENSITIES = {"PC": 3150, "FA": 2200, "SC": 2900, "SF": 2200, "FAGG": 2630, "CAGG": 2710,
             "WATER": 1000, "AEA": 1010, "WR_HR": 1080, "WR": 1140, "ACC": 1340}
VFINAL_MIN, VFINAL_MAX = 0.95, 1.05
RCPT_LIMIT = 1200   # coulombs; "Low" or better (Port Authority class boundary)
DERIVED_VARS = ["w/b", "b/a", "SCM%", "CAGG%", "FAGG%", "PC%", "FA%", "SC%",
                "AEA_pct", "WR_HR_pct", "WR_pct", "ACC_pct"]


def pa_class(coulomb: pd.Series) -> pd.Series:
    """Chloride permeability classes recommended by the Port Authority (RCPT, coulombs)."""
    bins = [-np.inf, 800, 1200, 2000, np.inf]
    labels = ["Very Low", "Low", "Moderate", "High"]
    return pd.cut(coulomb, bins=bins, labels=labels, right=False).astype(str)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tb = df["PC"] + df["FA"] + df["SC"]
    agg = df["FAGG"] + df["CAGG"]
    df["TOTAL_BINDER"] = tb
    df["w/b"] = df["WATER"] / tb
    df["b/a"] = tb / agg
    df["SCM%"] = (df["FA"] + df["SC"]) / tb
    df["CAGG%"] = df["CAGG"] / agg
    df["FAGG%"] = df["FAGG"] / agg
    for v in ["PC", "FA", "SC"]:
        df[f"{v}%"] = df[v] / tb
    for v in ADMIX_VARS:
        df[f"{v}_pct"] = df[v] / (tb + 1e-9)
    return df


def vfinal(df: pd.DataFrame) -> pd.Series:
    vm = sum(df[v] / DENSITIES[v] for v in DENSITIES if v in df)
    air = np.where(df["AEA"] / (df["PC"] + 1e-9) >= 0.000244, 0.07, 0.03)
    return vm + air


def load_strength() -> pd.DataFrame:
    raw = pd.read_csv(RAW_PATH)
    raw = raw.rename(columns={"WR": "WR_HR", "WR_HR": "WR"})        # step 1
    return raw[MIX_VARS + ["7day", "28day", "56day"]].copy()


def match_chloride(raw_imp: pd.DataFrame) -> pd.DataFrame:
    cl = pd.read_excel(CL_PATH, sheet_name="all_data").rename(columns={"SS": "SC"})
    cl["PC"] = cl["PC1"] + cl["PC2"]
    n0 = len(cl)
    special = (cl["PC2"] > 0) | (cl["IS_LATEX"] > 0) | (cl["IS_LIGHT_WEIGHT"] > 0) | (cl["FIBER"] > 0)
    dist, idx = cKDTree(raw_imp[MIX_VARS].values).query(cl[MIX_VARS].values)
    cl["mix_id"] = raw_imp.index.values[idx]
    cl = cl[(dist <= 0.5) & ~special.values]
    print(f"Chloride tests: {n0} -> matched to strength mixes and standard: {len(cl)}")
    out = cl[["mix_id", "AGE", "COULOMB_TEST"]].rename(columns={"AGE": "age", "COULOMB_TEST": "coulomb"})
    out["rcpt_class"] = pa_class(out["coulomb"])
    out["pass_rcpt"] = (out["coulomb"] < RCPT_LIMIT).astype(int)
    return out


def main():
    mix = load_strength()
    mix.index.name = "mix_id"
    print(f"Strength mixes: {len(mix)}")
    chl = match_chloride(mix)                      # matching on imperial values

    for v in BULK_VARS:                            # step 2
        mix[v] = mix[v] * LB_YD3_TO_KG_M3
    for v in ADMIX_VARS:
        mix[v] = mix[v] * OZ_YD3_TO_KG_M3

    # strength must not decrease with age (7d <= 28d <= 56d), else the record is inconsistent
    bad = (mix["7day"] > mix["28day"]) | (mix["28day"] > mix["56day"])
    print(f"Dropped {int(bad.sum())} mixes with non-monotonic strength")
    mix = mix[~bad]

    mix["Vfinal"] = vfinal(mix)                    # step 3
    mix = mix[mix["Vfinal"].between(VFINAL_MIN, VFINAL_MAX)]
    print(f"After Vfinal in [{VFINAL_MIN}, {VFINAL_MAX}]: {len(mix)}")

    mix = add_derived(mix)                         # step 4
    mix["GWP"] = sum(mix[k] * f for k, f in GWP_FACTORS.items())

    chl = chl[chl["mix_id"].isin(mix.index)]       # chloride tests of kept mixes
    print(f"Chloride tests on kept mixes: {len(chl)} ({chl['mix_id'].nunique()} mixes)")
    wide = chl.groupby(["mix_id", "age"])["coulomb"].mean().unstack()
    wide.columns = [f"RCPT_{int(a)}d" for a in wide.columns]
    mix = mix.join(wide)

    cols = (MIX_VARS + ["TOTAL_BINDER", "GWP", "Vfinal"] + DERIVED_VARS
            + ["7day", "28day", "56day"] + list(wide.columns))
    mix = mix[cols].reset_index()
    feat = ["mix_id"] + MIX_VARS + ["TOTAL_BINDER", "w/b", "b/a", "SCM%", "CAGG%", "FAGG%",
                                    "PC%", "FA%", "SC%", "AEA_pct", "WR_HR_pct", "WR_pct", "ACC_pct"]
    long = chl.merge(mix[feat], on="mix_id")

    readme = pd.DataFrame({"item": [
        "units", "WR / WR_HR", "strength cleaning", "Vfinal", "chloride matching", "RCPT class", "pass_rcpt",
        "mix_level", "chloride_tests"], "description": [
        "kg/m3 (lb/yd3 x 0.5933 for binders, aggregates, water; oz/yd3 x 0.03708 for admixtures)",
        "labels follow the chloride file; they are swapped in the original strength file",
        "non-monotonic strength (7d>28d or 28d>56d) removed before the Vfinal filter",
        f"Pfeiffer 2024; kept in [{VFINAL_MIN}, {VFINAL_MAX}]",
        "all 11 ingredient quantities within 0.5 (imperial units); second-cement, latex, "
        "lightweight and fibre mixes dropped",
        "Port Authority: <800 Very Low, 800-1200 Low, 1200-2000 Moderate, >2000 High (coulombs)",
        f"1 if coulomb < {RCPT_LIMIT} (Low or better)",
        "one row per mix: strength + RCPT_28d/90d/120d (mean coulomb, NaN if not tested)",
        "one row per RCPT test (age in days) with mix composition"]})
    with pd.ExcelWriter(OUT_XLSX) as xw:
        mix.to_excel(xw, sheet_name="mix_level", index=False)
        long.to_excel(xw, sheet_name="chloride_tests", index=False)
        readme.to_excel(xw, sheet_name="readme", index=False)
    print(f"Saved {OUT_XLSX}")

    vagg = mix["FAGG"] / DENSITIES["FAGG"] + mix["CAGG"] / DENSITIES["CAGG"]
    cons = {
        "unit": "kg/m3", "n_rows": int(len(mix)),
        "raw": {v: {"min": float(mix[v].min()), "max": float(mix[v].max())} for v in RAW_VARS},
        "derived": {v: {"min": float(mix[v].min()), "max": float(mix[v].max())} for v in DERIVED_VARS},
        "physics": {"Vfinal": {"min": VFINAL_MIN, "max": VFINAL_MAX},
                    "Vagg": {"min": float(vagg.min()), "max": float(vagg.max())},
                    "TOTAL_BINDER": {"min": float(mix["TOTAL_BINDER"].min()),
                                     "max": float(mix["TOTAL_BINDER"].max())}},
        "durability": {"rcpt_limit_coulomb": RCPT_LIMIT, "age_days": 28, "min_pass_probability": 0.7},
        "densities": {k: DENSITIES[k] for k in RAW_VARS},
        "gwp_factors": GWP_FACTORS,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(cons, f, indent=2)
    print(f"Saved {OUT_JSON}")
    print(mix[RAW_VARS + ["GWP", "28day"]].describe().T[["min", "mean", "max"]].round(3))
    print(chl.groupby(["age", "rcpt_class"]).size().unstack(fill_value=0))
    print(chl.groupby("age")["pass_rcpt"].mean().round(2).to_dict())


if __name__ == "__main__":
    main()

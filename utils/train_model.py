"""
train_model.py
==============
Train a chained CatBoost surrogate model for concrete compressive strength.

Architecture (CatBoost-Chain, as described in the paper):
  Stage 1: predict 7-day  strength  <- raw mix features only
  Stage 2: predict 28-day strength  <- raw mix features + predicted 7-day
  Stage 3: predict 56-day strength  <- raw mix features + predicted 28-day

Unit convention:
  Raw data is in lb/yd³. This script converts to kg/m³ at load time
  (× 0.5933) so the saved model expects kg/m³ inputs — consistent with
  optimizer_core.py after its load_df() conversion.

Output:
  ../concrete_catboost_optimized.pkl
    keys: models, feature_names, unit

Usage:
  cd utils/
  python train_model.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ─────────────────────────────────────────────────────────────
# CONSTANTS  (must match optimizer_core.py)
# ─────────────────────────────────────────────────────────────

LB_YD3_TO_KG_M3 = 0.5933
RAW_VARS = ["PC", "FA", "SC", "FAGG", "CAGG", "WATER", "AEA", "WR_HR", "WR", "ACC"]
DATA_PATH = "../data/Super_Cleaned_Concrete_Data_model_train.csv"
OUT_PKL   = "../concrete_catboost_optimized.pkl"


# ─────────────────────────────────────────────────────────────
# 1. LOAD & CONVERT DATA
# ─────────────────────────────────────────────────────────────

print("Loading data ...")
df = pd.read_csv(DATA_PATH)
print(f"  Raw rows: {len(df)}")

# Convert ingredients from lb/yd³ to kg/m³
for col in RAW_VARS:
    if col in df.columns:
        df[col] = df[col] * LB_YD3_TO_KG_M3

print(f"  Unit conversion applied: lb/yd³ × {LB_YD3_TO_KG_M3} = kg/m³")
print(f"  PC range: [{df['PC'].min():.1f}, {df['PC'].max():.1f}] kg/m³")
print(f"  WATER range: [{df['WATER'].min():.1f}, {df['WATER'].max():.1f}] kg/m³")


# ─────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING  (identical to optimizer_core._engineer_one)
# ─────────────────────────────────────────────────────────────

e = 1e-9
tb  = df["PC"] + df["FA"] + df["SC"]
agg = df["FAGG"] + df["CAGG"]

df["TOTAL_BINDER"] = tb
df["w/b"]   = df["WATER"] / (tb + e)
df["b/a"]   = tb / (agg + e)
df["SCM%"]  = (df["FA"] + df["SC"]) / (tb + e)
df["CAGG%"] = df["CAGG"] / (agg + e)
df["FAGG%"] = df["FAGG"] / (agg + e)
df["PC%"]   = df["PC"]   / (tb + e)
df["FA%"]   = df["FA"]   / (tb + e)
df["SC%"]   = df["SC"]   / (tb + e)

# Base feature set: raw ingredients + derived ratios (no strength columns)
base_features = [col for col in df.columns
                 if col not in ["7day", "28day", "56day"]]

print(f"\nFeatures ({len(base_features)}): {base_features}")


# ─────────────────────────────────────────────────────────────
# 3. CATBOOST HYPERPARAMETER GRID
# ─────────────────────────────────────────────────────────────

param_grid = {
    "iterations":    [500, 1000],
    "learning_rate": [0.05, 0.1],
    "depth":         [6, 8],
    "l2_leaf_reg":   [3, 5, 10],
}


def tune_catboost(X_train, y_train, name: str) -> CatBoostRegressor:
    print(f"\n>>> Tuning CatBoost for [{name}] (n={len(X_train)}) ...")
    base_model = CatBoostRegressor(
        random_seed=42,
        verbose=0,
        eval_metric="R2",
    )
    gs = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    print(f"  Best params : {gs.best_params_}")
    print(f"  CV R²       : {gs.best_score_:.4f}")
    return gs.best_estimator_


# ─────────────────────────────────────────────────────────────
# 4. CHAINED TRAINING
# ─────────────────────────────────────────────────────────────
#
#  Stage 1: 7-day   <- base_features
#  Stage 2: 28-day  <- base_features + [7day]
#  Stage 3: 56-day  <- base_features + [28day]
#
# Each stage only uses rows that have a valid target value,
# maximising training data at each stage.

models        = {}
train_test_data = {}

for target, prev_target in [("7day", None), ("28day", "7day"), ("56day", "28day")]:
    if prev_target is None:
        current_features = base_features
    else:
        current_features = base_features + [prev_target]

    # Keep only rows with valid target (and valid chain input if needed)
    subset = df.dropna(subset=[target]).copy()
    if prev_target:
        subset = subset.dropna(subset=[prev_target])

    X = subset[current_features]
    y = subset[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = tune_catboost(X_train, y_train, target)
    models[target] = model
    train_test_data[target] = (X_train, X_test, y_train, y_test,
                               current_features)


# ─────────────────────────────────────────────────────────────
# 5. EVALUATION — INDEPENDENT (each stage uses true prior value)
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("Stage-wise evaluation (true prior-stage values as input):")
print("=" * 55)
for target, _, in [("7day", None), ("28day", "7day"), ("56day", "28day")]:
    X_train, X_test, y_train, y_test, _ = train_test_data[target]
    train_r2 = r2_score(y_train, models[target].predict(X_train))
    test_r2  = r2_score(y_test,  models[target].predict(X_test))
    test_mae = mean_absolute_error(y_test, models[target].predict(X_test))
    print(f"  [{target:5s}]  Train R²={train_r2:.4f}  "
          f"Test R²={test_r2:.4f}  Test MAE={test_mae:.2f} MPa")


# ─────────────────────────────────────────────────────────────
# 6. EVALUATION — CHAINED (simulates real inference with error propagation)
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("Chained evaluation (simulates real deployment, error propagates):")
print("=" * 55)

# Use rows that have all three strength values
eval_df = df.dropna(subset=["7day", "28day", "56day"]).copy()
_, test_df = train_test_split(eval_df, test_size=0.2, random_state=42)

# Stage 1
test_df["pred_7day"] = models["7day"].predict(test_df[base_features])

# Stage 2: use predicted 7-day
X28 = test_df[base_features].copy()
X28["7day"] = test_df["pred_7day"]
test_df["pred_28day"] = models["28day"].predict(X28)

# Stage 3: use predicted 28-day
X56 = test_df[base_features].copy()
X56["28day"] = test_df["pred_28day"]
test_df["pred_56day"] = models["56day"].predict(X56)

rows = []
for day in ["7day", "28day", "56day"]:
    r2  = r2_score(test_df[day], test_df[f"pred_{day}"])
    mae = mean_absolute_error(test_df[day], test_df[f"pred_{day}"])
    rows.append({"Stage": day, "Chained R²": round(r2, 4), "MAE (MPa)": round(mae, 2)})

print(pd.DataFrame(rows).to_string(index=False))


# ─────────────────────────────────────────────────────────────
# 7. SAVE
# ─────────────────────────────────────────────────────────────

meta = {
    "models":        models,           # {"7day": model, "28day": model, "56day": model}
    "feature_names": base_features,    # features for stage 1 (no prior-stage strength)
    "unit":          "kg/m3",          # inputs must be in kg/m³
}
joblib.dump(meta, OUT_PKL)
print(f"\n✅ Model saved to: {OUT_PKL}")
print(f"   Feature count : {len(base_features)}")
print(f"   Input unit    : kg/m³")
print(f"   Predict call  : predict(meta, mix)  where mix values are in kg/m³")

"""
optimizer_core.py
=================
Core functions for the LLM-based single-objective concrete mix optimizer.
This file contains all stable logic and should NOT be edited between experiments.

To run different ablation experiments, use run_experiment.py.

Variables
---------
  11 raw: PC, FA, SC, FAGG, CAGG, WATER, AEA, WR_HR, WR, ACC
   9 derived: w/b, b/a, SCM%, CAGG%, FAGG%, PC%, FA%, SC%  (SF removed)

Objective : minimize GWP (kg CO2-eq/m3)
Constraint: predicted 28d strength >= STRENGTH_MIN MPa

Unit convention
---------------
  All ingredient quantities are stored and displayed in kg/m³.
  Raw data (lb/yd³) is converted at load time via LB_YD3_TO_KG_M3 = 0.5933.
  The CatBoost model was trained on lb/yd³ data, so predict() converts
  back to lb/yd³ internally before calling the model.
"""

import json
import os
import re
import time
import warnings
import joblib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import google.generativeai as genai

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONSTANTS (physical, never change between experiments)
# ─────────────────────────────────────────────────────────────

# Unit conversion: raw data is in lb/yd³; multiply by this to get kg/m³
LB_YD3_TO_KG_M3 = 0.5933  # 1 lb/yd³ = 0.4536 kg/lb ÷ 0.7646 m³/yd³

# GWP emission factors in kg CO₂-eq / kg material (from PA database)
# With inputs in kg/m³: GWP = Σ(ingredient[kg/m³] × factor[kg CO₂/kg]) = kg CO₂/m³
GWP_FACTORS = {
    "PC": 1.048, "FA": 0.328, "SC": 0.264,
    "FAGG": 0.0026, "CAGG": 0.0037,
    "WATER": 0.0, "AEA": 0.0, "WR_HR": 0.0, "WR": 0.0, "ACC": 0.0,
}

RAW_VARS     = ["PC", "FA", "SC", "FAGG", "CAGG", "WATER", "AEA", "WR_HR", "WR", "ACC"]
DERIVED_VARS = ["w/b", "b/a", "SCM%", "CAGG%", "FAGG%", "PC%", "FA%", "SC%"]


# ─────────────────────────────────────────────────────────────
# EXPERIMENT CONFIG DATACLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """
    All parameters that vary between ablation experiments.
    Pass an instance of this to run_llm() and related functions.
    """
    # Experiment identity
    name: str = "baseline"
    description: str = ""

    # Problem parameters
    strength_min: float = 55.0
    max_iters: int = 30

    # LLM settings
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.9

    # GA settings
    ga_gens: int = 200
    ga_pop: int = 100

    # Stagnation / restart
    stag_window: int = 5
    stag_threshold: float = 0.005  # relative improvement (fraction of current best GWP)
    stag_min_best: float = 350.0
    max_restarts: int = 2
    restart_temp: float = 1.3

    # Anti-oscillation
    anti_osc_window: int = 3
    anti_osc_tol: float = 5.0

    # RAG / prompt ablation flags
    use_few_shot: bool = True        # include static few-shot examples
    use_knowledge_table: bool = True # include Material Effects table
    use_situation_rules: bool = True # include Situation A-D strategies
    use_memory: bool = False         # cross-run memory (off for ablation)
    rag_mode: str = "static"         # "static" | "dynamic" | "none"
    rag_k: int = 3
    rag_pool: str = "feasible"       # "feasible" | "full"
    rag_format: str = "tabular"      # "tabular" | "text"

    # File paths
    data_path: str = "data/Super_Cleaned_Concrete_Data.csv"
    model_pkl: str = "concrete_catboost_optimized.pkl"
    ga_csv: str = "ga_reference_solution.csv"
    output_prefix: str = "results"   # CSVs and reports named {output_prefix}_*


# ─────────────────────────────────────────────────────────────
# 1. DATA & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Convert ingredient quantities from lb/yd³ (raw data) to kg/m³ (SI)
    for col in RAW_VARS:
        if col in df.columns:
            df[col] = df[col] * LB_YD3_TO_KG_M3
    return _add_derived(df)


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tb  = df["PC"] + df["FA"] + df["SC"]
    agg = df["FAGG"] + df["CAGG"]
    df["TOTAL_BINDER"] = tb
    df["w/b"]   = df["WATER"] / tb
    df["b/a"]   = tb / agg
    df["SCM%"]  = (df["FA"] + df["SC"]) / tb
    df["CAGG%"] = df["CAGG"] / agg
    df["FAGG%"] = df["FAGG"] / agg
    df["PC%"]   = df["PC"]   / tb
    df["FA%"]   = df["FA"]   / tb
    df["SC%"]   = df["SC"]   / tb
    return df


def get_bounds(df: pd.DataFrame):
    raw = {v: {"min": float(df[v].min()), "max": float(df[v].max())}
           for v in RAW_VARS}
    der = {v: {"min": float(df[v].min()), "max": float(df[v].max())}
           for v in DERIVED_VARS}
    return raw, der


# ─────────────────────────────────────────────────────────────
# 2. CATBOOST SURROGATE
# ─────────────────────────────────────────────────────────────

def load_surrogate(pkl: str) -> dict:
    meta = joblib.load(pkl)
    assert "models" in meta and "feature_names" in meta
    return meta


def _engineer_one(mix: dict) -> dict:
    m  = dict(mix)
    tb = m["PC"] + m["FA"] + m["SC"]
    ag = m["FAGG"] + m["CAGG"]
    e  = 1e-9
    m["TOTAL_BINDER"] = tb
    m["w/b"]   = m["WATER"] / (tb + e)
    m["b/a"]   = tb / (ag + e)
    m["SCM%"]  = (m["FA"] + m["SC"]) / (tb + e)
    m["CAGG%"] = m["CAGG"] / (ag + e)
    m["FAGG%"] = m["FAGG"] / (ag + e)
    m["PC%"]   = m["PC"]   / (tb + e)
    m["FA%"]   = m["FA"]   / (tb + e)
    m["SC%"]   = m["SC"]   / (tb + e)
    return m


def predict(meta: dict, mix: dict) -> dict:
    fn  = meta["feature_names"]
    mdl = meta["models"]
    # If model was trained on lb/yd³ (old model, no 'unit' key), convert back.
    # New models trained by utils/train_model.py store unit='kg/m3' and need no conversion.
    if meta.get("unit", "lb/yd3") == "kg/m3":
        mix_in = mix
    else:
        mix_in = {k: (v / LB_YD3_TO_KG_M3 if k in RAW_VARS else v)
                  for k, v in mix.items()}
    m   = _engineer_one(mix_in)

    r7  = pd.DataFrame([{k: m.get(k, 0.) for k in fn}])
    p7  = float(mdl["7day"].predict(r7)[0])

    m["7day"] = p7
    r28 = pd.DataFrame([{k: m.get(k, 0.) for k in fn + ["7day"]}])
    p28 = float(mdl["28day"].predict(r28)[0])

    m["28day"] = p28
    r56 = pd.DataFrame([{k: m.get(k, 0.) for k in fn + ["28day"]}])
    p56 = float(mdl["56day"].predict(r56)[0])

    return {"7day": round(p7, 2), "28day": round(p28, 2), "56day": round(p56, 2)}


def compute_gwp(mix: dict) -> float:
    return round(sum(mix.get(k, 0.) * v for k, v in GWP_FACTORS.items()), 2)


def get_derived(mix: dict) -> dict:
    m = _engineer_one(mix)
    return {k: round(m[k], 5) for k in DERIVED_VARS}


# ─────────────────────────────────────────────────────────────
# 3. FEASIBILITY CHECK
# ─────────────────────────────────────────────────────────────

def check_feasibility(mix: dict, raw_b: dict, der_b: dict,
                      p28: float, strength_min: float) -> dict:
    rv = {v: {"val": mix[v], "min": b["min"], "max": b["max"]}
          for v, b in raw_b.items()
          if mix.get(v, 0) < b["min"] - 0.5 or mix.get(v, 0) > b["max"] + 0.5}

    dv_vals = get_derived(mix)
    dv = {}
    for v, b in der_b.items():
        val = dv_vals[v]
        tol = (b["max"] - b["min"]) * 0.01 + 1e-6
        if val < b["min"] - tol or val > b["max"] + tol:
            dv[v] = {"val": round(val, 4), "min": round(b["min"], 4),
                     "max": round(b["max"], 4)}

    sv = p28 < strength_min
    return {"raw_v": rv, "der_v": dv, "str_v": sv,
            "feasible": not rv and not dv and not sv}


# ─────────────────────────────────────────────────────────────
# 4. GA REFERENCE
# ─────────────────────────────────────────────────────────────

def run_ga(raw_b: dict, der_b: dict, meta: dict, cfg: ExperimentConfig) -> dict:
    try:
        from pymoo.algorithms.soo.nonconvex.ga import GA
        from pymoo.core.problem import Problem
        from pymoo.optimize import minimize as pymoo_min
        from pymoo.termination import get_termination
    except ImportError:
        raise ImportError("pymoo not installed — run: pip install pymoo")

    xl  = np.array([raw_b[v]["min"] for v in RAW_VARS])
    xu  = np.array([raw_b[v]["max"] for v in RAW_VARS])
    n_c = 1 + len(DERIVED_VARS) * 2

    class ConcreteProblem(Problem):
        def __init__(self):
            super().__init__(n_var=len(RAW_VARS), n_obj=1,
                             n_ieq_constr=n_c, xl=xl, xu=xu)

        def _evaluate(self, X, out, *args, **kwargs):
            F, G = [], []
            for row in X:
                mix = dict(zip(RAW_VARS, row))
                pr  = predict(meta, mix)
                g   = compute_gwp(mix)
                F.append([g])
                gc  = [cfg.strength_min - pr["28day"]]
                dv  = get_derived(mix)
                for v in DERIVED_VARS:
                    b = der_b[v]
                    gc += [b["min"] - dv[v], dv[v] - b["max"]]
                G.append(gc)
            out["F"] = np.array(F)
            out["G"] = np.array(G)

    print(f"\n[GA] Running {cfg.ga_gens} generations x pop={cfg.ga_pop} ...")
    res = pymoo_min(
        ConcreteProblem(), GA(pop_size=cfg.ga_pop),
        termination=get_termination("n_gen", cfg.ga_gens),
        seed=42, verbose=False,
    )

    best = None
    if res.X is not None:
        candidates = [res.X] if res.X.ndim == 1 else res.X
        for x in candidates:
            mix  = dict(zip(RAW_VARS, x))
            pr   = predict(meta, mix)
            g    = compute_gwp(mix)
            feas = check_feasibility(mix, raw_b, der_b, pr["28day"], cfg.strength_min)
            if feas["feasible"] and (best is None or g < best["gwp"]):
                dv = get_derived(mix)
                tb = mix["PC"] + mix["FA"] + mix["SC"]
                best = {
                    **{k: round(float(v), 2) for k, v in mix.items()},
                    **{k: round(float(v), 5) for k, v in dv.items()},
                    "total_binder": round(tb, 2),
                    "pred_7day":    pr["7day"],
                    "pred_28day":   pr["28day"],
                    "pred_56day":   pr["56day"],
                    "gwp":          g,
                }

    if best:
        print(f"[GA] Best: GWP={best['gwp']:.2f} kg/m³  28d={best['pred_28day']:.2f} MPa")
    else:
        print("[GA] No feasible solution found.")
    return best


# ─────────────────────────────────────────────────────────────
# 5. FEW-SHOT & DYNAMIC RAG
# ─────────────────────────────────────────────────────────────

def select_few_shot(df: pd.DataFrame, strength_min: float, n: int = 3) -> list:
    sub = df.dropna(subset=["28day"]).copy()
    sub = sub[sub["28day"] >= strength_min]
    sub["gwp"] = sub.apply(compute_gwp, axis=1)

    if len(sub) == 0:
        print(f"  [Warning] No rows satisfy 28d >= {strength_min}. Using top-5 by strength.")
        sub = df.dropna(subset=["28day"]).copy()
        sub["gwp"] = sub.apply(compute_gwp, axis=1)
        sub = sub.nlargest(5, "28day")

    examples, labels = [], ["Lowest GWP", "Highest strength", "Balanced"]

    r1 = sub.loc[sub["gwp"].idxmin()]
    examples.append(r1)
    sub = sub.drop(r1.name)

    if len(sub) > 0:
        r2 = sub.loc[sub["28day"].idxmax()]
        examples.append(r2)
        sub = sub.drop(r2.name)

    if len(sub) > 0:
        s_n = (sub["28day"] - sub["28day"].min()) / (sub["28day"].max() - sub["28day"].min() + 1e-9)
        c_n = (sub["gwp"]   - sub["gwp"].min())   / (sub["gwp"].max()   - sub["gwp"].min()   + 1e-9)
        r3  = sub.loc[(s_n - c_n).abs().idxmin()]
        examples.append(r3)

    result = []
    for row, label in zip(examples[:n], labels[:n]):
        ex = {k: round(float(row[k]), 2) for k in RAW_VARS}
        ex["pred_28day"] = round(float(row["28day"]), 1)
        ex["gwp"]        = round(float(row["gwp"]), 1)
        ex["label"]      = label
        result.append(ex)
    return result


def retrieve_similar_mixes(current_mix: dict, df: pd.DataFrame,
                            strength_min: float, k: int = 3,
                            pool: str = "feasible") -> list:
    """Dynamic RAG: k-NN retrieval from dataset based on ingredient similarity."""
    features = ["PC", "SC", "FA", "WATER", "FAGG", "CAGG", "AEA", "WR", "WR_HR", "ACC"]

    sub = df.copy()
    if pool == "feasible":
        sub = sub[sub["28day"] >= strength_min].copy()
    sub = sub.dropna(subset=["28day"]).copy()
    sub["gwp"] = sub.apply(compute_gwp, axis=1)

    if len(sub) == 0:
        return []

    data_mat  = sub[features].values.astype(float)
    col_min   = data_mat.min(axis=0)
    col_range = data_mat.max(axis=0) - col_min + 1e-9

    cur_vec  = np.array([current_mix.get(f, 0) for f in features], dtype=float)
    cur_norm = (cur_vec   - col_min) / col_range
    dat_norm = (data_mat  - col_min) / col_range

    dists   = np.linalg.norm(dat_norm - cur_norm, axis=1)
    top_idx = np.argsort(dists)[:k]

    result = []
    for _, row in sub.iloc[top_idx].iterrows():
        ex = {v: round(float(row[v]), 2) for v in RAW_VARS}
        ex["pred_28day"] = round(float(row["28day"]), 1)
        ex["gwp"]        = round(float(row["gwp"]), 1)
        result.append(ex)
    return result


# ─────────────────────────────────────────────────────────────
# 6. PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────

KNOWLEDGE_TABLE = """\
MATERIAL EFFECTS — DECISION TABLE
===================================
Each material affects both GWP and 28-day strength. Use this table every step.
All quantities are in kg/m³; GWP is in kg CO₂-eq/m³.

  PC   (Portland cement)
       GWP factor : 1.048 kg CO₂/kg  — largest CO₂ contributor
       Strength   : STRONG positive — PC is the primary strength driver
       Strategy   : Reduce PC as much as possible, but never below what
                    strength requires. Each 10 kg/m³ PC reduced saves ~10.5 kg CO₂/m³.

  SC   (Slag cement / GGBS)
       GWP factor : 0.264 kg CO₂/kg  — most efficient binder for CO₂ reduction
       Strength   : MODERATE positive at 28d — activates via PC hydration products;
                    high SC substitution (>60%) may slightly reduce 28d strength.
       Strategy   : PREFERRED substitute for PC.
                    Net saving per kg/m³ PC→SC swap = 1.048 - 0.264 = 0.784 kg CO₂/m³.

  FA   (Fly ash)
       GWP factor : 0.328 kg CO₂/kg  — second best option for CO₂ reduction
       Strength   : WEAK positive at 28d — FA is slow-reacting (pozzolanic).
       Strategy   : Use FA after SC is maximised. High FA risks falling below floor.

  WATER
       GWP factor : 0.000  ZERO
       Strength   : NEGATIVE — more water = higher w/b = lower strength
       Strategy   : Reduce WATER to improve strength at zero GWP cost.

  FAGG / CAGG  (Fine / Coarse aggregate)
       GWP factor : 0.0026 / 0.0037 kg CO₂/kg  — nearly zero
       Strength   : More aggregate = lower b/a = less paste per m³.
       Strategy   : Increasing FAGG+CAGG enables Path 2 (lower total binder).

  WR / WR_HR  (Water reducer / Superplasticiser)
       GWP factor : 0.000  ZERO
       Strength   : Indirect — allows lower WATER, enabling lower w/b.
       Strategy   : High WR+WR_HR (30-120 kg/m³) is essential for Path 2.

  ACC  (Accelerator)
       GWP factor : 0.000  ZERO
       Strength   : Direct positive — useful when total binder is low.
       Strategy   : ACC 120-450 kg/m³ can compensate strength loss on Path 2.

KEY INSIGHT — TWO PATHS TO LOW GWP:
  Path 1: substitute PC with SC/FA (total binder 300-360 kg/m³, GWP 165-240 kg/m³)
  Path 2: reduce total binder <250 kg/m³ + high FAGG/CAGG + high WR/WR_HR + ACC
          (GWP 120-165 kg/m³ — harder but lower)

STRENGTH-GWP RELATIONSHIP:
  Higher strength = higher binder = higher GWP.
  Target strength AS CLOSE AS POSSIBLE to the floor, NOT maximum strength.
"""

SITUATION_RULES = """\
HOW TO OPTIMISE — SITUATION-BASED STRATEGIES
=============================================
All quantities below are in kg/m³.

SITUATION A: Strength well above floor (margin > 8 MPa)
  -> Over-engineered. Reduce total binder.
  -> Reduce PC+SC by 12-18 kg/m³ each, OR increase FAGG+CAGG by 60-120 kg/m³,
     OR switch to Path 2 (total binder < 250 kg/m³ + WR 30-60 kg/m³).

SITUATION B: Strength close to floor (margin 0-5 MPa)  [efficient zone]
  -> Swap 6-12 kg/m³ PC -> SC (saves ~4.7-9.4 kg CO₂/m³).
  -> OR reduce WATER 3-6 kg/m³ first, then swap PC->SC.
  -> OR add ACC 30-60 kg/m³ to create headroom for further PC reduction.

SITUATION C: Strength below floor (infeasible)
  -> 1. Reduce WATER 6-9 kg/m³ (zero GWP cost, strong strength boost)
  -> 2. Add ACC 60-180 kg/m³ (zero GWP cost)
  -> 3. Increase WR/WR_HR 18-30 kg/m³ (allows more WATER reduction)
  -> 4. Increase PC 6-12 kg/m³ (last resort — raises GWP)
  -> Never increase SC or FA to recover strength at 28d.

SITUATION D: Stuck in local optimum (no GWP improvement)
  -> Check which variables have been CONSTANT in last 5 iterations.
  -> Option 1 (most impactful): Switch to Path 2
       total binder < 250 kg/m³, FAGG 470-590, CAGG 1010-1190,
       WR 60-120 kg/m³, WR_HR 30-60 kg/m³, ACC 120-300 kg/m³.
  -> Option 2: Add ACC 60-240 kg/m³ (free strength boost, try lower PC).
  -> Option 3: Different binder blend (high FA, or pure PC+SC).
  -> Option 4: Increase WR 30-60 kg/m³ to allow cutting WATER 12-18 kg/m³.
"""

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert concrete mix design engineer specialising in low-carbon concrete.

OPTIMISATION PROBLEM
====================
OBJECTIVE  : MINIMISE total GWP (kg CO₂-eq/m³)
CONSTRAINT : 28-day compressive strength >= {strength_min} MPa (HARD LIMIT)

GWP (kg CO₂/m³) = PC*1.048 + FA*0.328 + SC*0.264 + CAGG*0.0037 + FAGG*0.0026
  (all ingredient quantities in kg/m³)

{knowledge_block}

VARIABLE BOUNDS (kg/m³, from dataset)
=======================================
{raw_bounds}

DERIVED RATIO BOUNDS
====================
{der_bounds}
  w/b   = WATER / (PC+FA+SC)
  b/a   = (PC+FA+SC) / (FAGG+CAGG)
  SCM%  = (FA+SC) / (PC+FA+SC)
  CAGG% = CAGG / (FAGG+CAGG)
  FAGG% = FAGG / (FAGG+CAGG)
  PC%   = PC / (PC+FA+SC)
  FA%   = FA / (PC+FA+SC)
  SC%   = SC / (PC+FA+SC)

{situation_block}

{memory_block}

{few_shot_block}

OUTPUT FORMAT — STRICTLY REQUIRED
===================================
IMPORTANT: Before outputting, mentally verify 28d strength >= {strength_min} MPa.
NEVER output a mix you expect to be infeasible.

Return ONLY a valid JSON object. No markdown, no extra text.

{{
  "reasoning": "<what you changed and why, max 140 words>",
  "mix": {{
    "PC": <number>, "FA": <number>, "SC": <number>,
    "FAGG": <number>, "CAGG": <number>, "WATER": <number>,
    "AEA": <number>, "WR_HR": <number>, "WR": <number>, "ACC": <number>
  }}
}}
"""

FIRST_TURN = """\
Start optimisation. Propose an initial mix satisfying ALL of:
  - 28-day strength >= {strength_min} MPa  (HARD CONSTRAINT — most important)
  - GWP as low as possible
  - All variable bounds and derived ratio bounds satisfied

SAFE STARTING POINT to guarantee feasibility:
  Use PC >= 150 kg/m³ as a starting point — this reliably achieves {strength_min} MPa.
  Then reduce GWP by substituting PC with SC in later iterations.
  Do NOT start with PC < 120 kg/m³ — it risks falling below the strength floor.

Output ONLY the JSON object.\
"""

FEEDBACK_TEMPLATE = """\
=== ITERATION {it} / {max_it} ===

Last proposed mix:
{mix_json}

CatBoost evaluation:
  7-day  : {p7:.2f} MPa
  28-day : {p28:.2f} MPa   {str_status}
  56-day : {p56:.2f} MPa

GWP breakdown:
{gwp_breakdown}
  TOTAL GWP : {gwp:.2f} kg CO₂-eq/m³

Derived ratios:
{ratio_check}

Feasibility: {feas_str}

=== PROGRESS ===
  Previous iter {prev_iter}: GWP={prev_gwp:.2f}  28d={prev_28:.2f} MPa
  This iter     {it}       : GWP={gwp:.2f}  28d={p28:.2f} MPa
  GWP change    : {gwp_change:+.2f} kg  {gwp_trend}
  Str margin    : {str_margin:+.2f} MPa above {strength_min} MPa floor

Best so far (iter {best_iter}):
  PC={best_PC:.0f}  SC={best_SC:.0f}  FA={best_FA:.0f}
  FAGG={best_FAGG:.0f}  CAGG={best_CAGG:.0f}  WATER={best_WATER:.0f}
  AEA={best_AEA:.1f}  WR_HR={best_WR_HR:.1f}  WR={best_WR:.1f}  ACC={best_ACC:.0f}
  -> GWP={best_gwp:.2f} kg  28d={best_28:.2f} MPa

Current vs best:
  PC={cur_PC:.0f} vs {best_PC:.0f} ({pc_diff:+.0f})    SC={cur_SC:.0f} vs {best_SC:.0f} ({sc_diff:+.0f})
  FA={cur_FA:.0f} vs {best_FA:.0f} ({fa_diff:+.0f})    WATER={cur_WATER:.0f} vs {best_WATER:.0f} ({water_diff:+.0f})
  FAGG={cur_FAGG:.0f} vs {best_FAGG:.0f} ({fagg_diff:+.0f})  CAGG={cur_CAGG:.0f} vs {best_CAGG:.0f} ({cagg_diff:+.0f})
  WR={cur_WR:.0f} vs {best_WR:.0f} ({wr_diff:+.0f})    WR_HR={cur_WR_HR:.0f} vs {best_WR_HR:.0f} ({wr_hr_diff:+.0f})
  ACC={cur_ACC:.0f} vs {best_ACC:.0f} ({acc_diff:+.0f})
  GWP: {gwp:.2f} vs {best_gwp:.2f} ({gwp_vs_best:+.2f} kg)

{rag_block}{infeas_warning}{osc_warning}=== ACTION REQUIRED ===
{feedback}

Propose the NEXT mix to reduce GWP. Output ONLY the JSON object.\
"""


def build_system_prompt(raw_b: dict, der_b: dict, few_shot: list,
                        cfg: ExperimentConfig, memory: list = None) -> str:
    raw_lines = [f"  {v:<8} [{b['min']:8.2f}, {b['max']:8.2f}]"
                 for v, b in raw_b.items()]
    der_lines = [f"  {v:<8} [{b['min']:8.5f}, {b['max']:8.5f}]"
                 for v, b in der_b.items()]

    # Knowledge table block
    knowledge_block = KNOWLEDGE_TABLE if cfg.use_knowledge_table else \
        "GWP formula: GWP = PC*1.048 + FA*0.328 + SC*0.264 + CAGG*0.0037 + FAGG*0.0026\n"

    # Situation rules block
    situation_block = SITUATION_RULES if cfg.use_situation_rules else ""

    # Memory block
    memory_block = ""
    if cfg.use_memory and memory:
        memory_block = _format_memory(memory)

    # Few-shot block
    few_shot_block = ""
    if cfg.use_few_shot and cfg.rag_mode == "static" and few_shot:
        parts = []
        for ex in few_shot:
            mix_str = "  ".join(f"{k}={ex[k]}" for k in RAW_VARS)
            parts.append(
                f"[{ex['label']}]\n  {mix_str}\n"
                f"  -> 28d={ex['pred_28day']} MPa  GWP={ex['gwp']} kg CO₂/m³"
            )
        few_shot_block = (
            f"REFERENCE MIXES FROM DATASET (all satisfy 28d >= {cfg.strength_min} MPa)\n"
            "=" * 70 + "\n"
            + "\n\n".join(parts)
        )

    return SYSTEM_PROMPT_TEMPLATE.format(
        strength_min=cfg.strength_min,
        raw_bounds="\n".join(raw_lines),
        der_bounds="\n".join(der_lines),
        knowledge_block=knowledge_block,
        situation_block=situation_block,
        memory_block=memory_block,
        few_shot_block=few_shot_block,
    )


def _format_memory(memory: list) -> str:
    if not memory:
        return ""
    lines = ["EXPERIENCE FROM PREVIOUS RUNS\n" + "=" * 40]
    for rec in memory[-3:]:
        lines.append(
            f"Run #{rec['run_id']} ({rec.get('timestamp','')})  "
            f"Best GWP={rec['best_gwp']:.2f} kg"
        )
        bm = rec.get("best_mix", {})
        lines.append(
            f"  Best mix: PC={bm.get('PC',0):.0f} SC={bm.get('SC',0):.0f} "
            f"FA={bm.get('FA',0):.0f} WATER={bm.get('WATER',0):.0f} "
            f"ACC={bm.get('ACC',0):.0f} WR={bm.get('WR',0):.0f}"
        )
        if rec.get("llm_learned"):
            lines.append("  LLM learned:")
            for line in rec["llm_learned"].split("\n")[:5]:
                if line.strip():
                    lines.append(f"    {line.strip()}")
    return "\n".join(lines) + "\n"


def build_feedback(it: int, max_it: int, mix: dict, preds: dict,
                   gwp: float, feas: dict, trajectory: list,
                   der_b: dict, cfg: ExperimentConfig,
                   df: pd.DataFrame = None) -> str:
    p28 = preds["28day"]

    str_status = "OK" if p28 >= cfg.strength_min \
        else f"INFEASIBLE -- {p28 - cfg.strength_min:+.2f} MPa below floor"

    contribs = sorted(
        [(k, mix.get(k, 0) * GWP_FACTORS.get(k, 0)) for k in RAW_VARS],
        key=lambda x: -x[1]
    )
    gwp_breakdown = "\n".join(
        f"  {k:<6} {mix.get(k,0):6.1f} kg/m³ x {GWP_FACTORS[k]:.4f} = {c:6.2f} kg CO₂/m³"
        for k, c in contribs if c > 0.05
    )

    dv = get_derived(mix)
    ratio_lines = []
    for v, b in der_b.items():
        val    = dv[v]
        tol    = (b["max"] - b["min"]) * 0.01 + 1e-6
        ok     = b["min"] - tol <= val <= b["max"] + tol
        status = "OK" if ok else f"VIOLATION [{b['min']:.4f},{b['max']:.4f}]"
        ratio_lines.append(f"  {v:<8}= {val:.4f}  {status}")

    if feas["feasible"]:
        feas_str = "FEASIBLE"
    else:
        issues = (list(feas["raw_v"]) + list(feas["der_v"])
                  + (["strength"] if feas["str_v"] else []))
        feas_str = "INFEASIBLE (" + ", ".join(issues) + ")"

    prev = trajectory[-2] if len(trajectory) >= 2 else None
    prev_gwp  = prev["gwp"]        if prev else gwp
    prev_28   = prev["pred_28day"] if prev else p28
    prev_iter = prev["iteration"]  if prev else it
    gwp_change = round(gwp - prev_gwp, 2)
    str_change = round(p28 - prev_28, 2)

    if gwp_change < -0.5:
        gwp_trend = f"DECREASED {abs(gwp_change):.2f} kg/m³ -- good"
    elif gwp_change > 0.5:
        gwp_trend = f"INCREASED {gwp_change:.2f} kg/m³ -- WRONG DIRECTION"
    else:
        gwp_trend = "barely changed -- need a bigger move"

    feasible_traj = [r for r in trajectory if r["feasible"]]
    best_rec   = min(feasible_traj, key=lambda r: r["gwp"]) if feasible_traj else None
    best_gwp   = best_rec["gwp"]           if best_rec else gwp
    best_28    = best_rec["pred_28day"]    if best_rec else p28
    best_iter  = best_rec["iteration"]     if best_rec else it
    best_PC    = best_rec.get("PC", 0)    if best_rec else mix.get("PC", 0)
    best_SC    = best_rec.get("SC", 0)    if best_rec else mix.get("SC", 0)
    best_FA    = best_rec.get("FA", 0)    if best_rec else mix.get("FA", 0)
    best_WATER = best_rec.get("WATER", 0) if best_rec else mix.get("WATER", 0)
    best_WR_HR = best_rec.get("WR_HR", 0) if best_rec else mix.get("WR_HR", 0)
    best_WR    = best_rec.get("WR", 0)    if best_rec else mix.get("WR", 0)
    best_FAGG  = best_rec.get("FAGG", 0)  if best_rec else mix.get("FAGG", 0)
    best_CAGG  = best_rec.get("CAGG", 0)  if best_rec else mix.get("CAGG", 0)
    best_ACC   = best_rec.get("ACC", 0)   if best_rec else mix.get("ACC", 0)
    best_AEA   = best_rec.get("AEA", 0)   if best_rec else mix.get("AEA", 0)

    str_margin  = round(p28 - cfg.strength_min, 2)
    gwp_vs_best = round(gwp - best_gwp, 2)

    def diff(a, b): return round(a - b, 0)
    pc_diff    = diff(mix.get("PC", 0),    best_PC)
    sc_diff    = diff(mix.get("SC", 0),    best_SC)
    fa_diff    = diff(mix.get("FA", 0),    best_FA)
    water_diff = diff(mix.get("WATER", 0), best_WATER)
    fagg_diff  = diff(mix.get("FAGG", 0),  best_FAGG)
    cagg_diff  = diff(mix.get("CAGG", 0),  best_CAGG)
    wr_diff    = diff(mix.get("WR", 0),    best_WR)
    wr_hr_diff = diff(mix.get("WR_HR", 0), best_WR_HR)
    acc_diff   = diff(mix.get("ACC", 0),   best_ACC)

    # Anti-oscillation
    osc_warning = ""
    recent = trajectory[-cfg.anti_osc_window:] if len(trajectory) >= cfg.anti_osc_window else []
    if len(recent) == cfg.anti_osc_window:
        oscillating = all(
            abs(recent[i].get(v, 0) - recent[j].get(v, 0)) < cfg.anti_osc_tol
            for v in ["PC", "SC", "FA"]
            for i in range(len(recent))
            for j in range(i + 1, len(recent))
        )
        if oscillating:
            osc_warning = (
                "*** OSCILLATION WARNING ***\n"
                "Last 3 proposals nearly identical. "
                "Change at least 2 ingredients by > 12 kg/m³.\n\n"
            )

    infeas_warning = ""
    if not feas["feasible"]:
        infeas_warning = (
            f"*** INFEASIBLE — DO NOT REPEAT ***\n"
            f"Mentally verify 28d >= {cfg.strength_min} MPa before proposing.\n\n"
        )

    # Dynamic RAG block
    rag_block = ""
    if cfg.rag_mode == "dynamic" and df is not None:
        similar = retrieve_similar_mixes(mix, df, cfg.strength_min,
                                         k=cfg.rag_k, pool=cfg.rag_pool)
        if similar:
            if cfg.rag_format == "text":
                rag_lines = [
                    "=== SIMILAR MIXES FROM DATASET (retrieved based on your current proposal) ===",
                    "These are real historical mixes closest to what you just proposed.",
                    "Study their GWP and strength outcomes to guide your next step:\n",
                ]
                for i, s in enumerate(similar, 1):
                    tb = s.get("PC", 0) + s.get("FA", 0) + s.get("SC", 0)
                    wb = s.get("WATER", 0) / (tb + 1e-9)
                    rag_lines.append(
                        f"  [{i}] A mix with {s.get('PC',0):.0f} kg/m³ Portland cement, "
                        f"{s.get('SC',0):.0f} kg/m³ slag cement, and {s.get('FA',0):.0f} kg/m³ "
                        f"fly ash achieves {s['pred_28day']:.1f} MPa 28-day strength "
                        f"with GWP of {s['gwp']:.1f} kg CO₂/m³. "
                        f"Total binder: {tb:.0f} kg/m³, w/b ratio: {wb:.2f}, "
                        f"FAGG: {s.get('FAGG',0):.0f}, CAGG: {s.get('CAGG',0):.0f}, "
                        f"ACC: {s.get('ACC',0):.0f}, WR: {s.get('WR',0):.0f} kg/m³.\n"
                    )
            else:  # tabular (default)
                rag_lines = [
                    "SIMILAR MIXES FROM DATASET (k-NN retrieval based on your current proposal):"
                ]
                for i, s in enumerate(similar, 1):
                    mix_str = "  ".join(f"{v}={s[v]}" for v in RAW_VARS)
                    rag_lines.append(f"  [{i}] {mix_str}")
                    rag_lines.append(f"       -> 28d={s['pred_28day']} MPa  GWP={s['gwp']:.1f} kg/m³")
            rag_block = "\n".join(rag_lines) + "\n\n"

    # Directional feedback
    fb = []
    if not feas["feasible"]:
        if feas["str_v"]:
            fb.append(
                f"  Strength {p28:.1f} MPa below floor. "
                "Reduce WATER first, then add ACC, then increase PC."
            )
        for v, info in feas["der_v"].items():
            fb.append(f"  {v}={info['val']:.4f} violates [{info['min']:.4f},{info['max']:.4f}].")
    else:
        if gwp_change > 0.5:
            fb.append("  GWP went UP — wrong direction. Swap PC->SC or reduce WATER.")
        elif abs(gwp_change) <= 0.5:
            fb.append("  GWP barely changed — make a BIGGER move (12-18 kg/m³ increments).")
        else:
            fb.append(f"  GWP decreased {abs(gwp_change):.2f} kg/m³ — keep going.")

        if str_margin > 8 and gwp > best_gwp + 5:
            fb.append(
                f"  Over-engineered: {str_margin:.1f} MPa above floor. "
                "Reduce PC+SC by 12 kg/m³ each, or increase aggregates."
            )
        elif str_margin > 3:
            fb.append(f"  Good headroom ({str_margin:.1f} MPa). Swap PC->SC or increase WR.")
        elif str_margin < 2:
            fb.append("  Tight margin. Reduce WATER before cutting PC further.")

        cur_wr = mix.get("WR", 0) + mix.get("WR_HR", 0)
        if cur_wr > 120:
            fb.append(f"  WR+WR_HR={cur_wr:.0f} kg/m³ — very high. Do NOT increase further.")
        elif cur_wr < 30 and mix.get("WATER", 120) > 100:
            fb.append("  WR low — increase WR 30-60 kg/m³ to allow cutting WATER.")

        if mix.get("ACC", 0) < 60 and str_margin < 3:
            fb.append("  ACC low — adding ACC 120-240 kg/m³ boosts strength at zero GWP cost.")

        if mix.get("FAGG", 0) + mix.get("CAGG", 0) < 1200 and str_margin > 5:
            fb.append("  Aggregates low — increase FAGG+CAGG to dilute binder and cut GWP.")

    return FEEDBACK_TEMPLATE.format(
        it=it, max_it=max_it,
        mix_json=json.dumps(mix, indent=4),
        p7=preds["7day"], p28=p28, p56=preds["56day"],
        str_status=str_status,
        gwp_breakdown=gwp_breakdown,
        gwp=gwp,
        ratio_check="\n".join(ratio_lines),
        feas_str=feas_str,
        prev_iter=prev_iter, prev_gwp=prev_gwp, prev_28=prev_28,
        gwp_change=gwp_change, gwp_trend=gwp_trend,
        str_change=str_change, str_margin=str_margin,
        strength_min=cfg.strength_min,
        best_iter=best_iter,
        best_PC=best_PC, best_SC=best_SC, best_FA=best_FA,
        best_WATER=best_WATER, best_WR_HR=best_WR_HR, best_WR=best_WR,
        best_FAGG=best_FAGG, best_CAGG=best_CAGG,
        best_ACC=best_ACC, best_AEA=best_AEA, best_gwp=best_gwp, best_28=best_28,
        cur_PC=mix.get("PC",0), cur_SC=mix.get("SC",0), cur_FA=mix.get("FA",0),
        cur_WATER=mix.get("WATER",0), cur_FAGG=mix.get("FAGG",0),
        cur_CAGG=mix.get("CAGG",0), cur_WR=mix.get("WR",0),
        cur_WR_HR=mix.get("WR_HR",0), cur_ACC=mix.get("ACC",0),
        pc_diff=pc_diff, sc_diff=sc_diff, fa_diff=fa_diff,
        water_diff=water_diff, fagg_diff=fagg_diff, cagg_diff=cagg_diff,
        wr_diff=wr_diff, wr_hr_diff=wr_hr_diff, acc_diff=acc_diff,
        gwp_vs_best=gwp_vs_best,
        rag_block=rag_block,
        infeas_warning=infeas_warning,
        osc_warning=osc_warning,
        feedback="\n".join(fb),
    )


# ─────────────────────────────────────────────────────────────
# 7. HELPERS
# ─────────────────────────────────────────────────────────────

def clip_mix(mix: dict, raw_b: dict) -> tuple:
    clean, notes = {}, []
    for v in RAW_VARS:
        b   = raw_b[v]
        val = float(mix.get(v, b["min"]))
        clp = float(np.clip(val, b["min"], b["max"]))
        if abs(clp - val) > 0.5:
            notes.append(
                f"  {v}: proposed {val:.0f} clipped to [{b['min']:.0f},{b['max']:.0f}] -> {clp:.0f}"
            )
        clean[v] = round(clp, 2)
    return clean, notes


def parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def detect_stagnation(trajectory: list, cfg: ExperimentConfig) -> bool:
    feasible = [r for r in trajectory if r["feasible"]]
    if len(feasible) < cfg.stag_window:
        return False
    best_gwp = min(r["gwp"] for r in feasible)
    if best_gwp >= cfg.stag_min_best:
        return False
    recent = feasible[-cfg.stag_window:]
    # threshold is relative: fraction of current best GWP
    abs_threshold = cfg.stag_threshold * best_gwp
    improvements = [recent[i-1]["gwp"] - recent[i]["gwp"]
                    for i in range(1, len(recent))]
    return all(imp < abs_threshold for imp in improvements)


def build_restart_msg(trajectory: list, ga_ref: dict, restart_num: int) -> str:
    feasible = [r for r in trajectory if r["feasible"]]
    top5     = sorted(feasible, key=lambda r: r["gwp"])[:5]
    top5_str = ""
    for i, r in enumerate(top5, 1):
        top5_str += (
            f"  #{i} iter={r['iteration']:2d}: "
            f"PC={r['PC']:.0f} SC={r['SC']:.0f} FA={r['FA']:.0f} "
            f"WATER={r['WATER']:.0f} ACC={r.get('ACC',0):.0f} "
            f"-> GWP={r['gwp']:.2f} 28d={r['pred_28day']:.2f} MPa\n"
        )
    center = {v: round(float(np.mean([r[v] for r in feasible[-10:]])), 1)
              for v in ["PC", "SC", "FA", "WATER"]}
    ga_str = (f"GWP={ga_ref['gwp']:.2f} PC={ga_ref.get('PC',0):.0f} "
              f"SC={ga_ref.get('SC',0):.0f}" if ga_ref else "N/A")

    return f"""
=== RESTART #{restart_num} — STAGNATION ===
Search center (avg last 10): {center}
DO NOT propose similar mixes.

Top-5 best so far:
{top5_str}
GA reference: {ga_str}

Try ONE bold strategy:
  A: total binder < 420 kg + FAGG 800-1000 + CAGG 1700-2000 + WR 100-200 + ACC 200-500
  B: high FA (80-150 kg) replacing SC as primary SCM
  C: very low WATER (153-165 kg) + WR_HR 80-120 kg
  D: PC+SC pure binary, FA=0, very low w/b

Output ONLY the JSON object.
"""


# ─────────────────────────────────────────────────────────────
# 8. EVALUATION METRICS
# ─────────────────────────────────────────────────────────────

def compute_metrics(trajectory: list, ga_ref: dict,
                    raw_b: dict, total_catboost_calls: int) -> dict:
    """
    Compute all evaluation metrics for a completed run.

    OGR  — Optimality Gap Ratio
           (best_llm_gwp - ga_gwp) / ga_gwp
           Lower is better. 0 = matches GA.

    QER  — Query Efficiency Ratio
           delta_gwp_total / N_catboost_calls
           How much GWP reduction per CatBoost call.
           Higher is better (more improvement per call).

    MCE  — Mix Composition Entropy
           Sum of normalized std deviations across all 10 variables.
           Higher = more diverse search space explored.

    Also returns: feasibility_rate, convergence_iter, best_gwp, str_margin_at_best
    """
    feasible = [r for r in trajectory if r["feasible"]]
    if not feasible:
        return {
            "OGR": float("nan"), "QER": float("nan"), "MCE": float("nan"),
            "feasibility_rate": 0.0, "convergence_iter": None,
            "best_gwp": float("nan"), "str_margin_at_best": float("nan"),
            "total_catboost_calls": total_catboost_calls,
        }

    ga_gwp     = ga_ref["gwp"] if ga_ref else float("nan")
    best_rec   = min(feasible, key=lambda r: r["gwp"])
    best_gwp   = best_rec["gwp"]
    first_gwp  = feasible[0]["gwp"]

    # OGR
    OGR = (best_gwp - ga_gwp) / ga_gwp if ga_ref else float("nan")

    # QER
    delta_gwp = first_gwp - best_gwp          # total improvement from first feasible
    QER = delta_gwp / total_catboost_calls if total_catboost_calls > 0 else float("nan")

    # MCE: normalized std of each variable across all feasible iterations
    mce_total = 0.0
    for v in RAW_VARS:
        vals = [r[v] for r in feasible if v in r]
        if len(vals) < 2:
            continue
        std = float(np.std(vals))
        v_range = raw_b[v]["max"] - raw_b[v]["min"]
        if v_range > 0:
            mce_total += std / v_range   # normalized to [0, 1] scale

    return {
        "OGR":                  round(OGR, 4),
        "QER":                  round(QER, 4),
        "MCE":                  round(mce_total, 4),
        "feasibility_rate":     round(len(feasible) / len(trajectory), 4),
        "convergence_iter":     best_rec["iteration"],
        "best_gwp":             best_gwp,
        "ga_gwp":               ga_gwp,
        "gwp_gap":              round(best_gwp - ga_gwp, 2),
        "str_margin_at_best":   best_rec.get("str_margin", float("nan")),
        "total_feasible":       len(feasible),
        "total_iters":          len(trajectory),
        "total_catboost_calls": total_catboost_calls,
    }


# ─────────────────────────────────────────────────────────────
# 9. MAIN LLM LOOP
# ─────────────────────────────────────────────────────────────

def run_llm(raw_b: dict, der_b: dict, meta: dict, ga_ref: dict,
            few_shot: list, cfg: ExperimentConfig,
            memory: list = None, df: pd.DataFrame = None):
    """
    Run the LLM iterative optimizer.
    Returns: (trajectory, run_summary, total_catboost_calls)
    """
    print(f"\n{'='*62}")
    print(f"  Experiment: {cfg.name}")
    print(f"  Model: {cfg.gemini_model}  |  Constraint: 28d >= {cfg.strength_min} MPa")
    print(f"  RAG: {cfg.rag_mode}  |  Knowledge: {cfg.use_knowledge_table}"
          f"  |  Rules: {cfg.use_situation_rules}  |  Few-shot: {cfg.use_few_shot}")
    if ga_ref:
        print(f"  GA target: GWP={ga_ref['gwp']:.2f}  28d={ga_ref['pred_28day']:.2f} MPa")
    print(f"{'='*62}")

    sys_prompt = build_system_prompt(raw_b, der_b, few_shot, cfg, memory)
    genai.configure(api_key=cfg.gemini_api_key)

    def _make_model(temp):
        return genai.GenerativeModel(
            model_name=cfg.gemini_model,
            system_instruction=sys_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temp, max_output_tokens=1024),
        )

    cur_temp      = cfg.temperature
    model         = _make_model(cur_temp)
    chat          = model.start_chat(history=[])
    trajectory    = []
    parse_fails   = 0
    restart_count = 0
    catboost_calls = 0    # tracks EVERY predict() call including retries

    cur_mix   = {v: 0.0 for v in RAW_VARS}
    cur_preds = {"7day": 0.0, "28day": 0.0, "56day": 0.0}
    cur_gwp   = 0.0
    cur_feas  = {"feasible": True, "raw_v": {}, "der_v": {}, "str_v": False}
    clip_notes = []

    ga_gwp = ga_ref["gwp"] if ga_ref else float("nan")

    # NOTE: max_iters counts FEASIBLE iterations only.
    # Infeasible proposals trigger inner retries (up to 5 each) and are not counted.
    # Total API calls may be up to max_iters * 6 in the worst case.
    print(f"\n  {'Iter':>4}  {'28d MPa':>8}  {'GWP':>8}  {'Gap':>9}  Mode")
    print(f"  (max_iters={cfg.max_iters} feasible; up to {cfg.max_iters * 6} total API calls)")
    print("  " + "-"*50)

    it             = 0
    total_attempts = 0
    max_attempts   = cfg.max_iters * 6
    consec_fail    = 0

    while it < cfg.max_iters and total_attempts < max_attempts:
        total_attempts += 1

        # ── Stagnation / restart ──────────────────────────────
        mode = "exploit"
        if (it > 1 and restart_count < cfg.max_restarts
                and detect_stagnation(trajectory, cfg)):
            restart_count += 1
            mode      = f"RESTART#{restart_count}"
            cur_temp  = cfg.restart_temp
            model     = _make_model(cur_temp)
            chat      = model.start_chat(history=[])
            user_msg  = build_restart_msg(trajectory, ga_ref, restart_count)
            print(f"\n  [!] Stagnation — restart #{restart_count}")

        elif it == 0:
            user_msg = FIRST_TURN.format(strength_min=cfg.strength_min)

        else:
            if cur_temp != cfg.temperature:
                cur_temp = cfg.temperature
                model    = _make_model(cur_temp)
                chat     = model.start_chat(history=[])
                feas_sf  = [r for r in trajectory if r["feasible"]]
                if feas_sf:
                    best = min(feas_sf, key=lambda r: r["gwp"])
                    try:
                        chat.send_message(
                            f"Resume. Best so far: PC={best['PC']:.0f} SC={best['SC']:.0f} "
                            f"FA={best['FA']:.0f} WATER={best['WATER']:.0f} "
                            f"GWP={best['gwp']:.2f} 28d={best['pred_28day']:.2f} MPa. "
                            "Improve from here. Output ONLY the JSON object."
                        )
                    except Exception:
                        pass

            user_msg = build_feedback(
                it, cfg.max_iters, cur_mix, cur_preds, cur_gwp,
                cur_feas, trajectory, der_b, cfg, df=df,
            )
            if clip_notes:
                user_msg = (
                    "*** BOUNDS VIOLATION ***\n"
                    + "\n".join(clip_notes)
                    + "\nStay within bounds in system prompt.\n\n"
                    + user_msg
                )

        # ── Call Gemini ───────────────────────────────────────
        try:
            resp     = chat.send_message(user_msg)
            raw_text = resp.text
        except Exception as exc:
            err = str(exc)
            wait = 60 if "429" in err else 15
            print(f"  API error ({err[:40]}) — waiting {wait}s ...")
            time.sleep(wait)
            continue

        # ── Parse ─────────────────────────────────────────────
        parsed = parse_json(raw_text)
        if parsed is None or "mix" not in parsed:
            parse_fails += 1
            try:
                resp2  = chat.send_message(
                    "Output ONLY the JSON object with keys 'reasoning' and 'mix'."
                )
                parsed = parse_json(resp2.text)
            except Exception:
                pass
            if parsed is None or "mix" not in parsed:
                continue

        reasoning  = parsed.get("reasoning", "")
        mix, clip_notes = clip_mix(parsed["mix"], raw_b)
        preds      = predict(meta, mix)
        catboost_calls += 1
        g          = compute_gwp(mix)
        dv         = get_derived(mix)
        feas       = check_feasibility(mix, raw_b, der_b, preds["28day"], cfg.strength_min)
        tb         = mix["PC"] + mix["FA"] + mix["SC"]
        gwp_gap    = round(g - ga_gwp, 2) if not np.isnan(ga_gwp) else float("nan")

        # ── Infeasible handling ───────────────────────────────
        # ── Infeasible inner retry loop (up to 5 attempts) ───
        inner_retry = 0
        while not feas["feasible"] and inner_retry < 5:
            inner_retry += 1
            consec_fail += 1
            cur_mix, cur_preds, cur_gwp, cur_feas = mix, preds, g, feas
            print(f"  {'--':>4}  {preds['28day']:8.2f}  {g:8.2f}  "
                  f"{'infeasible':>9}  {mode}[retry {inner_retry}]")

            if consec_fail >= 30:
                print(f"  [!] 30 consecutive infeasible attempts — aborting.")
                break

            issues = []
            if feas["str_v"]:
                issues.append(
                    f"- Strength {preds['28day']:.1f} MPa is {cfg.strength_min - preds['28day']:.1f} MPa"
                    f" BELOW the {cfg.strength_min} MPa floor.\n"
                    f"  REQUIRED FIXES (choose one or combine):\n"
                    f"    a) Increase PC by {max(20, int((cfg.strength_min - preds['28day']) * 15))}-"
                    f"{max(40, int((cfg.strength_min - preds['28day']) * 25))} kg\n"
                    f"    b) Reduce WATER by 10-20 kg (current WATER={mix.get('WATER', 0):.0f})\n"
                    f"    c) Add ACC 200-500 kg (zero GWP cost, direct strength boost)\n"
                    f"    d) Combine: increase PC by 15 kg + reduce WATER by 10 kg + add ACC 200 kg"
                )
            for v, info in feas["der_v"].items():
                issues.append(
                    f"- {v}={info['val']:.4f} violates [{info['min']:.4f},{info['max']:.4f}].\n"
                    f"  Adjust proportions to bring {v} within bounds."
                )
            for v, info in feas["raw_v"].items():
                issues.append(
                    f"- {v}={info['val']:.1f} is outside [{info['min']:.1f},{info['max']:.1f}]."
                )

            retry_msg = (
                    f"ATTEMPT {inner_retry}/5 — Still INFEASIBLE.\n"
                    f"Current result: 28d={preds['28day']:.1f} MPa  GWP={g:.1f} kg\n\n"
                    f"Problems to fix:\n" + "\n".join(issues) + "\n\n"
                                                                f"IMPORTANT: Do NOT keep increasing GWP. Fix strength first.\n"
                                                                f"The MINIMUM viable approach: increase PC to at least "
                                                                f"{int(mix.get('PC', 164) + max(20, (cfg.strength_min - preds['28day']) * 15)):.0f} kg.\n\n"
                                                                f"Output ONLY the JSON object."
            )
            try:
                retry_resp = chat.send_message(retry_msg)
                retry_parsed = parse_json(retry_resp.text)
                if retry_parsed and "mix" in retry_parsed:
                    mix, clip_notes = clip_mix(retry_parsed["mix"], raw_b)
                    preds = predict(meta, mix)
                    catboost_calls += 1
                    g = compute_gwp(mix)
                    dv = get_derived(mix)
                    feas = check_feasibility(mix, raw_b, der_b,
                                             preds["28day"], cfg.strength_min)
                    tb = mix["PC"] + mix["FA"] + mix["SC"]
                    gwp_gap = round(g - ga_gwp, 2) if not np.isnan(ga_gwp) else float("nan")
                    reasoning = retry_parsed.get("reasoning", reasoning)
                else:
                    time.sleep(3)
                    break
            except Exception:
                time.sleep(3)
                break

        # After inner loop: if still infeasible, update state and continue outer loop
        if not feas["feasible"]:
            cur_mix, cur_preds, cur_gwp, cur_feas = mix, preds, g, feas
            if consec_fail >= 30:
                break
            time.sleep(2)
            continue

        # ── Feasible — record ─────────────────────────────────
        consec_fail = 0
        it += 1
        cur_mix, cur_preds, cur_gwp, cur_feas = mix, preds, g, feas

        record = {
            "iteration":    it,
            "mode":         mode,
            "reasoning":    reasoning,
            **mix,
            **{k: round(v, 5) for k, v in dv.items()},
            "total_binder": round(tb, 2),
            "pred_7day":    preds["7day"],
            "pred_28day":   preds["28day"],
            "pred_56day":   preds["56day"],
            "gwp":          g,
            "gwp_gap":      gwp_gap,
            "str_margin":   round(preds["28day"] - cfg.strength_min, 2),
            "feasible":     True,
        }
        trajectory.append(record)

        gap_s = f"{gwp_gap:+.2f}" if not np.isnan(gwp_gap) else "   n/a"
        print(f"  {it:4d}  {preds['28day']:8.2f}  {g:8.2f}  {gap_s:>9}  {mode}")
        time.sleep(3)

    print(f"\n  Restarts: {restart_count}  Failures: {parse_fails}"
          f"  Attempts: {total_attempts}  CatBoost calls: {catboost_calls}"
          f"  Feasible iters: {it}")

    # ── LLM self-summary ──────────────────────────────────────
    run_summary = ""
    if trajectory:
        try:
            traj_str = "\n".join([
                f"iter={r['iteration']} PC={r['PC']:.0f} SC={r['SC']:.0f} "
                f"FA={r['FA']:.0f} WATER={r['WATER']:.0f} ACC={r.get('ACC',0):.0f} "
                f"WR={r.get('WR',0):.0f} FAGG={r.get('FAGG',0):.0f} "
                f"GWP={r['gwp']:.2f} 28d={r['pred_28day']:.2f}"
                for r in trajectory
            ])
            summary_resp = chat.send_message(
                f"Run complete. Strength floor: {cfg.strength_min} MPa.\n"
                f"Trajectory:\n{traj_str}\n\n"
                "Give 6-10 concise bullet points on: what worked, what failed, "
                "local optima found, what to do differently. Be specific and quantitative."
                " Output ONLY bullet points."
            )
            run_summary = summary_resp.text.strip()
        except Exception as e:
            print(f"  [Warning] Summary failed: {e}")

    # ── Print best ────────────────────────────────────────────
    feas_all = [r for r in trajectory if r["feasible"]]
    if feas_all:
        best = min(feas_all, key=lambda r: r["gwp"])
        print(f"\n  BEST: GWP={best['gwp']:.2f} kg  28d={best['pred_28day']:.2f} MPa  "
              f"iter={best['iteration']}")
        print(f"  PC={best['PC']:.0f} SC={best['SC']:.0f} FA={best['FA']:.0f} "
              f"WATER={best['WATER']:.0f} ACC={best.get('ACC',0):.0f} "
              f"WR={best.get('WR',0):.0f}")

    return trajectory, run_summary, catboost_calls


# ─────────────────────────────────────────────────────────────
# 10. SAVE RESULTS
# ─────────────────────────────────────────────────────────────

def save_results(trajectory: list, ga_ref: dict, metrics: dict,
                 cfg: ExperimentConfig, run_summary: str = "") -> None:
    if not trajectory:
        print("  [Warning] Empty trajectory — nothing to save.")
        return

    prefix = cfg.output_prefix

    # CSV
    save_cols = [c for c in trajectory[0] if c != "reasoning"]
    pd.DataFrame(trajectory)[save_cols].to_csv(f"{prefix}_results.csv", index=False)

    # GA reference
    if ga_ref:
        pd.DataFrame([ga_ref]).to_csv(f"{prefix}_ga_ref.csv", index=False)

    # Metrics CSV
    metrics_row = {"experiment": cfg.name, **metrics}
    pd.DataFrame([metrics_row]).to_csv(f"{prefix}_metrics.csv", index=False)

    # Text report
    report_lines = [
        "=" * 62,
        f"  Experiment: {cfg.name}",
        f"  {cfg.description}",
        f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 62,
        f"  Strength floor    : >= {cfg.strength_min} MPa",
        f"  RAG mode          : {cfg.rag_mode}",
        f"  Knowledge table   : {cfg.use_knowledge_table}",
        f"  Situation rules   : {cfg.use_situation_rules}",
        f"  Few-shot          : {cfg.use_few_shot}",
        "-" * 62,
        "  METRICS",
        "-" * 62,
    ]
    for k, v in metrics.items():
        report_lines.append(f"  {k:<25}: {v}")

    if run_summary:
        report_lines += ["-" * 62, "  LLM SELF-SUMMARY", "-" * 62, run_summary]

    report_lines += ["-" * 62, "  TRAJECTORY", "-" * 62,
                     f"  {'Iter':>4}  {'28d':>7}  {'GWP':>8}  {'Gap':>7}  "
                     f"{'PC':>5}  {'SC':>5}  {'FA':>5}  {'w/b':>6}  Mode"]
    for r in trajectory:
        gap_s = f"{r.get('gwp_gap', float('nan')):+.2f}" \
            if not np.isnan(r.get("gwp_gap", float("nan"))) else "  n/a"
        report_lines.append(
            f"  {r['iteration']:4d}  {r['pred_28day']:7.2f}  {r['gwp']:8.2f}  "
            f"{gap_s:>7}  {r['PC']:5.0f}  {r['SC']:5.0f}  {r['FA']:5.0f}  "
            f"{r.get('w/b',0):6.4f}  {r.get('mode','')}"
        )
    report_lines.append("=" * 62)

    with open(f"{prefix}_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n  Saved: {prefix}_results.csv  {prefix}_metrics.csv  {prefix}_report.txt")
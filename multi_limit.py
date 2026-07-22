"""
LLM-Based Single-Objective Optimizer for Low-Carbon Concrete Mix Design
========================================================================
Problem definition
------------------
  Objective  : MINIMIZE total GWP  (kg CO2-eq / yd3)
  Constraint : predicted 28-day compressive strength >= STRENGTH_MIN (default 55 MPa)
  Variables  : 11 raw mix ingredients, bounded by dataset range
               + 8 derived ratio constraints (w/b, b/a, SCM%, etc.)

Reference solution
------------------
  A single-objective GA (pymoo) finds the minimum-GWP feasible solution first.
  This is the ground-truth reference (ga_ref).  The LLM then runs independently
  and each iteration logs:
    gwp_gap   = current_gwp - ga_ref_gwp   (how far from optimal, kg)
    gwp_ratio = current_gwp / ga_ref_gwp   (1.0 = matches GA)
    str_margin= current_28d - STRENGTH_MIN (MPa above the floor)

Inspired by
-----------
  Forootani (2025) IEEE TAI  — LLM fuzzy control loop (BESS example)
  Yang et al. (2023) OPRO   — LLM as optimizer with memory + restart

Usage
-----
  python llm_concrete_optimizer.py
  python llm_concrete_optimizer.py --strength 60 --iters 40
  python llm_concrete_optimizer.py --skip-ga      # reuse saved GA result

Dependencies
------------
  pip install google-generativeai catboost joblib pandas numpy pymoo
"""

import json
import re
import time
import warnings
import joblib
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import google.generativeai as genai
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────

GEMINI_API_KEY  = "AIzaSyDnV_LdQ2aztxCjwuEckEFFYQfc-se4ERA"
GEMINI_MODEL    = "gemini-2.5-flash-lite"
TEMPERATURE     = 0.9

STRENGTH_MIN    = 55.0      # 28-day strength floor (MPa) — hard constraint
STRENGTH_7D_MIN  = 30    # 7-day strength floor (MPa), set 0 to disable
STRENGTH_56D_MIN = 0.0    # 56-day strength floor (MPa), set 0 to disable
MAX_ITERATIONS  = 30
GA_GENS         = 200
GA_POP          = 100

# Stagnation / restart (OPRO-style)
STAG_WINDOW     = 5         # consecutive feasible iters with no improvement
STAG_THRESHOLD  = 1.0       # kg — improvement < this = no progress
STAG_MIN_BEST   = 350.0     # only restart if best GWP < this (LLM found good solution)
MAX_RESTARTS    = 2
RESTART_TEMP    = 1.3

# Anti-oscillation
ANTI_OSC_WINDOW = 3
ANTI_OSC_TOL    = 5.0       # kg per ingredient
# RAG settings
RAG_MODE = "text"    # "none" | "tabular" | "text"
RAG_K    = 5

DATA_PATH  = "Super_Cleaned_Concrete_Data_model_train.csv"
MODEL_PKL  = "concrete_catboost_optimized.pkl"
GA_CSV     = "ga_reference_solution.csv"
LLM_CSV    = "llm_optimizer_results.csv"
REPORT_TXT = "llm_optimizer_report.txt"
MEMORY_FILE = "llm_run_memory.json"   # cross-run experience memory

GWP_FACTORS = {
    "PC": 1.048, "FA": 0.328, "SC": 0.264,
    "CAGG": 0.0037, "FAGG": 0.0026,
    "WATER": 0.0, "AEA": 0.0, "WR_HR": 0.0, "WR": 0.0, "ACC": 0.0,
}

RAW_VARS     = ["PC","FA","SC","FAGG","CAGG","WATER","AEA","WR_HR","WR","ACC"]
DERIVED_VARS = ["w/b","b/a","SCM%","CAGG%","FAGG%","PC%","FA%","SC%"]


# ─────────────────────────────────────────────────────────────
# 1.  DATA & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
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
# 2.  CATBOOST SURROGATE
# ─────────────────────────────────────────────────────────────

def load_surrogate(pkl: str) -> dict:
    meta = joblib.load(pkl)
    assert "models" in meta and "feature_names" in meta
    return meta


def _engineer_one(mix: dict) -> dict:
    m  = dict(mix)
    tb = m["PC"] + m["FA"] + m["SC"]
    ag = m["FAGG"] + m["CAGG"]
    e  = 0
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
    m   = _engineer_one(mix)

    r7  = pd.DataFrame([{k: m.get(k, 0.) for k in fn}])
    p7  = float(mdl["7day"].predict(r7)[0])

    m["7day"] = p7
    r28 = pd.DataFrame([{k: m.get(k, 0.) for k in fn + ["7day"]}])
    p28 = float(mdl["28day"].predict(r28)[0])

    m["28day"] = p28
    r56 = pd.DataFrame([{k: m.get(k, 0.) for k in fn + ["28day"]}])
    p56 = float(mdl["56day"].predict(r56)[0])

    return {"7day": round(p7,2), "28day": round(p28,2), "56day": round(p56,2)}


def compute_gwp(mix: dict) -> float:
    return round(sum(mix.get(k, 0.) * v for k, v in GWP_FACTORS.items()), 2)


def get_derived(mix: dict) -> dict:
    m = _engineer_one(mix)
    return {k: round(m[k], 5) for k in DERIVED_VARS}


# ─────────────────────────────────────────────────────────────
# 3.  FEASIBILITY CHECK
# ─────────────────────────────────────────────────────────────

def check_feasibility(mix: dict, raw_b: dict, der_b: dict,
                      p28: float, preds: dict = None) -> dict:
    rv = {v: {"val": mix[v], "min": b["min"], "max": b["max"]}
          for v, b in raw_b.items()
          if mix.get(v, 0) < b["min"] - 0.5 or mix.get(v, 0) > b["max"] + 0.5}

    dv_vals = get_derived(mix)
    dv = {}
    for v, b in der_b.items():
        val = dv_vals[v]
        tol = (b["max"] - b["min"]) * 0.01 + 1e-6
        if val < b["min"] - tol or val > b["max"] + tol:
            dv[v] = {"val": round(val,4), "min": round(b["min"],4),
                     "max": round(b["max"],4)}

    sv28 = p28 < STRENGTH_MIN
    sv7  = (preds["7day"]  < STRENGTH_7D_MIN
            if (preds and STRENGTH_7D_MIN  > 0) else False)
    sv56 = (preds["56day"] < STRENGTH_56D_MIN
            if (preds and STRENGTH_56D_MIN > 0) else False)

    return {
        "raw_v":    rv,
        "der_v":    dv,
        "str_v":    sv28,
        "str7_v":   sv7,
        "str56_v":  sv56,
        "feasible": not rv and not dv and not sv28 and not sv7 and not sv56,
    }


# ─────────────────────────────────────────────────────────────
# 4.  GA REFERENCE  (single-objective: minimize GWP)
# ─────────────────────────────────────────────────────────────

def run_ga(raw_b: dict, der_b: dict, meta: dict,
           n_gen: int = GA_GENS, pop: int = GA_POP) -> dict:
    try:
        from pymoo.algorithms.soo.nonconvex.ga import GA
        from pymoo.core.problem import Problem
        from pymoo.optimize import minimize as pymoo_min
        from pymoo.termination import get_termination
    except ImportError:
        raise ImportError("pymoo not installed — run: pip install pymoo")

    xl = np.array([raw_b[v]["min"] for v in RAW_VARS])
    xu = np.array([raw_b[v]["max"] for v in RAW_VARS])
    n_c = 1 + len(DERIVED_VARS) * 2
    if STRENGTH_7D_MIN > 0:
        n_c += 1
    if STRENGTH_56D_MIN > 0:
        n_c += 1

    class ConcreteProblem(Problem):
        def __init__(self):
            super().__init__(n_var=len(RAW_VARS), n_obj=1,
                             n_ieq_constr=n_c, xl=xl, xu=xu)
            self.eval_count = 0

        def _evaluate(self, X, out, *args, **kwargs):
            F, G = [], []
            for row in X:
                mix = dict(zip(RAW_VARS, row))
                pr  = predict(meta, mix)
                g   = compute_gwp(mix)
                F.append([g])
                gc = [STRENGTH_MIN - pr["28day"]]
                if STRENGTH_7D_MIN > 0:
                    gc.append(STRENGTH_7D_MIN - pr["7day"])
                if STRENGTH_56D_MIN > 0:
                    gc.append(STRENGTH_56D_MIN - pr["56day"])
                dv  = get_derived(mix)
                for v in DERIVED_VARS:
                    b = der_b[v]
                    gc += [b["min"] - dv[v], dv[v] - b["max"]]
                G.append(gc)
            self.eval_count += len(X)
            out["F"] = np.array(F)
            out["G"] = np.array(G)

    print(f"\n[GA] Running {n_gen} generations x pop={pop} ...")

    problem = ConcreteProblem()
    res = pymoo_min(
        problem, GA(pop_size=pop),
        termination=get_termination("n_gen", n_gen),
        seed=42, verbose=False,
    )
    ga_catboost_calls = problem.eval_count

    best = None
    if res.X is not None:
        candidates = [res.X] if res.X.ndim == 1 else res.X
        for x in candidates:
            mix  = dict(zip(RAW_VARS, x))
            pr   = predict(meta, mix)
            g    = compute_gwp(mix)
            feas = check_feasibility(mix, raw_b, der_b, pr["28day"], pr)
            if feas["feasible"] and (best is None or g < best["gwp"]):
                dv = get_derived(mix)
                tb = mix["PC"]+mix["FA"]+mix["SC"]
                best = {
                    **{k: round(float(v),2) for k,v in mix.items()},
                    **{k: round(float(v),5) for k,v in dv.items()},
                    "total_binder": round(tb,2),
                    "pred_7day":    pr["7day"],
                    "pred_28day":   pr["28day"],
                    "pred_56day":   pr["56day"],
                    "gwp":          g,
                }

    if best:
        print(f"[GA] Best: GWP={best['gwp']:.2f} kg/yd3  "
              f"28d={best['pred_28day']:.2f} MPa")
    else:
        print("[GA] No feasible solution found.")

    return best, ga_catboost_calls


# ─────────────────────────────────────────────────────────────
# 5.  FEW-SHOT EXAMPLES FROM DATASET
# ─────────────────────────────────────────────────────────────

def select_few_shot(df: pd.DataFrame, n: int = 3) -> list:
    sub = df.dropna(subset=["28day"]).copy()
    sub = sub[sub["28day"] >= STRENGTH_MIN]
    sub["gwp"] = sub.apply(compute_gwp, axis=1)
    if len(sub) == 0:
        print(f"  [Warning] No dataset rows satisfy 28d >= {STRENGTH_MIN} MPa. "
              f"Using top-5 closest rows as examples.")
        sub = df.dropna(subset=["28day"]).copy()
        sub["gwp"] = sub.apply(compute_gwp, axis=1)
        sub = sub.nlargest(5, "28day")

    examples = []
    labels   = ["Lowest GWP in dataset", "Highest strength in dataset", "Balanced"]

    # 1. Lowest GWP
    r1 = sub.loc[sub["gwp"].idxmin()]
    examples.append(r1)
    sub = sub.drop(r1.name)

    # 2. Highest strength
    if len(sub) > 0:
        r2 = sub.loc[sub["28day"].idxmax()]
        examples.append(r2)
        sub = sub.drop(r2.name)

    # 3. Balanced
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
                            k: int = 5, pool: str = "feasible") -> list:
    features = ["PC", "SC", "FA", "WATER", "FAGG", "CAGG", "WR", "WR_HR", "ACC"]
    sub = df.copy()
    if pool == "feasible":
        sub = sub[sub["28day"] >= STRENGTH_MIN].copy()
    sub = sub.dropna(subset=["28day"]).copy()
    sub["gwp"] = sub.apply(compute_gwp, axis=1)
    if len(sub) == 0:
        return []
    data_mat  = sub[features].values.astype(float)
    col_min   = data_mat.min(axis=0)
    col_range = data_mat.max(axis=0) - col_min + 1e-9
    cur_vec   = np.array([current_mix.get(f, 0) for f in features], dtype=float)
    cur_norm  = (cur_vec   - col_min) / col_range
    dat_norm  = (data_mat  - col_min) / col_range
    dists     = np.linalg.norm(dat_norm - cur_norm, axis=1)
    top_idx   = np.argsort(dists)[:k]
    result = []
    for _, row in sub.iloc[top_idx].iterrows():
        ex = {v: round(float(row[v]), 2) for v in RAW_VARS}
        ex["pred_28day"] = round(float(row["28day"]), 1)
        ex["gwp"]        = round(float(row["gwp"]), 1)
        result.append(ex)
    return result
# ─────────────────────────────────────────────────────────────
# 6.  PROMPTS
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert concrete mix design engineer specialising in low-carbon concrete.

OPTIMISATION PROBLEM
====================
OBJECTIVE  : MINIMISE total embodied carbon (GWP, kg CO2-eq/yd3) — as low as possible
CONSTRAINTS:
  - Predicted 28-day strength >= {strength_min} MPa    (always active)
  - Predicted 7-day  strength >= {strength_7d_min} MPa (active if > 0)
  - Predicted 56-day strength >= {strength_56d_min} MPa (active if > 0)

This is a SINGLE-OBJECTIVE problem. Strength just needs to meet the floor.
Every decision should focus purely on reducing GWP.

GWP FORMULA
===========
GWP = PC*1.048 + FA*0.328 + SC*0.264 
    + CAGG*0.0037 + FAGG*0.0026

KEY INSIGHT — TWO PATHS TO LOW GWP
====================================
Most GWP comes from binder materials (PC, SC, FA, FAGG, CAGG).
There are TWO fundamentally different strategies to reduce GWP:

Path 1 — Substitute PC with lower-GWP binders:
  replace PC with SC/FA, since FA and SC has lower GWP factor.

Path 2 — Reduce total binder quantity:
  Use MORE aggregate (FAGG+CAGG) and MORE water reducers (WR, WR_HR)
  to achieve strength with LESS total binder.
  Combined with ACC to compensate strength loss.


STRENGTH-GWP RELATIONSHIP — CRITICAL INSIGHT
=============================================
Higher strength generally means higher GWP, because strength requires
more or higher-quality binder.

THEREFORE: The target is NOT to maximise strength.
The optimal mix has strength AS CLOSE AS POSSIBLE to {strength_min} MPa
(the floor), not far above it.

If your current strength is well above the floor,
you are likely over-using binder and wasting GWP budget.
Try reducing total binder to bring strength closer to {strength_min} MPa.

MATERIAL EFFECTS — DECISION TABLE
===================================
Each material affects both GWP and 28-day strength. Use this table every step:

  PC   (Portland cement)
       GWP factor : 1.048  HIGH — largest CO2 contributor
       Strength   : STRONG positive — PC is the primary strength driver
       Strategy   : Reduce PC as much as possible, but never below what
                    strength requires. Each 10 kg PC reduced saves 10.48 kg CO2.

  SC   (Slag cement)
       GWP factor : 0.264  LOW — most efficient binder for CO2 reduction
       Strength   : MODERATE positive at 28d — activates via PC hydration products;
                    high SC substitution (>60%) may slightly reduce 28d strength
                    but improves 56d strength.
       Strategy   : PREFERRED substitute for PC.
                    Net saving per kg PC→SC swap = 1.048 - 0.264 = 0.784 kg CO2.

  FA   (Fly ash)
       GWP factor : 0.328  LOW — second best option for CO2 reduction
       Strength   : WEAK positive at 28d — FA is slow-reacting (pozzolanic);
                    primarily improves long-term strength, not 28d.
       Strategy   : Use FA after SC is maximised. Be careful: high FA can
                    push 28d strength below the floor. Test cautiously.


  WATER
       GWP factor : 0.000  ZERO
       Strength   : NEGATIVE — more water = higher w/b = lower strength
       Strategy   : Reduce water to improve strength without raising GWP.
                    Reducing WATER by 10 kg (lower w/b) can recover strength
                    lost from reducing PC, at zero GWP cost.

  FAGG / CAGG  (Fine / Coarse aggregate)
       GWP factor : 0.0026 / 0.0037  — nearly zero
       Strength   : More aggregate = lower b/a ratio = less paste per yd3.
                    This ALLOWS you to reduce total binder while keeping
                    strength, because the mix is more efficiently packed.
       Strategy   : Increasing FAGG+CAGG is a key enabler of Path 2.
                    Try FAGG=800-1000, CAGG=1700-2000 with reduced total binder.
                    FAGG/(FAGG+CAGG) must stay between 0.28 and 0.69 (dataset bounds).

  WR / WR_HR  (Water reducer / Superplasticiser)
       GWP factor : 0.000  — zero GWP impact
       Strength   : Indirect — allows much lower WATER while maintaining
                    workability, enabling lower w/b and lower total binder.
       Strategy   : High WR+WR_HR (50-150 kg total) is essential for Path 2.
                    If you want to reduce total binder below 420 kg,
                    you MUST increase WR/WR_HR to compensate.
                    WR range: 0-211 kg. WR_HR range: 0-127 kg.

  AEA  (Air-entraining admixture)
       GWP factor : 0.000  ZERO
       Strength   : Slight negative at high doses (air voids reduce strength)
       Strategy   : Keep AEA low (0-5 kg). Not a useful optimisation lever.

  ACC  (Accelerator)
       GWP factor : 0.000  — zero GWP impact
       Strength   : Direct positive effect on both 7d and 28d strength.
                    Especially useful when total binder is low.
       Strategy   : ACC 200-750 kg can compensate for strength loss when
                    reducing total binder on Path 2. Zero GWP cost.
                    If strength is below floor and you do not want to add PC,
                    try ACC 100-400 kg first.

MOST EFFICIENT MOVES (ranked by CO2 saving per unit):
  1. Replace 10 kg PC with 10 kg SC  → saves 7.84 kg CO2, mild strength loss
  2. Reduce WATER by 10 kg           → saves 0 kg CO2, RECOVERS strength
  3. Replace 10 kg PC with 10 kg FA  → saves 7.20 kg CO2, more strength loss than SC

VARIABLE BOUNDS  (kg/yd3, from dataset)
=======================================
{raw_bounds}

DERIVED RATIO BOUNDS  (must be satisfied after computing from raw ingredients)
===============================================================================
{der_bounds}
  Formulas:
    w/b   = WATER / (PC+FA+SC)
    b/a   = (PC+FA+SC) / (FAGG+CAGG)
    SCM%  = (FA+SC) / (PC+FA+SC)
    CAGG% = CAGG / (FAGG+CAGG)
    FAGG% = FAGG / (FAGG+CAGG)
    PC%   = PC / (PC+FA+SC)
    FA%   = FA / (PC+FA+SC)
    SC%   = SC / (PC+FA+SC)

your design need to make sure are in the variable bounds and derived ratio bounds as well as above the strength limit.

Examples: 

SITUATION A: Strength is well above floor (str_margin > 8 MPa)
  -> Over-engineered. Too much binder. High strength = high GWP.
  -> Goal: bring strength DOWN towards {strength_min} MPa.
  -> Options (try multiple simultaneously):
       - Reduce PC AND reduce SC (reduce total binder)
       - Increase FAGG+CAGG (dilutes paste, same strength with less binder)

SITUATION B: Strength is close to floor (str_margin 0-5 MPa)
  -> In the efficient zone. Fine-tune carefully.
  -> Options:
       - Swap PC -> SC (saves CO2, slight strength risk)
       - Reduce WATER first (recovers strength at zero GWP cost),
         then try another PC->SC swap
       - Add ACC to create headroom for further PC reduction
       - If on Path 1, check if Path 2 could achieve same strength with less binder:
         try reducing total binder while adding WR

SITUATION C: Strength below floor (infeasible)
  -> Try to increase the strength above the floor. Use these levers in order of effectiveness:
       1. Reduce WATER (lowers w/b, strong strength boost)
       2. Add ACC (direct strength boost)
       3. Increase WR or WR_HR (allows further WATER reduction)
       4. Increase PC (last resort)

SITUATION D: Stuck in local optimum (GWP not improving for several iters)
  -> You MUST explore variables you have not changed recently.
  -> Check which variables have been CONSTANT across your last 5 proposals.
     If FAGG, CAGG, WR, WR_HR, or ACC have not changed, try them NOW.
  -> Options (try in this order):

       Option 1 — Switch to Path 2 (most impactful, often missed):
         Set total binder (PC+SC+FA) < 420 kg,
         increase FAGG, CAGG
         increase WR and/or WR_HR 
         add ACC to recover strength at zero GWP cost.
         This is the strategy that finds the lowest GWP solutions.

       Option 2 — Use ACC as a free strength booster:
         Add ACC. Zero GWP cost, direct strength gain.
         This lets you reduce PC further without losing strength.

       Option 3 — Different binder composition:
         Try high FA with lower SC.
         Try pure PC+SC with no FA.
         Try reducing total binder with lower WATER.

       Option 4 — Adjust WR/WR_HR:
         Increase WR to allow cutting WATER ,
         recovering strength lost from PC reduction at zero GWP cost.

GENERAL PRINCIPLES:
  - Every feasible solution should have lower GWP than your previous feasible solution
  - Strength should stay between {strength_min} and {strength_min_plus5} MPa — not far above
  - SCM% = (FA+SC)/(PC+FA+SC) must stay below {scm_max:.1%} — do NOT push SC too high
  - When SC is near maximum, switch strategy: reduce FA+SC together, explore Path 2 instead
  - WR/WR_HR should not exceed 200 kg total
  - Make changes of at least 15 kg per iteration — small tweaks stall progress
  - If GWP is not improving for 3 iterations, you are stuck — change FAGG, CAGG, or ACC


REFERENCE MIXES FROM DATASET  (all satisfy 28d >= {strength_min} MPa)
=======================================================================
{few_shot_block}

OUTPUT FORMAT — STRICTLY REQUIRED
===================================

IMPORTANT: Before outputting your mix, mentally calculate the predicted
28-day strength. If you believe it will be below {strength_min} MPa,
adjust the mix first. NEVER output a mix you expect to be infeasible.

Return ONLY a valid JSON object. No markdown, no text outside the JSON.

{{
  "reasoning": "<what you changed vs last iter and why, referencing the decision table, max 140 words>",
  "mix": {{
    "PC": <number>, "FA": <number>, "SC": <number>, 
    "FAGG": <number>, "CAGG": <number>, "WATER": <number>,
    "AEA": <number>, "WR_HR": <number>, "WR": <number>, "ACC": <number>
  }}
}}

Verify all derived ratios are within bounds before responding.
"""

FIRST_TURN = """\
Start optimisation. Propose an initial mix that:
  - Satisfies 28-day strength >= {strength_min} MPa  (hard constraint)
  - Has as low a GWP as possible
  - Respects all variable bounds and derived-ratio bounds

Use the reference mixes for inspiration. Output ONLY the JSON object.\
"""

FEEDBACK_TEMPLATE = """\
=== ITERATION {it} / {max_it} ===

Your last proposed mix:
{mix_json}

CatBoost evaluation:
  7-day  strength : {p7:.2f} MPa
  28-day strength : {p28:.2f} MPa   {str_status}
  56-day strength : {p56:.2f} MPa

GWP breakdown:
{gwp_breakdown}
  TOTAL GWP : {gwp:.2f} kg CO2-eq/yd3

Derived ratios:
{ratio_check}

Feasibility: {feas_str}

=== PROGRESS vs PREVIOUS ITERATION ===
  Previous iter {prev_iter}  : GWP={prev_gwp:.2f} kg   28d={prev_28:.2f} MPa
  This iter     {it}  : GWP={gwp:.2f} kg   28d={p28:.2f} MPa
  GWP change          : {gwp_change:+.2f} kg   {gwp_trend}
  Strength change     : {str_change:+.2f} MPa
  Strength margin     : {str_margin:+.2f} MPa above the {strength_min} MPa floor

Best solution so far (iter {best_iter}):
  PC={best_PC:.0f}  SC={best_SC:.0f}  FA={best_FA:.0f} 
  FAGG={best_FAGG:.0f}  CAGG={best_CAGG:.0f}  WATER={best_WATER:.0f}
  AEA={best_AEA:.1f}  WR_HR={best_WR_HR:.1f}  WR={best_WR:.1f}  ACC={best_ACC:.0f}
  -> GWP={best_gwp:.2f} kg   28d={best_28:.2f} MPa

Your current vs best (all variables):
  PC={cur_PC:.0f} vs {best_PC:.0f} ({pc_diff:+.0f})  SC={cur_SC:.0f} vs {best_SC:.0f} ({sc_diff:+.0f})
  FA={cur_FA:.0f} vs {best_FA:.0f} ({fa_diff:+.0f})  FAGG={cur_FAGG:.0f} vs {best_FAGG:.0f} ({fagg_diff:+.0f})
  CAGG={cur_CAGG:.0f} vs {best_CAGG:.0f} ({cagg_diff:+.0f})  WATER={cur_WATER:.0f} vs {best_WATER:.0f} ({water_diff:+.0f})
  WR={cur_WR:.0f} vs {best_WR:.0f} ({wr_diff:+.0f})  WR_HR={cur_WR_HR:.0f} vs {best_WR_HR:.0f} ({wr_hr_diff:+.0f})
  ACC={cur_ACC:.0f} vs {best_ACC:.0f} ({acc_diff:+.0f})
  GWP: {gwp:.2f} vs {best_gwp:.2f} ({gwp_vs_best:+.2f} kg)

{infeas_warning}{osc_warning}=== ACTION REQUIRED ===
{feedback}

Propose the NEXT mix to reduce GWP further. Output ONLY the JSON object.\
"""


def _build_system_prompt(raw_b, der_b, few_shot, strength_min, memory=None):
    raw_lines = [f"  {v:<8} [{b['min']:8.2f}, {b['max']:8.2f}]"
                 for v, b in raw_b.items()]
    der_lines = [f"  {v:<8} [{b['min']:8.5f}, {b['max']:8.5f}]"
                 for v, b in der_b.items()]
    fs_parts  = []
    for ex in few_shot:
        mix_str = "  ".join(f"{k}={ex[k]}" for k in RAW_VARS)
        fs_parts.append(
            f"[{ex['label']}]\n  {mix_str}\n"
            f"  -> 28d={ex['pred_28day']} MPa   GWP={ex['gwp']} kg CO2/yd3"
        )
    return SYSTEM_PROMPT.format(
        strength_min=strength_min,
        strength_min_plus5=strength_min + 5,
        strength_7d_min=STRENGTH_7D_MIN,
        strength_56d_min=STRENGTH_56D_MIN,
        scm_max=der_b["SCM%"]["max"],  # 从数据集实际上限取值
        raw_bounds="\n".join(raw_lines),
        der_bounds="\n".join(der_lines),
        few_shot_block="\n\n".join(fs_parts),
        memory_block=format_memory_for_prompt(memory or []),
    )


def _build_feedback(it, max_it, mix, preds, gwp, feas,
                    trajectory, der_b, df=None):
    p28 = preds["28day"]

    # ── Strength status ───────────────────────────────────────
    str_status = "OK" if p28 >= STRENGTH_MIN \
        else f"INFEASIBLE -- {p28 - STRENGTH_MIN:+.2f} MPa below floor"

    # ── GWP breakdown ─────────────────────────────────────────
    contribs = sorted(
        [(k, mix.get(k, 0) * GWP_FACTORS.get(k, 0)) for k in RAW_VARS],
        key=lambda x: -x[1]
    )
    gwp_breakdown = "\n".join(
        f"  {k:<6} {mix.get(k,0):6.1f} kg x {GWP_FACTORS[k]:.4f} = {c:6.2f} kg CO2"
        for k, c in contribs if c > 0.05
    )

    # ── Derived ratio check ───────────────────────────────────
    dv = get_derived(mix)
    ratio_lines = []
    for v, b in der_b.items():
        val   = dv[v]
        tol   = (b["max"] - b["min"]) * 0.01 + 1e-6
        ok    = b["min"] - tol <= val <= b["max"] + tol
        b_min = b["min"]
        b_max = b["max"]
        status = "OK" if ok else f"VIOLATION [{b_min:.4f},{b_max:.4f}]"
        ratio_lines.append(f"  {v:<8}= {val:.4f}  {status}")

    # ── Feasibility ───────────────────────────────────────────
    if feas["feasible"]:
        feas_str = "FEASIBLE"
    else:
        issues = (list(feas["raw_v"]) + list(feas["der_v"])
                  + (["strength"] if feas["str_v"] else []))
        feas_str = "INFEASIBLE (" + ", ".join(issues) + ")"

    # ── Previous iteration comparison ────────────────────────
    # trajectory already contains the current record as its last entry,
    # so "previous" is the second-to-last
    prev = trajectory[-2] if len(trajectory) >= 2 else None
    prev_gwp  = prev["gwp"]        if prev else gwp
    prev_28   = prev["pred_28day"] if prev else p28
    prev_iter = prev["iteration"]  if prev else it
    gwp_change = round(gwp - prev_gwp, 2)
    str_change = round(p28 - prev_28, 2)

    if gwp_change < -0.5:
        gwp_trend = f"DECREASED {abs(gwp_change):.2f} kg -- good progress"
    elif gwp_change > 0.5:
        gwp_trend = f"INCREASED {gwp_change:.2f} kg -- WRONG DIRECTION"
    else:
        gwp_trend = "barely changed -- need a bigger adjustment"

    # ── Best feasible solution so far ────────────────────────
    feasible_traj = [r for r in trajectory if r["feasible"]]
    if feasible_traj:
        best_rec = min(feasible_traj, key=lambda r: r["gwp"])
    else:
        best_rec = None

    best_gwp   = best_rec["gwp"]        if best_rec else gwp
    best_28    = best_rec["pred_28day"] if best_rec else p28
    best_iter  = best_rec["iteration"]  if best_rec else it
    best_PC    = best_rec["PC"]         if best_rec else mix["PC"]
    best_SC    = best_rec["SC"]         if best_rec else mix["SC"]
    best_FA    = best_rec["FA"]         if best_rec else mix["FA"]

    best_WATER = best_rec["WATER"]      if best_rec else mix["WATER"]
    best_WR_HR = best_rec.get("WR_HR", 0) if best_rec else mix.get("WR_HR", 0)
    best_WR    = best_rec.get("WR", 0)    if best_rec else mix.get("WR", 0)

    best_FAGG = best_rec.get("FAGG", 0) if best_rec else mix.get("FAGG", 0)
    best_CAGG = best_rec.get("CAGG", 0) if best_rec else mix.get("CAGG", 0)
    best_ACC = best_rec.get("ACC", 0) if best_rec else mix.get("ACC", 0)
    best_AEA = best_rec.get("AEA", 0) if best_rec else mix.get("AEA", 0)

    str_margin  = round(p28 - STRENGTH_MIN, 2)
    gwp_vs_best = round(gwp - best_gwp, 2)
    pc_diff = round(mix["PC"] - best_PC, 0)
    sc_diff = round(mix["SC"] - best_SC, 0)
    fa_diff = round(mix["FA"] - best_FA, 0)
    fagg_diff = round(mix.get("FAGG", 0) - best_rec.get("FAGG", 0) if best_rec else 0, 0)
    cagg_diff = round(mix.get("CAGG", 0) - best_rec.get("CAGG", 0) if best_rec else 0, 0)
    water_diff = round(mix.get("WATER", 0) - best_WATER, 0)
    wr_diff = round(mix.get("WR", 0) - best_WR, 0)
    wr_hr_diff = round(mix.get("WR_HR", 0) - best_WR_HR, 0)
    acc_diff = round(mix.get("ACC", 0) - best_rec.get("ACC", 0) if best_rec else 0, 0)

    # ── Anti-oscillation ──────────────────────────────────────
    osc_warning = ""
    infeas_warning = ""
    recent = trajectory[-ANTI_OSC_WINDOW:] if len(trajectory) >= ANTI_OSC_WINDOW else []
    if len(recent) == ANTI_OSC_WINDOW:
        oscillating = all(
            abs(recent[i].get(v, 0) - recent[j].get(v, 0)) < ANTI_OSC_TOL
            for v in ["PC", "SC", "FA"]
            for i in range(len(recent))
            for j in range(i + 1, len(recent))
        )
        # Also detect SCM% boundary oscillation
        scm_vals = [r.get("SCM%", 0) for r in recent]
        scm_near_limit = all(v > 0.68 for v in scm_vals)
        if scm_near_limit:
            oscillating = True
        if oscillating:
            if scm_near_limit:
                osc_warning = (
                    "*** SCM% BOUNDARY TRAP ***\n"
                    f"You are stuck near the SCM% upper limit ({der_b['SCM%']['max'] * 100:.1f}%).\n"
                    "Adding more SC is NOT possible — you have hit the constraint.\n"
                    "You MUST try a completely different strategy:\n"
                    "  - Switch to Path 2: reduce total binder, increase FAGG+CAGG\n"
                    "  - Try adding FA instead of SC\n"
                    "  - Reduce both PC AND SC, compensate with ACC\n\n"
                )
            else:
                osc_warning = (
                    "*** OSCILLATION WARNING ***\n"
                    "Last 3 proposals nearly identical. Change at least 2 variables by > 20 kg.\n\n"
                )

    # ── Directional feedback ──────────────────────────────────
    fb = []
    if not feas["feasible"]:
        infeas_warning = (
            "*** YOU WASTED AN ITERATION ***\n"
            "The mix you proposed was INFEASIBLE. This iteration produced no useful result.\n"
            "Before outputting your next mix, you MUST mentally verify:\n"
            "  1. Will 28-day strength be >= {strength_min} MPa? If unsure, increase PC slightly.\n"
            "  2. Are all derived ratios (w/b, SCM%, PC%) within bounds?\n"
            "Do NOT guess. A wrong answer wastes another iteration.\n\n"
        ).format(strength_min=STRENGTH_MIN)
        if feas["str_v"]:
            fb.append(
                f"  Strength {p28:.1f} MPa is BELOW the {STRENGTH_MIN} MPa floor.\n"
                " try reducing WATER (costs 0 GWP).\n"
                " increase PC/FA/SC/CAGG/FAGG if reducing WATER is not enough."
            )
        if feas.get("str7_v"):
            fb.append(
                f"  7-day strength {preds['7day']:.1f} MPa is BELOW "
                f"the {STRENGTH_7D_MIN} MPa floor.\n"
                f"  To improve 7-day strength: increase PC, or reduce SC/FA "
                f"(SC and FA are slow-reacting and mainly contribute to later-age strength)."
            )
        if feas.get("str56_v"):
            fb.append(
                f"  56-day strength {preds['56day']:.1f} MPa is BELOW "
                f"the {STRENGTH_56D_MIN} MPa floor.\n"
                f"  To improve 56-day strength: increase SC or FA "
                f"(both contribute strongly to long-term strength)."
            )
        for v, info in feas["der_v"].items():
            fb.append(
                f"  {v}={info['val']:.4f} violates "
                f"[{info['min']:.4f},{info['max']:.4f}]. Adjust proportions."
            )
    else:
        # GWP trend feedback
        if gwp_change > 0.5:
            fb.append(
                f"  GWP went UP by {gwp_change:.2f} kg — this is the wrong direction.\n"
                "  You must reduce GWP. Use the decision table:\n"
                "    - Replace more PC with SC (saves 0.784 kg CO2 per kg swapped)\n"
                "    - Reduce WATER to recover any strength lost (zero GWP cost)"
            )
        elif abs(gwp_change) <= 0.5:
            fb.append(
                f"  GWP barely changed ({gwp_change:+.2f} kg). You need a BIGGER move.\n"
                "  Try replacing PC with SC in increments of 20-30 kg at a time."
            )
        else:
            fb.append(f"  GWP decreased {abs(gwp_change):.2f} kg — good. Keep going.")

        # Strength headroom feedback
        if str_margin > 8 and gwp > best_gwp + 5:
            fb.append(
                f"  *** OVER-ENGINEERED ***\n"
                f"  Strength is {str_margin:.1f} MPa above the floor — far too high.\n"
                f"  High strength = high binder content = high GWP. This is wasteful.\n"
                f"  Your current GWP ({gwp:.1f}) is {gwp - best_gwp:.1f} kg above best.\n"
                f"  Target: bring strength DOWN towards {STRENGTH_MIN} MPa.\n"
                f"  Options:\n"
                f"    - Reduce PC by 20 kg AND SC by 20 kg (saves ~{20 * 1.048 + 20 * 0.264:.1f} kg CO2)\n"
                f"    - Increase FAGG+CAGG by 200 kg (dilutes binder, reduces total paste)\n"
                f"    - Reduce total binder below 420 kg, compensate with ACC or WR increase"
            )
        elif str_margin > 3:
            fb.append(
                f"  Strength margin is {str_margin:.1f} MPa — reasonable headroom.\n"
                f"  Options:\n"
                f"    - Swap 15 kg PC -> SC (saves ~{15 * 0.784:.1f} kg CO2)\n"
                f"    - Increase WR by 30 kg to allow cutting WATER by 15 kg\n"
                f"    - Try adding ACC 100-200 kg to create room for more PC reduction"
            )
        elif str_margin < 2:
            fb.append(
                f"  Strength margin is only {str_margin:.1f} MPa — very tight.\n"
                "  Do NOT reduce PC further. Instead:\n"
                "    - Reduce WATER by 5-10 kg to recover strength margin\n"
                "    - Then try another PC->SC swap"
            )

        # WR discipline
        # WR discipline
        cur_wr = mix.get("WR", 0) + mix.get("WR_HR", 0)
        if cur_wr > 150:
            fb.append(
                f"  WR+WR_HR = {cur_wr:.0f} kg — already very high. Do NOT increase further."
            )
        elif cur_wr < 50 and mix.get("WATER", 200) > 170:
            fb.append(
                f"  WR+WR_HR = {cur_wr:.0f} kg — currently low.\n"
                f"  Consider increasing WR to 50-100 kg: this allows reducing WATER\n"
                f"  by 20-30 kg, recovering strength at zero GWP cost."
            )

        # ACC suggestion
        cur_acc = mix.get("ACC", 0)
        if cur_acc < 100 and str_margin < 3:
            fb.append(
                f"  ACC = {cur_acc:.0f} kg — very low.\n"
                f"  Adding ACC 200-400 kg boosts strength at zero GWP cost.\n"
                f"  This would give you room to reduce PC further."
            )

        # FAGG/CAGG suggestion
        cur_fagg = mix.get("FAGG", 0)
        cur_cagg = mix.get("CAGG", 0)
        if cur_fagg + cur_cagg < 2000 and str_margin > 5:
            fb.append(
                f"  FAGG+CAGG = {cur_fagg + cur_cagg:.0f} kg — consider increasing.\n"
                f"  More aggregate dilutes binder, allowing less PC+SC for same strength."
            )

        # ── Dynamic RAG block ─────────────────────────────────────
    rag_block = ""
    if RAG_MODE != "none" and df is not None:
        similar = retrieve_similar_mixes(mix, df, k=RAG_K, pool="feasible")
        if similar:
            if RAG_MODE == "tabular":
                rag_lines = [
                    "\n=== SIMILAR MIXES FROM DATASET (k-NN retrieval) ===",
                    "These are real mixes closest to your current proposal:\n"
                ]
                for i, s in enumerate(similar, 1):
                    mix_str = "  ".join(f"{v}={s[v]}" for v in RAW_VARS)
                    rag_lines.append(f"  [{i}] {mix_str}")
                    rag_lines.append(
                        f"       -> 28d={s['pred_28day']} MPa   "
                        f"GWP={s['gwp']:.1f} kg CO2/yd3\n"
                    )
            else:  # text
                rag_lines = [
                    "\n=== SIMILAR MIXES FROM DATASET (k-NN retrieval) ===",
                    "These are real mixes closest to your current proposal:\n"
                ]
                for i, s in enumerate(similar, 1):
                    tb = s.get("PC", 0) + s.get("FA", 0) + s.get("SC", 0)
                    wb = s.get("WATER", 0) / (tb + 1e-9)
                    rag_lines.append(
                        f"  [{i}] A mix with {s.get('PC', 0):.0f} kg/m³ Portland cement, "
                        f"{s.get('SC', 0):.0f} kg/m³ slag cement, and {s.get('FA', 0):.0f} kg/m³ "
                        f"fly ash achieves {s['pred_28day']:.1f} MPa 28-day strength "
                        f"with GWP of {s['gwp']:.1f} kg CO₂/m³. "
                        f"Total binder: {tb:.0f} kg/m³, w/b ratio: {wb:.2f}, "
                        f"FAGG: {s.get('FAGG', 0):.0f}, CAGG: {s.get('CAGG', 0):.0f}, "
                        f"ACC: {s.get('ACC', 0):.0f}, WR: {s.get('WR', 0):.0f} kg/m³.\n"
                    )
            rag_block = "\n".join(rag_lines)
    fb.append(rag_block)

    return FEEDBACK_TEMPLATE.format(
        it=it, max_it=max_it,
        mix_json=json.dumps(mix, indent=4),
        p7=preds["7day"], p28=p28, p56=preds["56day"],
        str_status=str_status,
        gwp_breakdown=gwp_breakdown,
        gwp=gwp,
        ratio_check="\n".join(ratio_lines),
        feas_str=feas_str,
        # Previous iter comparison
        prev_iter=prev_iter,
        prev_gwp=prev_gwp,
        prev_28=prev_28,
        gwp_change=gwp_change,
        gwp_trend=gwp_trend,
        str_change=str_change,
        str_margin=str_margin,
        strength_min=STRENGTH_MIN,
        # Best so far
        best_iter=best_iter,
        best_PC=best_PC, best_SC=best_SC, best_FA=best_FA,
        best_WATER=best_WATER, best_WR_HR=best_WR_HR, best_WR=best_WR,
        best_gwp=best_gwp, best_28=best_28,
        # Current vs best diff
        cur_PC=mix["PC"], cur_SC=mix["SC"],
        pc_diff=pc_diff, sc_diff=sc_diff, gwp_vs_best=gwp_vs_best,
        # Oscillation
        osc_warning=osc_warning,
        feedback="\n".join(fb),
        infeas_warning=infeas_warning,
        cur_FA=mix["FA"], cur_FAGG=mix.get("FAGG", 0),
        cur_CAGG=mix.get("CAGG", 0), cur_WATER=mix.get("WATER", 0),
        cur_WR=mix.get("WR", 0), cur_WR_HR=mix.get("WR_HR", 0),
        cur_ACC=mix.get("ACC", 0),
        best_FAGG=best_FAGG, best_CAGG=best_CAGG,
        best_ACC=best_ACC,
        fa_diff=fa_diff, fagg_diff=fagg_diff, cagg_diff=cagg_diff,
        water_diff=water_diff, wr_diff=wr_diff,
        wr_hr_diff=wr_hr_diff, acc_diff=acc_diff,best_AEA=best_AEA,
    )
# ─────────────────────────────────────────────────────────────
# 6b.  CROSS-RUN MEMORY  (persistent experience across runs)
# ─────────────────────────────────────────────────────────────

def load_memory(path: str) -> list:
    """
    Load experience records from previous runs.
    Each record contains: run_id, best_gwp, best_mix, what_worked,
    what_failed, local_optima_encountered.
    Returns empty list if file does not exist.
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_memory(path: str, trajectory: list, strength_min: float,
                ga_ref: dict = None, run_summary: str = "") -> None:
    """
    Analyse the completed trajectory and append a summary record to memory.
    Records all variable changes (not just PC/SC/FA/WATER).
    """
    ALL_VARS = ["PC","FA","SC","FAGG","CAGG","WATER","AEA","WR_HR","WR","ACC"]

    memory  = load_memory(path)
    feasible = [r for r in trajectory if r["feasible"]]
    if not feasible:
        return

    best = min(feasible, key=lambda r: r["gwp"])

    # ── What worked: steps where GWP decreased meaningfully ──
    worked = []
    for i in range(1, len(trajectory)):
        prev = trajectory[i - 1]
        curr = trajectory[i]
        if curr["feasible"] and prev["feasible"]:
            delta = prev["gwp"] - curr["gwp"]
            if delta > 2.0:
                # Record ALL variable changes, skip unchanged ones
                move_parts = []
                for v in ALL_VARS:
                    diff = curr.get(v, 0) - prev.get(v, 0)
                    move_parts.append(f"{v} {diff:+.0f}")
                move_str = "  ".join(move_parts)

                worked.append({
                    "from": {v: prev.get(v, 0) for v in ALL_VARS}
                             | {"GWP": prev["gwp"], "28d": prev["pred_28day"]},
                    "to":   {v: curr.get(v, 0) for v in ALL_VARS}
                             | {"GWP": curr["gwp"], "28d": curr["pred_28day"]},
                    "gwp_saved": round(delta, 2),
                    "move": move_str,
                })

    # ── What failed: infeasible or GWP increase ───────────────
    failed = []
    for i in range(1, len(trajectory)):
        prev = trajectory[i - 1]
        curr = trajectory[i]
        if not curr["feasible"]:
            failed.append({
                "type": "infeasible",
                "mix":  {v: curr.get(v, 0) for v in ALL_VARS}
                        | {"28d_predicted": curr["pred_28day"]},
                "reason": ("strength_violation"
                           if curr["pred_28day"] < strength_min
                           else "ratio_violation"),
            })
        elif prev["feasible"] and curr["gwp"] > prev["gwp"] + 3.0:
            move_parts = []
            for v in ALL_VARS:
                diff = curr.get(v, 0) - prev.get(v, 0)
                move_parts.append(f"{v} {diff:+.0f}")
            failed.append({
                "type": "gwp_increase",
                "move": "  ".join(move_parts),
                "gwp_change": round(curr["gwp"] - prev["gwp"], 2),
            })

    # ── Local optima: regions where progress stalled ─────────
    local_optima = []
    feas_only = [r for r in trajectory if r["feasible"]]
    for i in range(4, len(feas_only)):
        window    = feas_only[i - 4:i + 1]
        gwp_range = max(r["gwp"] for r in window) - min(r["gwp"] for r in window)
        if gwp_range < 3.0:
            centre = window[2]
            sig = (f"PC~{round(centre['PC']/10)*10:.0f}  "
                   f"SC~{round(centre['SC']/10)*10:.0f}  "
                   f"FA~{round(centre['FA']/10)*10:.0f}  "
                   f"FAGG~{round(centre.get('FAGG',0)/100)*100:.0f}  "
                   f"ACC~{round(centre.get('ACC',0)/100)*100:.0f}  "
                   f"GWP~{round(centre['gwp']/5)*5:.0f}")
            if sig not in [lo["signature"] for lo in local_optima]:
                local_optima.append({
                    "signature": sig,
                    "gwp":       round(centre["gwp"], 1),
                    "note":      "stalled here — avoid this region next run",
                })

    record = {
        "run_id":          len(memory) + 1,
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M"),
        "strength_min":    strength_min,
        "total_iters":     len(trajectory),
        "feasible_count":  len(feasible),
        "best_gwp":        best["gwp"],
        "best_mix":        {v: best.get(v, 0) for v in ALL_VARS}
                           | {"pred_28day": best["pred_28day"],
                              "gwp":        best["gwp"]},
        "ga_best":         ga_ref if ga_ref else None,
        "gwp_gap_to_ga":   round(best["gwp"] - ga_ref["gwp"], 2)
                           if ga_ref else None,
        "what_worked":     worked[:5],
        "what_failed":     failed[:10],
        "local_optima":    local_optima,
        "llm_learned":     run_summary,
    }

    memory.append(record)
    memory = memory[-5:]   # keep last 5 runs only

    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)

    print(f"\n  Memory saved -> '{path}'  (total runs stored: {len(memory)})")


def format_memory_for_prompt(memory: list) -> str:
    """
    Convert memory records into a compact prompt-friendly string.
    Shows full compositions and all variable changes.
    Called once when building the system prompt.
    """
    ALL_VARS = ["PC","FA","SC","FAGG","CAGG","WATER","AEA","WR_HR","WR","ACC"]

    if not memory:
        return "(No previous run history — this is the first run.)"

    lines = []
    for rec in memory:
        lines.append(
            f"Run #{rec['run_id']} ({rec['timestamp']})  "
            f"iters={rec['total_iters']}  feasible={rec['feasible_count']}"
        )
        lines.append(
            f"  Best achieved: GWP={rec['best_gwp']:.2f} kg  "
            f"28d={rec['best_mix'].get('pred_28day', 0):.2f} MPa"
        )

        # Full best mix — all variables
        bm = rec["best_mix"]
        lines.append(
            f"  Best mix (full):\n"
            f"    PC={bm.get('PC',0):.0f}  SC={bm.get('SC',0):.0f}  "
            f"FA={bm.get('FA',0):.0f}\n"
            f"    FAGG={bm.get('FAGG',0):.0f}  CAGG={bm.get('CAGG',0):.0f}  "
            f"WATER={bm.get('WATER',0):.0f}\n"
            f"    AEA={bm.get('AEA',0):.1f}  WR_HR={bm.get('WR_HR',0):.1f}  "
            f"WR={bm.get('WR',0):.1f}  ACC={bm.get('ACC',0):.0f}\n"
            f"    -> GWP={bm.get('gwp',0):.2f} kg  "
            f"28d={bm.get('pred_28day',0):.2f} MPa"
        )

        # Full GA best — all variables
        if rec.get("ga_best"):
            ga = rec["ga_best"]
            lines.append(
                f"  GA optimal (full):\n"
                f"    PC={ga.get('PC',0):.0f}  SC={ga.get('SC',0):.0f}  "
                f"FA={ga.get('FA',0):.0f}\n"
                f"    FAGG={ga.get('FAGG',0):.0f}  CAGG={ga.get('CAGG',0):.0f}  "
                f"WATER={ga.get('WATER',0):.0f}\n"
                f"    AEA={ga.get('AEA',0):.1f}  WR_HR={ga.get('WR_HR',0):.1f}  "
                f"WR={ga.get('WR',0):.1f}  ACC={ga.get('ACC',0):.0f}\n"
                f"    -> GWP={ga.get('gwp',0):.2f} kg  "
                f"28d={ga.get('pred_28day',0):.2f} MPa"
            )
            if rec.get("gwp_gap_to_ga") is not None:
                gap = rec["gwp_gap_to_ga"]
                if gap > 0:
                    lines.append(
                        f"  LLM was {gap:.2f} kg ABOVE GA last run — "
                        f"try to close or beat this gap."
                    )
                else:
                    lines.append(
                        f"  LLM BEAT GA by {abs(gap):.2f} kg last run — "
                        f"try to maintain or improve further."
                    )

        # What worked — full variable changes
        if rec.get("what_worked"):
            lines.append("  MOVES THAT REDUCED GWP (all variable changes shown):")
            for w in rec["what_worked"][:3]:
                lines.append(
                    f"    [{w['move']}]  "
                    f"saved {w['gwp_saved']:.1f} kg CO2  "
                    f"(GWP {w['from']['GWP']:.1f} -> {w['to']['GWP']:.1f}  "
                    f"28d {w['from']['28d']:.1f} -> {w['to']['28d']:.1f} MPa)"
                )

        # What failed
        if rec.get("what_failed"):
            infeas = [f for f in rec["what_failed"] if f["type"] == "infeasible"]
            gwp_up = [f for f in rec["what_failed"] if f["type"] == "gwp_increase"]
            lines.append(
                f"  Failures: {len(infeas)} infeasible, "
                f"{len(gwp_up)} moves that raised GWP"
            )
            # Show a couple of infeasible examples so LLM knows what to avoid
            for f in infeas[:2]:
                mix = f["mix"]
                lines.append(
                    f"    Infeasible example: "
                    f"PC={mix.get('PC',0):.0f}  SC={mix.get('SC',0):.0f}  "
                    f"FA={mix.get('FA',0):.0f}  FAGG={mix.get('FAGG',0):.0f}  "
                    f"CAGG={mix.get('CAGG',0):.0f}  "
                    f"WATER={mix.get('WATER',0):.0f}  "
                    f"ACC={mix.get('ACC',0):.0f}  "
                    f"-> 28d={mix.get('28d_predicted',0):.1f} MPa "
                    f"({f['reason']})"
                )

        # Local optima
        if rec.get("local_optima"):
            lines.append("  LOCAL OPTIMA TO AVOID:")
            for lo in rec["local_optima"][:3]:
                lines.append(f"    {lo['signature']}  <- {lo['note']}")

        # LLM self-summary
        if rec.get("llm_learned"):
            lines.append("  WHAT LLM LEARNED (apply to this run):")
            for line in rec["llm_learned"].split("\n"):
                line = line.strip()
                if line:
                    lines.append(f"    {line}")

        lines.append("")

    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────
# 7.  STAGNATION DETECTION & RESTART
# ─────────────────────────────────────────────────────────────

def detect_stagnation(trajectory: list) -> bool:
    """
    True if BOTH conditions hold:
      (1) LLM has already found a good solution (best GWP < STAG_MIN_BEST)
      (2) No meaningful GWP improvement in the last STAG_WINDOW feasible iters
    """
    feasible = [r for r in trajectory if r["feasible"]]
    if len(feasible) < STAG_WINDOW:
        return False
    if min(r["gwp"] for r in feasible) >= STAG_MIN_BEST:
        return False
    recent = feasible[-STAG_WINDOW:]
    improvements = [recent[i-1]["gwp"] - recent[i]["gwp"]
                    for i in range(1, len(recent))]
    return all(imp < STAG_THRESHOLD for imp in improvements)


def _build_restart_msg(trajectory: list, ga_ref: dict,
                       restart_num: int) -> str:
    feasible = [r for r in trajectory if r["feasible"]]
    top5     = sorted(feasible, key=lambda r: r["gwp"])[:5]

    top5_str = ""
    for i, r in enumerate(top5, 1):
        line = (f"  #{i} iter={r['iteration']:2d}: "
                f"PC={r['PC']:.0f}  SC={r['SC']:.0f}  FA={r['FA']:.0f}  "
                f"WATER={r['WATER']:.0f}  WR_HR={r.get('WR_HR',0):.1f}"
                f"  -> GWP={r['gwp']:.2f}  28d={r['pred_28day']:.2f} MPa")
        top5_str += line + "\n"

    center = {v: round(float(np.mean([r[v] for r in feasible[-10:]])), 1)
              for v in ["PC","SC","FA","WATER"]}
    center_str = "  ".join(f"{k}={v}" for k, v in center.items())

    ga_str = (f"GWP={ga_ref['gwp']:.2f}  PC={ga_ref['PC']:.0f}  "
              f"SC={ga_ref['SC']:.0f}  FA={ga_ref['FA']:.0f}"
              if ga_ref else "N/A")

    return f"""
=== RESTART #{restart_num} -- STAGNATION DETECTED ===

You have been stuck in a local optimum with no GWP improvement.

CURRENT SEARCH CENTER (average of last 10 feasible proposals):
  {center_str}
DO NOT propose mixes similar to this region.

TOP-5 BEST SOLUTIONS SO FAR:
{top5_str}
GA REFERENCE: {ga_str}

Try ONE of these escape strategies:
  A -- Push SC to maximum allowed, reduce PC aggressively
  B -- Use FA as primary SCM instead of SC (higher FA, lower SC)
  C -- Reduce total binder by 80-120 kg, compensate with lower w/b
  D -- Try a completely different w/b range (e.g. 0.38-0.45)

Be BOLD -- small changes will not escape the local optimum.
Output ONLY the JSON object.
"""


def clip_mix(mix: dict, raw_b: dict) -> tuple:
    clean      = {}
    clip_notes = []
    for v in RAW_VARS:
        b   = raw_b[v]
        val = float(mix.get(v, b["min"]))
        clp = float(np.clip(val, b["min"], b["max"]))
        if abs(clp - val) > 0.5:
            print(f"    [clip] {v}: {val:.1f} -> {clp:.1f}")
            clip_notes.append(
                f"  {v}: you proposed {val:.0f} but bound is "
                f"[{b['min']:.0f}, {b['max']:.0f}], clipped to {clp:.0f}"
            )
        clean[v] = round(clp, 2)
    return clean, clip_notes


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


# ─────────────────────────────────────────────────────────────
# 8.  MAIN LLM LOOP
# ─────────────────────────────────────────────────────────────

def run_llm(raw_b, der_b, meta, ga_ref, few_shot, memory,
            max_iters=MAX_ITERATIONS, df=None):

    print(f"\n{'='*62}")
    print("  LLM Single-Objective Optimizer -- Minimize GWP")
    print(f"{'='*62}")
    print(f"  Model      : {GEMINI_MODEL}")
    print(f"  Objective  : minimize GWP  (kg CO2-eq/yd3)")
    constraints = [f"28-day >= {STRENGTH_MIN} MPa"]
    if STRENGTH_7D_MIN > 0:
        constraints.append(f"7-day >= {STRENGTH_7D_MIN} MPa")
    if STRENGTH_56D_MIN > 0:
        constraints.append(f"56-day >= {STRENGTH_56D_MIN} MPa")
    print(f"  Constraint : {' | '.join(constraints)}")
    if ga_ref:
        print(f"  GA target  : GWP={ga_ref['gwp']:.2f}  "
              f"28d={ga_ref['pred_28day']:.2f} MPa")
    print(f"  Max iters  : {max_iters}")
    print(f"{'='*62}")

    sys_prompt = _build_system_prompt(raw_b, der_b, few_shot, STRENGTH_MIN, memory)
    genai.configure(api_key=GEMINI_API_KEY)

    def _make_model(temp):
        return genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=sys_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temp, max_output_tokens=1024),
        )

    cur_temp      = TEMPERATURE
    model         = _make_model(cur_temp)
    chat          = model.start_chat(history=[])
    trajectory    = []
    catboost_calls = 0
    parse_fails   = 0
    restart_count = 0

    # 修改后的初始化
    cur_mix = {v: 0.0 for v in RAW_VARS}
    cur_preds = {"7day": 0.0, "28day": 0.0, "56day": 0.0}
    cur_gwp = 0.0
    cur_feas = {"feasible": True}

    ga_gwp = ga_ref["gwp"] if ga_ref else float("nan")

    print(f"\n  {'Iter':>4}  {'28d MPa':>8}  {'GWP':>8}  "
          f"{'Gap to GA':>10}  {'Feas':>4}  Mode")
    print("  " + "-"*56)

    it = 0
    already_started = False  # ← 加这行
    total_attempts = 0
    max_attempts = max_iters * 3  # safety cap to avoid infinite loop
    consec_fail = 0

    while it < max_iters and total_attempts < max_attempts:
        total_attempts += 1

        # ── Stagnation / restart ──────────────────────────────
        mode = "exploit"
        if it > 1 and restart_count < MAX_RESTARTS \
                and detect_stagnation(trajectory):
            restart_count += 1
            mode     = f"RESTART#{restart_count}"
            cur_temp = RESTART_TEMP
            model    = _make_model(cur_temp)
            chat     = model.start_chat(history=[])
            user_msg = _build_restart_msg(trajectory, ga_ref, restart_count)
            print(f"\n  [!] Stagnation -- restart #{restart_count} "
                  f"(temp -> {RESTART_TEMP})")

        # ── First iteration ───────────────────────────────────
        elif not already_started:
            user_msg = FIRST_TURN.format(strength_min=STRENGTH_MIN)
            already_started = True

        # ── Normal feedback ───────────────────────────────────
        else:
            # Return to normal temp after a restart
            if cur_temp != TEMPERATURE:
                cur_temp = TEMPERATURE
                model    = _make_model(cur_temp)
                chat     = model.start_chat(history=[])
                feas_so_far = [r for r in trajectory if r["feasible"]]
                if feas_so_far:
                    best = min(feas_so_far, key=lambda r: r["gwp"])
                    seed = (
                        f"Resume optimisation. Best solution so far:\n"
                        f"  PC={best['PC']:.0f}  SC={best['SC']:.0f}  "
                        f"FA={best['FA']:.0f}  WATER={best['WATER']:.0f}  "
                        f"WR_HR={best.get('WR_HR',0):.1f}  "
                        f"WR={best.get('WR',0):.1f}\n"
                        f"  GWP={best['gwp']:.2f} kg  "
                        f"28d={best['pred_28day']:.2f} MPa\n"
                        f"Improve from here. Output ONLY the JSON object."
                    )
                    try:
                        chat.send_message(seed)
                    except Exception:
                        pass

            user_msg = _build_feedback(
                it, max_iters, cur_mix, cur_preds, cur_gwp,
                cur_feas, trajectory, der_b, df=df,
            )
            if clip_notes:
                clip_warning = (
                        "*** BOUNDS VIOLATION IN YOUR LAST PROPOSAL ***\n"
                        "These values were outside the allowed range and were clipped:\n"
                        + "\n".join(clip_notes)
                        + "\nStay within the bounds in the system prompt.\n\n"
                )
                user_msg = clip_warning + user_msg

        # ── Call Gemini ───────────────────────────────────────
        try:
            resp     = chat.send_message(user_msg)
            raw_text = resp.text
        except Exception as exc:
            print(f"  [iter {it:02d}] API error: {exc} -- waiting 15 s ...")
            time.sleep(15)
            continue

        # ── Parse ─────────────────────────────────────────────
        parsed = parse_json(raw_text)
        if parsed is None or "mix" not in parsed:
            parse_fails += 1
            print(f"  [iter {it:02d}] JSON parse failure ({parse_fails})")
            try:
                resp2 = chat.send_message(
                    "Output ONLY the JSON object with keys 'reasoning' and 'mix'. "
                    "No markdown, no extra text."
                )
                parsed = parse_json(resp2.text)
            except Exception:
                pass
            if parsed is None or "mix" not in parsed:
                continue  # still failed, skip this attempt entirely

        reasoning = parsed.get("reasoning", "")
        mix, clip_notes = clip_mix(parsed["mix"], raw_b)
        preds     = predict(meta, mix)
        catboost_calls += 1
        g         = compute_gwp(mix)
        dv        = get_derived(mix)
        feas = check_feasibility(mix, raw_b, der_b, preds["28day"], preds)
        tb        = mix["PC"] + mix["FA"] + mix["SC"]
        gwp_gap   = round(g - ga_gwp, 2) if not np.isnan(ga_gwp) else float("nan")

        # If infeasible, do not count this as an iteration
        # Feed back the infeasible result and retry
        if not feas["feasible"]:
            consec_fail += 1
            cur_mix, cur_preds, cur_gwp, cur_feas = mix, preds, g, feas
            feas_s = "NO (retrying...)"
            print(f"  {'--':>4}  {preds['28day']:8.2f}  {g:8.2f}  "
                  f"{'infeasible':>10}  {feas_s:>8}  {mode} [retry]")
            if consec_fail >= 15:
                print(f"\n  [!] {consec_fail} consecutive infeasible attempts. Aborting.")
                break

            # Build specific infeasibility feedback and send to LLM
            issues = []
            if feas["str_v"]:
                issues.append(
                    f"- Strength {preds['28day']:.1f} MPa is below the {STRENGTH_MIN} MPa floor.\n"
                    f"  Fix: increase PC by 20-30 kg, OR reduce WATER by 10-15 kg, "
                    f"OR add ACC 200-400 kg."
                )
            for v, info in feas["der_v"].items():
                issues.append(
                    f"- {v}={info['val']:.4f} violates [{info['min']:.4f}, {info['max']:.4f}]."
                )
            for v, info in feas["raw_v"].items():
                issues.append(
                    f"- {v}={info['val']:.1f} is outside [{info['min']:.1f}, {info['max']:.1f}]."
                )

            retry_msg = (
                    f"Your proposed mix is INFEASIBLE. Do NOT repeat the same mix.\n"
                    f"Issues found:\n" + "\n".join(issues) + "\n\n"
                                                             f"Predicted 28d strength: {preds['28day']:.1f} MPa "
                                                             f"(need >= {STRENGTH_MIN} MPa)\n"
                                                             f"GWP: {g:.1f} kg\n\n"
                                                             f"Propose a DIFFERENT mix that fixes these issues. "
                                                             f"Output ONLY the JSON object."
            )
            try:
                retry_resp = chat.send_message(retry_msg)
                # Immediately parse the retry response
                retry_parsed = parse_json(retry_resp.text)
                if retry_parsed and "mix" in retry_parsed:
                    # Overwrite parsed so the next evaluation uses this response
                    parsed = retry_parsed
                    reasoning = parsed.get("reasoning", "")
                    mix, clip_notes = clip_mix(parsed["mix"], raw_b)
                    preds = predict(meta, mix)
                    catboost_calls += 1
                    g = compute_gwp(mix)
                    dv = get_derived(mix)
                    feas = check_feasibility(mix, raw_b, der_b, preds["28day"], preds)
                    tb = mix["PC"] + mix["FA"] + mix["SC"]
                    gwp_gap = round(g - ga_gwp, 2) if not np.isnan(ga_gwp) else float("nan")
                    # Loop back to feasibility check with new values
                    # by NOT continuing — fall through to the feasible block below
                    # If retry is still infeasible, go back to outer loop
                    if not feas["feasible"]:
                        time.sleep(3)
                        continue
                    # Otherwise fall through to the feasible record block below
                else:
                    time.sleep(3)
                    continue
            except Exception:
                time.sleep(3)
                continue

        # Feasible — count this as a completed iteration
        # ── Feasible — record and count ──────────────────────
        consec_fail = 0
        it += 1
        cur_mix, cur_preds, cur_gwp, cur_feas = mix, preds, g, feas

        record = {
            "iteration": it,
            "mode": mode,
            "reasoning": reasoning,
            **mix,
            **{k: round(v, 5) for k, v in dv.items()},
            "total_binder": round(tb, 2),
            "pred_7day": preds["7day"],
            "pred_28day": preds["28day"],
            "pred_56day": preds["56day"],
            "gwp": g,
            "gwp_gap": gwp_gap,
            "str_margin": round(preds["28day"] - STRENGTH_MIN, 2),
            "feasible": True,
        }
        trajectory.append(record)

        gap_s = f"{gwp_gap:+.2f}" if not np.isnan(gwp_gap) else "   n/a"
        print(f"  {it:4d}  {preds['28day']:8.2f}  {g:8.2f}  "
              f"{gap_s:>10}  {'YES':>4}  {mode}")

        time.sleep(4)

    print(f"\n  Total restarts: {restart_count}  |  "
          f"Parse failures: {parse_fails}  |  "
          f"Total attempts: {total_attempts}  |  "
          f"Feasible iters: {it}")

    # ── Ask LLM to summarise what it learned this run ────────
    run_summary = ""
    try:
        traj_summary = "\n".join([
            f"iter={r['iteration']} PC={r['PC']:.0f} SC={r['SC']:.0f} "
            f"FA={r['FA']:.0f} WATER={r['WATER']:.0f} "
            f"WR={r.get('WR', 0):.0f} GWP={r['gwp']:.2f} "
            f"28d={r['pred_28day']:.2f} feasible={r['feasible']} "
            f"reasoning={r.get('reasoning', '')}"
            for r in trajectory
        ])
        summary_prompt = f"""You just completed a concrete mix optimisation run.
    Strength constraint: 28-day >= {STRENGTH_MIN} MPa
    Total iterations: {len(trajectory)}
    Feasible solutions: {len([r for r in trajectory if r['feasible']])}

    Full trajectory (iteration, mix, GWP, strength, feasibility, your reasoning):
    {traj_summary}

    Summarise what you learned in 8-12 bullet points. Focus on:
    - Which ingredient changes reliably reduced GWP
    - Which changes caused infeasible solutions and why
    - What local optima you got stuck in and what their signatures are
    - What strategies worked vs failed
    - Any non-obvious relationships you observed between ingredients
    - What you would do differently next time

    Write concise, specific, quantitative bullets. 
    Example: "Reducing PC by 20 kg and increasing SC by 20 kg saved ~15 kg GWP 
    while keeping 28d strength above the floor"
    Not: "SC substitution is good"

    Output ONLY the bullet points, no preamble."""

        summary_resp = chat.send_message(summary_prompt)
        run_summary = summary_resp.text.strip()
        print(f"\n  LLM run summary:\n{run_summary}")
    except Exception as e:
        print(f"\n  [Warning] Could not get run summary from LLM: {e}")
        run_summary = ""

    # ── Print best solution found ─────────────────────────────
    feasible_all = [r for r in trajectory if r["feasible"]]
    if feasible_all:
        best = min(feasible_all, key=lambda r: r["gwp"])
        print(f"\n  {'=' * 50}")
        print(f"  BEST SOLUTION (GWP={best['gwp']:.2f} kg, "
              f"28d={best['pred_28day']:.2f} MPa, iter={best['iteration']})")
        print(f"  PC={best['PC']:.0f}  SC={best['SC']:.0f}  "
              f"FA={best['FA']:.0f}  "
              f"WATER={best['WATER']:.0f}  WR={best.get('WR', 0):.0f}")
        print(f"  {'=' * 50}\n")
    return trajectory, run_summary, catboost_calls


# ─────────────────────────────────────────────────────────────
# 9.  REPORT & SAVE
# ─────────────────────────────────────────────────────────────

def build_report(trajectory: list, ga_ref: dict) -> str:
    feasible = [r for r in trajectory if r["feasible"]]
    best     = min(feasible, key=lambda r: r["gwp"]) if feasible else None
    ga_gwp   = ga_ref["gwp"] if ga_ref else float("nan")

    SEP  = "=" * 62
    sep  = "-" * 62
    skip = {"reasoning"}
    restart_iters = [r["iteration"] for r in trajectory
                     if r.get("mode","").startswith("RESTART")]

    lines = [
        SEP,
        "  LLM Single-Objective Optimizer -- Results Report",
        f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        SEP,
        f"  Objective         : minimize GWP",
        f"  Strength floor    : >= {STRENGTH_MIN} MPa",
        f"  Total iterations  : {len(trajectory)}",
        f"  Feasible solutions: {len(feasible)} / {len(trajectory)}",
        f"  Restarts triggered: {len(restart_iters)}  (iters: {restart_iters})",
        sep,
        "  GA REFERENCE SOLUTION",
        sep,
    ]
    if ga_ref:
        for k, v in ga_ref.items():
            lines.append(f"    {k:<18}: {v}")

    lines += [sep, "  BEST LLM SOLUTION  (lowest feasible GWP)", sep]
    if best:
        for k, v in best.items():
            if k not in skip:
                lines.append(f"    {k:<18}: {v}")
        lines += [
            sep,
            f"    GWP gap to GA    : {best['gwp'] - ga_gwp:+.2f} kg CO2/yd3",
            f"    GWP ratio        : {best['gwp']/ga_gwp:.4f}  (1.0 = matches GA)",
            f"    Strength margin  : {best['str_margin']:+.2f} MPa above floor",
        ]
    else:
        lines.append("    No feasible solution found.")

    lines += [
        sep,
        "  FULL TRAJECTORY",
        sep,
        f"  {'Iter':>4}  {'28d':>7}  {'GWP':>8}  {'Gap':>7}  "
        f"{'PC':>5}  {'SC':>5}  {'FA':>5}  "
        f"{'FAGG':>6}  {'CAGG':>6}  {'ACC':>5}  "
        f"{'w/b':>6}  {'SCM%':>6}  Feas  Mode",
        sep,
    ]
    for r in trajectory:
        gap_s = f"{r.get('gwp_gap', float('nan')):+.2f}" \
            if not np.isnan(r.get("gwp_gap", float("nan"))) else "   n/a"
        lines.append(
            f"  {r['iteration']:4d}  {r['pred_28day']:7.2f}  {r['gwp']:8.2f}  "
            f"{gap_s:>7}  {r['PC']:5.0f}  {r['SC']:5.0f}  {r['FA']:5.0f}  "
            f"{r.get('FAGG', 0):6.0f}  {r.get('CAGG', 0):6.0f}  "
            f"{r.get('ACC', 0):5.0f}  "
            f"{r.get('w/b', 0):6.4f}  "
            f"{r.get('SCM%', 0) * 100:5.1f}%  "
            f"{'Y' if r['feasible'] else 'N':4s}  {r.get('mode', '')}"
        )

    lines += [sep, "  CHAIN-OF-THOUGHT LOG", sep]
    for r in trajectory:
        lines.append(f"  Iter {r['iteration']:2d}: {r.get('reasoning','')}")
    lines.append(SEP)

    return "\n".join(lines)

def compute_metrics(trajectory, ga_ref, raw_b,
                    total_catboost_calls, ga_catboost_calls=20000) -> dict:
    """
    OGR: Optimality Gap Ratio = (best_llm_gwp - ga_gwp) / ga_gwp
    QER: Query Efficiency Ratio = delta_gwp_total / N_catboost_calls
    MCE: Mix Composition Entropy = sum of normalized std dev across variables
    """
    feasible = [r for r in trajectory if r["feasible"]]
    if not feasible:
        return {"OGR": float("nan"), "QER": float("nan"),
                "MCE": float("nan"), "note": "no feasible solutions"}

    ga_gwp    = ga_ref["gwp"] if ga_ref else float("nan")
    best_gwp  = min(r["gwp"] for r in feasible)
    first_gwp = feasible[0]["gwp"]

    # OGR
    OGR = (best_gwp - ga_gwp) / ga_gwp if ga_ref else float("nan")

    # QER
    delta_gwp = first_gwp - best_gwp
    QER = delta_gwp / total_catboost_calls if total_catboost_calls > 0 else float("nan")

    # MCE: normalized std across feasible iterations
    mce = 0.0
    for v in RAW_VARS:
        vals    = [r[v] for r in feasible if v in r]
        if len(vals) < 2:
            continue
        std     = float(np.std(vals))
        v_range = raw_b[v]["max"] - raw_b[v]["min"]
        if v_range > 0:
            mce += std / v_range

    return {
        "OGR":                  round(OGR, 4),
        "QER":                  round(QER, 4),
        "MCE":                  round(mce, 4),
        "best_gwp":             best_gwp,
        "ga_gwp":               ga_gwp,
        "gwp_gap":              round(best_gwp - ga_gwp, 2),
        "feasibility_rate":     round(len(feasible) / len(trajectory), 4),
        "convergence_iter":     min(feasible, key=lambda r: r["gwp"])["iteration"],
        "total_catboost_calls": total_catboost_calls,
        "ga_catboost_calls": ga_catboost_calls,
        "calls_ratio": round(total_catboost_calls / ga_catboost_calls, 4),
    }




def save_all(trajectory: list, ga_ref: dict) -> None:
    if not trajectory:
        print("\n  [Warning] No feasible solutions found — nothing to save.")
        return
    save_cols = [c for c in trajectory[0] if c != "reasoning"]
    pd.DataFrame(trajectory)[save_cols].to_csv(LLM_CSV, index=False)
    print(f"\n  Trajectory -> '{LLM_CSV}'")

    if ga_ref:
        pd.DataFrame([ga_ref]).to_csv(GA_CSV, index=False)
        print(f"  GA reference -> '{GA_CSV}'")

    report = build_report(trajectory, ga_ref)
    print("\n" + report)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report -> '{REPORT_TXT}'")


# ─────────────────────────────────────────────────────────────
# 10.  ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    global STRENGTH_MIN, MAX_ITERATIONS, MAX_RESTARTS, STRENGTH_7D_MIN, STRENGTH_56D_MIN

    parser = argparse.ArgumentParser(
        description="LLM Single-Objective Optimizer -- Minimize GWP (28d >= floor)"
    )
    parser.add_argument("--strength",     type=float, default=STRENGTH_MIN,
                        help=f"28-day strength floor MPa (default {STRENGTH_MIN})")
    parser.add_argument("--strength-7d", type=float, default=STRENGTH_7D_MIN,
                        help=f"7-day strength floor MPa (default {STRENGTH_7D_MIN}, 0=disabled)")
    parser.add_argument("--strength-56d", type=float, default=STRENGTH_56D_MIN,
                        help=f"56-day strength floor MPa (default {STRENGTH_56D_MIN}, 0=disabled)")
    parser.add_argument("--iters",        type=int,   default=MAX_ITERATIONS,
                        help=f"LLM iterations (default {MAX_ITERATIONS})")
    parser.add_argument("--ga-gen",       type=int,   default=GA_GENS,
                        help=f"GA generations (default {GA_GENS})")
    parser.add_argument("--ga-pop",       type=int,   default=GA_POP,
                        help=f"GA population size (default {GA_POP})")
    parser.add_argument("--max-restarts", type=int,   default=MAX_RESTARTS,
                        help=f"Max OPRO restarts (default {MAX_RESTARTS})")
    parser.add_argument("--skip-ga",      action="store_true",
                        help="Skip GA phase (reuse saved ga_reference_solution.csv)")
    args = parser.parse_args()

    STRENGTH_MIN   = args.strength
    MAX_ITERATIONS = args.iters
    MAX_RESTARTS   = args.max_restarts
    STRENGTH_7D_MIN = args.strength_7d
    STRENGTH_56D_MIN = args.strength_56d

    # ── Create output folder for this run ────────────────────
    from datetime import datetime
    import os
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = (f"run_s28d{int(STRENGTH_MIN)}"
                  f"_s7d{int(STRENGTH_7D_MIN)}"
                  f"_s56d{int(STRENGTH_56D_MIN)}"
                  f"_i{MAX_ITERATIONS}_{RAG_MODE}_{run_id}")
    os.makedirs(run_folder, exist_ok=True)
    print(f"\n  Output folder: '{run_folder}'")

    # Redirect output file paths into this folder
    global LLM_CSV, GA_CSV, REPORT_TXT
    LLM_CSV = os.path.join(run_folder, "llm_optimizer_results.csv")
    GA_CSV = os.path.join(run_folder, "ga_reference_solution.csv")
    REPORT_TXT = os.path.join(run_folder, "llm_optimizer_report.txt")

    print("\n[1/3] Loading dataset and CatBoost surrogate ...")
    df   = load_df(DATA_PATH)
    raw_b, der_b = get_bounds(df)
    meta = load_surrogate(MODEL_PKL)
    print(f"      Dataset: {len(df)} rows  |  "
          f"Feasible (28d>={STRENGTH_MIN}): "
          f"{df.dropna(subset=['28day'])[df['28day']>=STRENGTH_MIN].shape[0]} rows")

    GA_CSV_SAVED = "ga_reference_solution.csv"  # fixed location for reuse
    if args.skip_ga and pd.io.common.file_exists(GA_CSV_SAVED):
        print(f"\n[2/3] Loading saved GA reference from '{GA_CSV_SAVED}' ...")
        ga_ref = pd.read_csv(GA_CSV_SAVED).iloc[0].to_dict()
        print(f"      GWP={ga_ref['gwp']:.2f}  28d={ga_ref['pred_28day']:.2f} MPa")
        ga_catboost_calls = GA_GENS * GA_POP
    else:
        print("\n[2/3] Running GA reference optimisation ...")
        ga_ref, ga_catboost_calls = run_ga(raw_b, der_b, meta,
                                           n_gen=args.ga_gen, pop=args.ga_pop)

    few_shot = select_few_shot(df, n=3)

    print("\n[3/3] Starting LLM optimisation ...")
    trajectory, run_summary, catboost_calls = run_llm(
        raw_b, der_b, meta, ga_ref, few_shot,
        None, max_iters=args.iters, df=df
    )
    save_all(trajectory, ga_ref)

    # Compute and print metrics
    raw_b_for_metrics, _ = get_bounds(df)
    metrics = compute_metrics(trajectory, ga_ref, raw_b_for_metrics, catboost_calls)
    print("\n" + "=" * 40)
    print("  EVALUATION METRICS")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"  {k:<25}: {v}")

    import json
    metrics_path = os.path.join(run_folder, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved -> '{metrics_path}'")



if __name__ == "__main__":
    main()
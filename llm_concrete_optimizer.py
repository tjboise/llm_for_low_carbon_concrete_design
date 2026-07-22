"""
LLM-Based Iterative Fuzzy Optimizer for Low-Carbon Concrete Mix Design
=======================================================================
Reference: Forootani (2025) IEEE TAI  — LLM fuzzy control loop
           Ramos et al. (2023) ICLR   — LLAMBO, LLM-enhanced BO
           Völker et al. (2024)       — LLMs for sustainable concrete (KDD)

Pipeline
--------
PHASE 1 — NSGA-II baseline
  Run NSGA-II (pymoo) to obtain the true Pareto front.
  Apply TOPSIS on the Pareto front to select a single "best balanced" solution
  (equal weights on maximise-strength and minimise-GWP).
  This TOPSIS point is the ground-truth reference for evaluating the LLM.

PHASE 2 — LLM optimiser  (Gemini 2.0 Flash, multi-turn)
  System prompt includes:
    • Variable bounds (raw + derived)
    • GWP formula and fuzzy design rules
    • 3 few-shot ICL examples extracted from Phase 1 trajectory
      (showing good iterative reasoning patterns)
  Each iteration:
    1. Gemini proposes a mix (JSON)
    2. CatBoost evaluates 7d / 28d / 56d strength
    3. Compute GWP, feasibility, and FIVE per-iteration metrics:
         d_topsis   — normalised Euclidean distance to TOPSIS best point
         d_pareto   — normalised distance to nearest point on NSGA-II front
         hv_contrib — 2-D hypervolume contribution of this solution
         domination_rank — how many NSGA-II solutions this solution dominates
         improvement_rate — relative improvement in d_topsis vs previous iter
    4. Feedback returned to Gemini includes d_topsis and directional hint
  Stopping: fixed MAX_ITERATIONS (default 30).  d_topsis logged every round.

PHASE 3 — Summary
  Convergence plot data (d_topsis per iter), final Pareto comparison,
  IGD / GD+ / HV-ratio between LLM Pareto and NSGA-II Pareto.
  All results saved to CSV + TXT report.

Usage
-----
  python llm_concrete_optimizer.py                  # full run
  python llm_concrete_optimizer.py --iters 40       # more LLM iterations
  python llm_concrete_optimizer.py --strength 50    # tighter strength floor
  python llm_concrete_optimizer.py --nsga2-gen 300  # more NSGA-II generations
  python llm_concrete_optimizer.py --skip-nsga2     # skip Phase 1 (debug only)

Dependencies
------------
  pip install google-generativeai pandas numpy scikit-learn catboost joblib pymoo
"""

import json
import re
import time
import warnings
import joblib
import argparse
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd

import google.generativeai as genai

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 0.  GLOBAL CONFIGURATION
# ─────────────────────────────────────────────────────────────

GEMINI_API_KEY   = "AIzaSyDnV_LdQ2aztxCjwuEckEFFYQfc-se4ERA"
GEMINI_MODEL     = "gemini-2.0-flash"
TEMPERATURE      = 0.9

STRENGTH_MIN     = 40.0        # feasibility floor (MPa)
MAX_ITERATIONS   = 60          # LLM iterations
NSGA2_GENS       = 200         # NSGA-II generations
NSGA2_POP        = 100         # NSGA-II population size
TOPSIS_W_STR     = 0.5         # TOPSIS weight for strength (0-1)
TOPSIS_W_GWP     = 0.5         # TOPSIS weight for GWP     (must sum to 1)

# ── OPRO-style restart parameters (Yang et al. 2023, "Large Language Models as Optimizers")
STAG_WINDOW      = 5           # consecutive iters with no d_topsis improvement -> restart
STAG_THRESHOLD   = 0.005       # minimum improvement per iter to count as "progress"
STAG_MIN_D_EVER  = 0.25        # restart only fires if LLM has already found a good solution
                               # (d_topsis < this value at least once). Prevents early panic.
RESTART_TEMP     = 1.3         # Gemini temperature during restart (promotes diversity)
NORMAL_TEMP      = TEMPERATURE # temperature for normal exploitation
MAX_RESTARTS     = 3           # max restarts allowed per run
RESTART_TOP_K    = 5           # top-K historical solutions shown in restart prompt
ANTI_OSC_WINDOW  = 3           # detect oscillation: N consecutive near-identical proposals

DATA_PATH        = "Super_Cleaned_Concrete_Data.csv"
MODEL_PKL        = "concrete_catboost_optimized.pkl"
NSGA2_CSV        = "nsga2_pareto_front.csv"
LLM_CSV          = "llm_optimizer_results.csv"
REPORT_TXT       = "llm_optimizer_report.txt"

# GWP emission factors  (lb CO2-eq / lb material)
GWP = {
    "PC": 1.048, "FA": 0.328, "SC": 0.264, "SF": 0.850,
    "CAGG": 0.0037, "FAGG": 0.0026,
    "WATER": 0.0, "AEA": 0.0, "WR_HR": 0.0, "WR": 0.0, "ACC": 0.0,
}

RAW_VARS     = ["PC","FA","SC","SF","FAGG","CAGG","WATER","AEA","WR_HR","WR","ACC"]
DERIVED_VARS = ["w/b","b/a","SCM%","CAGG%","FAGG%","PC%","FA%","SC%"]


# ─────────────────────────────────────────────────────────────
# 1.  DATA & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return add_derived(df)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tb  = df["PC"] + df["FA"] + df["SC"] + df["SF"]
    agg = df["FAGG"] + df["CAGG"]
    df["TOTAL_BINDER"] = tb
    df["w/b"]   = df["WATER"] / tb
    df["b/a"]   = tb / agg
    df["SCM%"]  = (df["FA"] + df["SC"] + df["SF"]) / tb
    df["CAGG%"] = df["CAGG"] / agg
    df["FAGG%"] = df["FAGG"] / agg
    df["PC%"]   = df["PC"]   / tb
    df["FA%"]   = df["FA"]   / tb
    df["SC%"]   = df["SC"]   / tb
    return df


def get_bounds(df: pd.DataFrame):
    raw = {v: {"min": float(df[v].min()), "max": float(df[v].max())} for v in RAW_VARS}
    der = {v: {"min": float(df[v].min()), "max": float(df[v].max())} for v in DERIVED_VARS}
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
    tb = m["PC"] + m["FA"] + m["SC"] + m["SF"]
    ag = m["FAGG"] + m["CAGG"]
    e  = 0
    m["TOTAL_BINDER"] = tb
    m["w/b"]   = m["WATER"] / (tb + e)
    m["b/a"]   = tb / (ag + e)
    m["SCM%"]  = (m["FA"] + m["SC"] + m["SF"]) / (tb + e)
    m["CAGG%"] = m["CAGG"] / (ag + e)
    m["FAGG%"] = m["FAGG"] / (ag + e)
    m["PC%"]   = m["PC"]   / (tb + e)
    m["FA%"]   = m["FA"]   / (tb + e)
    m["SC%"]   = m["SC"]   / (tb + e)
    return m


def predict(meta: dict, mix: dict) -> dict:
    """Chained CatBoost: mix -> 7d -> 28d -> 56d."""
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


def gwp(mix: dict) -> float:
    return round(sum(mix.get(k,0.) * v for k, v in GWP.items()), 2)


def derived_vals(mix: dict) -> dict:
    m = _engineer_one(mix)
    return {k: round(m[k], 5) for k in DERIVED_VARS}


# ─────────────────────────────────────────────────────────────
# 3.  FEASIBILITY CHECK
# ─────────────────────────────────────────────────────────────

def check_feasibility(mix: dict, raw_b: dict, der_b: dict, p28: float) -> dict:
    rv = {}
    for v, b in raw_b.items():
        val = mix.get(v, 0.)
        if val < b["min"] - 0.5 or val > b["max"] + 0.5:
            rv[v] = {"val": val, "min": b["min"], "max": b["max"]}

    dv_vals = derived_vals(mix)
    dv = {}
    for v, b in der_b.items():
        val = dv_vals[v]
        tol = (b["max"] - b["min"]) * 0.01 + 1e-6
        if val < b["min"] - tol or val > b["max"] + tol:
            dv[v] = {"val": round(val,4), "min": round(b["min"],4), "max": round(b["max"],4)}

    sv = p28 < STRENGTH_MIN
    return {
        "raw_violations":     rv,
        "derived_violations": dv,
        "strength_violation": sv,
        "feasible": not rv and not dv and not sv,
    }


# ─────────────────────────────────────────────────────────────
# 4.  NSGA-II  (Phase 1)
# ─────────────────────────────────────────────────────────────

def run_nsga2(raw_b: dict, der_b: dict, meta: dict,
              n_gen: int = NSGA2_GENS, pop: int = NSGA2_POP) -> list:
    """
    Bi-objective NSGA-II:
      f1 = -pred_28day  (minimise -> maximise strength)
      f2 = total GWP    (minimise)
    Constraints:
      pred_28day >= STRENGTH_MIN
      all derived ratios within dataset bounds
    Returns list of record dicts.
    """
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem
        from pymoo.optimize import minimize as pymoo_min
        from pymoo.termination import get_termination
    except ImportError:
        raise ImportError("pymoo not installed. Run: pip install pymoo")

    xl = np.array([raw_b[v]["min"] for v in RAW_VARS])
    xu = np.array([raw_b[v]["max"] for v in RAW_VARS])
    n_der_c = len(DERIVED_VARS) * 2   # lower + upper per derived var

    class Prob(Problem):
        def __init__(self):
            super().__init__(n_var=len(RAW_VARS), n_obj=2,
                             n_ieq_constr=1 + n_der_c, xl=xl, xu=xu)

        def _evaluate(self, X, out, *args, **kwargs):
            F, G = [], []
            for row in X:
                mix  = dict(zip(RAW_VARS, row))
                pr   = predict(meta, mix)
                g_   = gwp(mix)
                F.append([-pr["28day"], g_])
                gc   = [STRENGTH_MIN - pr["28day"]]
                dv   = derived_vals(mix)
                for v in DERIVED_VARS:
                    b = der_b[v]
                    gc += [b["min"] - dv[v], dv[v] - b["max"]]
                G.append(gc)
            out["F"] = np.array(F)
            out["G"] = np.array(G)

    print(f"\n[NSGA-II] Running {n_gen} generations × pop={pop} ...")
    res = pymoo_min(Prob(), NSGA2(pop_size=pop),
                    termination=get_termination("n_gen", n_gen),
                    seed=42, verbose=False)

    records = []
    if res.X is not None:
        for x in res.X:
            mix = dict(zip(RAW_VARS, x))
            pr  = predict(meta, mix)
            g_  = gwp(mix)
            dv  = derived_vals(mix)
            feas = check_feasibility(mix, raw_b, der_b, pr["28day"])
            tb  = mix["PC"] + mix["FA"] + mix["SC"] + mix["SF"]
            records.append({
                **{k: round(float(v), 2) for k, v in mix.items()},
                **{k: round(float(v), 5) for k, v in dv.items()},
                "total_binder": round(tb, 2),
                "pred_7day":  pr["7day"],
                "pred_28day": pr["28day"],
                "pred_56day": pr["56day"],
                "gwp":        g_,
                "feasible":   feas["feasible"],
            })

    feasible = [r for r in records if r["feasible"]]
    print(f"[NSGA-II] Done. {len(records)} solutions, {len(feasible)} feasible.")
    return records


# ─────────────────────────────────────────────────────────────
# 5.  TOPSIS
# ─────────────────────────────────────────────────────────────

def topsis(records: list, w_str: float = TOPSIS_W_STR,
           w_gwp: float = TOPSIS_W_GWP) -> dict:
    """
    Apply TOPSIS on a set of records.
    Objectives: maximise pred_28day, minimise gwp.
    Weights: w_str + w_gwp = 1.
    Returns the single best record.
    """
    if not records:
        raise ValueError("Empty records list passed to topsis()")

    s_arr = np.array([r["pred_28day"] for r in records])
    c_arr = np.array([r["gwp"]        for r in records])

    # Normalise
    s_n = s_arr / (np.sqrt(np.sum(s_arr**2)) + 1e-9)
    c_n = c_arr / (np.sqrt(np.sum(c_arr**2)) + 1e-9)

    # Weighted normalised matrix
    ws = s_n * w_str
    wc = c_n * w_gwp

    # Ideal best / worst
    # strength: higher is better  -> ideal = max, anti = min
    # GWP:      lower  is better  -> ideal = min, anti = max
    ideal_s, ideal_c  = ws.max(), wc.min()
    anti_s,  anti_c   = ws.min(), wc.max()

    d_pos = np.sqrt((ws - ideal_s)**2 + (wc - ideal_c)**2)
    d_neg = np.sqrt((ws - anti_s )**2 + (wc - anti_c )**2)

    score = d_neg / (d_pos + d_neg + 1e-9)
    best_idx = int(np.argmax(score))
    best = dict(records[best_idx])
    best["topsis_score"] = round(float(score[best_idx]), 4)
    return best


# ─────────────────────────────────────────────────────────────
# 6.  PER-ITERATION METRICS
# ─────────────────────────────────────────────────────────────

def _norm_pt(pt: dict, ref_pts: list) -> tuple:
    """Normalise a (strength, gwp) point using the combined reference set."""
    s_all = [r["pred_28day"] for r in ref_pts]
    c_all = [r["gwp"]        for r in ref_pts]
    s_min, s_max = min(s_all), max(s_all)
    c_min, c_max = min(c_all), max(c_all)
    eps = 1e-9
    return (
        (pt["pred_28day"] - s_min) / (s_max - s_min + eps),
        (pt["gwp"]        - c_min) / (c_max - c_min + eps),
    )


def dist_to_topsis(record: dict, topsis_pt: dict, all_ref: list) -> float:
    """
    Normalised Euclidean distance from record to TOPSIS optimum.
    Normalisation uses combined range of [record] + all_ref.
    Range: [0, sqrt(2)]; 0 = perfect match.
    """
    pool = all_ref + [record, topsis_pt]
    rn   = _norm_pt(record,    pool)
    tn   = _norm_pt(topsis_pt, pool)
    return round(float(np.sqrt((rn[0]-tn[0])**2 + (rn[1]-tn[1])**2)), 4)


def dist_to_nearest_pareto(record: dict, nsga_front: list, all_ref: list) -> float:
    """
    Normalised distance from record to nearest point on the NSGA-II Pareto front.
    """
    if not nsga_front:
        return float("nan")
    pool = all_ref + [record]
    rn   = _norm_pt(record, pool)
    dists = []
    for npt in nsga_front:
        nn = _norm_pt(npt, pool)
        dists.append(np.sqrt((rn[0]-nn[0])**2 + (rn[1]-nn[1])**2))
    return round(float(min(dists)), 4)


def hv_contribution(record: dict, existing: list, all_ref: list) -> float:
    """
    2-D hypervolume contribution of record given existing non-dominated set.
    Reference point = (0, 1.1) in normalised minimisation space
    where obj1 = -strength_norm, obj2 = gwp_norm.
    """
    def hv2d(pts_norm, ref):
        sorted_pts = sorted(pts_norm, key=lambda p: p[0])
        hv, prev_x = 0., ref[0]
        for x, y in sorted_pts:
            if y < ref[1]:
                hv    += (prev_x - x) * (ref[1] - y)
                prev_x = x
        return hv

    pool   = all_ref + [record] + existing
    ref_pt = (
        max(-_norm_pt(r, pool)[0] for r in pool) * 1.1,
        max( _norm_pt(r, pool)[1] for r in pool) * 1.1,
    )

    def to_min(r):
        n = _norm_pt(r, pool)
        return (-n[0], n[1])  # minimise -strength_norm, gwp_norm

    pts_with    = [to_min(r) for r in existing + [record]]
    pts_without = [to_min(r) for r in existing]

    hv_w  = hv2d(pts_with,    ref_pt)
    hv_wo = hv2d(pts_without, ref_pt)
    return round(float(hv_w - hv_wo), 6)


def domination_count(record: dict, nsga_front: list) -> int:
    """
    How many NSGA-II Pareto solutions does this record dominate?
    (record dominates npt if record.str >= npt.str AND record.gwp <= npt.gwp,
     with at least one strict.)
    """
    count = 0
    for npt in nsga_front:
        if (record["pred_28day"] >= npt["pred_28day"] and
                record["gwp"] <= npt["gwp"] and
                (record["pred_28day"] > npt["pred_28day"] or
                 record["gwp"] < npt["gwp"])):
            count += 1
    return count


def compute_all_metrics(record: dict, topsis_pt: dict,
                        nsga_front: list, llm_traj_so_far: list) -> dict:
    """
    Compute all five per-iteration metrics.
    llm_traj_so_far: feasible LLM solutions seen so far (for HV contribution).
    """
    all_ref = nsga_front + [topsis_pt]

    d_top  = dist_to_topsis(record, topsis_pt, all_ref)
    d_par  = dist_to_nearest_pareto(record, nsga_front, all_ref)
    hv_con = hv_contribution(record, llm_traj_so_far, all_ref)
    dom_n  = domination_count(record, nsga_front)

    prev_d = llm_traj_so_far[-1].get("d_topsis", None) if llm_traj_so_far else None
    impr   = round((prev_d - d_top) / (prev_d + 1e-9), 4) if prev_d else float("nan")

    return {
        "d_topsis":        d_top,   # main convergence metric
        "d_pareto":        d_par,   # distance to NSGA-II front
        "hv_contrib":      hv_con,  # hypervolume contribution
        "dom_count":       dom_n,   # how many NSGA-II pts dominated
        "improvement":     impr,    # relative improvement vs prev iter
    }


# ─────────────────────────────────────────────────────────────
# 7.  AGGREGATE METRICS  (end-of-run)
# ─────────────────────────────────────────────────────────────

def _normalise_front(pts: list, all_pts: list) -> list:
    s = [p["pred_28day"] for p in all_pts]
    c = [p["gwp"]        for p in all_pts]
    s0, s1 = min(s), max(s)
    c0, c1 = min(c), max(c)
    e = 1e-9
    return [((p["pred_28day"]-s0)/(s1-s0+e), (p["gwp"]-c0)/(c1-c0+e))
            for p in pts]


def extract_pareto(records: list) -> list:
    feas = [r for r in records if r.get("feasible", True)]
    out  = []
    for i, ri in enumerate(feas):
        if not any(
            feas[j]["pred_28day"] >= ri["pred_28day"] and
            feas[j]["gwp"]        <= ri["gwp"] and
            (feas[j]["pred_28day"] > ri["pred_28day"] or feas[j]["gwp"] < ri["gwp"])
            for j in range(len(feas)) if j != i
        ):
            out.append(ri)
    return out


def igd(llm_p: list, nsga_p: list) -> float:
    if not llm_p or not nsga_p:
        return float("nan")
    all_ = llm_p + nsga_p
    ln = _normalise_front(llm_p,  all_)
    rn = _normalise_front(nsga_p, all_)
    return round(
        sum(min(np.sqrt((rp[0]-lp[0])**2+(rp[1]-lp[1])**2) for lp in ln)
            for rp in rn) / len(rn), 4)


def gd_plus(llm_p: list, nsga_p: list) -> float:
    if not llm_p or not nsga_p:
        return float("nan")
    all_ = llm_p + nsga_p
    ln = _normalise_front(llm_p,  all_)
    rn = _normalise_front(nsga_p, all_)
    return round(
        sum(min(np.sqrt((lp[0]-rp[0])**2+(lp[1]-rp[1])**2) for rp in rn)
            for lp in ln) / len(ln), 4)


def hv_ratio(llm_p: list, nsga_p: list) -> float:
    if not llm_p or not nsga_p:
        return float("nan")

    def hv2d(pts, ref):
        sp = sorted([(-r["pred_28day"], r["gwp"]) for r in pts], key=lambda x: x[0])
        hv, px = 0., ref[0]
        for x, y in sp:
            if y < ref[1]:
                hv += (px - x) * (ref[1] - y)
                px  = x
        return hv

    all_ = llm_p + nsga_p
    ref  = (max(-r["pred_28day"] for r in all_)*1.1,
            max(r["gwp"]         for r in all_)*1.1)
    hl   = hv2d(llm_p,  ref)
    hn   = hv2d(nsga_p, ref)
    return round(hl / hn, 4) if hn > 1e-9 else float("nan")


# ─────────────────────────────────────────────────────────────
# 8.  FEW-SHOT ICL EXAMPLE BUILDER
# ─────────────────────────────────────────────────────────────

def build_icl_examples(nsga_records: list, topsis_pt: dict) -> str:
    """
    Build 3 ICL examples from NSGA-II trajectory demonstrating good
    reasoning patterns for the LLM:
      Example 1 — high-strength starting region
      Example 2 — GWP reduction via SC substitution
      Example 3 — handling w/b constraint violation (recovery)
    These illustrate the optimisation trajectory without revealing the target.
    """
    feasible = [r for r in nsga_records if r["feasible"]]
    if len(feasible) < 3:
        return "(No ICL examples available — NSGA-II produced too few feasible solutions.)"

    # Sort by strength desc
    by_str = sorted(feasible, key=lambda r: -r["pred_28day"])
    # Sort by GWP asc
    by_gwp = sorted(feasible, key=lambda r:  r["gwp"])
    # Middle-ground (closest to TOPSIS)
    all_ref = feasible + [topsis_pt]
    by_topsis = sorted(feasible,
                       key=lambda r: dist_to_topsis(r, topsis_pt, all_ref))

    ex1 = by_str[0]    # best strength
    ex3 = by_gwp[0]    # lowest GWP
    ex2 = by_topsis[0] # closest to TOPSIS (balanced)

    def fmt(r, label, note):
        return (
            f"  [{label}]\n"
            f"  Mix:  PC={r['PC']:.0f}  FA={r['FA']:.0f}  SC={r['SC']:.0f}  "
            f"SF={r['SF']:.0f}  WATER={r['WATER']:.0f}  WR_HR={r['WR_HR']:.1f}  WR={r['WR']:.1f}\n"
            f"  w/b={r['w/b']:.3f}  SCM%={r['SCM%']*100:.1f}%  b/a={r['b/a']:.3f}\n"
            f"  -> 28-day={r['pred_28day']:.1f} MPa   GWP={r['gwp']:.1f} lb CO2/yd3\n"
            f"  Lesson: {note}\n"
        )

    block = (
        "The following examples are taken from a high-quality reference optimisation run.\n"
        "Study the mix ratios and their outcomes to understand effective design patterns:\n\n"
        + fmt(ex1, "High-strength region",
              "High total binder (PC+SC) with low w/b achieves peak strength. "
              "Use as a starting point when strength is far below target.")
        + fmt(ex2, "Balanced trade-off (closest to ideal)",
              "Moderate PC with high SC substitution balances strength and GWP well. "
              "SC (GWP=0.264) is the most efficient binder for carbon reduction.")
        + fmt(ex3, "Low-GWP region",
              "Minimum PC with maximum SC and minimal FA achieves lowest GWP. "
              "Note: very low w/b requires WR_HR >= 40 for workability.")
    )
    return block


# ─────────────────────────────────────────────────────────────
# 9.  PROMPT CONSTRUCTION
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert concrete mix design engineer specialising in low-carbon,
high-performance concrete.

OPTIMISATION OBJECTIVE
======================
Bi-objective Pareto optimisation — find mixes that simultaneously:
  (A) MAXIMISE  28-day compressive strength  (MPa)
  (B) MINIMISE  total embodied carbon / GWP  (lb CO2-eq/yd3)

WHAT "OPTIMAL" MEANS IN THIS PROBLEM
======================================
The evaluation uses TOPSIS with EQUAL weights (50% strength / 50% GWP).
This means:
  - A mix with moderate strength AND low GWP beats a mix with maximum
    strength but high GWP.
  - Do NOT chase peak strength alone — the goal is to improve BOTH
    objectives simultaneously.
  - At each step: does my proposal improve strength AND GWP vs the last?
    Or at least improve one without worsening the other?

GWP formula:
  GWP = PC*1.048 + FA*0.328 + SC*0.264 + SF*0.850
      + CAGG*0.0037 + FAGG*0.0026

CRITICAL: SF has GWP=0.850 lb CO2/lb, almost as high as PC=1.048.
Preferred substitution order to reduce GWP: SC (0.264) > FA (0.328) >> SF (0.850).

FEASIBILITY CONSTRAINTS — ALL must be satisfied
================================================
(a) Raw ingredient bounds (kg/yd3):
{raw_bounds}

(b) Derived ratio bounds (computed from mix — must stay within):
{der_bounds}
    w/b   = WATER / (PC+FA+SC+SF)
    b/a   = (PC+FA+SC+SF) / (FAGG+CAGG)
    SCM%  = (FA+SC+SF) / (PC+FA+SC+SF)
    CAGG% = CAGG / (FAGG+CAGG)
    FAGG% = FAGG / (FAGG+CAGG)
    PC%   = PC / (PC+FA+SC+SF)
    FA%   = FA / (PC+FA+SC+SF)
    SC%   = SC / (PC+FA+SC+SF)

(c) Strength floor: predicted 28-day >= {strength_min} MPa
    (Mixes below this are INFEASIBLE — never propose them)

DESIGN RULES (enforced — check each one before proposing)
==========================================================
Rule 1 — Strength check:
  IF your proposed 28-day is predicted to be < {strength_min} MPa
  THEN increase PC or reduce w/b.  If w/b < 0.30, also raise WR or WR_HR.

Rule 2 — GWP check (apply when strength is adequate):
  Substitute PC with SC first (GWP=0.264), FA second (GWP=0.328).
  NEVER increase SF to compensate — SF GWP=0.850 is almost as bad as PC.
  Keep SCM% within dataset bounds.

Rule 3 — Workability check:
  IF w/b < 0.30 AND (WR + WR_HR) < 1.0 THEN increase WR or WR_HR.
  Do NOT add water to fix workability — it destroys strength.

Rule 4 — Progress check (most important):
  IF your proposal is very similar to your last proposal (same PC/SC/FA)
  THEN you MUST make a larger change — shift one ingredient by at least 20 kg.

REFERENCE EXAMPLES (from high-quality optimisation run — study these)
======================================================================
{icl_block}

OUTPUT FORMAT — STRICTLY REQUIRED
===================================
Return ONLY a valid JSON object. No markdown. No text outside the JSON.

{{
  "reasoning": "<Chain-of-Thought: rules applied and why, max 140 words>",
  "mix": {{
    "PC": <number>, "FA": <number>, "SC": <number>, "SF": <number>,
    "FAGG": <number>, "CAGG": <number>, "WATER": <number>,
    "AEA": <number>, "WR_HR": <number>, "WR": <number>, "ACC": <number>
  }}
}}

Before finalising, mentally verify all derived ratios are within bounds.
Do NOT include strength or GWP predictions — computed externally.
"""

FIRST_TURN = """\
Start the optimisation. Propose an initial mix that is:
  - Feasible (28-day >= {strength_min} MPa, all bounds satisfied)
  - A strong start on the Pareto front: high strength AND low GWP

Use the reference examples as inspiration. Output ONLY the JSON object.\
"""

FEEDBACK_TEMPLATE = """\
=== ITERATION {it} / {max_it} ===

Last proposed mix:
{mix_json}

CatBoost evaluation:
  Predicted  7-day: {p7:.2f} MPa
  Predicted 28-day: {p28:.2f} MPa   {str_flag}
  Predicted 56-day: {p56:.2f} MPa
  Total GWP       : {gwp:.2f} kg CO2-eq/yd3

GWP breakdown (top contributors):
{gwp_break}

Derived ratio check:
{ratio_check}

Feasibility: {feas_str}

--- TOPSIS TARGET (EQUAL WEIGHTS 50% strength / 50% GWP) ---
  Target composition: PC={t_PC:.0f}  SC={t_SC:.0f}  FA={t_FA:.0f}  SF={t_SF:.0f}
                      WATER={t_WATER:.0f}  WR_HR={t_WR_HR:.1f}  WR={t_WR:.1f}
  Target performance: 28d={t_str:.2f} MPa  |  GWP={t_gwp:.2f} kg CO2/yd3
  Your vs target:     28d gap={str_dir}{str_gap:.1f} MPa  |  GWP gap={gwp_dir}{gwp_gap:.1f} kg
  d_topsis = {d_top:.4f}  (0=perfect match, lower=better)

--- CONVERGENCE METRICS ---
  d_pareto   = {d_par:.4f}   (distance to nearest NSGA-II Pareto point)
  hv_contrib = {hv:.6f}   (hypervolume contribution; higher is better)
  dom_count  = {dom}          (# NSGA-II solutions this mix dominates)
  improvement vs prev = {impr}

Best d_topsis so far: {best_d:.4f}  (iteration {best_it})

{osc_warning}Directional feedback:
{feedback}

Propose the NEXT mix. Output ONLY the JSON object.\
"""


def build_sys_prompt(raw_b: dict, der_b: dict, topsis_pt: dict,
                     icl_block: str, strength_min: float) -> str:
    raw_lines = [f"  {v:<8} [{b['min']:8.2f}, {b['max']:8.2f}]"
                 for v, b in raw_b.items()]
    der_lines = [f"  {v:<8} [{b['min']:8.5f}, {b['max']:8.5f}]"
                 for v, b in der_b.items()]
    return SYSTEM_PROMPT_TEMPLATE.format(
        raw_bounds="\n".join(raw_lines),
        der_bounds="\n".join(der_lines),
        strength_min=strength_min,
        icl_block=icl_block,
    )


def build_feedback(it: int, max_it: int, mix: dict, preds: dict,
                   g: float, feas: dict, metrics: dict,
                   topsis_pt: dict, der_b: dict,
                   llm_traj: list) -> str:

    # Flags
    str_flag = "OK [feasible]" if preds["28day"] >= STRENGTH_MIN \
        else f"INFEASIBLE — {preds['28day'] - STRENGTH_MIN:+.2f} MPa vs floor"
    feas_str = "FEASIBLE" if feas["feasible"] else \
        "INFEASIBLE (" + ", ".join(
            list(feas["raw_violations"]) + list(feas["derived_violations"])
            + (["strength"] if feas["strength_violation"] else [])
        ) + ")"

    # GWP breakdown
    contribs = sorted(
        [(k, mix.get(k,0)*GWP.get(k,0)) for k in RAW_VARS],
        key=lambda x: -x[1]
    )
    gwp_break = "\n".join(
        f"  {k:<6} {mix.get(k,0):6.1f} kg × {GWP.get(k,0):.4f} = {c:6.2f} kg CO2"
        for k, c in contribs if c > 0.01
    )

    # Derived ratio check
    dv = derived_vals(mix)
    ratio_lines = []
    for v, b in der_b.items():
        val     = dv[v]
        tol     = (b["max"] - b["min"]) * 0.01 + 1e-6
        ok      = b["min"] - tol <= val <= b["max"] + tol
        b_min   = b["min"]
        b_max   = b["max"]
        status  = "OK" if ok else f"VIOLATION [{b_min:.4f},{b_max:.4f}]"
        ratio_lines.append(f"  {v:<8}= {val:.4f}  {status}")

    # Best d_topsis so far
    feasible_traj = [r for r in llm_traj if r.get("feasible")]
    if feasible_traj:
        best_rec = min(feasible_traj, key=lambda r: r.get("d_topsis", 9999))
        best_d   = best_rec["d_topsis"]
        best_it  = best_rec["iteration"]
    else:
        best_d, best_it = float("nan"), 0

    # Directional hint from TOPSIS
    str_gap  = abs(topsis_pt["pred_28day"] - preds["28day"])
    gwp_gap  = abs(topsis_pt["gwp"]        - g)
    str_dir  = "UP"   if preds["28day"] < topsis_pt["pred_28day"] else "DOWN"
    gwp_dir  = "DOWN" if g > topsis_pt["gwp"] else "UP"

    # Targeted feedback
    fb = []
    if not feas["feasible"]:
        if feas["strength_violation"]:
            fb.append(f"  -> Strength {preds['28day']:.1f} MPa below floor {STRENGTH_MIN}. "
                      "Rule 1: increase PC or reduce WATER.")
        for v, info in feas["derived_violations"].items():
            fb.append(f"  -> {v}={info['val']:.4f} out of "
                      f"[{info['min']:.4f},{info['max']:.4f}]. Adjust proportions.")
    else:
        d = metrics["d_topsis"]
        if d > 0.3:
            fb.append(f"  -> d_topsis={d:.4f} is large. Push both objectives harder.")
        elif d > 0.15:
            fb.append(f"  -> d_topsis={d:.4f}. Getting closer — keep pushing "
                      f"strength {str_dir} and GWP {gwp_dir}.")
        else:
            fb.append(f"  -> d_topsis={d:.4f}. Very close to optimal! "
                      "Fine-tune while preserving both objectives.")
        if str_dir == "UP":
            fb.append(f"  -> Strength needs +{str_gap:.1f} MPa. "
                      "Try raising PC slightly or reducing w/b.")
        if gwp_dir == "DOWN":
            fb.append(f"  -> GWP needs -{gwp_gap:.1f} kg. "
                      "Substitute more PC with SC (GWP=0.264).")

    impr_str = f"{metrics['improvement']:+.4f}" \
        if not np.isnan(metrics["improvement"]) else "n/a (first iter)"

    # Anti-oscillation warning
    osc_warning = ""
    if detect_oscillation(llm_traj):
        osc_warning = (
            "*** OSCILLATION DETECTED ***\n"
            "Your last 3 proposals are nearly identical (PC/SC/FA within 5 kg).\n"
            "You MUST change at least 2 ingredients by MORE THAN 20 kg each.\n"
            "Small tweaks cannot escape this local optimum.\n\n"
        )

    return FEEDBACK_TEMPLATE.format(
        it=it, max_it=max_it,
        mix_json=json.dumps(mix, indent=4),
        p7=preds["7day"], p28=preds["28day"], p56=preds["56day"],
        str_flag=str_flag, gwp=g,
        gwp_break=gwp_break,
        ratio_check="\n".join(ratio_lines),
        feas_str=feas_str,
        # TOPSIS anchor fields
        t_PC=topsis_pt.get("PC", 0),
        t_SC=topsis_pt.get("SC", 0),
        t_FA=topsis_pt.get("FA", 0),
        t_SF=topsis_pt.get("SF", 0),
        t_WATER=topsis_pt.get("WATER", 0),
        t_WR_HR=topsis_pt.get("WR_HR", 0),
        t_WR=topsis_pt.get("WR", 0),
        t_str=topsis_pt.get("pred_28day", 0),
        t_gwp=topsis_pt.get("gwp", 0),
        str_dir="+" if str_dir == "UP" else "-",
        str_gap=str_gap,
        gwp_dir="+" if gwp_dir == "UP" else "-",
        gwp_gap=gwp_gap,
        d_top=metrics["d_topsis"],
        d_par=metrics["d_pareto"],
        hv=metrics["hv_contrib"],
        dom=metrics["dom_count"],
        impr=impr_str,
        best_d=best_d, best_it=best_it,
        osc_warning=osc_warning,
        feedback="\n".join(fb),
    )


# ─────────────────────────────────────────────────────────────
# 10.  PARSE & CLIP
# ─────────────────────────────────────────────────────────────

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


def clip_mix(mix: dict, raw_b: dict) -> dict:
    clean = {}
    for v in RAW_VARS:
        b   = raw_b[v]
        val = float(mix.get(v, b["min"]))
        clp = float(np.clip(val, b["min"], b["max"]))
        if abs(clp - val) > 0.5:
            print(f"    [clip] {v}: {val:.1f} -> {clp:.1f}")
        clean[v] = round(clp, 2)
    return clean


# ─────────────────────────────────────────────────────────────
# 10b.  OPRO-STYLE STAGNATION DETECTION & RESTART
# ─────────────────────────────────────────────────────────────
# Reference: Yang et al. (2023) OPRO — "Large Language Models as Optimizers"
#   "Empirically, diverse LLMs make monotonic progress until stagnation;
#    local optima possible, especially under context bottlenecks."
#   Remedy: inject diverse historical examples + raise temperature.

def detect_stagnation(trajectory: list, window: int = STAG_WINDOW,
                      threshold: float = STAG_THRESHOLD) -> bool:
    """
    Return True if the LLM is genuinely stuck — TWO conditions both required:
      (1) The LLM has already found at least one good solution (d_topsis < STAG_MIN_D_EVER).
          This prevents panic-restarts when the LLM is still exploring in the early phase.
      (2) The last `window` feasible iterations show no meaningful improvement
          (all per-step improvements < threshold).
    Only feasible iterations count to avoid false triggers from infeasible bounces.
    """
    feasible = [r for r in trajectory if r.get("feasible") and
                not np.isnan(r.get("d_topsis", float("nan")))]
    if len(feasible) < window:
        return False

    # Condition 1: LLM must have already found a reasonably good solution
    min_d_ever = min(r["d_topsis"] for r in feasible)
    if min_d_ever >= STAG_MIN_D_EVER:
        return False   # still exploring — too early to restart

    # Condition 2: no meaningful improvement in recent window
    recent = feasible[-window:]
    improvements = [
        recent[i - 1]["d_topsis"] - recent[i]["d_topsis"]
        for i in range(1, len(recent))
    ]
    return all(imp < threshold for imp in improvements)


def detect_oscillation(trajectory: list,
                       window: int = ANTI_OSC_WINDOW) -> bool:
    """
    Return True if the last `window` proposals are nearly identical
    (PC, SC, FA all within 5 kg of each other), indicating the LLM is
    stuck in an infeasible<->feasible loop without real progress.
    """
    recent = trajectory[-window:] if len(trajectory) >= window else []
    if len(recent) < window:
        return False
    for var in ["PC", "SC", "FA"]:
        vals = [r.get(var, 0) for r in recent]
        if max(vals) - min(vals) > 5:
            return False   # at least one variable is changing meaningfully
    return True


def compute_search_center(trajectory: list, n: int = 10) -> dict:
    """
    Compute the centroid of the last n feasible solutions.
    Used to tell LLM 'you have been searching around this region'.
    """
    feasible = [r for r in trajectory if r.get("feasible")][-n:]
    if not feasible:
        return {}
    center = {}
    for v in RAW_VARS:
        vals = [r[v] for r in feasible if v in r]
        if vals:
            center[v] = round(float(np.mean(vals)), 1)
    return center


def build_restart_message(trajectory: list, topsis_pt: dict,
                          restart_num: int) -> str:
    """
    Build a forced-restart user message following OPRO principles:
      1. Explicitly acknowledge stagnation
      2. Show the top-K best solutions found so far (memory)
      3. Show the current search center (region to ESCAPE from)
      4. Instruct LLM to propose a structurally DIFFERENT mix
    """
    feasible = [r for r in trajectory if r.get("feasible")]

    # Top-K by d_topsis
    top_k = sorted(feasible, key=lambda r: r.get("d_topsis", 9999))[:RESTART_TOP_K]

    top_k_str = ""
    for i, r in enumerate(top_k, 1):
        line1 = (
            f"  #{i} (iter {r['iteration']:2d}): "
            f"PC={r['PC']:.0f}  SC={r['SC']:.0f}  FA={r['FA']:.0f}  "
            f"SF={r['SF']:.0f}  WATER={r['WATER']:.0f}  "
            f"WR_HR={r.get('WR_HR',0):.1f}  WR={r.get('WR',0):.1f}"
        )
        line2 = (
            f"          -> 28d={r['pred_28day']:.2f} MPa  "
            f"GWP={r['gwp']:.2f} kg  d_topsis={r.get('d_topsis',float('nan')):.4f}"
        )
        top_k_str += line1 + "\n" + line2 + "\n"

    # Search center
    center = compute_search_center(trajectory)
    center_str = "  ".join(f"{k}={v}" for k, v in center.items()
                           if k in ["PC", "SC", "FA", "SF", "WATER"])

    # TOPSIS target
    t_str = (f"28d={topsis_pt['pred_28day']:.2f} MPa, "
             f"GWP={topsis_pt['gwp']:.2f} kg CO2/yd3")

    return f"""
=== RESTART #{restart_num} — STAGNATION DETECTED ===

The optimisation has stagnated. Your recent proposals have been making
very small improvements (d_topsis improvement < {STAG_THRESHOLD} per step).

CURRENT SEARCH CENTER (average of last 10 feasible solutions):
  {center_str}
You have been exploring this region exhaustively. It is a LOCAL OPTIMUM
for your current strategy. DO NOT propose mixes similar to this region.

TOP-{RESTART_TOP_K} BEST SOLUTIONS FOUND SO FAR (for reference only — do not copy):
{top_k_str}
TOPSIS TARGET: {t_str}

YOUR TASK: Propose a mix that is STRUCTURALLY DIFFERENT from the search center above.
Specifically, try ONE of these unexplored strategies:
  Strategy A — Low binder, high aggregate:  reduce total binder by 100+ kg,
               increase FAGG/CAGG, rely on low w/b for strength.
  Strategy B — High FA, low SC:  use FA as primary SCM instead of SC
               (FA is cheaper but needs longer curing — check if 28d still feasible).
  Strategy C — Higher PC, zero FA/SF:  pure PC+SC binary system, no FA or SF at all.
  Strategy D — Completely different w/b:  try w/b around 0.35-0.40 with higher binder
               total to achieve strength through paste volume rather than low w/b.

Pick whichever strategy is most likely to find new feasible solutions closer to the
TOPSIS target. Be BOLD — small adjustments will not escape the local optimum.

Output ONLY the JSON object.
"""


# ─────────────────────────────────────────────────────────────
# 11.  MAIN LLM LOOP  (Phase 2)
# ─────────────────────────────────────────────────────────────

def run_llm(raw_b: dict, der_b: dict, meta: dict,
            nsga_records: list, topsis_pt: dict,
            icl_block: str, max_iters: int = MAX_ITERATIONS) -> list:

    nsga_pareto = extract_pareto(nsga_records)

    print(f"\n{'='*64}")
    print("  LLM Iterative Fuzzy Optimiser — Phase 2  (OPRO restart enabled)")
    print(f"{'='*64}")
    print(f"  Model       : {GEMINI_MODEL}")
    print(f"  Objectives  : maximise 28d strength  |  minimise GWP")
    print(f"  Strength min: {STRENGTH_MIN} MPa  (feasibility floor)")
    print(f"  Max iters   : {max_iters}")
    print(f"  TOPSIS ref  : 28d={topsis_pt['pred_28day']:.2f} MPa  "
          f"GWP={topsis_pt['gwp']:.2f} kg/yd3")
    print(f"  Stagnation  : window={STAG_WINDOW} iters, threshold={STAG_THRESHOLD}")
    print(f"  Max restarts: {MAX_RESTARTS}  (temp {NORMAL_TEMP:.1f} -> {RESTART_TEMP:.1f} on restart)")
    print(f"{'='*64}")

    sys_prompt = build_sys_prompt(raw_b, der_b, topsis_pt, icl_block, STRENGTH_MIN)

    genai.configure(api_key=GEMINI_API_KEY)

    def _make_model(temp: float):
        """Create a fresh Gemini model with the given temperature."""
        return genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=sys_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temp, max_output_tokens=1024),
        )

    cur_temp  = NORMAL_TEMP
    model     = _make_model(cur_temp)
    chat      = model.start_chat(history=[])

    print(f"\n  {'Iter':>4}  {'28d MPa':>8}  {'GWP':>8}  "
          f"{'d_topsis':>9}  {'d_pareto':>9}  "
          f"{'hv_cont':>9}  {'dom':>4}  {'Feas':>4}  {'Mode'}")
    print("  " + "-"*84)

    trajectory    = []
    parse_fails   = 0
    restart_count = 0
    cur_mix       = None
    cur_preds     = None
    cur_gwp       = None
    cur_feas      = None

    for it in range(1, max_iters + 1):

        # ── OPRO stagnation check & restart ──────────────────
        mode = "exploit"
        if (it > 1
                and restart_count < MAX_RESTARTS
                and detect_stagnation(trajectory)):

            restart_count += 1
            mode = f"RESTART #{restart_count}"
            print(f"\n  [!] Stagnation detected at iter {it}. "
                  f"Triggering restart #{restart_count} "
                  f"(temp {cur_temp:.1f} -> {RESTART_TEMP:.1f}) ...")

            # Switch to high-temperature model + fresh conversation
            # (fresh chat = clear context bias; history retained in trajectory)
            cur_temp = RESTART_TEMP
            model    = _make_model(cur_temp)
            chat     = model.start_chat(history=[])

            user_msg = build_restart_message(trajectory, topsis_pt, restart_count)

        # ── Normal feedback ───────────────────────────────────
        elif it == 1:
            user_msg = FIRST_TURN.format(strength_min=STRENGTH_MIN)
        else:
            # Return to normal temperature if we were in restart mode
            if cur_temp != NORMAL_TEMP:
                cur_temp = NORMAL_TEMP
                model    = _make_model(cur_temp)
                chat     = model.start_chat(history=[])
                # Re-seed with current best so new conversation isn't cold
                best_feas = [r for r in trajectory if r.get("feasible")]
                if best_feas:
                    seed_rec = min(best_feas, key=lambda r: r.get("d_topsis", 9999))
                    seed_msg = (
                        f"Resume optimisation. Current best feasible solution:\n"
                        f"  PC={seed_rec['PC']:.0f}  SC={seed_rec['SC']:.0f}  "
                        f"FA={seed_rec['FA']:.0f}  SF={seed_rec['SF']:.0f}  "
                        f"WATER={seed_rec['WATER']:.0f}  "
                        f"WR_HR={seed_rec.get('WR_HR',0):.1f}  "
                        f"WR={seed_rec.get('WR',0):.1f}\n"
                        f"  28d={seed_rec['pred_28day']:.2f} MPa  "
                        f"GWP={seed_rec['gwp']:.2f} kg  "
                        f"d_topsis={seed_rec.get('d_topsis',float('nan')):.4f}\n"
                        f"Continue improving from here. Output ONLY the JSON object."
                    )
                    try:
                        chat.send_message(seed_msg)
                    except Exception:
                        pass

            m_obj = compute_all_metrics(
                {**cur_mix,
                 "pred_28day": cur_preds["28day"],
                 "gwp": cur_gwp,
                 "feasible": cur_feas["feasible"]},
                topsis_pt, nsga_pareto,
                [{**r, "pred_28day": r["pred_28day"], "gwp": r["gwp"]}
                 for r in trajectory if r["feasible"]],
            )
            user_msg = build_feedback(
                it, max_iters, cur_mix, cur_preds, cur_gwp,
                cur_feas, m_obj, topsis_pt, der_b, trajectory,
            )

        # ── Call Gemini ───────────────────────────────────────
        try:
            resp     = chat.send_message(user_msg)
            raw_text = resp.text
        except Exception as exc:
            print(f"  [iter {it:02d}] API error: {exc} — waiting 15 s ...")
            time.sleep(15)
            continue

        # ── Parse ─────────────────────────────────────────────
        parsed = parse_json(raw_text)
        if parsed is None or "mix" not in parsed:
            parse_fails += 1
            print(f"  [iter {it:02d}] JSON parse failure ({parse_fails}) — correcting ...")
            try:
                chat.send_message(
                    "Your response could not be parsed as JSON. "
                    "Output ONLY the JSON object with keys 'reasoning' and 'mix'."
                )
            except Exception:
                pass
            continue

        reasoning = parsed.get("reasoning", "")
        mix       = clip_mix(parsed["mix"], raw_b)
        preds     = predict(meta, mix)
        g         = gwp(mix)
        dv        = derived_vals(mix)
        feas      = check_feasibility(mix, raw_b, der_b, preds["28day"])
        tb        = mix["PC"] + mix["FA"] + mix["SC"] + mix["SF"]

        # ── Compute metrics ───────────────────────────────────
        rec_for_metrics = {
            "pred_28day": preds["28day"], "gwp": g, "feasible": feas["feasible"]
        }
        feasible_so_far = [
            {"pred_28day": r["pred_28day"], "gwp": r["gwp"]}
            for r in trajectory if r["feasible"]
        ]
        metrics = compute_all_metrics(
            rec_for_metrics, topsis_pt, nsga_pareto, feasible_so_far
        )

        record = {
            "iteration":    it,
            "reasoning":    reasoning,
            "mode":         mode,
            **mix,
            **{k: round(v, 5) for k, v in dv.items()},
            "total_binder": round(tb, 2),
            "pred_7day":    preds["7day"],
            "pred_28day":   preds["28day"],
            "pred_56day":   preds["56day"],
            "gwp":          g,
            "feasible":     feas["feasible"],
            **metrics,
        }
        trajectory.append(record)

        cur_mix, cur_preds, cur_gwp, cur_feas = mix, preds, g, feas

        # ── Print row ─────────────────────────────────────────
        fstr   = "Y" if feas["feasible"] else "N"
        impr_s = f"{metrics['improvement']:+.4f}" \
            if not np.isnan(metrics["improvement"]) else "  n/a"
        print(
            f"  {it:4d}  {preds['28day']:8.2f}  {g:8.2f}  "
            f"{metrics['d_topsis']:9.4f}  {metrics['d_pareto']:9.4f}  "
            f"{metrics['hv_contrib']:9.5f}  {metrics['dom_count']:4d}  "
            f"{fstr:4s}  {mode}"
        )

        time.sleep(0.4)

    print(f"\n  Total restarts triggered: {restart_count}")
    return trajectory


# ─────────────────────────────────────────────────────────────
# 12.  REPORT & SAVE
# ─────────────────────────────────────────────────────────────

def build_report(llm_traj: list, nsga_records: list,
                 topsis_pt: dict) -> str:
    nsga_pareto = extract_pareto(nsga_records)
    llm_pareto  = extract_pareto(llm_traj)
    feasible    = [r for r in llm_traj if r["feasible"]]

    best_str = max(feasible, key=lambda r: r["pred_28day"]) if feasible else None
    best_gwp = min(feasible, key=lambda r: r["gwp"])        if feasible else None
    best_dtop = min(feasible, key=lambda r: r.get("d_topsis", 9999)) if feasible else None

    SEP = "="*66
    sep = "-"*66
    skip = {"reasoning"}

    restart_iters = [r["iteration"] for r in llm_traj if r.get("mode","").startswith("RESTART")]
    lines = [
        SEP,
        "  LLM Fuzzy Optimiser — Full Results Report  (OPRO restart enabled)",
        f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        SEP,
        f"  LLM iterations    : {len(llm_traj)}",
        f"  Parse failures    : {sum(1 for r in llm_traj if r.get('d_topsis') is None)}",
        f"  Feasible / total  : {len(feasible)} / {len(llm_traj)}",
        f"  LLM Pareto size   : {len(llm_pareto)}",
        f"  NSGA-II Pareto    : {len(nsga_pareto)}",
        f"  Restarts triggered: {len(restart_iters)}  (at iters: {restart_iters})",
        sep,
        "  TOPSIS REFERENCE POINT",
        sep,
        f"    28-day strength : {topsis_pt['pred_28day']:.2f} MPa",
        f"    GWP             : {topsis_pt['gwp']:.2f} kg CO2-eq/yd3",
        f"    TOPSIS score    : {topsis_pt.get('topsis_score','n/a')}",
        sep,
        "  AGGREGATE COMPARISON METRICS  (LLM Pareto vs NSGA-II Pareto)",
        sep,
        f"    IGD  (LLM -> NSGA-II) : {igd(llm_pareto, nsga_pareto)}",
        f"    GD+  (LLM -> NSGA-II) : {gd_plus(llm_pareto, nsga_pareto)}",
        f"    HV ratio (LLM/NSGA)   : {hv_ratio(llm_pareto, nsga_pareto)}",
        sep,
    ]

    if best_dtop:
        lines += ["  BEST d_topsis SOLUTION", sep]
        for k, v in best_dtop.items():
            if k not in skip:
                lines.append(f"    {k:<20}: {v}")
    if best_str:
        lines += [sep, "  BEST STRENGTH SOLUTION", sep]
        for k, v in best_str.items():
            if k not in skip:
                lines.append(f"    {k:<20}: {v}")
    if best_gwp:
        lines += [sep, "  BEST GWP SOLUTION", sep]
        for k, v in best_gwp.items():
            if k not in skip:
                lines.append(f"    {k:<20}: {v}")

    lines += [
        sep,
        "  FULL LLM TRAJECTORY",
        sep,
        f"  {'Iter':>4}  {'28d':>7}  {'GWP':>8}  "
        f"{'d_topsis':>9}  {'d_pareto':>9}  {'hv':>9}  {'dom':>4}  "
        f"{'improve':>8}  F  P",
        sep,
    ]
    for r in llm_traj:
        on_p = any(p["iteration"] == r["iteration"] for p in llm_pareto)
        impr = f"{r['improvement']:+.4f}" \
            if not np.isnan(r.get("improvement", float("nan"))) else "    n/a"
        lines.append(
            f"  {r['iteration']:4d}  {r['pred_28day']:7.2f}  {r['gwp']:8.2f}  "
            f"{r.get('d_topsis',float('nan')):9.4f}  "
            f"{r.get('d_pareto',float('nan')):9.4f}  "
            f"{r.get('hv_contrib',float('nan')):9.5f}  "
            f"{r.get('dom_count',0):4d}  "
            f"{impr:>8}  "
            f"{'Y' if r['feasible'] else 'N'}  "
            f"{'Y' if on_p else '-'}"
        )

    lines += [sep, "  LLM PARETO FRONT DETAIL", sep]
    for r in sorted(llm_pareto, key=lambda x: -x["pred_28day"]):
        lines.append(
            f"  Iter {r['iteration']:2d} | 28d={r['pred_28day']:.2f} MPa | "
            f"GWP={r['gwp']:.2f} | "
            f"PC={r['PC']:.0f} SC={r['SC']:.0f} FA={r['FA']:.0f} SF={r['SF']:.0f} | "
            f"d_topsis={r.get('d_topsis',float('nan')):.4f}"
        )

    lines += [sep, "  CHAIN-OF-THOUGHT LOG", sep]
    for r in llm_traj:
        lines.append(f"  Iter {r['iteration']:2d}: {r.get('reasoning','')}")
    lines.append(SEP)
    return "\n".join(lines)


def save_results(llm_traj: list, nsga_records: list,
                 topsis_pt: dict) -> None:
    # LLM trajectory CSV
    save_cols = [c for c in llm_traj[0].keys() if c != "reasoning"]
    pd.DataFrame(llm_traj)[save_cols].to_csv(LLM_CSV, index=False)
    print(f"\n  LLM trajectory  -> '{LLM_CSV}'")

    # NSGA-II Pareto CSV
    nsga_pareto = extract_pareto(nsga_records)
    if nsga_pareto:
        pd.DataFrame(nsga_pareto).to_csv(NSGA2_CSV, index=False)
        print(f"  NSGA-II Pareto  -> '{NSGA2_CSV}'")

    # Report
    report = build_report(llm_traj, nsga_records, topsis_pt)
    print("\n" + report)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report          -> '{REPORT_TXT}'")


# ─────────────────────────────────────────────────────────────
# 13.  ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    global STRENGTH_MIN, MAX_RESTARTS, STAG_WINDOW, STAG_MIN_D_EVER   # declared first

    parser = argparse.ArgumentParser(
        description="LLM Fuzzy Optimiser -- Low-Carbon Concrete (bi-objective)"
    )
    parser.add_argument("--strength",   type=float, default=STRENGTH_MIN,
                        help=f"28-day strength floor (default {STRENGTH_MIN} MPa)")
    parser.add_argument("--iters",      type=int,   default=MAX_ITERATIONS,
                        help=f"LLM iterations (default {MAX_ITERATIONS})")
    parser.add_argument("--nsga2-gen",  type=int,   default=NSGA2_GENS,
                        help=f"NSGA-II generations (default {NSGA2_GENS})")
    parser.add_argument("--nsga2-pop",  type=int,   default=NSGA2_POP,
                        help=f"NSGA-II pop size (default {NSGA2_POP})")
    parser.add_argument("--w-str",      type=float, default=TOPSIS_W_STR,
                        help=f"TOPSIS weight for strength (default {TOPSIS_W_STR})")
    parser.add_argument("--w-gwp",      type=float, default=TOPSIS_W_GWP,
                        help=f"TOPSIS weight for GWP (default {TOPSIS_W_GWP})")
    parser.add_argument("--skip-nsga2", action="store_true",
                        help="Skip NSGA-II phase (use saved nsga2_pareto_front.csv)")
    parser.add_argument("--max-restarts", type=int, default=MAX_RESTARTS,
                        help=f"Max OPRO restarts (default {MAX_RESTARTS})")
    parser.add_argument("--stag-window",  type=int, default=STAG_WINDOW,
                        help=f"Stagnation detection window (default {STAG_WINDOW} iters)")
    parser.add_argument("--stag-min-d",   type=float, default=STAG_MIN_D_EVER,
                        help=f"Min d_topsis ever before restart fires (default {STAG_MIN_D_EVER})")
    args = parser.parse_args()

    STRENGTH_MIN   = args.strength
    MAX_RESTARTS   = args.max_restarts
    STAG_WINDOW    = args.stag_window
    STAG_MIN_D_EVER = args.stag_min_d
    # Load data & surrogate
    print("\n[1/4] Loading data and CatBoost surrogate ...")
    df   = load_df(DATA_PATH)
    raw_b, der_b = get_bounds(df)
    meta = load_surrogate(MODEL_PKL)
    print(f"      Dataset: {len(df)} rows | Raw vars: {len(RAW_VARS)}")

    # Phase 1: NSGA-II
    if args.skip_nsga2 and pd.io.common.file_exists(NSGA2_CSV):
        print(f"\n[2/4] Loading existing NSGA-II results from '{NSGA2_CSV}' ...")
        nsga_records = pd.read_csv(NSGA2_CSV).to_dict("records")
        # Re-mark feasibility
        for r in nsga_records:
            mix  = {v: r[v] for v in RAW_VARS}
            pr   = predict(meta, mix)
            feas = check_feasibility(mix, raw_b, der_b, pr["pred_28day"]
                                     if "pred_28day" in pr else r["pred_28day"])
            r["feasible"] = feas["feasible"]
        print(f"      Loaded {len(nsga_records)} solutions.")
    else:
        print("\n[2/4] Running NSGA-II ...")
        nsga_records = run_nsga2(raw_b, der_b, meta,
                                 n_gen=args.nsga2_gen, pop=args.nsga2_pop)

    # TOPSIS on NSGA-II Pareto
    nsga_pareto = extract_pareto(nsga_records)
    if not nsga_pareto:
        raise RuntimeError("NSGA-II produced no feasible Pareto solutions.")
    topsis_pt = topsis(nsga_pareto, w_str=args.w_str, w_gwp=args.w_gwp)
    print(f"\n[3/4] TOPSIS optimum: "
          f"28d={topsis_pt['pred_28day']:.2f} MPa | "
          f"GWP={topsis_pt['gwp']:.2f} kg/yd3 | "
          f"score={topsis_pt.get('topsis_score','')}")

    # Build ICL examples from NSGA-II
    icl_block = build_icl_examples(nsga_records, topsis_pt)

    # Phase 2: LLM optimiser
    print("\n[4/4] Starting LLM optimiser ...")
    llm_traj = run_llm(
        raw_b, der_b, meta, nsga_records, topsis_pt, icl_block,
        max_iters=args.iters,
    )

    # Save & report
    save_results(llm_traj, nsga_records, topsis_pt)


if __name__ == "__main__":
    main()

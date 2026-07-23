"""
optimizer_mo.py
===============
Multi-objective (and single-objective) LLM-based optimizer for low-carbon
concrete mix design.

Objectives
----------
  Single-objective : minimize GWP  (kg CO2-eq/m³)
                     subject to 28d strength >= strength_min
  Multi-objective  : minimize GWP  +  maximize 56d strength simultaneously
                     (no hard strength floor — trade-off is explored)

Differences from optimizer_core.py
-----------------------------------
  - Maintains a Pareto archive of non-dominated solutions
  - LLM feedback shows the current Pareto front and a scalarised target
    that shifts each iteration to cover different Pareto regions
  - Evaluation metrics: Hypervolume Indicator (HVI) and Pareto Coverage Rate
  - GA baseline uses NSGA-II (pymoo) for multi-objective mode

Usage
-----
  python optimizer_mo.py                          # multi-obj, all defaults
  python optimizer_mo.py --mode single            # single-objective
  python optimizer_mo.py --mode multi --iters 40
  python optimizer_mo.py --skip-ga

Dependencies
------------
  pip install google-generativeai catboost joblib pandas numpy pymoo python-dotenv
"""

import argparse
import json
import os
import re
import time
import warnings
import joblib
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# Re-use shared utilities from optimizer_core
from optimizer_core import (
    LB_YD3_TO_KG_M3, GWP_FACTORS, RAW_VARS, DERIVED_VARS,
    load_df, get_bounds, load_surrogate, predict, compute_gwp,
    get_derived, check_feasibility, select_few_shot,
    KNOWLEDGE_TABLE, SITUATION_RULES,
)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

@dataclass
class MOConfig:
    """Parameters for multi/single objective LLM optimization."""
    name: str = "mo_baseline"
    mode: str = "multi"              # "single" | "multi"

    # Objectives (multi mode)
    # obj1: minimize GWP (kg CO2/m³)
    # obj2: maximize 56d strength (MPa)

    # Single-objective constraint
    strength_min: float = 55.0       # only used in single mode

    # LLM settings
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.9

    # Optimization loop
    max_iters: int = 40
    max_restarts: int = 2
    restart_temp: float = 1.3

    # Stagnation
    stag_window: int = 5
    stag_threshold: float = 0.005    # relative HV improvement

    # Anti-oscillation
    anti_osc_window: int = 3
    anti_osc_tol: float = 3.0        # kg/m³

    # GA settings
    ga_gens: int = 200
    ga_pop: int = 100

    # Prompt components
    use_knowledge_table: bool = True
    use_situation_rules: bool = True
    use_few_shot: bool = True

    # Pareto scalarization sweep (lambda cycles through these weights)
    # Each value is (w_gwp, w_strength) — must sum to 1
    lambda_schedule: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.7, 0.3), (0.5, 0.5), (0.3, 0.7),
        (0.8, 0.2), (0.2, 0.8), (0.6, 0.4),
        (0.4, 0.6), (0.9, 0.1), (0.1, 0.9),
        (0.5, 0.5),
    ])

    # Hypervolume reference point (worst case, slightly beyond data bounds)
    hv_ref_gwp: float = 600.0        # kg CO2/m³  (worse than worst mix)
    hv_ref_str: float = 0.0          # MPa        (worse than worst strength)

    # File paths
    data_path: str = "data/Super_Cleaned_Concrete_Data_model_train.csv"
    model_pkl: str = "concrete_catboost_optimized.pkl"
    output_prefix: str = "mo_results"


# ─────────────────────────────────────────────────────────────
# PARETO UTILITIES
# ─────────────────────────────────────────────────────────────

def dominates(a: dict, b: dict) -> bool:
    """Return True if solution a Pareto-dominates b.
    Objectives: minimize GWP, maximize strength_56d.
    a dominates b if a is no worse on all objectives and better on at least one.
    """
    a_gwp, a_str = a["gwp"], a["pred_56day"]
    b_gwp, b_str = b["gwp"], b["pred_56day"]
    # a no worse than b on both
    no_worse = (a_gwp <= b_gwp) and (a_str >= b_str)
    # a strictly better on at least one
    strictly_better = (a_gwp < b_gwp) or (a_str > b_str)
    return no_worse and strictly_better


def update_pareto(archive: list, candidate: dict) -> Tuple[list, bool]:
    """Add candidate to Pareto archive if non-dominated.
    Returns (new_archive, was_added).
    """
    # Check if candidate is dominated by any existing solution
    for sol in archive:
        if dominates(sol, candidate):
            return archive, False

    # Remove solutions dominated by the candidate
    new_archive = [sol for sol in archive if not dominates(candidate, sol)]
    new_archive.append(candidate)
    return new_archive, True


def compute_hypervolume(pareto: list, ref_gwp: float, ref_str: float) -> float:
    """2D hypervolume indicator.
    Objectives: (GWP to minimise, strength to maximise).
    Transform to minimisation: obj2 = -strength, ref_obj2 = -ref_str.
    Uses sweep-line algorithm for 2D.
    """
    if not pareto:
        return 0.0

    # Transform: both objectives to minimise
    pts = sorted([(s["gwp"], -s["pred_56day"]) for s in pareto],
                 key=lambda p: p[0])
    ref = (ref_gwp, -ref_str)

    hv = 0.0
    prev_x = ref[0]
    for x, y in pts:
        if x >= ref[0] or y >= ref[1]:
            continue
        width  = prev_x - x
        height = ref[1] - y
        if width > 0 and height > 0:
            hv += width * height
        prev_x = x

    return round(hv, 4)


def scalarize(sol: dict, w_gwp: float, w_str: float,
              gwp_range: tuple, str_range: tuple) -> float:
    """Weighted Chebyshev scalarization (lower = better)."""
    gwp_n = (sol["gwp"] - gwp_range[0]) / (gwp_range[1] - gwp_range[0] + 1e-9)
    str_n = 1 - (sol["pred_56day"] - str_range[0]) / (str_range[1] - str_range[0] + 1e-9)
    return max(w_gwp * gwp_n, w_str * str_n)


# ─────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────

MO_SYSTEM_PROMPT = """\
You are an expert concrete mix design engineer specialising in low-carbon, \
high-performance concrete.

OPTIMISATION PROBLEM ({mode})
==============================
{objective_block}

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

{few_shot_block}

OUTPUT FORMAT — STRICTLY REQUIRED
===================================
Return ONLY a valid JSON object. No markdown, no extra text.

{{
  "reasoning": "<what you changed and why, referencing both GWP and strength trade-offs, max 150 words>",
  "mix": {{
    "PC": <number>, "FA": <number>, "SC": <number>,
    "FAGG": <number>, "CAGG": <number>, "WATER": <number>,
    "AEA": <number>, "WR_HR": <number>, "WR": <number>, "ACC": <number>
  }}
}}
"""

MO_OBJECTIVE_SINGLE = """\
MODE       : SINGLE-OBJECTIVE
OBJECTIVE  : MINIMISE GWP (kg CO₂/m³)
CONSTRAINT : 28-day strength >= {strength_min} MPa (HARD LIMIT)
Strategy   : Push GWP as low as possible. Strength just needs to meet the floor."""

MO_OBJECTIVE_MULTI = """\
MODE       : MULTI-OBJECTIVE (Pareto optimisation)
OBJECTIVE 1: MINIMISE GWP (kg CO₂/m³)   — lower is better
OBJECTIVE 2: MAXIMISE 56-day strength (MPa) — higher is better
NO hard strength floor. You are exploring the Pareto front — the set of
solutions where you cannot improve one objective without worsening the other.

THIS ITERATION TARGET (scalarised weight):
  Prioritise GWP reduction by {w_gwp:.0%} and strength gain by {w_str:.0%}.
  This weight shifts across iterations to cover different Pareto regions."""

MO_FEEDBACK_TEMPLATE = """\
=== ITERATION {it} / {max_it} ===

Last proposed mix:
{mix_json}

CatBoost evaluation:
  7-day  : {p7:.2f} MPa
  28-day : {p28:.2f} MPa
  56-day : {p56:.2f} MPa

GWP breakdown:
{gwp_breakdown}
  TOTAL GWP : {gwp:.2f} kg CO₂/m³

Derived ratios:
{ratio_check}

Feasibility: {feas_str}

=== PROGRESS ===
  Previous iter {prev_iter}: GWP={prev_gwp:.2f}  56d={prev_56:.2f} MPa
  This iter     {it}       : GWP={gwp:.2f}  56d={p56:.2f} MPa
  GWP change    : {gwp_change:+.2f} kg/m³  {gwp_trend}
  Strength change: {str_change:+.2f} MPa

{pareto_block}
{infeas_warning}{osc_warning}\
=== ACTION REQUIRED ===
{feedback}
THIS ITERATION: prioritise GWP {w_gwp:.0%} / strength {w_str:.0%}.
Propose the NEXT mix. Output ONLY the JSON object.\
"""

MO_FIRST_TURN = """\
Start optimisation. Propose an initial mix satisfying ALL variable and ratio bounds.

{start_hint}

Output ONLY the JSON object.\
"""


def _format_pareto_block(pareto: list, hv: float, cfg: MOConfig) -> str:
    if not pareto:
        return "Pareto archive: empty (no feasible solutions yet)\n"

    sorted_front = sorted(pareto, key=lambda s: s["gwp"])
    lines = [
        f"Current Pareto front ({len(pareto)} solutions)  |  Hypervolume: {hv:.1f}",
        f"  {'GWP (kg/m³)':<16} {'56d (MPa)':<12} {'PC':<8} {'SC':<8} {'FA':<8}",
        "  " + "-" * 52,
    ]
    for s in sorted_front[:8]:   # show at most 8
        lines.append(
            f"  {s['gwp']:<16.1f} {s['pred_56day']:<12.1f} "
            f"{s.get('PC',0):<8.0f} {s.get('SC',0):<8.0f} {s.get('FA',0):<8.0f}"
        )
    if len(sorted_front) > 8:
        lines.append(f"  ... ({len(sorted_front)-8} more solutions)")

    best_gwp = min(s["gwp"] for s in pareto)
    best_str = max(s["pred_56day"] for s in pareto)
    lines.append(f"\n  Best GWP achieved : {best_gwp:.2f} kg/m³")
    lines.append(f"  Best 56d achieved : {best_str:.2f} MPa")
    return "\n".join(lines) + "\n"


def build_mo_system_prompt(raw_b: dict, der_b: dict, few_shot: list,
                           cfg: MOConfig, w_gwp: float, w_str: float) -> str:
    raw_lines = [f"  {v:<8} [{b['min']:8.2f}, {b['max']:8.2f}]"
                 for v, b in raw_b.items()]
    der_lines = [f"  {v:<8} [{b['min']:8.5f}, {b['max']:8.5f}]"
                 for v, b in der_b.items()]

    if cfg.mode == "single":
        obj_block = MO_OBJECTIVE_SINGLE.format(strength_min=cfg.strength_min)
    else:
        obj_block = MO_OBJECTIVE_MULTI.format(w_gwp=w_gwp, w_str=w_str)

    knowledge_block = KNOWLEDGE_TABLE if cfg.use_knowledge_table else \
        "GWP = PC*1.048 + FA*0.328 + SC*0.264 + CAGG*0.0037 + FAGG*0.0026\n"

    situation_block = SITUATION_RULES if cfg.use_situation_rules else ""

    few_shot_block = ""
    if cfg.use_few_shot and few_shot:
        parts = []
        for ex in few_shot:
            mix_str = "  ".join(f"{k}={ex[k]}" for k in RAW_VARS)
            parts.append(
                f"[{ex['label']}]\n  {mix_str}\n"
                f"  -> 28d={ex['pred_28day']} MPa  GWP={ex['gwp']} kg CO₂/m³"
            )
        few_shot_block = (
            "REFERENCE MIXES FROM DATASET\n" + "=" * 60 + "\n"
            + "\n\n".join(parts)
        )

    return MO_SYSTEM_PROMPT.format(
        mode=cfg.mode.upper(),
        objective_block=obj_block,
        raw_bounds="\n".join(raw_lines),
        der_bounds="\n".join(der_lines),
        knowledge_block=knowledge_block,
        situation_block=situation_block,
        few_shot_block=few_shot_block,
    )


def build_mo_feedback(it: int, max_it: int, mix: dict, preds: dict,
                      gwp: float, feas: dict, trajectory: list,
                      pareto: list, hv: float, der_b: dict,
                      cfg: MOConfig, w_gwp: float, w_str: float) -> str:
    p28 = preds["28day"]
    p56 = preds["56day"]

    if cfg.mode == "single":
        feas_str = "FEASIBLE" if feas["feasible"] else \
            "INFEASIBLE (" + ", ".join(
                list(feas["raw_v"]) + list(feas["der_v"])
                + (["strength"] if feas["str_v"] else [])
            ) + ")"
    else:
        # Multi-obj: no hard strength floor — only check bounds
        bound_issues = list(feas["raw_v"]) + list(feas["der_v"])
        feas_str = "FEASIBLE" if not bound_issues else \
            "INFEASIBLE (" + ", ".join(bound_issues) + ")"

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
        val = dv[v]
        tol = (b["max"] - b["min"]) * 0.01 + 1e-6
        ok  = b["min"] - tol <= val <= b["max"] + tol
        status = "OK" if ok else f"VIOLATION [{b['min']:.4f},{b['max']:.4f}]"
        ratio_lines.append(f"  {v:<8}= {val:.4f}  {status}")

    prev = trajectory[-2] if len(trajectory) >= 2 else None
    prev_gwp  = prev["gwp"]        if prev else gwp
    prev_56   = prev["pred_56day"] if prev else p56
    prev_iter = prev["iteration"]  if prev else it
    gwp_change = round(gwp - prev_gwp, 2)
    str_change = round(p56 - prev_56, 2)

    if gwp_change < -0.5:
        gwp_trend = f"DECREASED {abs(gwp_change):.2f} — good"
    elif gwp_change > 0.5:
        gwp_trend = f"INCREASED {gwp_change:.2f} — check if strength improved"
    else:
        gwp_trend = "barely changed"

    # Pareto status of this solution
    pareto_block = _format_pareto_block(pareto, hv, cfg)
    is_pareto = any(
        s["gwp"] == gwp and s["pred_56day"] == p56 for s in pareto
    )
    pareto_status = "✓ This solution IS on the Pareto front." if is_pareto \
        else "✗ This solution is NOT on the Pareto front (dominated)."
    pareto_block = pareto_status + "\n\n" + pareto_block

    # Directional feedback
    fb = []
    bound_issues = list(feas["raw_v"]) + list(feas["der_v"])
    if bound_issues:
        for v, info in feas["raw_v"].items():
            fb.append(f"  {v}={info['val']:.1f} out of bounds [{info['min']:.1f},{info['max']:.1f}].")
        for v, info in feas["der_v"].items():
            fb.append(f"  {v}={info['val']:.4f} violates [{info['min']:.4f},{info['max']:.4f}].")
    else:
        if w_gwp >= 0.6:
            # GWP-focused iteration
            if gwp_change > 0.5:
                fb.append("  GWP went UP — swap more PC→SC or reduce total binder.")
            elif abs(gwp_change) <= 0.5:
                fb.append("  GWP barely moved — make a bolder substitution (>15 kg/m³ PC→SC).")
            else:
                fb.append(f"  GWP reduced by {abs(gwp_change):.1f} kg/m³ — keep pushing lower.")

            if p56 < 30:
                fb.append("  Warning: 56d strength very low. Don't sacrifice strength further.")
        else:
            # Strength-focused iteration
            if str_change < -1:
                fb.append("  Strength dropped — reduce WATER or add ACC to recover.")
            elif str_change > 1:
                fb.append(f"  Strength gained {str_change:.1f} MPa — can you hold this and cut GWP?")
            else:
                fb.append("  Strength unchanged — try reducing WATER or increasing PC/SC.")

        # Gap to Pareto front
        if pareto:
            best_gwp_pareto = min(s["gwp"] for s in pareto)
            if gwp > best_gwp_pareto + 10:
                fb.append(
                    f"  GWP is {gwp - best_gwp_pareto:.1f} kg/m³ above the Pareto-best. "
                    "More aggressive PC→SC substitution needed."
                )
            best_str_pareto = max(s["pred_56day"] for s in pareto)
            if p56 < best_str_pareto - 5 and w_str >= 0.4:
                fb.append(
                    f"  56d strength is {best_str_pareto - p56:.1f} MPa below Pareto-best. "
                    "Increase PC or reduce water to boost strength."
                )

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
    if bound_issues:
        infeas_warning = "*** BOUND VIOLATION — fix before next proposal ***\n\n"

    return MO_FEEDBACK_TEMPLATE.format(
        it=it, max_it=max_it,
        mix_json=json.dumps(mix, indent=4),
        p7=preds["7day"], p28=p28, p56=p56,
        gwp_breakdown=gwp_breakdown,
        gwp=gwp,
        ratio_check="\n".join(ratio_lines),
        feas_str=feas_str,
        prev_iter=prev_iter, prev_gwp=prev_gwp, prev_56=prev_56,
        gwp_change=gwp_change, gwp_trend=gwp_trend, str_change=str_change,
        pareto_block=pareto_block,
        infeas_warning=infeas_warning,
        osc_warning=osc_warning,
        feedback="\n".join(fb) if fb else "  Looking good — keep exploring.",
        w_gwp=w_gwp, w_str=w_str,
    )


# ─────────────────────────────────────────────────────────────
# GA BASELINE
# ─────────────────────────────────────────────────────────────

def run_mo_ga(raw_b: dict, der_b: dict, meta: dict, cfg: MOConfig) -> dict:
    """Run GA baseline. Single-obj GA for single mode, NSGA-II for multi mode."""
    try:
        from pymoo.optimize import minimize as pymoo_min
        from pymoo.termination import get_termination
        from pymoo.core.problem import Problem
    except ImportError:
        raise ImportError("pymoo not installed — run: pip install pymoo")

    xl = np.array([raw_b[v]["min"] for v in RAW_VARS])
    xu = np.array([raw_b[v]["max"] for v in RAW_VARS])
    n_c = 1 + len(DERIVED_VARS) * 2   # derived ratio constraints

    if cfg.mode == "single":
        from pymoo.algorithms.soo.nonconvex.ga import GA

        class SingleProblem(Problem):
            def __init__(self):
                super().__init__(n_var=len(RAW_VARS), n_obj=1,
                                 n_ieq_constr=n_c + 1, xl=xl, xu=xu)

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

        print(f"\n[GA-single] {cfg.ga_gens} gen x pop={cfg.ga_pop} ...")
        res = pymoo_min(SingleProblem(), GA(pop_size=cfg.ga_pop),
                        termination=get_termination("n_gen", cfg.ga_gens),
                        seed=42, verbose=False)

        best = None
        if res.X is not None:
            candidates = [res.X] if res.X.ndim == 1 else res.X
            for x in candidates:
                mix = dict(zip(RAW_VARS, x))
                pr  = predict(meta, mix)
                g   = compute_gwp(mix)
                feas = check_feasibility(mix, raw_b, der_b, pr["28day"], cfg.strength_min)
                if feas["feasible"] and (best is None or g < best["gwp"]):
                    best = {**{k: round(float(v), 2) for k, v in mix.items()},
                            "pred_28day": pr["28day"], "pred_56day": pr["56day"],
                            "gwp": g}
        if best:
            print(f"[GA-single] Best: GWP={best['gwp']:.2f} kg/m³  "
                  f"28d={best['pred_28day']:.2f} MPa")
        return {"mode": "single", "best": best, "pareto": None}

    else:
        from pymoo.algorithms.moo.nsga2 import NSGA2

        class MultiProblem(Problem):
            def __init__(self):
                super().__init__(n_var=len(RAW_VARS), n_obj=2,
                                 n_ieq_constr=n_c, xl=xl, xu=xu)

            def _evaluate(self, X, out, *args, **kwargs):
                F, G = [], []
                for row in X:
                    mix = dict(zip(RAW_VARS, row))
                    pr  = predict(meta, mix)
                    g   = compute_gwp(mix)
                    F.append([g, -pr["56day"]])   # minimise GWP, maximise 56d
                    gc  = []
                    dv  = get_derived(mix)
                    for v in DERIVED_VARS:
                        b = der_b[v]
                        gc += [b["min"] - dv[v], dv[v] - b["max"]]
                    G.append(gc)
                out["F"] = np.array(F)
                out["G"] = np.array(G)

        print(f"\n[NSGA-II] {cfg.ga_gens} gen x pop={cfg.ga_pop} ...")
        res = pymoo_min(MultiProblem(), NSGA2(pop_size=cfg.ga_pop),
                        termination=get_termination("n_gen", cfg.ga_gens),
                        seed=42, verbose=False)

        pareto_ga = []
        if res.X is not None:
            feasible_mask = np.all(res.G <= 0, axis=1) if res.G is not None else \
                np.ones(len(res.X), dtype=bool)
            for i, x in enumerate(res.X):
                if not feasible_mask[i]:
                    continue
                mix = dict(zip(RAW_VARS, x))
                pr  = predict(meta, mix)
                g   = compute_gwp(mix)
                pareto_ga.append({
                    **{k: round(float(v), 2) for k, v in mix.items()},
                    "pred_28day": pr["28day"], "pred_56day": pr["56day"], "gwp": g,
                })

        hv = compute_hypervolume(pareto_ga, cfg.hv_ref_gwp, cfg.hv_ref_str)
        print(f"[NSGA-II] Pareto front: {len(pareto_ga)} solutions  HV={hv:.1f}")
        return {"mode": "multi", "best": None, "pareto": pareto_ga, "hv": hv}


# ─────────────────────────────────────────────────────────────
# MAIN LLM LOOP
# ─────────────────────────────────────────────────────────────

def _parse_llm_json(text: str) -> Optional[dict]:
    """Extract JSON from LLM response."""
    text = re.sub(r"```(?:json)?", "", text).strip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                return None
    return None


def _clip_mix(mix: dict, raw_b: dict) -> dict:
    return {v: float(np.clip(mix.get(v, raw_b[v]["min"]),
                             raw_b[v]["min"], raw_b[v]["max"]))
            for v in RAW_VARS}


def run_mo_llm(raw_b: dict, der_b: dict, meta: dict,
               few_shot: list, cfg: MOConfig,
               ga_ref: dict = None) -> Tuple[list, list, int]:
    """
    Main LLM optimization loop for multi/single objective.

    Returns
    -------
    trajectory   : list of all evaluated mixes (including infeasible)
    pareto       : Pareto archive (multi mode) or [best] (single mode)
    catboost_calls : total surrogate evaluations
    """
    genai.configure(api_key=cfg.gemini_api_key)

    trajectory   = []
    pareto       = []
    catboost_calls = 0
    n_restarts   = 0
    history      = []   # LLM conversation history

    # Normalisation range for scalarization (updated dynamically)
    gwp_range = (raw_b["PC"]["min"] * GWP_FACTORS["PC"],
                 cfg.hv_ref_gwp * 0.8)
    str_range = (20.0, 80.0)

    prev_hv = 0.0
    stag_count = 0

    def get_weights(it: int) -> Tuple[float, float]:
        if cfg.mode == "single":
            return 1.0, 0.0
        idx = it % len(cfg.lambda_schedule)
        return cfg.lambda_schedule[idx]

    def start_session(restart_msg: str = None):
        nonlocal history
        w_gwp, w_str = get_weights(len(trajectory))
        sys_prompt = build_mo_system_prompt(raw_b, der_b, few_shot, cfg, w_gwp, w_str)

        model = genai.GenerativeModel(
            model_name=cfg.gemini_model,
            system_instruction=sys_prompt,
            generation_config=genai.GenerationConfig(
                temperature=cfg.temperature if not restart_msg else cfg.restart_temp,
                max_output_tokens=1024,
            ),
        )
        chat = model.start_chat(history=[])

        if restart_msg:
            first = restart_msg
        else:
            if cfg.mode == "single":
                hint = f"Use PC >= 150 kg/m³ as a safe starting point to meet {cfg.strength_min} MPa."
            else:
                hint = (
                    "Propose an initial mix that balances GWP and 56d strength.\n"
                    "A good starting point: PC~200 kg/m³, SC~150 kg/m³, FA~50 kg/m³, "
                    "WATER~140 kg/m³, FAGG~600 kg/m³, CAGG~900 kg/m³."
                )
            first = MO_FIRST_TURN.format(start_hint=hint)

        history = [chat, first]
        return chat, first

    chat, first_msg = start_session()
    feasible_iters = 0
    it = 0

    while it < cfg.max_iters:
        w_gwp, w_str = get_weights(it)

        # Send message
        msg = first_msg if it == 0 else history[-1]
        try:
            resp = chat.send_message(msg)
            raw_text = resp.text
        except Exception as e:
            print(f"  [LLM error] {e}. Skipping iteration.")
            it += 1
            continue

        parsed = _parse_llm_json(raw_text)
        if parsed is None or "mix" not in parsed:
            print(f"  [Parse error iter {it}] Could not parse JSON.")
            it += 1
            continue

        mix = _clip_mix(parsed["mix"], raw_b)
        reasoning = parsed.get("reasoning", "")

        # Evaluate
        preds = predict(meta, mix)
        catboost_calls += 1
        gwp   = compute_gwp(mix)
        feas  = check_feasibility(mix, raw_b, der_b, preds["28day"],
                                  cfg.strength_min if cfg.mode == "single" else 0.0)

        is_bound_ok = not feas["raw_v"] and not feas["der_v"]

        # Log
        record = {
            "iteration": it,
            "feasible": feas["feasible"] if cfg.mode == "single" else is_bound_ok,
            "gwp": gwp,
            "pred_7day":  preds["7day"],
            "pred_28day": preds["28day"],
            "pred_56day": preds["56day"],
            "reasoning": reasoning,
            **mix,
        }
        trajectory.append(record)

        # Update Pareto (multi mode) or best (single mode)
        if is_bound_ok:
            if cfg.mode == "multi":
                pareto, added = update_pareto(pareto, record)
                hv = compute_hypervolume(pareto, cfg.hv_ref_gwp, cfg.hv_ref_str)
                hv_improvement = (hv - prev_hv) / (prev_hv + 1e-9)
                if added:
                    feasible_iters += 1
                    print(f"  iter {it:2d} | GWP={gwp:.1f}  56d={preds['56day']:.1f} MPa "
                          f"| Pareto: {len(pareto)} sols  HV={hv:.1f} "
                          f"{'[+Pareto]' if added else ''}")
                    stag_count = 0 if hv_improvement > cfg.stag_threshold else stag_count + 1
                    prev_hv = hv
                else:
                    print(f"  iter {it:2d} | GWP={gwp:.1f}  56d={preds['56day']:.1f} MPa "
                          f"| dominated  HV={hv:.1f}")
                    stag_count += 1
            else:
                if feas["feasible"]:
                    feasible_iters += 1
                    best_gwp = min((r["gwp"] for r in trajectory
                                   if r.get("feasible")), default=gwp)
                    if gwp <= best_gwp:
                        pareto = [record]
                    print(f"  iter {it:2d} | GWP={gwp:.1f}  28d={preds['28day']:.1f} MPa "
                          f"[feasible iter {feasible_iters}]")
                else:
                    print(f"  iter {it:2d} | GWP={gwp:.1f}  28d={preds['28day']:.1f} MPa "
                          f"[infeasible: strength]")

        # Stagnation restart
        if (stag_count >= cfg.stag_window and n_restarts < cfg.max_restarts
                and len(pareto) > 0):
            n_restarts += 1
            stag_count = 0
            print(f"\n  [Restart {n_restarts}] Stagnation detected. Restarting with T={cfg.restart_temp}")

            if cfg.mode == "multi":
                front_summary = "\n".join(
                    f"  GWP={s['gwp']:.1f}  56d={s['pred_56day']:.1f} MPa  "
                    f"PC={s.get('PC',0):.0f} SC={s.get('SC',0):.0f}"
                    for s in sorted(pareto, key=lambda s: s["gwp"])[:5]
                )
                restart_msg = (
                    f"RESTART #{n_restarts}: Current Pareto front ({len(pareto)} solutions):\n"
                    f"{front_summary}\n\n"
                    "Propose a STRUCTURALLY DIFFERENT mix to fill gaps in the Pareto front. "
                    "Try a region you have NOT explored: e.g., very high SC + low FA, "
                    "or Path 2 (low total binder + high aggregates + ACC).\n"
                    "Output ONLY the JSON object."
                )
            else:
                best = pareto[0] if pareto else trajectory[-1]
                restart_msg = (
                    f"RESTART #{n_restarts}: Best so far: GWP={best['gwp']:.1f} kg/m³  "
                    f"28d={best['pred_28day']:.1f} MPa\n"
                    "Try a COMPLETELY DIFFERENT strategy (Path 2: low binder + high aggregates). "
                    "Output ONLY the JSON object."
                )

            chat, _ = start_session(restart_msg)
            history.append(restart_msg)
            it += 1
            continue

        # Build feedback for next iter
        hv_for_display = compute_hypervolume(pareto, cfg.hv_ref_gwp, cfg.hv_ref_str) \
            if pareto else 0.0
        next_w_gwp, next_w_str = get_weights(it + 1)
        feedback_msg = build_mo_feedback(
            it, cfg.max_iters, mix, preds, gwp, feas, trajectory,
            pareto, hv_for_display, der_b, cfg, next_w_gwp, next_w_str,
        )
        history.append(feedback_msg)
        it += 1

    return trajectory, pareto, catboost_calls


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def compute_mo_metrics(trajectory: list, pareto_llm: list,
                       ga_ref: dict, cfg: MOConfig) -> dict:
    if not pareto_llm:
        return {"error": "no feasible solutions found"}

    hv_llm = compute_hypervolume(pareto_llm, cfg.hv_ref_gwp, cfg.hv_ref_str)
    feasible = [r for r in trajectory if r.get("feasible", False)]
    catboost_calls = len(trajectory)

    metrics = {
        "mode": cfg.mode,
        "pareto_size": len(pareto_llm),
        "hypervolume": round(hv_llm, 2),
        "feasible_iters": len(feasible),
        "total_iters": len(trajectory),
        "feasibility_rate": round(len(feasible) / max(len(trajectory), 1), 3),
        "catboost_calls": catboost_calls,
    }

    if cfg.mode == "single" and ga_ref and ga_ref.get("best"):
        ga_gwp = ga_ref["best"]["gwp"]
        best_llm_gwp = pareto_llm[0]["gwp"] if pareto_llm else None
        if best_llm_gwp is not None:
            metrics["best_gwp"] = round(best_llm_gwp, 2)
            metrics["ga_gwp"]   = round(ga_gwp, 2)
            metrics["OGR"]      = round((best_llm_gwp - ga_gwp) / (ga_gwp + 1e-9), 4)
            metrics["surrogate_ratio"] = round(catboost_calls / 20000, 5)

    if cfg.mode == "multi" and ga_ref and ga_ref.get("pareto"):
        ga_pareto = ga_ref["pareto"]
        hv_ga = compute_hypervolume(ga_pareto, cfg.hv_ref_gwp, cfg.hv_ref_str)
        metrics["hv_ga"]    = round(hv_ga, 2)
        metrics["hv_ratio"] = round(hv_llm / (hv_ga + 1e-9), 4)

        # Pareto Coverage Rate: fraction of GA Pareto front dominated by LLM front
        dominated_count = sum(
            1 for ga_sol in ga_pareto
            if any(dominates(llm_sol, ga_sol) for llm_sol in pareto_llm)
        )
        metrics["pareto_coverage_rate"] = round(dominated_count / max(len(ga_pareto), 1), 3)
        metrics["surrogate_ratio"] = round(catboost_calls / 20000, 5)

    return metrics


# ─────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────

def save_mo_results(trajectory: list, pareto: list, metrics: dict,
                    ga_ref: dict, cfg: MOConfig):
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    prefix = f"{cfg.output_prefix}_{ts}"

    # Trajectory CSV
    traj_df = pd.DataFrame(trajectory)
    traj_path = f"{prefix}_trajectory.csv"
    traj_df.to_csv(traj_path, index=False)

    # Pareto front CSV
    if pareto:
        pareto_df = pd.DataFrame(pareto)
        pareto_path = f"{prefix}_pareto.csv"
        pareto_df.to_csv(pareto_path, index=False)
    else:
        pareto_path = None

    # Metrics CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_path = f"{prefix}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Text report
    report_path = f"{prefix}_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment : {cfg.name}\n")
        f.write(f"Mode       : {cfg.mode}\n")
        f.write(f"Timestamp  : {ts}\n")
        f.write(f"Model      : {cfg.gemini_model}\n")
        f.write("=" * 60 + "\n\nMETRICS\n")
        for k, v in metrics.items():
            f.write(f"  {k:<30}: {v}\n")

        if pareto:
            f.write("\nPARETO FRONT\n" + "-" * 40 + "\n")
            for s in sorted(pareto, key=lambda x: x["gwp"]):
                f.write(
                    f"  GWP={s['gwp']:.2f}  56d={s['pred_56day']:.2f} MPa  "
                    f"PC={s.get('PC',0):.1f}  SC={s.get('SC',0):.1f}  "
                    f"FA={s.get('FA',0):.1f}\n"
                )

    print(f"\n  Trajectory  -> {traj_path}")
    if pareto_path:
        print(f"  Pareto front-> {pareto_path}")
    print(f"  Metrics     -> {metrics_path}")
    print(f"  Report      -> {report_path}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi/single objective LLM concrete optimizer")
    parser.add_argument("--mode",     default="multi", choices=["single", "multi"])
    parser.add_argument("--iters",    type=int,   default=40)
    parser.add_argument("--strength", type=float, default=55.0,
                        help="Strength floor for single-obj mode (MPa)")
    parser.add_argument("--skip-ga",  action="store_true")
    args = parser.parse_args()

    cfg = MOConfig(
        name=f"mo_{args.mode}",
        mode=args.mode,
        strength_min=args.strength,
        max_iters=args.iters,
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        output_prefix=f"results/mo_{args.mode}",
    )

    print(f"\n{'='*60}")
    print(f"  LLM Concrete Optimizer — {cfg.mode.upper()} mode")
    print(f"  Model: {cfg.gemini_model}  |  Iters: {cfg.max_iters}")
    print(f"{'='*60}")

    # Load data and model
    print("\n[Setup] Loading data and surrogate ...")
    df       = load_df(cfg.data_path)
    raw_b, der_b = get_bounds(df)
    meta     = load_surrogate(cfg.model_pkl)
    few_shot = select_few_shot(df, cfg.strength_min if cfg.mode == "single" else 40.0, n=3)

    print(f"  Dataset: {len(df)} rows (kg/m³)")
    print(f"  PC range: [{raw_b['PC']['min']:.1f}, {raw_b['PC']['max']:.1f}] kg/m³")

    # GA reference
    ga_ref = None
    ga_csv = f"ga_reference_{cfg.mode}.csv"
    if args.skip_ga and os.path.exists(ga_csv):
        print(f"\n[GA] Loading saved reference from '{ga_csv}' ...")
        # (simplified — load from CSV if available)
    else:
        ga_ref = run_mo_ga(raw_b, der_b, meta, cfg)

    # LLM optimization
    print(f"\n[LLM] Starting {cfg.mode} optimization ({cfg.max_iters} iterations) ...")
    trajectory, pareto, catboost_calls = run_mo_llm(
        raw_b, der_b, meta, few_shot, cfg, ga_ref
    )

    # Metrics
    metrics = compute_mo_metrics(trajectory, pareto, ga_ref, cfg)

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:<30}: {v}")

    # Save
    os.makedirs("results", exist_ok=True)
    save_mo_results(trajectory, pareto, metrics, ga_ref, cfg)


if __name__ == "__main__":
    main()

"""
optimizer_core.py
=================
Core of the LLM-based single-objective concrete mix optimizer and its GA reference.
Stable logic only; experiments are defined in run_experiment.py.

Problem
-------
  minimise  GWP (kg CO2-eq/m3)
  subject to (checked by ONE function, check_feasibility, used by the GA and the LLM alike)
    - 10 ingredient bounds                       (data/constraints.json -> raw)
    - 12 derived-ratio bounds                    (data/constraints.json -> derived)
    - volume balance 0.95 <= Vfinal <= 1.05       (data/constraints.json -> physics)
    - predicted 28-day strength >= strength_min
    - predicted P(28-day RCPT < 1200 C) >= 0.7    (only when use_durability=True)

Units: all ingredient quantities kg/m3, strength MPa. See docs/DATA_AND_MODELS.md.
"""

import json
import os
import re
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
CONSTRAINTS_PATH = os.path.join(ROOT, "data", "constraints.json")
DATASET_PATH = os.path.join(ROOT, "data", "Concrete_Dataset_SI.xlsx")
STRENGTH_PKL = os.path.join(ROOT, "models", "strength_chain.pkl")
CHLORIDE_PKL = os.path.join(ROOT, "models", "chloride_clf.pkl")

RAW_VARS = ["PC", "FA", "SC", "FAGG", "CAGG", "WATER", "AEA", "WR_HR", "WR", "ACC"]
ADMIX_VARS = ["AEA", "WR_HR", "WR", "ACC"]
DERIVED_VARS = ["w/b", "b/a", "SCM%", "CAGG%", "FAGG%", "PC%", "FA%", "SC%",
                "AEA_pct", "WR_HR_pct", "WR_pct", "ACC_pct"]
MIX_DECIMALS = 3
EPS = 1e-9
_PRED_LOCK = threading.Lock()   # CatBoost prediction is serialised so runs can use threads


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """Everything that varies between experiments. Saved as config.json with every run."""
    name: str = "baseline"
    description: str = ""

    # problem
    strength_min: float = 50.0
    use_durability: bool = False
    max_iters: int = 30                # feasible proposals to collect

    # LLM
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.9
    max_output_tokens: int = 1024
    restart_temp: float = 1.3

    # GA reference
    ga_gens: int = 200
    ga_pop: int = 100
    ga_seeds: tuple = (1, 2, 3, 4, 5)

    # stagnation / restart
    stag_window: int = 5
    stag_threshold: float = 0.005      # relative improvement over the window
    stag_min_best: float = 330.0       # only declare stagnation once GWP is below this
    max_restarts: int = 2
    anti_osc_window: int = 3
    anti_osc_tol: float = 5.0

    # prompt / retrieval ablation flags
    use_few_shot: bool = True
    use_knowledge_table: bool = True
    use_situation_rules: bool = True
    use_directional_hints: bool = True   # prescriptive advice in feedback, retry, restart and first-turn messages
    rag_low_gwp_half: bool = False       # retrieve only from the lower-GWP half of the candidate pool
    rag_mode: str = "static"           # "static" | "dynamic" | "none"
    rag_k: int = 5
    rag_format: str = "tabular"        # "tabular" | "text"
    max_attempt_factor: int = 6        # attempts cap = max_iters * factor


# ─────────────────────────────────────────────────────────────
# 1. PROBLEM SPECIFICATION (constraints + surrogates + dataset statistics)
# ─────────────────────────────────────────────────────────────

class Spec:
    """Constraint bounds, surrogate models and dataset statistics. Built once per process."""

    def __init__(self, constraints_path=CONSTRAINTS_PATH, strength_pkl=STRENGTH_PKL,
                 chloride_pkl=CHLORIDE_PKL, dataset_path=DATASET_PATH):
        with open(constraints_path, encoding="utf-8") as f:
            c = json.load(f)
        self.raw_b = c["raw"]
        self.der_b = c["derived"]
        self.phys_b = c["physics"]
        self.densities = c["densities"]
        self.gwp_factors = {v: c["gwp_factors"].get(v, 0.0) for v in RAW_VARS}
        self.rcpt_limit = c["durability"]["rcpt_limit_coulomb"]
        self.min_pass_prob = c["durability"]["min_pass_probability"]
        self.strength = joblib.load(strength_pkl)
        self.chloride = joblib.load(chloride_pkl)
        assert self.strength["unit"] == "kg/m3" and self.chloride["unit"] == "kg/m3"
        self.features = self.strength["feature_names"]
        self.df = load_df(dataset_path)
        self.stats = {
            "binder_p5": float(self.df["TOTAL_BINDER"].quantile(0.05)),
            "binder_p50": float(self.df["TOTAL_BINDER"].median()),
            "binder_p95": float(self.df["TOTAL_BINDER"].quantile(0.95)),
            "gwp_p5": float(self.df["GWP"].quantile(0.05)),
            "gwp_p50": float(self.df["GWP"].median()),
        }


def load_df(path: str = DATASET_PATH) -> pd.DataFrame:
    """Mix-level dataset (kg/m3), without silica-fume mixes (SF is not a design variable)."""
    df = pd.read_excel(path, sheet_name="mix_level")
    return df[df["SF"] == 0].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 2. FEATURES, SURROGATES, GWP, PHYSICS
# ─────────────────────────────────────────────────────────────

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add TOTAL_BINDER and the derived ratios (same definitions as utils/prepare_data.py)."""
    d = df.copy()
    tb = d["PC"] + d["FA"] + d["SC"]
    ag = d["FAGG"] + d["CAGG"]
    d["TOTAL_BINDER"] = tb
    d["w/b"] = d["WATER"] / (tb + EPS)
    d["b/a"] = tb / (ag + EPS)
    d["SCM%"] = (d["FA"] + d["SC"]) / (tb + EPS)
    d["CAGG%"] = d["CAGG"] / (ag + EPS)
    d["FAGG%"] = d["FAGG"] / (ag + EPS)
    for v in ["PC", "FA", "SC"]:
        d[f"{v}%"] = d[v] / (tb + EPS)
    for v in ADMIX_VARS:
        d[f"{v}_pct"] = d[v] / (tb + EPS)
    return d


def predict_batch(spec: Spec, mixes: pd.DataFrame) -> pd.DataFrame:
    """Chained strength prediction (predicted previous stage is fed forward) and chloride pass probability."""
    x = engineer(mixes[RAW_VARS].astype(float))
    fn, mdl = spec.features, spec.strength["models"]
    with _PRED_LOCK:
        p7 = mdl["7day"].predict(x[fn])
        p28 = mdl["28day"].predict(x[fn].assign(**{"7day": p7}))
        p56 = mdl["56day"].predict(x[fn].assign(**{"28day": p28}))
        proba = spec.chloride["model"].predict_proba(x[spec.chloride["feature_names"]])
    k = spec.chloride["pass_classes"]
    prob = proba[:, :k].sum(axis=1)
    # Class label consistent with feasibility: if P(pass) >= threshold, the likelier of Very Low / Low;
    # otherwise the likelier of Moderate / High.
    names = np.array(spec.chloride["classes"])
    pass_lab = names[proba[:, :k].argmax(axis=1)]
    fail_lab = names[k + proba[:, k:].argmax(axis=1)]
    cls = np.where(prob >= spec.min_pass_prob, pass_lab, fail_lab)
    return pd.DataFrame({"pred_7day": p7, "pred_28day": p28, "pred_56day": p56,
                         "chloride_prob": prob, "chloride_class": cls}, index=mixes.index)


def predict(spec: Spec, mix: dict) -> dict:
    r = predict_batch(spec, pd.DataFrame([mix])).iloc[0]
    return {"7day": round(float(r["pred_7day"]), 2), "28day": round(float(r["pred_28day"]), 2),
            "56day": round(float(r["pred_56day"]), 2), "chloride_prob": round(float(r["chloride_prob"]), 3),
            "chloride_class": str(r["chloride_class"])}


def compute_gwp(mix: dict, spec: Spec = None) -> float:
    f = spec.gwp_factors if spec else {"PC": 1.048, "FA": 0.328, "SC": 0.264,
                                       "FAGG": 0.0026, "CAGG": 0.0037}
    return round(sum(mix.get(k, 0.0) * v for k, v in f.items()), 2)


def get_derived(mix: dict) -> dict:
    m = engineer(pd.DataFrame([{v: mix.get(v, 0.0) for v in RAW_VARS}])).iloc[0]
    return {k: float(m[k]) for k in DERIVED_VARS}


def get_physics(spec: Spec, mix: dict) -> dict:
    """Vfinal = sum(mass/density) + air (7% if AEA/PC >= 0.000244 else 3%)."""
    vm = sum(mix.get(v, 0.0) / spec.densities[v] for v in RAW_VARS)
    air = 0.07 if mix.get("AEA", 0.0) / (mix.get("PC", 0.0) + EPS) >= 0.000244 else 0.03
    return {"Vm": vm, "air": air, "Vfinal": vm + air}


# ─────────────────────────────────────────────────────────────
# 3. FEASIBILITY (single source of truth, used by GA and LLM)
# ─────────────────────────────────────────────────────────────

def check_feasibility(spec: Spec, mix: dict, preds: dict, strength_min: float,
                      use_durability: bool) -> dict:
    """Exact bounds, no tolerance. Returns violated constraints and an overall flag."""
    raw_v, der_v = {}, {}
    for v in RAW_VARS:
        b = spec.raw_b[v]
        x = mix.get(v, 0.0)
        if x < b["min"] - EPS or x > b["max"] + EPS:
            raw_v[v] = {"val": x, "min": b["min"], "max": b["max"]}
    dv = get_derived(mix)
    for v in DERIVED_VARS:
        b = spec.der_b[v]
        if dv[v] < b["min"] - EPS or dv[v] > b["max"] + EPS:
            der_v[v] = {"val": dv[v], "min": b["min"], "max": b["max"]}
    vf = get_physics(spec, mix)["Vfinal"]
    pb = spec.phys_b["Vfinal"]
    phys_v = {} if pb["min"] - EPS <= vf <= pb["max"] + EPS else \
        {"Vfinal": {"val": vf, "min": pb["min"], "max": pb["max"]}}
    str_v = preds["28day"] < strength_min
    dur_v = bool(use_durability and preds["chloride_prob"] < spec.min_pass_prob)
    return {"raw_v": raw_v, "der_v": der_v, "phys_v": phys_v, "str_v": str_v, "dur_v": dur_v,
            "feasible": not (raw_v or der_v or phys_v or str_v or dur_v)}


def clip_mix(mix: dict, spec: Spec) -> tuple:
    """Clip a proposal to the ingredient bounds and round it. Returns (mix, notes)."""
    clean, notes = {}, []
    for v in RAW_VARS:
        b = spec.raw_b[v]
        try:
            val = float(mix.get(v, b["min"]))
        except (TypeError, ValueError):
            val = b["min"]
        clp = float(np.clip(val, b["min"], b["max"]))
        if abs(clp - val) > 1e-6:
            notes.append(f"  {v}: proposed {val:.3f} clipped to [{b['min']:.3f},{b['max']:.3f}] -> {clp:.3f}")
        clean[v] = round(clp, MIX_DECIMALS)
        # rounding must not push a value outside its bound
        clean[v] = float(np.clip(clean[v], b["min"], b["max"]))
    return clean, notes


def verify_solution(spec: Spec, mix: dict, strength_min: float, use_durability: bool) -> dict:
    """Independent re-check of a reported solution from its mix only. Raises if infeasible."""
    mix = {v: float(mix[v]) for v in RAW_VARS}
    preds = predict(spec, mix)
    feas = check_feasibility(spec, mix, preds, strength_min, use_durability)
    if not feas["feasible"]:
        raise AssertionError(f"Reported solution violates constraints: {feas}")
    return {"preds": preds, "gwp": compute_gwp(mix, spec)}


# ─────────────────────────────────────────────────────────────
# 4. GA REFERENCE (same constraints as the LLM)
# ─────────────────────────────────────────────────────────────

def run_ga(spec: Spec, cfg: ExperimentConfig, seed: int) -> dict:
    """Single-objective GA (pymoo). Returns the best feasible solution, its cost and run statistics."""
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize as pymoo_min
    from pymoo.termination import get_termination

    xl = np.array([spec.raw_b[v]["min"] for v in RAW_VARS])
    xu = np.array([spec.raw_b[v]["max"] for v in RAW_VARS])
    der_rng = {v: max(spec.der_b[v]["max"] - spec.der_b[v]["min"], EPS) for v in DERIVED_VARS}
    n_c = 1 + (1 if cfg.use_durability else 0) + 2 * len(DERIVED_VARS) + 2
    counter = {"evals": 0}

    class ConcreteProblem(Problem):
        def __init__(self):
            super().__init__(n_var=len(RAW_VARS), n_obj=1, n_ieq_constr=n_c, xl=xl, xu=xu)

        def _evaluate(self, X, out, *args, **kwargs):
            X = np.clip(np.round(X, MIX_DECIMALS), xl, xu)
            mixes = pd.DataFrame(X, columns=RAW_VARS)
            pr = predict_batch(spec, mixes)
            counter["evals"] += len(mixes)
            eng = engineer(mixes)
            out["F"] = (mixes[RAW_VARS] * pd.Series(spec.gwp_factors)).sum(axis=1).values[:, None]
            G = [(cfg.strength_min - pr["pred_28day"]).values]
            if cfg.use_durability:
                G.append((spec.min_pass_prob - pr["chloride_prob"]).values * 10.0)
            for v in DERIVED_VARS:
                b = spec.der_b[v]
                G += [((b["min"] - eng[v]) / der_rng[v]).values, ((eng[v] - b["max"]) / der_rng[v]).values]
            vm = sum(mixes[v] / spec.densities[v] for v in RAW_VARS)
            air = np.where(mixes["AEA"] / (mixes["PC"] + EPS) >= 0.000244, 0.07, 0.03)
            vf = (vm + air).values
            pb = spec.phys_b["Vfinal"]
            G += [(pb["min"] - vf) * 10.0, (vf - pb["max"]) * 10.0]
            out["G"] = np.column_stack(G)

    t0 = time.time()
    res = pymoo_min(ConcreteProblem(), GA(pop_size=cfg.ga_pop),
                    termination=get_termination("n_gen", cfg.ga_gens), seed=seed, verbose=False)
    wall = time.time() - t0

    best = None
    if res.X is not None:
        x = res.X if res.X.ndim == 1 else res.X[0]
        mix, _ = clip_mix(dict(zip(RAW_VARS, np.round(x, MIX_DECIMALS))), spec)
        preds = predict(spec, mix)
        feas = check_feasibility(spec, mix, preds, cfg.strength_min, cfg.use_durability)
        if feas["feasible"]:
            tb = mix["PC"] + mix["FA"] + mix["SC"]
            best = {**mix, **{k: round(v, 5) for k, v in get_derived(mix).items()},
                    "total_binder": round(tb, 3), "Vfinal": round(get_physics(spec, mix)["Vfinal"], 4),
                    "pred_7day": preds["7day"], "pred_28day": preds["28day"],
                    "pred_56day": preds["56day"], "chloride_prob": preds["chloride_prob"],
                    "chloride_class": preds["chloride_class"],
                    "gwp": compute_gwp(mix, spec)}
    return {"best": best, "seed": seed, "n_evals": counter["evals"], "wall_time_s": round(wall, 2)}


def select_ga_reference(runs: list) -> dict:
    """Reference optimum = lowest-GWP feasible solution over all GA seeds."""
    ok = [r for r in runs if r["best"]]
    return min(ok, key=lambda r: r["best"]["gwp"])["best"] if ok else None


# ─────────────────────────────────────────────────────────────
# 5. FEW-SHOT AND DYNAMIC RAG (dataset pools)
# ─────────────────────────────────────────────────────────────

def _pool(spec: Spec, cfg: ExperimentConfig) -> pd.DataFrame:
    """Dataset mixes that meet the strength floor (and the chloride screen when durability is on)."""
    sub = spec.df.dropna(subset=["28day"]).copy()
    sub = sub[sub["28day"] >= cfg.strength_min]
    if cfg.use_durability and len(sub):
        sub = sub[predict_batch(spec, sub)["chloride_prob"].values >= spec.min_pass_prob]
    sub["gwp"] = sub.apply(lambda r: compute_gwp(r.to_dict(), spec), axis=1)
    return sub


def select_few_shot(spec: Spec, cfg: ExperimentConfig, n: int = 3) -> list:
    sub = _pool(spec, cfg)
    if len(sub) == 0:
        return []
    labels = ["Lowest GWP", "Highest strength", "Balanced"]
    rows = []
    r1 = sub.loc[sub["gwp"].idxmin()]
    rows.append(r1)
    sub = sub.drop(r1.name)
    if len(sub):
        r2 = sub.loc[sub["28day"].idxmax()]
        rows.append(r2)
        sub = sub.drop(r2.name)
    if len(sub):
        s_n = (sub["28day"] - sub["28day"].min()) / (sub["28day"].max() - sub["28day"].min() + EPS)
        c_n = (sub["gwp"] - sub["gwp"].min()) / (sub["gwp"].max() - sub["gwp"].min() + EPS)
        rows.append(sub.loc[(s_n - c_n).abs().idxmin()])
    out = []
    for row, label in zip(rows[:n], labels[:n]):
        ex = {k: round(float(row[k]), 3) for k in RAW_VARS}
        ex.update({"pred_28day": round(float(row["28day"]), 1), "gwp": round(float(row["gwp"]), 1),
                   "label": label})
        out.append(ex)
    return out


def retrieve_similar_mixes(spec: Spec, cfg: ExperimentConfig, current_mix: dict, pool: pd.DataFrame) -> list:
    """Dynamic RAG: k nearest dataset mixes (min-max normalised ingredient space)."""
    if len(pool) == 0:
        return []
    data = pool[RAW_VARS].values.astype(float)
    lo, rng = data.min(axis=0), data.max(axis=0) - data.min(axis=0) + EPS
    cur = (np.array([current_mix.get(f, 0) for f in RAW_VARS], dtype=float) - lo) / rng
    idx = np.argsort(np.linalg.norm((data - lo) / rng - cur, axis=1))[:cfg.rag_k]
    out = []
    for _, row in pool.iloc[idx].iterrows():
        ex = {v: round(float(row[v]), 3) for v in RAW_VARS}
        ex["pred_28day"] = round(float(row["28day"]), 1)
        ex["gwp"] = round(float(row["gwp"]), 1)
        out.append(ex)
    return out


# ─────────────────────────────────────────────────────────────
# 6. PROMPTS
# ─────────────────────────────────────────────────────────────

def build_knowledge(spec: Spec, cfg: ExperimentConfig) -> str:
    s, rb, d = spec.stats, spec.raw_b, spec.densities
    dur = ""
    if cfg.use_durability:
        dur = (
            "\n  CHLORIDE RESISTANCE (durability constraint)\n"
            "       In the dataset, mixes with SCM >= 50% of the binder (mostly slag) pass the chloride\n"
            "       test far more often than PC-only mixes; w/b above ~0.45 rarely passes.\n"
            "       Strategy   : keep substituting PC with SC (it lowers GWP AND helps chloride resistance);\n"
            "                    avoid raising WATER.\n")
    return f"""\
MATERIAL EFFECTS — DECISION TABLE
===================================
Each material affects GWP, 28-day strength and the volume balance. Use this table every step.
All quantities are in kg/m³; GWP is in kg CO₂-eq/m³.

  PC   (Portland cement)
       GWP factor : 1.048 kg CO₂/kg  — largest CO₂ contributor
       Strength   : STRONG positive — PC is the primary strength driver
       Strategy   : Reduce PC as much as possible, but never below what strength requires.
                    Each 10 kg/m³ PC reduced saves ~10.5 kg CO₂/m³.

  SC   (Slag cement / GGBS)
       GWP factor : 0.264 kg CO₂/kg  — most efficient binder for CO₂ reduction
       Strength   : MODERATE positive at 28d — activates via PC hydration products;
                    very high SC substitution (>60%) may slightly reduce 28d strength.
       Strategy   : PREFERRED substitute for PC.
                    Net saving per kg/m³ PC→SC swap = 1.048 - 0.264 = 0.784 kg CO₂/m³.

  FA   (Fly ash)
       GWP factor : 0.328 kg CO₂/kg  — second best option for CO₂ reduction
       Strength   : WEAK positive at 28d — FA is slow-reacting (pozzolanic).
       Strategy   : Use FA after SC is maximised. High FA risks falling below the floor.

  WATER
       GWP factor : 0.000  ZERO
       Strength   : NEGATIVE — more water = higher w/b = lower strength
       Strategy   : Reduce WATER to improve strength at zero GWP cost.

  FAGG / CAGG  (Fine / Coarse aggregate)
       GWP factor : 0.0026 / 0.0037 kg CO₂/kg  — nearly zero
       Strength   : More aggregate = lower b/a = less paste per m³.
       Strategy   : Increasing FAGG+CAGG enables Path 2 (lower total binder).
                    Aggregates also fill the volume: Vfinal must stay in [0.95, 1.05].

  WR_HR / WR  (High-range / normal water reducer)  range WR_HR {rb['WR_HR']['min']:.1f}-{rb['WR_HR']['max']:.1f}, WR {rb['WR']['min']:.1f}-{rb['WR']['max']:.1f}
       GWP factor : 0.000  ZERO
       Strength   : Indirect — allows lower WATER, enabling lower w/b.
       Strategy   : Typical dosage is 1-5 kg/m³ WR_HR; use it to cut WATER on Path 2.

  AEA  (Air-entraining agent)  range {rb['AEA']['min']:.2f}-{rb['AEA']['max']:.2f}
       GWP factor : 0.000  ZERO
       Strength   : Entrained air lowers strength. If AEA/PC >= 0.000244 the mix is air-entrained
                    (air = 7%), otherwise air = 3%. Air counts in Vfinal.

  ACC  (Accelerator)  range {rb['ACC']['min']:.0f}-{rb['ACC']['max']:.0f}
       GWP factor : 0.000  ZERO
       Strength   : Direct positive — useful when total binder is low.
       Strategy   : ACC 5-15 kg/m³ can compensate strength loss on Path 2.
{dur}
KEY INSIGHT — TWO PATHS TO LOW GWP:
  Path 1: substitute PC with SC/FA (total binder {s['binder_p50']:.0f}-{s['binder_p95']:.0f} kg/m³)
  Path 2: reduce total binder below {s['binder_p5']:.0f} kg/m³ + high FAGG/CAGG + WR_HR + ACC
          (lowest GWP, but harder to keep strength)

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
  -> Reduce PC+SC by 12-18 kg/m³ each, OR increase FAGG+CAGG by 60-120 kg/m³ (watch Vfinal),
     OR switch to Path 2 (lower total binder + more WR_HR).

SITUATION B: Strength close to floor (margin 0-5 MPa)  [efficient zone]
  -> Swap 6-12 kg/m³ PC -> SC (saves ~4.7-9.4 kg CO₂/m³).
  -> OR reduce WATER 3-6 kg/m³ first, then swap PC->SC.
  -> OR add ACC 2-5 kg/m³ to create headroom for further PC reduction.

SITUATION C: A constraint is violated (infeasible)
  -> Strength too low: 1. reduce WATER 6-9 kg/m³  2. add ACC 3-8 kg/m³
     3. raise WR_HR 0.5-1.5 kg/m³ to allow less water  4. raise PC 6-12 kg/m³ (last resort).
     Never increase SC or FA to recover strength at 28d.
  -> Vfinal outside [0.95, 1.05]: add or remove aggregate (or water) so total volume balances.
  -> Chloride probability too low: raise SC share of the binder, lower WATER.

SITUATION D: Stuck in local optimum (no GWP improvement)
  -> Check which variables have been CONSTANT in the last 5 iterations.
  -> Option 1 (most impactful): switch to Path 2 (total binder well below the dataset median,
       more aggregate, WR_HR 2-5, ACC 5-15).
  -> Option 2: add ACC 3-10 kg/m³ (free strength boost, try lower PC).
  -> Option 3: different binder blend (high FA, or pure PC+SC).
  -> Option 4: raise WR 0.5-1.5 kg/m³ to allow cutting WATER 10-15 kg/m³.
"""

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert concrete mix design engineer specialising in low-carbon concrete.

OPTIMISATION PROBLEM
====================
OBJECTIVE  : MINIMISE total GWP (kg CO₂-eq/m³)
CONSTRAINTS (ALL are HARD LIMITS):
  1. 28-day compressive strength >= {strength_min} MPa
  2. every variable and ratio inside the bounds below
  3. volume balance: 0.95 <= Vfinal <= 1.05
{durability_line}
GWP (kg CO₂/m³) = PC*1.048 + FA*0.328 + SC*0.264 + CAGG*0.0037 + FAGG*0.0026
  (all ingredient quantities in kg/m³)

VOLUME BALANCE
==============
Vfinal = sum(mass_i / density_i) + air        (mass in kg/m³, density in kg/m³)
  densities: {densities}
  air = 0.07 if AEA/PC >= 0.000244 else 0.03

{knowledge_block}

VARIABLE BOUNDS (kg/m³, from dataset)
=======================================
{raw_bounds}

DERIVED RATIO BOUNDS
====================
{der_bounds}
  w/b = WATER/(PC+FA+SC)   b/a = (PC+FA+SC)/(FAGG+CAGG)   SCM% = (FA+SC)/(PC+FA+SC)
  CAGG% = CAGG/(FAGG+CAGG)   FAGG% = FAGG/(FAGG+CAGG)   PC%, FA%, SC% = share of binder
  AEA_pct, WR_HR_pct, WR_pct, ACC_pct = admixture / (PC+FA+SC)

{situation_block}

{few_shot_block}

OUTPUT FORMAT — STRICTLY REQUIRED
===================================
IMPORTANT: Before outputting, mentally verify all constraints. NEVER output a mix you expect to be infeasible.

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
Start optimisation. Propose an initial mix satisfying ALL constraints:
  - 28-day strength >= {strength_min} MPa  (HARD CONSTRAINT — most important)
{first_dur}  - Vfinal in [0.95, 1.05] and all variable / ratio bounds satisfied
  - GWP as low as possible

SAFE STARTING POINT to make feasibility likely:
  Use PC >= {safe_pc:.0f} kg/m³ as a starting point (typical of dataset mixes that reach {strength_min} MPa).
  Then reduce GWP by substituting PC with SC in later iterations.

Output ONLY the JSON object.\
"""

FEEDBACK_TEMPLATE = """\
=== ITERATION {it} / {max_it} ===

Last proposed mix:
{mix_json}

Surrogate evaluation:
  7-day  : {p7:.2f} MPa
  28-day : {p28:.2f} MPa   {str_status}
  56-day : {p56:.2f} MPa
{chloride_line}  Vfinal : {vfinal:.4f}   {vf_status}

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

Best feasible so far (iter {best_iter}):
  PC={best_PC:.1f}  SC={best_SC:.1f}  FA={best_FA:.1f}  FAGG={best_FAGG:.1f}  CAGG={best_CAGG:.1f}  WATER={best_WATER:.1f}
  AEA={best_AEA:.2f}  WR_HR={best_WR_HR:.2f}  WR={best_WR:.2f}  ACC={best_ACC:.2f}
  -> GWP={best_gwp:.2f} kg  28d={best_28:.2f} MPa

Current vs best:
  PC {pc_diff:+.0f}   SC {sc_diff:+.0f}   FA {fa_diff:+.0f}   WATER {water_diff:+.0f}   FAGG {fagg_diff:+.0f}   CAGG {cagg_diff:+.0f}
  GWP: {gwp:.2f} vs {best_gwp:.2f} ({gwp_vs_best:+.2f} kg)

{rag_block}{infeas_warning}{osc_warning}=== ACTION REQUIRED ===
{feedback}

Propose the NEXT mix to reduce GWP. Output ONLY the JSON object.\
"""


def build_system_prompt(spec: Spec, cfg: ExperimentConfig, few_shot: list) -> str:
    raw_lines = [f"  {v:<8} [{b['min']:9.3f}, {b['max']:9.3f}]" for v, b in spec.raw_b.items()]
    der_lines = [f"  {v:<10} [{b['min']:8.5f}, {b['max']:8.5f}]" for v, b in spec.der_b.items()]
    knowledge = build_knowledge(spec, cfg) if cfg.use_knowledge_table else \
        "GWP formula: GWP = PC*1.048 + FA*0.328 + SC*0.264 + CAGG*0.0037 + FAGG*0.0026\n"
    situation = SITUATION_RULES if cfg.use_situation_rules else ""

    few_shot_block = ""
    if cfg.use_few_shot and few_shot:
        parts = []
        for ex in few_shot:
            mix_str = "  ".join(f"{k}={ex[k]}" for k in RAW_VARS)
            parts.append(f"[{ex['label']}]\n  {mix_str}\n"
                         f"  -> 28d={ex['pred_28day']} MPa  GWP={ex['gwp']} kg CO₂/m³")
        few_shot_block = (f"REFERENCE MIXES FROM DATASET (all satisfy 28d >= {cfg.strength_min} MPa)\n"
                          + "=" * 70 + "\n" + "\n\n".join(parts))

    dur_line = (f"  4. predicted probability of passing the 28-day chloride test (< {spec.rcpt_limit} C) "
                f">= {spec.min_pass_prob}\n") if cfg.use_durability else ""
    return SYSTEM_PROMPT_TEMPLATE.format(
        strength_min=cfg.strength_min, durability_line=dur_line,
        densities=", ".join(f"{k} {v}" for k, v in spec.densities.items()),
        knowledge_block=knowledge, raw_bounds="\n".join(raw_lines), der_bounds="\n".join(der_lines),
        situation_block=situation, few_shot_block=few_shot_block)


FIRST_TURN_PLAIN = """\
Start optimisation. Propose an initial mix satisfying ALL constraints:
  - 28-day strength >= {strength_min} MPa
{first_dur}  - Vfinal in [0.95, 1.05] and all variable / ratio bounds satisfied
  - GWP as low as possible

Output ONLY the JSON object.\
"""


def build_first_turn(spec: Spec, cfg: ExperimentConfig) -> str:
    if not cfg.use_directional_hints:
        first_dur = (f"  - chloride pass probability >= {spec.min_pass_prob}\n" if cfg.use_durability else "")
        return FIRST_TURN_PLAIN.format(strength_min=cfg.strength_min, first_dur=first_dur)
    pool = _pool(spec, cfg)
    safe_pc = float(pool["PC"].quantile(0.25)) if len(pool) else spec.raw_b["PC"]["max"]
    first_dur = (f"  - chloride pass probability >= {spec.min_pass_prob}\n" if cfg.use_durability else "")
    return FIRST_TURN.format(strength_min=cfg.strength_min, safe_pc=safe_pc, first_dur=first_dur)


def _violations(spec: Spec, cfg: ExperimentConfig, mix: dict, preds: dict, feas: dict) -> list:
    """Facts only: which constraints are violated and by how much (no remedy)."""
    out = []
    if feas["str_v"]:
        out.append(f"  Strength {preds['28day']:.1f} MPa is below the {cfg.strength_min} MPa floor.")
    if feas["dur_v"]:
        out.append(f"  Chloride pass probability {preds['chloride_prob']:.2f} < {spec.min_pass_prob}.")
    if feas["phys_v"]:
        out.append(f"  Vfinal={feas['phys_v']['Vfinal']['val']:.3f} is outside [0.95, 1.05].")
    for v, info in feas["der_v"].items():
        out.append(f"  {v}={info['val']:.4f} violates [{info['min']:.4f},{info['max']:.4f}].")
    for v, info in feas["raw_v"].items():
        out.append(f"  {v}={info['val']:.3f} is outside [{info['min']:.3f},{info['max']:.3f}].")
    return out


def _advice(spec: Spec, cfg: ExperimentConfig, mix: dict, preds: dict, feas: dict) -> list:
    """Directional advice for each violated constraint (facts only when hints are off)."""
    if not cfg.use_directional_hints:
        return _violations(spec, cfg, mix, preds, feas)
    out = []
    if feas["str_v"]:
        out.append(f"  Strength {preds['28day']:.1f} MPa is below the {cfg.strength_min} MPa floor. "
                   "Reduce WATER first, then add ACC, then increase PC.")
    if feas["dur_v"]:
        out.append(f"  Chloride pass probability {preds['chloride_prob']:.2f} < {spec.min_pass_prob}. "
                   "Raise the SC share of the binder (aim SCM >= 50%) and lower WATER.")
    if feas["phys_v"]:
        vf = feas["phys_v"]["Vfinal"]["val"]
        b = spec.phys_b["Vfinal"]
        if vf < b["min"]:
            add = (b["min"] - vf) * 2700
            out.append(f"  Vfinal={vf:.3f} is too LOW (volume short). Add material volume: about +{add:.0f} kg of "
                       "aggregate (FAGG/CAGG) restores 0.95.")
        else:
            rem = (vf - b["max"]) * 2700
            out.append(f"  Vfinal={vf:.3f} is too HIGH (over-filled). Remove about {rem:.0f} kg of aggregate.")
    for v, info in feas["der_v"].items():
        out.append(f"  {v}={info['val']:.4f} violates [{info['min']:.4f},{info['max']:.4f}].")
    for v, info in feas["raw_v"].items():
        out.append(f"  {v}={info['val']:.3f} is outside [{info['min']:.3f},{info['max']:.3f}].")
    return out


def build_feedback(spec: Spec, cfg: ExperimentConfig, it: int, mix: dict, preds: dict, gwp: float,
                   feas: dict, trajectory: list, pool: pd.DataFrame = None) -> str:
    p28 = preds["28day"]
    str_status = "OK" if p28 >= cfg.strength_min else f"INFEASIBLE -- {p28 - cfg.strength_min:+.2f} MPa below floor"
    chloride_line = ""
    if cfg.use_durability:
        ok = "OK" if preds["chloride_prob"] >= spec.min_pass_prob else "INFEASIBLE"
        chloride_line = (f"  Chloride class : {preds['chloride_class']}   "
                         f"(feasible classes: Very Low, Low)\n"
                         f"  Chloride pass probability : {preds['chloride_prob']:.2f}   "
                         f"[need >= {spec.min_pass_prob}]  {ok}\n")
    phys = get_physics(spec, mix)
    vf_status = "OK" if not feas["phys_v"] else "VIOLATION [0.95, 1.05]"

    contribs = sorted([(k, mix.get(k, 0) * spec.gwp_factors[k]) for k in RAW_VARS], key=lambda x: -x[1])
    gwp_breakdown = "\n".join(f"  {k:<6} {mix.get(k, 0):8.2f} kg/m³ x {spec.gwp_factors[k]:.4f} = {c:6.2f} kg CO₂/m³"
                              for k, c in contribs if c > 0.05)
    dv = get_derived(mix)
    ratio_lines = []
    for v, b in spec.der_b.items():
        bad = v in feas["der_v"]
        ratio_lines.append(f"  {v:<10}= {dv[v]:.4f}  " + (f"VIOLATION [{b['min']:.4f},{b['max']:.4f}]" if bad else "OK"))

    feas_str = "FEASIBLE" if feas["feasible"] else "INFEASIBLE (" + ", ".join(
        list(feas["raw_v"]) + list(feas["der_v"]) + list(feas["phys_v"])
        + (["strength"] if feas["str_v"] else []) + (["chloride"] if feas["dur_v"] else [])) + ")"

    prev = trajectory[-2] if len(trajectory) >= 2 else None
    prev_gwp = prev["gwp"] if prev else gwp
    prev_28 = prev["pred_28day"] if prev else p28
    prev_iter = prev["iteration"] if prev else it
    gwp_change = round(gwp - prev_gwp, 2)
    if not cfg.use_directional_hints:
        gwp_trend = ""
    elif gwp_change < -0.5:
        gwp_trend = f"DECREASED {abs(gwp_change):.2f} kg/m³ -- good"
    elif gwp_change > 0.5:
        gwp_trend = f"INCREASED {gwp_change:.2f} kg/m³ -- WRONG DIRECTION"
    else:
        gwp_trend = "barely changed -- need a bigger move"

    feasible_traj = [r for r in trajectory if r["feasible"]]
    best = min(feasible_traj, key=lambda r: r["gwp"]) if feasible_traj else None
    bm = best if best else {**mix, "gwp": gwp, "pred_28day": p28, "iteration": it}
    str_margin = round(p28 - cfg.strength_min, 2)

    osc_warning = ""
    recent = trajectory[-cfg.anti_osc_window:]
    if len(recent) == cfg.anti_osc_window and all(
            abs(recent[i].get(v, 0) - recent[j].get(v, 0)) < cfg.anti_osc_tol
            for v in ["PC", "SC", "FA"] for i in range(len(recent)) for j in range(i + 1, len(recent))):
        osc_warning = ("*** OSCILLATION WARNING ***\nLast proposals nearly identical. "
                       "Change at least 2 ingredients by > 12 kg/m³.\n\n")
    infeas_warning = "" if feas["feasible"] else "*** INFEASIBLE — DO NOT REPEAT ***\n\n"

    rag_block = ""
    if cfg.rag_mode == "dynamic" and pool is not None:
        similar = retrieve_similar_mixes(spec, cfg, mix, pool)
        if similar:
            if cfg.rag_format == "text":
                lines = ["=== SIMILAR MIXES FROM DATASET (retrieved based on your current proposal) ===",
                         "These are real historical mixes closest to what you just proposed.",
                         "Study their GWP and strength outcomes to guide your next step:\n"]
                for i, s in enumerate(similar, 1):
                    tb = s["PC"] + s["FA"] + s["SC"]
                    lines.append(
                        f"  [{i}] A mix with {s['PC']:.0f} kg/m³ Portland cement, {s['SC']:.0f} kg/m³ slag cement, "
                        f"and {s['FA']:.0f} kg/m³ fly ash achieves {s['pred_28day']:.1f} MPa 28-day strength "
                        f"with GWP of {s['gwp']:.1f} kg CO₂/m³. Total binder: {tb:.0f} kg/m³, w/b ratio: "
                        f"{s['WATER'] / (tb + EPS):.2f}, FAGG: {s['FAGG']:.0f}, CAGG: {s['CAGG']:.0f}, "
                        f"WR_HR: {s['WR_HR']:.2f}, WR: {s['WR']:.2f}, ACC: {s['ACC']:.2f} kg/m³.\n")
            else:
                lines = ["SIMILAR MIXES FROM DATASET (k-NN retrieval based on your current proposal):"]
                for i, s in enumerate(similar, 1):
                    lines.append(f"  [{i}] " + "  ".join(f"{v}={s[v]}" for v in RAW_VARS))
                    lines.append(f"       -> 28d={s['pred_28day']} MPa  GWP={s['gwp']:.1f} kg/m³")
            rag_block = "\n".join(lines) + "\n\n"

    fb = _advice(spec, cfg, mix, preds, feas)
    if feas["feasible"] and cfg.use_directional_hints:
        if gwp_change > 0.5:
            fb.append("  GWP went UP — wrong direction. Swap PC->SC or reduce WATER.")
        elif abs(gwp_change) <= 0.5:
            fb.append("  GWP barely changed — make a BIGGER move (12-18 kg/m³ increments).")
        else:
            fb.append(f"  GWP decreased {abs(gwp_change):.2f} kg/m³ — keep going.")
        if str_margin > 8 and gwp > bm["gwp"] + 5:
            fb.append(f"  Over-engineered: {str_margin:.1f} MPa above floor. Reduce PC+SC or add aggregate.")
        elif str_margin < 2:
            fb.append("  Tight strength margin. Reduce WATER before cutting PC further.")
        if cfg.use_durability and preds["chloride_prob"] < spec.min_pass_prob + 0.05:
            fb.append("  Chloride probability is close to its limit; do not lower the SC share.")

    g = lambda k: bm.get(k, 0)
    d = lambda k: round(mix.get(k, 0) - bm.get(k, 0), 0)
    return FEEDBACK_TEMPLATE.format(
        it=it, max_it=cfg.max_iters, mix_json=json.dumps(mix, indent=4),
        p7=preds["7day"], p28=p28, p56=preds["56day"], str_status=str_status,
        chloride_line=chloride_line, vfinal=phys["Vfinal"], vf_status=vf_status,
        gwp_breakdown=gwp_breakdown, gwp=gwp, ratio_check="\n".join(ratio_lines), feas_str=feas_str,
        prev_iter=prev_iter, prev_gwp=prev_gwp, prev_28=prev_28, gwp_change=gwp_change, gwp_trend=gwp_trend,
        str_margin=str_margin, strength_min=cfg.strength_min, best_iter=bm.get("iteration", it),
        best_PC=g("PC"), best_SC=g("SC"), best_FA=g("FA"), best_FAGG=g("FAGG"), best_CAGG=g("CAGG"),
        best_WATER=g("WATER"), best_AEA=g("AEA"), best_WR_HR=g("WR_HR"), best_WR=g("WR"), best_ACC=g("ACC"),
        best_gwp=bm["gwp"], best_28=bm["pred_28day"],
        pc_diff=d("PC"), sc_diff=d("SC"), fa_diff=d("FA"), water_diff=d("WATER"),
        fagg_diff=d("FAGG"), cagg_diff=d("CAGG"), gwp_vs_best=round(gwp - bm["gwp"], 2),
        rag_block=rag_block, infeas_warning=infeas_warning, osc_warning=osc_warning,
        feedback="\n".join(fb))


# ─────────────────────────────────────────────────────────────
# 7. HELPERS
# ─────────────────────────────────────────────────────────────

def parse_json(text: str):
    text = re.sub(r"```(?:json)?", "", text or "").strip().rstrip("`").strip()
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
    thr = cfg.stag_threshold * best_gwp
    return all(recent[i - 1]["gwp"] - recent[i]["gwp"] < thr for i in range(1, len(recent)))


def build_restart_msg(spec: Spec, trajectory: list, ga_ref: dict, restart_num: int, hints: bool = True) -> str:
    feasible = [r for r in trajectory if r["feasible"]]
    top5 = sorted(feasible, key=lambda r: r["gwp"])[:5]
    top5_str = "".join(
        f"  #{i} iter={r['iteration']:2d}: PC={r['PC']:.0f} SC={r['SC']:.0f} FA={r['FA']:.0f} "
        f"WATER={r['WATER']:.0f} ACC={r['ACC']:.1f} -> GWP={r['gwp']:.2f} 28d={r['pred_28day']:.2f} MPa\n"
        for i, r in enumerate(top5, 1))
    center = {v: round(float(np.mean([r[v] for r in feasible[-10:]])), 1) for v in ["PC", "SC", "FA", "WATER"]}
    s = spec.stats
    if not hints:
        return f"""
=== RESTART #{restart_num} — STAGNATION ===
Search center (avg last 10): {center}
The last proposals show no improvement.

Top-5 best so far:
{top5_str}
Propose a mix that differs substantially from these.

Output ONLY the JSON object.
"""
    return f"""
=== RESTART #{restart_num} — STAGNATION ===
Search center (avg last 10): {center}
DO NOT propose similar mixes.

Top-5 best so far:
{top5_str}
Try ONE bold strategy:
  A: total binder below {s['binder_p5']:.0f} kg + high FAGG/CAGG (keep Vfinal in range) + WR_HR 3-5 + ACC 8-15
  B: high FA (60-120 kg) replacing SC as primary SCM
  C: very low WATER (lower quartile of the bounds) + WR_HR 4-7 kg
  D: PC+SC pure binary, FA=0, very low w/b

Output ONLY the JSON object.
"""


# ─────────────────────────────────────────────────────────────
# 8. METRICS
# ─────────────────────────────────────────────────────────────

def compute_metrics(spec: Spec, trajectory: list, ga_ref: dict, ga_evals: int, stats: dict) -> dict:
    """
    OGR  (best_llm_gwp - ga_gwp) / ga_gwp                       lower is better
    QER  (first_feasible_gwp - best_gwp) / n_surrogate_evals     higher is better
    Rcalls  n_surrogate_evals / ga_evals
    MCE  sum over variables of std(feasible values) / bound range
    Cost columns (api_calls, tokens, llm_time_s) are reported separately from Rcalls.
    """
    feasible = [r for r in trajectory if r["feasible"]]
    calls = stats["surrogate_evals"]
    base = {"total_iters": len(trajectory), "n_feasible": len(feasible), "surrogate_evals": calls,
            "rcalls": round(calls / ga_evals, 6) if ga_evals else float("nan"),
            "api_calls": stats["api_calls"], "prompt_tokens": stats["prompt_tokens"],
            "completion_tokens": stats["completion_tokens"], "llm_time_s": round(stats["llm_time_s"], 2),
            "parse_fails": stats["parse_fails"], "restarts": stats["restarts"]}
    if not feasible:
        return {**base, "OGR": float("nan"), "QER": float("nan"), "MCE": float("nan"),
                "best_gwp": float("nan"), "ga_gwp": ga_ref["gwp"] if ga_ref else float("nan")}
    best = min(feasible, key=lambda r: r["gwp"])
    ga_gwp = ga_ref["gwp"] if ga_ref else float("nan")
    mce = 0.0
    for v in RAW_VARS:
        vals = [r[v] for r in feasible]
        rng = spec.raw_b[v]["max"] - spec.raw_b[v]["min"]
        if len(vals) > 1 and rng > 0:
            mce += float(np.std(vals)) / rng
    return {**base,
            "OGR": round((best["gwp"] - ga_gwp) / ga_gwp, 4) if ga_ref else float("nan"),
            "QER": round((feasible[0]["gwp"] - best["gwp"]) / calls, 4) if calls else float("nan"),
            "MCE": round(mce, 4), "best_gwp": best["gwp"], "ga_gwp": ga_gwp,
            "gwp_gap": round(best["gwp"] - ga_gwp, 2) if ga_ref else float("nan"),
            "convergence_iter": best["iteration"], "str_margin_at_best": best["str_margin"],
            "feasibility_rate": round(len(feasible) / max(stats["surrogate_evals"], 1), 4)}


# ─────────────────────────────────────────────────────────────
# 9. LLM LOOP
# ─────────────────────────────────────────────────────────────

def make_gemini_chat_factory(cfg: ExperimentConfig):
    from google import genai
    from google.genai import types as genai_types
    client = genai.Client(api_key=cfg.gemini_api_key)

    def factory(temperature: float, system_prompt: str):
        return client.chats.create(
            model=cfg.gemini_model,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt, temperature=temperature,
                max_output_tokens=cfg.max_output_tokens))
    return factory


def run_llm(spec: Spec, cfg: ExperimentConfig, ga_ref: dict, few_shot: list,
            on_iter=None, chat_factory=None, verbose: bool = True) -> dict:
    """
    Run the LLM optimizer. `max_iters` counts FEASIBLE proposals; infeasible ones trigger up to
    5 repair attempts each. Every proposal (feasible or not) costs one surrogate evaluation.
    `chat_factory(temperature, system_prompt)` returns an object with send_message(text) -> .text;
    the default builds a Gemini chat. Returns {"trajectory", "attempts", "stats"}.
    """
    chat_factory = chat_factory or make_gemini_chat_factory(cfg)
    sys_prompt = build_system_prompt(spec, cfg, few_shot)
    pool = None
    if cfg.rag_mode == "dynamic":
        pool = _pool(spec, cfg)
        if cfg.rag_low_gwp_half:
            pool = pool[pool["gwp"] <= pool["gwp"].median()]   # neighbours come from the lower-GWP half
    stats = {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "llm_time_s": 0.0,
             "surrogate_evals": 0, "parse_fails": 0, "restarts": 0}

    def send(chat, text):
        t0 = time.time()
        resp = chat.send_message(text)
        stats["llm_time_s"] += time.time() - t0
        stats["api_calls"] += 1
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            stats["prompt_tokens"] += getattr(um, "prompt_token_count", 0) or 0
            stats["completion_tokens"] += getattr(um, "candidates_token_count", 0) or 0
        return resp

    def evaluate(raw_mix):
        mix, notes = clip_mix(raw_mix, spec)
        preds = predict(spec, mix)
        stats["surrogate_evals"] += 1
        feas = check_feasibility(spec, mix, preds, cfg.strength_min, cfg.use_durability)
        return mix, notes, preds, feas

    def record(it, mode, reasoning, mix, preds, feas):
        tb = mix["PC"] + mix["FA"] + mix["SC"]
        g = compute_gwp(mix, spec)
        return {"iteration": it, "mode": mode, "reasoning": reasoning, **mix,
                **{k: round(v, 5) for k, v in get_derived(mix).items()},
                "total_binder": round(tb, 3), "Vfinal": round(get_physics(spec, mix)["Vfinal"], 4),
                "pred_7day": preds["7day"], "pred_28day": preds["28day"], "pred_56day": preds["56day"],
                "chloride_prob": preds["chloride_prob"], "chloride_class": preds["chloride_class"], "gwp": g,
                "gwp_gap": round(g - ga_ref["gwp"], 2) if ga_ref else float("nan"),
                "str_margin": round(preds["28day"] - cfg.strength_min, 2),
                "api_calls_so_far": stats["api_calls"], "surrogate_evals_so_far": stats["surrogate_evals"],
                "feasible": feas["feasible"]}

    def log(msg):
        if verbose:
            print(msg)

    cur_temp = cfg.temperature
    chat = chat_factory(cur_temp, sys_prompt)
    trajectory, attempts = [], []
    restart_count = 0
    cur_mix, cur_preds = None, None
    cur_feas, clip_notes = {"feasible": True}, []
    it, total_attempts, consec_fail = 0, 0, 0
    max_attempts = cfg.max_iters * cfg.max_attempt_factor

    log(f"\n{'=' * 62}\n  {cfg.name} | {cfg.gemini_model} | 28d >= {cfg.strength_min} MPa | "
        f"durability={cfg.use_durability}\n{'=' * 62}")

    while it < cfg.max_iters and total_attempts < max_attempts:
        total_attempts += 1
        mode = "exploit"
        if it > 1 and restart_count < cfg.max_restarts and detect_stagnation(trajectory, cfg):
            restart_count += 1
            stats["restarts"] = restart_count
            mode = f"RESTART#{restart_count}"
            cur_temp = cfg.restart_temp
            chat = chat_factory(cur_temp, sys_prompt)
            user_msg = build_restart_msg(spec, trajectory, ga_ref, restart_count, cfg.use_directional_hints)
        elif it == 0 and cur_mix is None:
            user_msg = build_first_turn(spec, cfg)
        else:
            if cur_temp != cfg.temperature:
                cur_temp = cfg.temperature
                chat = chat_factory(cur_temp, sys_prompt)
                feas_sf = [r for r in trajectory if r["feasible"]]
                if feas_sf:
                    b = min(feas_sf, key=lambda r: r["gwp"])
                    try:
                        send(chat, f"Resume. Best so far: PC={b['PC']:.0f} SC={b['SC']:.0f} FA={b['FA']:.0f} "
                                   f"WATER={b['WATER']:.0f} GWP={b['gwp']:.2f} 28d={b['pred_28day']:.2f} MPa. "
                                   "Improve from here. Output ONLY the JSON object.")
                    except Exception:
                        pass
            user_msg = build_feedback(spec, cfg, it, cur_mix, cur_preds, compute_gwp(cur_mix, spec),
                                      cur_feas, trajectory, pool)
            if clip_notes:
                user_msg = "*** BOUNDS VIOLATION ***\n" + "\n".join(clip_notes) + \
                           "\nStay within the bounds in the system prompt.\n\n" + user_msg

        try:
            raw_text = send(chat, user_msg).text
        except Exception as exc:
            err = str(exc)
            wait = 60 if "429" in err else 15
            log(f"  API error ({err[:60]}) — waiting {wait}s ...")
            time.sleep(wait)
            continue

        parsed = parse_json(raw_text)
        if parsed is None or "mix" not in parsed:
            stats["parse_fails"] += 1
            try:
                parsed = parse_json(send(chat, "Output ONLY the JSON object with keys 'reasoning' and 'mix'.").text)
            except Exception:
                parsed = None
            if parsed is None or "mix" not in parsed:
                continue

        reasoning = parsed.get("reasoning", "")
        mix, clip_notes, preds, feas = evaluate(parsed["mix"])

        retry = 0
        while not feas["feasible"] and retry < 5:
            retry += 1
            consec_fail += 1
            cur_mix, cur_preds, cur_feas = mix, preds, feas
            rec = record(it, f"{mode}[retry {retry}]", reasoning, mix, preds, feas)
            attempts.append(rec)
            if on_iter:
                on_iter(rec)
            log(f"  {'--':>4}  {preds['28day']:8.2f}  {rec['gwp']:8.2f}  infeasible  {mode}[retry {retry}]")
            if consec_fail >= 30:
                break
            retry_msg = (f"ATTEMPT {retry}/5 — still INFEASIBLE.\n"
                         f"Result: 28d={preds['28day']:.1f} MPa  GWP={rec['gwp']:.1f}  "
                         f"Vfinal={rec['Vfinal']:.3f}  chloride_prob={preds['chloride_prob']:.2f}\n\n"
                         "Problems to fix:\n" + "\n".join(_advice(spec, cfg, mix, preds, feas))
                         + "\n\nFix the violated constraints first. Output ONLY the JSON object.")
            try:
                rp = parse_json(send(chat, retry_msg).text)
            except Exception:
                time.sleep(3)
                break
            if rp and "mix" in rp:
                mix, clip_notes, preds, feas = evaluate(rp["mix"])
                reasoning = rp.get("reasoning", reasoning)
            else:
                stats["parse_fails"] += 1
                break

        cur_mix, cur_preds, cur_feas = mix, preds, feas
        if not feas["feasible"]:
            if consec_fail >= 30:
                log("  [!] 30 consecutive infeasible attempts — aborting.")
                break
            continue

        consec_fail = 0
        it += 1
        rec = record(it, mode, reasoning, mix, preds, feas)
        trajectory.append(rec)
        attempts.append(rec)
        if on_iter:
            on_iter(rec)
        gap = f"{rec['gwp_gap']:+.2f}" if ga_ref else "n/a"
        log(f"  {it:4d}  {preds['28day']:8.2f}  {rec['gwp']:8.2f}  {gap:>9}  {mode}")

    log(f"\n  Restarts: {restart_count}  Parse fails: {stats['parse_fails']}  API calls: {stats['api_calls']}"
        f"  Surrogate evals: {stats['surrogate_evals']}  Feasible iters: {it}")
    return {"trajectory": trajectory, "attempts": attempts, "stats": stats}


# ─────────────────────────────────────────────────────────────
# 10. SAVE RESULTS
# ─────────────────────────────────────────────────────────────

def save_run(out_dir: str, cfg: ExperimentConfig, result: dict, ga_ref: dict, metrics: dict) -> None:
    """One run -> out_dir/{config.json, trajectory.csv, attempts.csv, metrics.json, report.txt}."""
    os.makedirs(out_dir, exist_ok=True)
    c = asdict(cfg)
    c["gemini_api_key"] = ""
    c["saved_at"] = datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(c, f, indent=2)
    for name in ("trajectory", "attempts"):
        if result[name]:
            pd.DataFrame(result[name]).to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)
    lines = [f"Experiment: {cfg.name}   {cfg.description}",
             f"strength >= {cfg.strength_min} MPa   durability={cfg.use_durability}   model={cfg.gemini_model}",
             "-" * 60] + [f"  {k:<22}: {v}" for k, v in metrics.items()] + ["-" * 60, "Reasoning log:"]
    for r in result["trajectory"]:
        lines.append(f"[{r['iteration']}] GWP={r['gwp']:.2f} 28d={r['pred_28day']:.1f} ({r['mode']}): {r['reasoning']}")
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

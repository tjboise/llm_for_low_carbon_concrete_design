"""
run_experiment.py
=================
Runs all experiments of the paper and stores them grouped by scenario and method.

Scenarios : 28-day strength floor {45, 50, 55} MPa  x  durability constraint {off, on}
Methods   : baseline, no_knowledge, zero_shot, rag_tabular, rag_text
Repeats   : 5 independent runs per (scenario, method); GA reference uses 5 seeds per scenario

Layout
  results/ga/<scenario>/seed_<k>.json, reference.json
  results/llm/<scenario>/<method>/run_<k>/{config.json, trajectory.csv, attempts.csv, metrics.json, report.txt}
  results/summary/all_runs.csv

Usage
  python run_experiment.py --stage ga                       # GA reference (no API calls)
  python run_experiment.py --stage llm --workers 6          # all LLM runs (resumes; skips finished runs)
  python run_experiment.py --stage llm --scenario s50_dur --method baseline --repeats 1
  python run_experiment.py --stage llm --dry-run            # stub LLM, no API calls (pipeline test)
  python run_experiment.py --stage summary
"""

import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import optimizer_core as oc

load_dotenv()

RESULTS = os.path.join(oc.ROOT, "results")
STRENGTHS = [45, 50, 55]
GEMINI_MODEL = "gemini-2.5-flash-lite"
N_REPEATS = 5
MAX_ITERS = 30
STAG_FACTOR = 1.3     # stagnation is only declared once GWP < 1.3 x the GA reference GWP

METHODS = {
    "baseline":     dict(use_knowledge_table=True, use_situation_rules=True, use_few_shot=True, rag_mode="static"),
    "no_knowledge": dict(use_knowledge_table=False, use_situation_rules=True, use_few_shot=True, rag_mode="static",
                         use_directional_hints=False),
    "zero_shot":    dict(use_knowledge_table=False, use_situation_rules=False, use_few_shot=False, rag_mode="none"),
    "rag_tabular":  dict(use_knowledge_table=True, use_situation_rules=True, use_few_shot=True,
                         rag_mode="dynamic", rag_format="tabular"),
    "rag_text":     dict(use_knowledge_table=True, use_situation_rules=True, use_few_shot=True,
                         rag_mode="dynamic", rag_format="text"),
}


def scenarios():
    return {f"s{s}_{'dur' if d else 'nodur'}": (s, d) for s in STRENGTHS for d in (False, True)}


def ga_dir(scn):
    return os.path.join(RESULTS, "ga", scn)


# ─────────────────────────────────────────────────────────────
# GA STAGE
# ─────────────────────────────────────────────────────────────

def run_ga_stage(spec, selected):
    for scn, (smin, dur) in scenarios().items():
        if selected and scn not in selected:
            continue
        cfg = oc.ExperimentConfig(name="ga", strength_min=smin, use_durability=dur)
        d = ga_dir(scn)
        os.makedirs(d, exist_ok=True)
        runs = []
        for seed in cfg.ga_seeds:
            f = os.path.join(d, f"seed_{seed}.json")
            if os.path.exists(f):
                runs.append(json.load(open(f)))
                continue
            r = oc.run_ga(spec, cfg, seed)
            json.dump(r, open(f, "w"), indent=2)
            runs.append(r)
            g = r["best"]["gwp"] if r["best"] else None
            print(f"[GA] {scn} seed={seed}: GWP={g}  evals={r['n_evals']}  {r['wall_time_s']}s")
        ref = oc.select_ga_reference(runs)
        if ref is None:
            print(f"[GA] {scn}: NO FEASIBLE SOLUTION")
            continue
        oc.verify_solution(spec, ref, smin, dur)           # must satisfy every constraint
        gw = [r["best"]["gwp"] for r in runs if r["best"]]
        out = {"reference": ref, "n_feasible_seeds": len(gw), "gwp_mean": float(np.mean(gw)),
               "gwp_std": float(np.std(gw)), "n_evals_per_run": runs[0]["n_evals"],
               "wall_time_mean_s": float(np.mean([r["wall_time_s"] for r in runs]))}
        json.dump(out, open(os.path.join(d, "reference.json"), "w"), indent=2)
        print(f"[GA] {scn}: reference GWP={ref['gwp']:.2f}  mean={out['gwp_mean']:.2f}+-{out['gwp_std']:.2f}")


def load_reference(scn):
    f = os.path.join(ga_dir(scn), "reference.json")
    if not os.path.exists(f):
        raise FileNotFoundError(f"Run --stage ga first ({f})")
    return json.load(open(f))


# ─────────────────────────────────────────────────────────────
# LLM STAGE
# ─────────────────────────────────────────────────────────────

class StubChat:
    """Dry-run chat: proposes random perturbations of dataset mixes. No API calls."""
    def __init__(self, spec, rng):
        self.spec, self.rng = spec, rng

    def send_message(self, text):
        row = self.spec.df.sample(1, random_state=self.rng.randint(0, 10**6)).iloc[0]
        mix = {v: float(row[v]) * self.rng.uniform(0.9, 1.1) for v in oc.RAW_VARS}
        body = json.dumps({"reasoning": "stub", "mix": mix})
        return type("R", (), {"text": body, "usage_metadata": None})()


def run_one(spec, scn, method, rep, dry_run, lock):
    smin, dur = scenarios()[scn]
    out_dir = os.path.join(RESULTS, "llm", scn, method, f"run_{rep}")
    if os.path.exists(os.path.join(out_dir, "metrics.json")):
        return f"skip {scn}/{method}/run_{rep}"
    ref = load_reference(scn)
    ga_ref = ref["reference"]
    cfg = oc.ExperimentConfig(
        name=method, description=f"{method} | {scn}", strength_min=smin, use_durability=dur,
        max_iters=MAX_ITERS, gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        gemini_model=GEMINI_MODEL, stag_min_best=STAG_FACTOR * ga_ref["gwp"], **METHODS[method])
    few_shot = oc.select_few_shot(spec, cfg) if cfg.rag_mode == "static" and cfg.use_few_shot else []
    factory = None
    if dry_run:
        rng = random.Random(rep)
        factory = lambda temp, sp: StubChat(spec, rng)
        out_dir = os.path.join(RESULTS, "_dryrun", scn, method, f"run_{rep}")
    res = oc.run_llm(spec, cfg, ga_ref, few_shot, chat_factory=factory, verbose=False)
    metrics = oc.compute_metrics(spec, res["trajectory"], ga_ref, ref["n_evals_per_run"], res["stats"])
    metrics.update({"scenario": scn, "method": method, "run": rep, "strength_min": smin,
                    "durability": dur, "model": cfg.gemini_model})
    # every reported best solution must satisfy all constraints
    if res["trajectory"]:
        best = min(res["trajectory"], key=lambda r: r["gwp"])
        oc.verify_solution(spec, {v: best[v] for v in oc.RAW_VARS}, smin, dur)
    with lock:
        oc.save_run(out_dir, cfg, res, ga_ref, metrics)
    return f"done {scn}/{method}/run_{rep}  best={metrics['best_gwp']}  OGR={metrics['OGR']}  api={metrics['api_calls']}"


def run_llm_stage(spec, selected_scn, selected_methods, repeats, workers, dry_run):
    jobs = [(scn, m, r) for scn in scenarios() if not selected_scn or scn in selected_scn
            for m in METHODS if not selected_methods or m in selected_methods
            for r in range(1, repeats + 1)]
    print(f"{len(jobs)} runs, {workers} workers" + ("  [DRY RUN]" if dry_run else ""))
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, spec, s, m, r, dry_run, lock): (s, m, r) for s, m, r in jobs}
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as e:                      # keep going; report the failure
                print(f"FAILED {futs[f]}: {type(e).__name__}: {e}", flush=True)


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────

def summarise():
    rows = []
    base = os.path.join(RESULTS, "llm")
    for scn in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        for m in sorted(os.listdir(os.path.join(base, scn))):
            for run in sorted(os.listdir(os.path.join(base, scn, m))):
                f = os.path.join(base, scn, m, run, "metrics.json")
                if os.path.exists(f):
                    rows.append(json.load(open(f)))
    os.makedirs(os.path.join(RESULTS, "summary"), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "summary", "all_runs.csv"), index=False)
    if len(df):
        agg = df.groupby(["scenario", "method"])[["OGR", "QER", "MCE", "best_gwp", "rcalls", "api_calls",
                                                  "prompt_tokens", "completion_tokens", "llm_time_s"]].agg(["mean", "std"])
        agg.to_csv(os.path.join(RESULTS, "summary", "mean_std.csv"))
        print(agg.round(4).to_string())
    print(f"{len(df)} runs summarised")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["ga", "llm", "summary", "all"], default="all")
    ap.add_argument("--scenario", nargs="*")
    ap.add_argument("--method", nargs="*")
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    spec = oc.Spec()
    if a.stage in ("ga", "all"):
        run_ga_stage(spec, a.scenario)
    if a.stage in ("llm", "all"):
        if not a.dry_run and not os.environ.get("GEMINI_API_KEY"):
            sys.exit("GEMINI_API_KEY missing (set it in .env)")
        run_llm_stage(spec, a.scenario, a.method, a.repeats, a.workers, a.dry_run)
    if a.stage in ("summary", "all"):
        summarise()


if __name__ == "__main__":
    main()

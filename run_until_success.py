"""
run_until_success.py
====================
Main experiment (28-day strength >= 50 MPa, chloride constraint on): for every method, keep running until five
successful runs exist. A run is successful if it has at least MIN_FEASIBLE feasible solutions.

Selection rule (fixed in advance): the first five successful runs of each method, in run-index order, are used for
figures and tables. All runs, successful or not, stay in results/llm/ and are counted in the success rate.
Output: results/summary/main_s50_dur_selection.csv

Usage: python run_until_success.py [--workers 6] [--max-runs 30]
"""
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import optimizer_core as oc
import run_experiment as rx

SCENARIO = "s50_dur"
TARGET_SUCCESSES = 5
MIN_FEASIBLE = 5


def run_dir(method, run):
    return os.path.join(rx.RESULTS, "llm", SCENARIO, method, f"run_{run}")


def existing_runs(method):
    base = os.path.join(rx.RESULTS, "llm", SCENARIO, method)
    out = []
    if os.path.isdir(base):
        for r in sorted(os.listdir(base), key=lambda x: int(x.split("_")[1])):
            f = os.path.join(base, r, "metrics.json")
            if os.path.exists(f):
                out.append((int(r.split("_")[1]), json.load(open(f))))
    return out


def successes(method):
    return [(i, m) for i, m in existing_runs(method) if m["n_feasible"] >= MIN_FEASIBLE]


def write_selection():
    rows = []
    for m in rx.METHODS:
        runs = existing_runs(m)
        succ = [(i, x) for i, x in runs if x["n_feasible"] >= MIN_FEASIBLE]
        chosen = {i for i, _ in succ[:TARGET_SUCCESSES]}
        last = max((i for i in chosen), default=0)
        n_attempts = len([1 for i, _ in runs if i <= last])
        for i, x in runs:
            rows.append({"method": m, "run": i, "n_feasible": x["n_feasible"], "OGR": x["OGR"],
                         "best_gwp": x["best_gwp"], "success": x["n_feasible"] >= MIN_FEASIBLE,
                         "selected": i in chosen, "attempts_until_5_successes": n_attempts})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(rx.RESULTS, "summary"), exist_ok=True)
    df.to_csv(os.path.join(rx.RESULTS, "summary", "main_s50_dur_selection.csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-runs", type=int, default=30)
    a = ap.parse_args()
    spec = oc.Spec()
    lock = threading.Lock()
    while True:
        jobs = []
        for m in rx.METHODS:
            runs = existing_runs(m)
            need = TARGET_SUCCESSES - len(successes(m))
            nxt = (max((i for i, _ in runs), default=0)) + 1
            for k in range(max(need, 0)):
                if nxt + k <= a.max_runs:
                    jobs.append((m, nxt + k))
        if not jobs:
            break
        print(f"batch: {len(jobs)} runs -> {jobs}", flush=True)
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(rx.run_one, spec, SCENARIO, m, r, False, lock) for m, r in jobs]
            for f in futs:
                try:
                    print(f.result(), flush=True)
                except Exception as e:
                    print("FAILED:", type(e).__name__, e, flush=True)
    df = write_selection()
    print(df.groupby("method").agg(runs=("run", "count"), successes=("success", "sum"),
                                   selected=("selected", "sum")).to_string())


if __name__ == "__main__":
    main()

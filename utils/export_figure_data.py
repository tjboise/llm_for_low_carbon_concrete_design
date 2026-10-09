"""
export_figure_data.py
=====================
Writes the exact runs and the plotted numbers behind every figure, so the figures can be reproduced and checked.

Output
  results/tables/figure_runs.csv      one row per (figure, method, run): used in the figure or not, with its metrics
  results/tables/figure_summary.csv   mean and std per (figure, method) as plotted

Run selection (same rules as utils/make_figures.py)
  Figures 3, 4        scenario s50_nodur, method baseline, the run with the lowest OGR among runs 3-7
  Figures 5, 6, 7     scenario s50_nodur, runs 3-7 of every method in the figure
  Figure 7 (exploratory, not for publication)  baseline runs 2,4; rag_tabular runs 3,7; rag_text runs 2,7
  Other multi-panel figures (fig_knowledge, fig_fewshot, fig_rag, fig_durability, fig_sensitivity)  runs 3-7 of every cell

Usage: python utils/export_figure_data.py
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
SCN = "s50_nodur"
METRICS = ["rcalls", "gwp_gap", "OGR", "QER", "best_gwp", "n_feasible", "surrogate_evals"]

FIGS = {
    "fig5": {"methods": ["no_knowledge", "baseline"], "runs": range(3, 8)},
    "fig6": {"methods": ["zero_shot", "baseline"], "runs": range(3, 8)},
    "fig7": {"methods": ["baseline", "rag_tabular", "rag_text"], "runs": range(3, 8)},
    "fig7_exploratory_selected_runs": {"methods": ["baseline", "rag_tabular", "rag_text"],
                                       "runs": {"baseline": [2, 4], "rag_tabular": [3, 7], "rag_text": [2, 7]}},
}


def main():
    allr = pd.read_csv(os.path.join(RES, "summary", "all_runs.csv"))
    allr = allr[allr["scenario"] == SCN]
    rows, summ = [], []

    # Figures 3 and 4: the best run among runs 3-7
    b = allr[(allr["method"] == "baseline") & allr["run"].between(3, 7) & allr["OGR"].notna()]
    best_run = int(b.loc[b["OGR"].idxmin(), "run"])
    ga = json.load(open(os.path.join(RES, "ga", SCN, "reference.json")))
    for fig in ("fig3a", "fig3b", "fig4"):
        r = allr[(allr["method"] == "baseline") & (allr["run"] == best_run)].iloc[0]
        rows.append({"figure": fig, "scenario": SCN, "method": "baseline", "run": best_run, "used": True,
                     **{k: r[k] for k in METRICS},
                     "data_file": f"results/llm/{SCN}/baseline/run_{best_run}/trajectory.csv",
                     "ga_reference_gwp": ga["reference"]["gwp"]})

    for fig, spec in FIGS.items():
        for m in spec["methods"]:
            used_runs = spec["runs"][m] if isinstance(spec["runs"], dict) else list(spec["runs"])
            sub = allr[allr["method"] == m]
            for _, r in sub.iterrows():
                rows.append({"figure": fig, "scenario": SCN, "method": m, "run": int(r["run"]),
                             "used": int(r["run"]) in used_runs, **{k: r[k] for k in METRICS},
                             "data_file": f"results/llm/{SCN}/{m}/run_{int(r['run'])}/metrics.json"})
            u = sub[sub["run"].isin(used_runs)]
            entry = {"figure": fig, "scenario": SCN, "method": m, "runs_used": ",".join(map(str, used_runs)),
                     "n_runs": len(u), "n_with_feasible_solution": int(u["OGR"].notna().sum())}
            for k in ["rcalls", "gwp_gap", "OGR", "QER"]:
                v = u[k].dropna()
                entry[f"{k}_mean"] = round(float(v.mean()), 4) if len(v) else np.nan
                entry[f"{k}_std"] = round(float(v.std(ddof=1)), 4) if len(v) > 1 else np.nan
            summ.append(entry)

    os.makedirs(os.path.join(RES, "tables"), exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(RES, "tables", "figure_runs.csv"), index=False)
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(RES, "tables", "figure_summary.csv"), index=False)
    print(f"Figures 3/4 use run {best_run} (OGR {b['OGR'].min():.4f}); GA reference {ga['reference']['gwp']}")
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()

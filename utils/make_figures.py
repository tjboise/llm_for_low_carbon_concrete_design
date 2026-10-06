"""
make_figures.py
===============
All paper figures from results/ with one consistent style (large fonts, SI units).

  fig3a_gwp_iterations.png, fig3b_strength_iterations.png   iteration history of the best run (lowest OGR), old Figure 3 style
  fig_knowledge.png       baseline vs no-knowledge        (OGR, GWP gap, QER, Rcalls)
  fig_fewshot.png         zero-shot vs few-shot (baseline)
  fig_rag.png             no-RAG (baseline) vs RAG tabular vs RAG text
  fig_durability.png      best GWP without / with the chloride constraint (GA and LLM baseline)
  fig_models.png          model comparison (R2) from results/model_comparison
  fig_sensitivity.png     robustness of best solutions to prediction error and ingredient variability

Bars are mean +- std over the runs that found a feasible solution; the number of such runs is printed on each bar.
Usage: python utils/make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "pictures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14, "xtick.labelsize": 13,
                     "ytick.labelsize": 13, "legend.fontsize": 12, "figure.dpi": 150, "axes.spines.top": True,
                     "axes.spines.right": True})
COL = {"baseline": "#2a6f97", "no_knowledge": "#e07a5f", "zero_shot": "#9a8c98", "rag_tabular": "#81b29a",
       "rag_text": "#3d405b", "GA": "#222222"}
LABEL = {"baseline": "Baseline", "no_knowledge": "No knowledge", "zero_shot": "Zero-shot",
         "rag_tabular": "RAG (tabular)", "rag_text": "RAG (text)"}
GWP_UNIT = "kg CO$_2$-eq/m$^3$"


def load_runs():
    return pd.read_csv(os.path.join(RES, "summary", "all_runs.csv"))


def bars(df, methods, title, fname, scenarios_sets):
    """Grouped bars: x = strength floor, one panel per metric, one figure row per durability setting."""
    metrics = [("OGR", "OGR"), ("gwp_gap", f"GWP gap ({GWP_UNIT})"), ("QER", f"QER ({GWP_UNIT} per call)"),
               ("rcalls", "$R_{calls}$")]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
    for r, dur in enumerate([False, True]):
        sub = df[df["durability"] == dur]
        for c, (m, ylabel) in enumerate(metrics):
            ax = axes[r, c]
            w = 0.8 / len(methods)
            for i, meth in enumerate(methods):
                means, stds, ns = [], [], []
                for s in [45, 50, 55]:
                    v = sub[(sub["method"] == meth) & (sub["strength_min"] == s)][m].dropna()
                    means.append(v.mean() if len(v) else np.nan)
                    stds.append(v.std() if len(v) > 1 else 0)
                    ns.append(len(v))
                x = np.arange(3) + i * w - 0.4 + w / 2
                ax.bar(x, means, w, yerr=stds, capsize=3, color=COL[meth], label=LABEL[meth])
                if m == "OGR":
                    for xi, mi, n in zip(x, means, ns):
                        if not np.isnan(mi):
                            ax.text(xi, mi, f"{n}", ha="center", va="bottom", fontsize=9)
            ax.set_xticks(range(3))
            ax.set_xticklabels(["45", "50", "55"])
            ax.set_xlabel("28-day strength floor (MPa)")
            ax.set_ylabel(ylabel)
            if c == 0:
                ax.set_title(("With" if dur else "Without") + " chloride constraint", loc="left")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(title + "   (numbers above OGR bars = runs with a feasible solution, of 5)", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)


def trajectory(scn="s50_nodur", method="baseline"):
    """Figure 3 (a, b): iteration history of the run closest to the GA reference (lowest OGR)."""
    ref = json.load(open(os.path.join(RES, "ga", scn, "reference.json")))["reference"]
    base = os.path.join(RES, "llm", scn, method)
    best_run, best_ogr = None, np.inf
    for run in sorted(os.listdir(base)):
        m = json.load(open(os.path.join(base, run, "metrics.json")))
        if m.get("OGR") == m.get("OGR") and m["OGR"] < best_ogr:
            best_run, best_ogr = run, m["OGR"]
    t = pd.read_csv(os.path.join(base, best_run, "trajectory.csv"))
    smin = int(scn.split("_")[0][1:])
    it, g = t["iteration"].values, t["gwp"].values
    restart = t["mode"].astype(str).str.startswith("RESTART").values
    best_so_far = np.minimum.accumulate(g)
    b_i = int(np.argmin(g))
    purple, green, orange, red, blue = "#5b3fb5", "#1a9c78", "#ff8c00", "#e05555", "#2f80e0"

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.axhline(ref["gwp"], color=red, ls="--", lw=1.4, label=f"GA reference ({ref['gwp']:.1f} kg)")
    ax.plot(it, g, color=purple, alpha=0.35, lw=1.2, zorder=1)
    ax.scatter(it[~restart], g[~restart], s=70, color=purple, edgecolor="white", zorder=3, label="LLM solution")
    if restart.any():
        ax.scatter(it[restart], g[restart], s=130, marker="D", color=orange, edgecolor="white", zorder=4,
                   label="Restart")
    ax.plot(it, best_so_far, color=green, lw=3, zorder=2, label="Best GWP so far")
    ax.annotate(f"Best: {g[b_i]:.1f} kg\n(iter {it[b_i]})", (it[b_i], g[b_i]), xytext=(it[b_i] + 1, g[b_i] + 8),
                color=green, fontsize=11, arrowprops=dict(arrowstyle="-", color=green))
    ax.set_xlabel("Iteration")
    ax.set_ylabel(f"GWP ({GWP_UNIT})")
    ax.set_xlim(0, max(31, it.max() + 1))
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3a_gwp_iterations.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    s = t["pred_28day"].values
    ax.axhline(smin, color=red, ls="--", lw=1.4, label="Target strength")
    ax.axhspan(smin - 5, smin, color=red, alpha=0.06)
    ax.plot(it, s, color=blue, alpha=0.4, lw=1.5, zorder=1)
    ax.scatter(it, s, s=70, color=blue, edgecolor="white", zorder=3, label="Predicted 28-day strength")
    ax.set_ylim(smin - 5, max(s.max(), smin) + 4)
    ax.set_xlim(0, max(31, it.max() + 1))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Predicted 28-day strength (MPa)")
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3b_strength_iterations.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 3 from {scn}/{method}/{best_run}: OGR={best_ogr:.3f}, best GWP={g[b_i]:.2f}, GA={ref['gwp']:.2f}")


def durability(df):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    xs = np.arange(3)
    for k, (dur, hatch) in enumerate([(False, ""), (True, "//")]):
        ga, llm, llm_sd = [], [], []
        for s in [45, 50, 55]:
            scn = f"s{s}_{'dur' if dur else 'nodur'}"
            ga.append(json.load(open(os.path.join(RES, "ga", scn, "reference.json")))["reference"]["gwp"])
            v = df[(df["scenario"] == scn) & (df["method"] == "baseline")]["best_gwp"].dropna()
            llm.append(v.mean())
            llm_sd.append(v.std())
        ax.bar(xs + (k - 0.5) * 0.38 - 0.1, ga, 0.18, color=COL["GA"], hatch=hatch, label=f"GA {'with' if dur else 'without'}")
        ax.bar(xs + (k - 0.5) * 0.38 + 0.1, llm, 0.18, yerr=llm_sd, capsize=3, color=COL["baseline"], hatch=hatch,
               label=f"LLM {'with' if dur else 'without'}")
    ax.set_xticks(xs)
    ax.set_xticklabels(["45", "50", "55"])
    ax.set_xlabel("28-day strength floor (MPa)")
    ax.set_ylabel(f"Best GWP ({GWP_UNIT})")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_durability.png"), bbox_inches="tight")
    plt.close(fig)


def models():
    f = os.path.join(RES, "model_comparison", "raw.csv")
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    setups = [("independent", "Independent"), ("chain_true_input", "Chain, true previous stage"),
              ("chain_pred_input", "Chain, predicted previous stage")]
    mods = ["RandomForest", "XGBoost", "CatBoost", "MLP"]
    for ax, age in zip(axes, ["7day", "28day", "56day"]):
        for i, (sk, sl) in enumerate(setups):
            g = d[(d["age"] == age) & (d["setup"] == sk)].groupby("model")["r2"].agg(["mean", "std"]).reindex(mods)
            ax.bar(np.arange(4) + (i - 1) * 0.27, g["mean"].clip(lower=-0.2), 0.27, yerr=g["std"], capsize=3, label=sl)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["RF", "XGB", "CatBoost", "MLP"])
        ax.set_title(age.replace("day", "-day"), loc="left")
        ax.set_ylim(-0.2, 1.0)
    axes[0].set_ylabel("R$^2$ (mean $\\pm$ std, 10 splits)")
    axes[0].legend(frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_models.png"), bbox_inches="tight")
    plt.close(fig)


def sensitivity():
    f = os.path.join(RES, "sensitivity", "solutions.csv")
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    order = ["GA", "baseline", "no_knowledge", "zero_shot", "rag_tabular", "rag_text"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, (col, ttl) in zip(axes, [("p_strength_ok_pred_error", "Strength prediction error ($\\sigma$ = 5.7 MPa)"),
                                     ("p_strength_ok_cv2", "Ingredient variability, CV 2%"),
                                     ("p_strength_ok_cv5", "Ingredient variability, CV 5%")]):
        g = d.groupby("method")[col].agg(["mean", "std"]).reindex(order)
        ax.bar(range(len(order)), g["mean"], yerr=g["std"], capsize=3,
               color=[COL.get(m, "#999") for m in order])
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(["GA", "Base", "No-K", "Zero", "RAG-T", "RAG-X"])
        ax.set_title(ttl, loc="left", fontsize=13)
    axes[0].set_ylabel("P(strength $\\geq$ floor)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_sensitivity.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_runs()
    trajectory("s50_nodur")
    bars(df, ["no_knowledge", "baseline"], "Effect of domain knowledge", "fig_knowledge.png", None)
    bars(df, ["zero_shot", "baseline"], "Zero-shot vs few-shot", "fig_fewshot.png", None)
    bars(df, ["baseline", "rag_tabular", "rag_text"], "No RAG vs RAG", "fig_rag.png", None)
    durability(df)
    models()
    sensitivity()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()

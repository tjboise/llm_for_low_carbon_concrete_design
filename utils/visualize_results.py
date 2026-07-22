"""
Visualization: NSGA-II Pareto Front + LLM Iterative Trajectory
===============================================================
Reads:
  - nsga2_pareto_front.csv   (output from Phase 1)
  - llm_optimizer_results.csv (output from Phase 2)

Produces:
  - concrete_optimization_results.png  (high-res, publication-quality)
  - concrete_optimization_results.pdf  (vector, for paper)

The plot shows:
  1. NSGA-II Pareto front as a connected curve (reference)
  2. TOPSIS optimal point (star marker)
  3. LLM trajectory points colored by iteration number
  4. Arrows connecting consecutive LLM iterations (n -> n+1)
  5. Infeasible LLM points shown distinctly
  6. Per-iteration d_topsis convergence subplot

Usage:
  python visualize_results.py
  python visualize_results.py --nsga2 nsga2_pareto_front.csv --llm llm_optimizer_results.csv
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ── Publication style ────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

# ── GWP factors (must match optimizer) ──────────────────────
GWP_FACTORS = {
    "PC": 1.048, "FA": 0.328, "SC": 0.264, "SF": 0.850,
    "CAGG": 0.0037, "FAGG": 0.0026,
    "WATER": 0.0, "AEA": 0.0, "WR_HR": 0.0, "WR": 0.0, "ACC": 0.0,
}

RAW_VARS = ["PC","FA","SC","SF","FAGG","CAGG","WATER","AEA","WR_HR","WR","ACC"]


def compute_gwp(row) -> float:
    return sum(row.get(k, 0) * v for k, v in GWP_FACTORS.items())


def topsis_from_df(df: pd.DataFrame, w_str=0.5, w_gwp=0.5) -> dict:
    """Recompute TOPSIS on a dataframe of solutions."""
    s = df["pred_28day"].values
    c = df["gwp"].values
    sn = s / (np.sqrt(np.sum(s**2)) + 1e-9)
    cn = c / (np.sqrt(np.sum(c**2)) + 1e-9)
    ws, wc = sn * w_str, cn * w_gwp
    d_pos = np.sqrt((ws - ws.max())**2 + (wc - wc.min())**2)
    d_neg = np.sqrt((ws - ws.min())**2 + (wc - wc.max())**2)
    score = d_neg / (d_pos + d_neg + 1e-9)
    idx   = int(np.argmax(score))
    return df.iloc[idx].to_dict()


# ── Color helpers ────────────────────────────────────────────
NSGA_COLOR    = "#1D9E75"   # teal green
ARROW_FEAS    = "#534AB7"   # purple  (feasible step)
ARROW_INFEAS  = "#E24B4A"   # red     (infeasible step)
TOPSIS_COLOR  = "#BA7517"   # amber
BEST_LLM_CLR  = "#E24B4A"   # red star for best LLM solution


def load_data(nsga_path: str, llm_path: str):
    """Load both CSVs; compute GWP if column missing."""
    nsga = pd.read_csv(nsga_path)
    llm  = pd.read_csv(llm_path)

    for df in [nsga, llm]:
        if "gwp" not in df.columns:
            df["gwp"] = df.apply(compute_gwp, axis=1)
        if "feasible" not in df.columns:
            df["feasible"] = True

    # Sort NSGA Pareto by GWP for the curve
    nsga = nsga.sort_values("gwp").reset_index(drop=True)
    llm  = llm.sort_values("iteration").reset_index(drop=True)
    return nsga, llm


def make_figure(nsga: pd.DataFrame, llm: pd.DataFrame,
                out_prefix: str = "concrete_optimization_results"):
    """
    Build a 2-panel figure:
      Left  — Pareto front + LLM trajectory with arrows
      Right — d_topsis convergence over iterations
    """
    fig = plt.figure(figsize=(16, 7))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.55, 1],
                            left=0.07, right=0.97,
                            top=0.92, bottom=0.12, wspace=0.32)
    ax1 = fig.add_subplot(gs[0])   # Pareto + trajectory
    ax2 = fig.add_subplot(gs[1])   # convergence

    # ── Colormap for LLM iterations ──────────────────────────
    n_iters = len(llm)
    cmap    = plt.get_cmap("plasma")
    norm    = Normalize(vmin=1, vmax=n_iters)
    sm      = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # ── NSGA-II Pareto curve ─────────────────────────────────
    nsga_feas = nsga[nsga["feasible"] == True].copy()
    nsga_feas = nsga_feas.sort_values("gwp")

    ax1.plot(nsga_feas["gwp"], nsga_feas["pred_28day"],
             color=NSGA_COLOR, linewidth=2.0, zorder=2,
             label="NSGA-II Pareto front", alpha=0.85)
    ax1.scatter(nsga_feas["gwp"], nsga_feas["pred_28day"],
                color=NSGA_COLOR, s=28, zorder=3, alpha=0.6)

    # ── TOPSIS optimal point ─────────────────────────────────
    tpt = topsis_from_df(nsga_feas)
    ax1.scatter(tpt["gwp"], tpt["pred_28day"],
                marker="*", s=380, color=TOPSIS_COLOR,
                zorder=6, label=f"TOPSIS optimum\n(28d={tpt['pred_28day']:.1f} MPa, "
                                f"GWP={tpt['gwp']:.1f})",
                edgecolors="white", linewidths=0.8)

    # ── LLM trajectory ───────────────────────────────────────
    # Scatter points colored by iteration
    feas_llm   = llm[llm["feasible"] == True]
    infeas_llm = llm[llm["feasible"] == False]

    # Infeasible: gray X markers
    if len(infeas_llm) > 0:
        ax1.scatter(infeas_llm["gwp"], infeas_llm["pred_28day"],
                    marker="x", s=70, color="#AAAAAA", linewidths=1.5,
                    zorder=4, label="LLM solution (infeasible)", alpha=0.7)

    # Feasible: colored dots
    sc = ax1.scatter(feas_llm["gwp"], feas_llm["pred_28day"],
                     c=feas_llm["iteration"], cmap=cmap, norm=norm,
                     s=90, zorder=5, edgecolors="white", linewidths=0.6,
                     label="LLM solution (feasible, colored by iter)")

    # ── Arrows: iter n -> iter n+1 ───────────────────────────
    rows = llm.reset_index(drop=True)
    for i in range(len(rows) - 1):
        r0 = rows.iloc[i]
        r1 = rows.iloc[i + 1]
        x0, y0 = r0["gwp"], r0["pred_28day"]
        x1, y1 = r1["gwp"], r1["pred_28day"]

        # Skip arrows for very short moves (same point)
        dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        if dist < 0.5:
            continue

        # Color: purple if both feasible, red if either infeasible
        both_feas = bool(r0["feasible"]) and bool(r1["feasible"])
        a_color   = ARROW_FEAS if both_feas else ARROW_INFEAS
        alpha     = 0.65 if both_feas else 0.35

        ax1.annotate(
            "",
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=a_color,
                lw=1.2,
                mutation_scale=10,
                alpha=alpha,
            ),
            zorder=4,
        )

    # ── Iteration number labels (every 5th + first + last) ───
    label_iters = set([1, n_iters] + list(range(5, n_iters + 1, 5)))
    for _, row in llm.iterrows():
        it = int(row["iteration"])
        if it in label_iters and row["feasible"]:
            ax1.annotate(
                str(it),
                xy=(row["gwp"], row["pred_28day"]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7.5, color="#333333", zorder=7,
            )

    # ── Best LLM feasible solution ───────────────────────────
    if len(feas_llm) > 0:
        # Best d_topsis (if column exists), else best balanced
        if "d_topsis" in llm.columns:
            best_row = feas_llm.loc[feas_llm["d_topsis"].idxmin()]
        else:
            feas_llm = feas_llm.copy()
            feas_llm["_score"] = (
                (feas_llm["pred_28day"] - feas_llm["pred_28day"].min()) /
                (feas_llm["pred_28day"].max() - feas_llm["pred_28day"].min() + 1e-9)
                -
                (feas_llm["gwp"] - feas_llm["gwp"].min()) /
                (feas_llm["gwp"].max() - feas_llm["gwp"].min() + 1e-9)
            )
            best_row = feas_llm.loc[feas_llm["_score"].idxmax()]

        ax1.scatter(best_row["gwp"], best_row["pred_28day"],
                    marker="*", s=280, color=BEST_LLM_CLR,
                    zorder=8, edgecolors="white", linewidths=0.8,
                    label=f"Best LLM solution\n(iter {int(best_row['iteration'])},"
                          f" 28d={best_row['pred_28day']:.1f} MPa,"
                          f" GWP={best_row['gwp']:.1f})")

    # ── Colorbar ─────────────────────────────────────────────
    cbar = fig.colorbar(sm, ax=ax1, pad=0.02, shrink=0.75)
    cbar.set_label("LLM iteration", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # ── Axis labels & title ───────────────────────────────────
    ax1.set_xlabel("Total GWP  (lb CO₂-eq / m³)   ←  lower is better", fontsize=11)
    ax1.set_ylabel("Predicted 28-day strength  (MPa)   ↑  higher is better", fontsize=11)
    ax1.set_title("Pareto Front Comparison: NSGA-II vs LLM Iterative Optimiser", fontsize=13)

    # ── Legend ────────────────────────────────────────────────
    arrow_feas_patch = mpatches.FancyArrow(
        0, 0, 1, 0, width=0.3, color=ARROW_FEAS, alpha=0.7)
    arrow_inf_patch  = mpatches.FancyArrow(
        0, 0, 1, 0, width=0.3, color=ARROW_INFEAS, alpha=0.4)
    extra_handles = [
        Line2D([0],[0], marker=">", color=ARROW_FEAS,  lw=1.2,
               markersize=6, label="LLM step (feasible→feasible)"),
        Line2D([0],[0], marker=">", color=ARROW_INFEAS, lw=1.2,
               markersize=6, alpha=0.5, label="LLM step (involves infeasible)"),
    ]
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles + extra_handles,
               labels + [h.get_label() for h in extra_handles],
               loc="upper left", fontsize=8.5,
               framealpha=0.9, edgecolor="#cccccc",
               ncol=1)

    # ═══════════════════════════════════════════════════════
    # Right panel: d_topsis convergence
    # ═══════════════════════════════════════════════════════
    if "d_topsis" in llm.columns:
        iters    = llm["iteration"].values
        d_vals   = llm["d_topsis"].values
        feasible = llm["feasible"].values

        # Line for feasible only
        feas_mask = feasible == True
        if feas_mask.sum() > 1:
            ax2.plot(iters[feas_mask], d_vals[feas_mask],
                     color=ARROW_FEAS, linewidth=1.8,
                     label="d_topsis (feasible)", zorder=3)

        # Scatter: feasible purple, infeasible red X
        ax2.scatter(iters[feas_mask],  d_vals[feas_mask],
                    color=ARROW_FEAS, s=55, zorder=4,
                    edgecolors="white", linewidths=0.5)
        ax2.scatter(iters[~feas_mask], d_vals[~feas_mask],
                    marker="x", color=ARROW_INFEAS, s=55, zorder=4,
                    linewidths=1.5, label="d_topsis (infeasible)", alpha=0.6)

        # Rolling minimum (best d so far)
        best_so_far = np.minimum.accumulate(
            np.where(feas_mask, d_vals, np.inf))
        best_so_far[best_so_far == np.inf] = np.nan
        ax2.plot(iters, best_so_far,
                 color=TOPSIS_COLOR, linewidth=2, linestyle="--",
                 label="Best d_topsis so far", zorder=5)

        # Annotate final best
        final_best = np.nanmin(best_so_far)
        ax2.axhline(final_best, color=TOPSIS_COLOR,
                    linewidth=0.8, linestyle=":", alpha=0.6)
        ax2.text(iters[-1] + 0.3, final_best,
                 f"  {final_best:.4f}", va="center",
                 fontsize=8.5, color=TOPSIS_COLOR)

        ax2.set_xlabel("Iteration", fontsize=11)
        ax2.set_ylabel("Normalised distance to TOPSIS optimum\n(d_topsis)", fontsize=10)
        ax2.set_title("LLM Convergence towards TOPSIS Optimum", fontsize=13)
        ax2.legend(fontsize=8.5, framealpha=0.9)
        ax2.set_xlim(left=0)
        ax2.set_ylim(bottom=0)

    else:
        # Fallback: show GWP / strength trajectories
        ax2b = ax2.twinx()
        ax2.plot(llm["iteration"], llm["gwp"],
                 color="#E24B4A", linewidth=1.8, label="GWP")
        ax2b.plot(llm["iteration"], llm["pred_28day"],
                  color=ARROW_FEAS, linewidth=1.8, linestyle="--",
                  label="28d strength")
        ax2.set_xlabel("Iteration"); ax2.set_ylabel("GWP (lb CO₂/m³)")
        ax2b.set_ylabel("28-day strength (MPa)")
        ax2.set_title("LLM Trajectory over Iterations")
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1+lines2, labels1+labels2, fontsize=8.5)

    # ── Save ─────────────────────────────────────────────────
    for ext in ("png", "pdf"):
        path = f"{out_prefix}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved -> '{path}'")

    plt.show()
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Visualise NSGA-II Pareto front + LLM trajectory"
    )
    parser.add_argument("--nsga2", default="nsga2_pareto_front.csv",
                        help="Path to NSGA-II Pareto CSV (default: nsga2_pareto_front.csv)")
    parser.add_argument("--llm",   default="llm_optimizer_results.csv",
                        help="Path to LLM results CSV (default: llm_optimizer_results.csv)")
    parser.add_argument("--out",   default="concrete_optimization_results",
                        help="Output filename prefix (no extension)")
    args = parser.parse_args()

    # Check files exist
    missing = [p for p in [args.nsga2, args.llm] if not os.path.exists(p)]
    if missing:
        print(f"ERROR: File(s) not found: {missing}")
        print("Make sure you run llm_concrete_optimizer.py first.")
        return

    print(f"Loading NSGA-II data from '{args.nsga2}' ...")
    print(f"Loading LLM data from    '{args.llm}' ...")
    nsga, llm = load_data(args.nsga2, args.llm)

    print(f"NSGA-II solutions : {len(nsga)}")
    print(f"LLM iterations    : {len(llm)}  "
          f"(feasible: {llm['feasible'].sum()})")

    print("\nGenerating figure ...")
    make_figure(nsga, llm, out_prefix=args.out)


if __name__ == "__main__":
    main()
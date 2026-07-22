"""
plot.py
=======
Reads results from a run folder produced by single_target.py and generates
four publication-quality figures:

  1. GWP Convergence
  2. 28-day Strength Tracking
  3. Full Mix Composition (stacked bar)
  4. GWP Breakdown by Material (stacked bar)

Usage
-----
  # Auto-detect the most recent run folder
  python plot.py

  # Specify a folder explicitly
  python plot.py --folder run_s55_i40_20260423_143022

  # Specify individual files
  python plot.py --llm path/to/llm_optimizer_results.csv
                 --ga  path/to/ga_reference_solution.csv

Output
------
  Figures saved inside the run folder (or current directory if not found):
    28d_{N}MPa_gwp_convergence.png
    28d_{N}MPa_strength_tracking.png
    28d_{N}MPa_binder_composition.png
    28d_{N}MPa_gwp_breakdown.png
"""

import argparse
import os
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  True,
    "axes.spines.right":True,
    "axes.grid":        False,
})

# ── Colors ────────────────────────────────────────────────────
C_FEAS = "#534AB7"   # purple  — LLM feasible points
C_GA   = "#E24B4A"   # red     — GA reference line
C_BEST = "#1D9E75"   # teal    — best-so-far line
C_STR  = "#378ADD"   # blue    — strength line
C_PC   = "#E24B4A"
C_SC   = "#534AB7"
C_FA   = "#1D9E75"
C_SF   = "#BA7517"

GWP_FACTORS = {
    "PC": 1.048, "FA": 0.328, "SC": 0.264,
    "CAGG": 0.0037, "FAGG": 0.0026,
}


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_llm(path: str):
    df = pd.read_csv(path)

    # Recompute GWP if missing
    if "gwp" not in df.columns:
        df["gwp"] = df.apply(
            lambda r: sum(r.get(k, 0) * v for k, v in GWP_FACTORS.items()),
            axis=1)

    # Normalise feasible column
    if "feasible" in df.columns:
        df["feasible"] = df["feasible"].astype(str).str.upper() == "TRUE"
    else:
        df["feasible"] = True

    # Detect strength floor
    if "str_margin" in df.columns:
        df["_floor"]  = df["pred_28day"] - df["str_margin"]
        strength_min  = float(df["_floor"].median())
    else:
        strength_min = 55.0

    df["str_margin"] = df["pred_28day"] - strength_min

    # GWP contributions per material
    for mat, factor in GWP_FACTORS.items():
        df[f"gwp_{mat}"] = df[mat].fillna(0) * factor if mat in df.columns else 0.0

    return df, strength_min


def load_ga(path: str):
    if not path or not os.path.exists(path):
        return None
    row = pd.read_csv(path).iloc[0].to_dict()
    if "gwp" not in row:
        row["gwp"] = sum(row.get(k, 0) * v for k, v in GWP_FACTORS.items())
    return row


def find_latest_folder():
    """Return the most recently created run_* folder, or None."""
    folders = sorted(glob.glob("run_s*"), key=os.path.getmtime, reverse=True)
    return folders[0] if folders else None


# ─────────────────────────────────────────────────────────────
# FIGURE 1 — GWP CONVERGENCE
# ─────────────────────────────────────────────────────────────

def plot_gwp_convergence(ax, df, ga_ref, strength_min):
    iters    = df["iteration"].values
    gwp      = df["gwp"].values
    feasible = df["feasible"].values

    # Best-so-far (feasible only, position-based)
    best_so_far  = np.full(len(df), np.nan)
    current_best = np.inf
    for pos, (_, row) in enumerate(df.iterrows()):
        if row["feasible"] and row["gwp"] < current_best:
            current_best = row["gwp"]
        best_so_far[pos] = current_best if current_best < np.inf else np.nan

    # GA reference
    if ga_ref:
        ax.axhline(ga_ref["gwp"], color=C_GA, linewidth=1.2,
                   linestyle="--", label=f"GA reference ({ga_ref['gwp']:.1f} lb)",
                   zorder=2)

    # Feasible scatter
    feas_mask = feasible == True
    ax.scatter(iters[feas_mask], gwp[feas_mask],
               color=C_FEAS, s=70, zorder=5,
               edgecolors="white", linewidths=0.5,
               label="LLM solution")

    # Best-so-far line
    valid = ~np.isnan(best_so_far)
    if valid.sum() > 1:
        ax.plot(iters[valid], best_so_far[valid],
                color=C_BEST, linewidth=2.2,
                label="Best GWP so far", zorder=4)

    # Arrows between consecutive feasible points
    feas_df = df[df["feasible"]].reset_index(drop=True)
    for i in range(len(feas_df) - 1):
        x0, y0 = feas_df.loc[i,   "iteration"], feas_df.loc[i,   "gwp"]
        x1, y1 = feas_df.loc[i+1, "iteration"], feas_df.loc[i+1, "gwp"]
        if abs(x1 - x0) <= 3 and abs(y1 - y0) > 0.5:
            ax.annotate("",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=C_FEAS,
                                lw=0.9, mutation_scale=8, alpha=0.45),
                zorder=3)

    # Restart markers
    restarts = df[df["mode"].astype(str).str.startswith("RESTART")]
    if len(restarts) > 0:
        ax.scatter(restarts["iteration"], restarts["gwp"],
                   marker="D", s=90, color="#FF8C00",
                   zorder=6, edgecolors="white", linewidths=0.6,
                   label="Restart")

    # Annotate best
    feas_only = df[df["feasible"]]
    if len(feas_only) > 0:
        best_row = feas_only.loc[feas_only["gwp"].idxmin()]
        ax.annotate(
            f"Best: {best_row['gwp']:.1f} lb\n(iter {int(best_row['iteration'])})",
            xy=(best_row["iteration"], best_row["gwp"]),
            xytext=(10, -20), textcoords="offset points",
            fontsize=8, color=C_BEST,
            arrowprops=dict(arrowstyle="->", color=C_BEST, lw=0.8),
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("GWP  (lb CO₂-eq / yd³)")
    ax.set_xlim(0, iters.max() + 1)
    ax.legend(fontsize=8.5, framealpha=0.9, loc="upper left")


# ─────────────────────────────────────────────────────────────
# FIGURE 2 — STRENGTH TRACKING
# ─────────────────────────────────────────────────────────────

def plot_strength(ax, df, strength_min):
    iters    = df["iteration"].values
    s28      = df["pred_28day"].values
    feasible = df["feasible"].values

    ax.axhline(strength_min, color=C_GA, linewidth=1.5,
               linestyle="--", alpha=0.85, label="Target strength")

    feas_mask = feasible == True
    ax.plot(iters, s28, color=C_STR, linewidth=1.2, alpha=0.45, zorder=2)
    ax.scatter(iters[feas_mask], s28[feas_mask],
               color=C_STR, s=65, zorder=4,
               edgecolors="white", linewidths=0.5,
               label="Predicted 28-day strength")

    # Shade below floor
    ax.fill_between(
        [0, iters.max() + 1],
        strength_min - 5, strength_min,
        color=C_GA, alpha=0.06)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Predicted 28-day strength  (MPa)")
    ax.set_xlim(0, iters.max() + 1)
    ax.set_ylim(strength_min - 5, df["pred_28day"].max() + 5)
    ax.legend(fontsize=8.5, framealpha=0.9)


# ─────────────────────────────────────────────────────────────
# FIGURE 3 — FULL MIX COMPOSITION
# ─────────────────────────────────────────────────────────────

def plot_binder_composition(ax, df):
    feasible = df[df["feasible"]].copy()
    if len(feasible) == 0:
        ax.text(0.5, 0.5, "No feasible solutions",
                ha="center", va="center", transform=ax.transAxes)
        return

    iters = feasible["iteration"].values
    w     = 0.65

    def col(name):
        return feasible[name].values if name in feasible.columns \
               else np.zeros(len(feasible))

    pc    = col("PC")
    sc    = col("SC")
    fa    = col("FA")
    water = col("WATER")
    fagg  = col("FAGG")
    cagg  = col("CAGG")
    wr    = col("WR")
    wr_hr = col("WR_HR")
    acc   = col("ACC")
    aea   = col("AEA")
    admix = wr + wr_hr + acc + aea

    bot = np.zeros(len(feasible))

    def add_bar(values, color, label, alpha=0.85):
        nonlocal bot
        if values.max() > 0.5:
            ax.bar(iters, values, bottom=bot,
                   color=color, label=label, alpha=alpha, width=w)
        bot += values

    add_bar(pc,    C_PC,      "PC",          alpha=0.90)
    add_bar(sc,    C_SC,      "SC",          alpha=0.90)
    add_bar(fa,    C_FA,      "FA",          alpha=0.90)
    add_bar(water, "#378ADD", "Water",       alpha=0.75)
    add_bar(fagg,  "#8BC4A8", "FAGG",        alpha=0.70)
    add_bar(cagg,  "#5A9E7A", "CAGG",        alpha=0.70)
    add_bar(admix, "#D4A84B", "Admixtures",  alpha=0.75)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mix content  (lb/yd³)")
    ax.set_xlim(0, iters.max() + 1)
    ax.legend(fontsize=7.5, framealpha=0.9, ncol=4,
              loc="upper right", bbox_to_anchor=(1.0, 1.0))


# ─────────────────────────────────────────────────────────────
# FIGURE 4 — GWP BREAKDOWN
# ─────────────────────────────────────────────────────────────

def plot_gwp_breakdown(ax, df, ga_ref):
    feasible = df[df["feasible"]].copy()
    if len(feasible) == 0:
        ax.text(0.5, 0.5, "No feasible solutions",
                ha="center", va="center", transform=ax.transAxes)
        return

    iters   = feasible["iteration"].values
    gwp_pc  = feasible["gwp_PC"].values  if "gwp_PC"   in feasible.columns else np.zeros(len(feasible))
    gwp_sc  = feasible["gwp_SC"].values  if "gwp_SC"   in feasible.columns else np.zeros(len(feasible))
    gwp_fa  = feasible["gwp_FA"].values  if "gwp_FA"   in feasible.columns else np.zeros(len(feasible))
    gwp_agg = (feasible["gwp_CAGG"].values if "gwp_CAGG" in feasible.columns else np.zeros(len(feasible))) + \
              (feasible["gwp_FAGG"].values if "gwp_FAGG" in feasible.columns else np.zeros(len(feasible)))

    w = 0.7
    ax.bar(iters, gwp_pc,  color=C_PC, label="PC",  alpha=0.85, width=w)
    ax.bar(iters, gwp_sc,  bottom=gwp_pc,
           color=C_SC, label="SC",  alpha=0.85, width=w)
    ax.bar(iters, gwp_fa,  bottom=gwp_pc + gwp_sc,
           color=C_FA, label="FA",  alpha=0.85, width=w)
    ax.bar(iters, gwp_agg, bottom=gwp_pc + gwp_sc + gwp_fa,
           color="#AAAAAA", label="Aggregates", alpha=0.70, width=w)

    if ga_ref:
        ax.axhline(ga_ref["gwp"], color=C_GA, linewidth=1.2,
                   linestyle="--",
                   label=f"GA reference)")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("GWP  lb CO₂-eq / yd³)")
    ax.set_xlim(0, iters.max() + 1)
    ax.legend(fontsize=8.5, framealpha=0.9, ncol=3, loc="best")


def plot_comparison(ax, llm_best, ga_ref):
    """画出 LLM 最佳与 GA 参考的堆叠成分对比图"""
    # 定义成分列表及颜色
    components = [
        ("PC", C_PC), ("SC", C_SC), ("FA", C_FA),
        ("WATER", "#378ADD"), ("FAGG", "#8BC4A8"),
        ("CAGG", "#5A9E7A"), ("ADMIX", "#D4A84B")
    ]

    def get_val(row, comp):
        if comp == "ADMIX":
            return row.get("WR", 0) + row.get("WR_HR", 0) + row.get("ACC", 0) + row.get("AEA", 0)
        return row.get(comp, 0)

    # 准备数据矩阵 [成分, 方案]
    llm_vals = [get_val(llm_best, c[0]) for c in components]
    ga_vals = [get_val(ga_ref, c[0]) for c in components]

    # 绘图：利用累积高度实现堆叠
    bottom_llm = 0
    bottom_ga = 0

    for i, (comp_name, color) in enumerate(components):
        # 绘制 LLM 柱 (X=0)
        ax.bar(0, llm_vals[i], bottom=bottom_llm, color=color,
               label=comp_name, width=0.5, edgecolor='white', linewidth=0.3)
        bottom_llm += llm_vals[i]

        # 绘制 GA 柱 (X=1)
        ax.bar(1, ga_vals[i], bottom=bottom_ga, color=color,
               width=0.5, edgecolor='white', linewidth=0.3)
        bottom_ga += ga_vals[i]

    # 美化图表
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["LLM Best", "GA Ref"])
    ax.set_ylabel("Content (lb/yd³)")
    ax.set_xlim(-0.5, 1.5)
    ax.legend(loc="best", fontsize=7, ncol=1, bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def make_figures(llm_path: str, ga_path: str, out_dir: str) -> None:
    df, strength_min = load_llm(llm_path)
    ga_ref           = load_ga(ga_path)

    prefix = os.path.join(out_dir, f"28d_{int(strength_min)}MPa")

    n_feas   = df["feasible"].sum()
    n_total  = len(df)
    best_gwp = df[df["feasible"]]["gwp"].min() if n_feas > 0 else float("nan")
    ga_gwp   = ga_ref["gwp"] if ga_ref else float("nan")

    print(f"  Iterations   : {n_total}  (feasible: {n_feas})")
    print(f"  Best LLM GWP : {best_gwp:.2f} lb/yd³")
    if ga_ref:
        print(f"  GA ref GWP   : {ga_gwp:.2f} lb/yd³  "
              f"(gap: {best_gwp - ga_gwp:+.2f} lb)")
    print(f"  Output prefix: {prefix}")
    print()

    feas_df = df[df["feasible"]]
    llm_best = feas_df.loc[feas_df["gwp"].idxmin()] if not feas_df.empty else {}
    ga_ref_dict = ga_ref if ga_ref else {}

    figures = [
        ("gwp_convergence", plot_gwp_convergence, (df, ga_ref, strength_min)),
        ("strength_tracking", plot_strength, (df, strength_min)),
        ("binder_composition", plot_binder_composition, (df,)),
        ("gwp_breakdown", plot_gwp_breakdown, (df, ga_ref)),
        # 新增对比图
        ("comparison_llm_ga", plot_comparison, (llm_best, ga_ref_dict)),
    ]

    for name, plot_fn, args in figures:
        fig, ax = plt.subplots()
        plot_fn(ax, *args)
        plt.tight_layout()
        path = f"{prefix}_{name}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved -> '{path}'")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot results from a single_target.py run folder"
    )
    parser.add_argument("--folder", default=None,
                        help="Run folder (default: auto-detect latest run_* folder)")
    parser.add_argument("--llm",    default=None,
                        help="Path to llm_optimizer_results.csv (overrides --folder)")
    parser.add_argument("--ga",     default=None,
                        help="Path to ga_reference_solution.csv (overrides --folder)")
    parser.add_argument("--out",    default=None,
                        help="Output directory (default: same as --folder)")
    args = parser.parse_args()

    # Resolve paths
    if args.llm and args.ga:
        llm_path = args.llm
        ga_path  = args.ga
        out_dir  = args.out or os.path.dirname(args.llm) or "."
    else:
        folder = args.folder or find_latest_folder()
        if folder is None:
            print("ERROR: No run folder found. "
                  "Use --folder or --llm/--ga to specify paths.")
            return
        print(f"Using folder: '{folder}'")
        llm_path = os.path.join(folder, "llm_optimizer_results.csv")
        ga_path  = os.path.join(folder, "ga_reference_solution.csv")
        out_dir  = args.out or folder

    if not os.path.exists(llm_path):
        print(f"ERROR: '{llm_path}' not found.")
        return

    make_figures(llm_path, ga_path, out_dir)


if __name__ == "__main__":
    main()
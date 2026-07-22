"""
collect_metrics.py
==================
Scans all run_* folders in the current directory, reads each metrics.json,
and produces a summary CSV and a printed table.

Usage
-----
  python collect_metrics.py                    # scan current directory
  python collect_metrics.py --dir path/to/runs # scan specific directory
  python collect_metrics.py --out summary.csv  # custom output filename
"""

import os
import json
import argparse
import glob
from datetime import datetime

import pandas as pd


def parse_folder_name(folder: str) -> dict:
    """
    Extract experiment metadata from folder name.
    Expected format: run_s{strength}_i{iters}_{tag}_{timestamp}
    e.g. run_s55_i30_baseline_20260423_171932
         run_s55_i30_no_knowledge_20260423_180000
         run_s55_i30_20260423_171932  (no tag = baseline)
    """
    name = os.path.basename(folder)
    parts = name.split("_")

    info = {
        "folder":      name,
        "strength":    None,
        "iters":       None,
        "experiment":  "baseline",
        "timestamp":   None,
    }

    try:
        # strength: s55 -> 55
        for p in parts:
            if p.startswith("s") and p[1:].isdigit():
                info["strength"] = int(p[1:])
                break

        # iters: i30 -> 30
        for p in parts:
            if p.startswith("i") and p[1:].isdigit():
                info["iters"] = int(p[1:])
                break

        # timestamp: last two parts if they look like YYYYMMDD_HHMMSS
        if len(parts) >= 2:
            last_two = parts[-2] + "_" + parts[-1]
            try:
                datetime.strptime(last_two, "%Y%m%d_%H%M%S")
                info["timestamp"] = last_two
                # everything between iters and timestamp is the experiment tag
                tag_parts = []
                found_i = False
                for p in parts:
                    if p.startswith("i") and p[1:].isdigit():
                        found_i = True
                        continue
                    if found_i and not (p.isdigit() and len(p) == 8):
                        # stop when we hit the date part
                        try:
                            int(p)
                            if len(p) == 8:
                                break
                        except ValueError:
                            pass
                        tag_parts.append(p)
                if tag_parts:
                    info["experiment"] = "_".join(tag_parts)
            except ValueError:
                pass
    except Exception:
        pass

    return info


def load_metrics(folder: str) -> dict:
    """Load metrics.json from a run folder."""
    path = os.path.join(folder, "metrics.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def collect(base_dir: str) -> pd.DataFrame:
    """Collect all metrics from run_* folders."""
    folders = sorted(glob.glob(os.path.join(base_dir, "run_*")))

    if not folders:
        print(f"No run_* folders found in '{base_dir}'")
        return pd.DataFrame()

    rows = []
    for folder in folders:
        if not os.path.isdir(folder):
            continue

        meta    = parse_folder_name(folder)
        metrics = load_metrics(folder)

        if metrics is None:
            print(f"  [skip] No metrics.json in '{os.path.basename(folder)}'")
            continue

        # Skip runs with no feasible solutions
        if metrics.get("note") == "no feasible solutions":
            print(f"  [skip] No feasible solutions in '{os.path.basename(folder)}'")
            continue

        row = {
            "folder":              meta["folder"],
            "experiment":          meta["experiment"],
            "strength_min":        meta["strength"],
            "iters":               meta["iters"],
            "timestamp":           meta["timestamp"],
            # Key metrics
            "OGR":                 metrics.get("OGR"),
            "QER":                 metrics.get("QER"),
            "MCE":                 metrics.get("MCE"),
            "best_gwp":            metrics.get("best_gwp"),
            "ga_gwp":              metrics.get("ga_gwp"),
            "gwp_gap":             metrics.get("gwp_gap"),
            "feasibility_rate":    metrics.get("feasibility_rate"),
            "convergence_iter":    metrics.get("convergence_iter"),
            "llm_catboost_calls":  metrics.get("total_catboost_calls"),
            "ga_catboost_calls":   metrics.get("ga_catboost_calls"),
            "calls_ratio":         metrics.get("calls_ratio"),
        }
        rows.append(row)
        print(f"  [ok]   {os.path.basename(folder)}"
              f"  exp={meta['experiment']}"
              f"  s={meta['strength']}"
              f"  OGR={metrics.get('OGR', 'n/a')}"
              f"  GWP={metrics.get('best_gwp', 'n/a')}")

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
    """Print a formatted summary grouped by experiment and strength."""
    if df.empty:
        return

    print("\n" + "=" * 80)
    print("  SUMMARY TABLE")
    print("=" * 80)

    # Sort by experiment then strength
    df_sorted = df.sort_values(["experiment", "strength_min"])

    # Key columns to display
    display_cols = ["experiment", "strength_min", "best_gwp", "ga_gwp",
                    "gwp_gap", "OGR", "QER", "MCE",
                    "feasibility_rate", "convergence_iter", "llm_catboost_calls"]

    available = [c for c in display_cols if c in df_sorted.columns]
    print(df_sorted[available].to_string(index=False))

    # If multiple runs per (experiment, strength), show mean±std
    grouped = df.groupby(["experiment", "strength_min"])
    if any(len(g) > 1 for _, g in grouped):
        print("\n" + "=" * 80)
        print("  MEAN ± STD (for conditions with multiple runs)")
        print("=" * 80)
        metric_cols = ["OGR", "QER", "MCE", "best_gwp", "gwp_gap",
                       "feasibility_rate", "convergence_iter"]
        stats_rows = []
        for (exp, s), group in grouped:
            if len(group) < 2:
                continue
            row = {"experiment": exp, "strength_min": s, "n_runs": len(group)}
            for col in metric_cols:
                if col in group.columns:
                    vals = group[col].dropna()
                    row[f"{col}_mean"] = round(float(vals.mean()), 4)
                    row[f"{col}_std"]  = round(float(vals.std()),  4)
            stats_rows.append(row)

        if stats_rows:
            stats_df = pd.DataFrame(stats_rows)
            print(stats_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Collect metrics from all run_* folders"
    )
    parser.add_argument("--dir", default=".",
                        help="Directory to scan (default: current directory)")
    parser.add_argument("--out", default="metrics_summary.csv",
                        help="Output CSV filename (default: metrics_summary.csv)")
    args = parser.parse_args()

    print(f"Scanning '{os.path.abspath(args.dir)}' for run_* folders...\n")
    df = collect(args.dir)

    if df.empty:
        print("No valid metrics found.")
        return

    # Save full table
    df.to_csv(args.out, index=False)
    print(f"\n  Full table saved -> '{args.out}'  ({len(df)} runs)")

    # Print summary
    print_summary(df)

    # Also save a stats version if multiple runs exist
    grouped = df.groupby(["experiment", "strength_min"])
    has_repeats = any(len(g) > 1 for _, g in grouped)
    if has_repeats:
        metric_cols = ["OGR", "QER", "MCE", "best_gwp", "gwp_gap",
                       "feasibility_rate", "convergence_iter"]
        stats_rows = []
        for (exp, s), group in grouped:
            row = {"experiment": exp, "strength_min": s, "n_runs": len(group)}
            for col in metric_cols:
                if col in group.columns:
                    vals = group[col].dropna()
                    row[f"{col}_mean"] = round(float(vals.mean()), 4)
                    row[f"{col}_std"]  = round(float(vals.std()),  4)
            stats_rows.append(row)
        stats_df = pd.DataFrame(stats_rows)
        stats_out = args.out.replace(".csv", "_stats.csv")
        stats_df.to_csv(stats_out, index=False)
        print(f"  Stats table saved -> '{stats_out}'")


if __name__ == "__main__":
    main()
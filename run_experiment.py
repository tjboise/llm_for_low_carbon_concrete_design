"""
run_experiment.py
=================
Entry point for ablation experiments.
This is the ONLY file you need to edit between experiments.

Usage
-----
  python run_experiment.py                        # run all experiments defined below
  python run_experiment.py --exp baseline         # run one specific experiment
  python run_experiment.py --skip-ga              # reuse saved GA reference
  python run_experiment.py --exp baseline --iters 30

Ablation experiments defined here
----------------------------------
  baseline            Full prompt (knowledge + rules + static few-shot)  [= No-RAG in paper]
  no_knowledge        Remove Material Effects table
  no_rules            Remove Situation A-D strategies
  no_fewshot          Remove few-shot examples
  zero_shot           Remove all three (pure zero-shot)
  dynamic_rag_tabular Dynamic RAG with tabular format (variable=value pairs)
  dynamic_rag_text    Dynamic RAG with natural language text format
"""

import argparse
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from optimizer_core import (
    ExperimentConfig,
    load_df, get_bounds, load_surrogate,
    run_ga, select_few_shot,
    run_llm, compute_metrics, save_results,
)

# ─────────────────────────────────────────────────────────────
# SHARED SETTINGS  (change these between experimental conditions)
# ─────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash-lite"
STRENGTH_MIN   = 55.0
MAX_ITERS      = 40
GA_GENS        = 200
GA_POP         = 100

DATA_PATH  = "data/Super_Cleaned_Concrete_Data_model_train.csv"
MODEL_PKL  = "concrete_catboost_optimized.pkl"
GA_CSV     = "ga_reference_solution.csv"

# ─────────────────────────────────────────────────────────────
# EXPERIMENT DEFINITIONS
# ─────────────────────────────────────────────────────────────
# Each entry is an ExperimentConfig.
# Only the flags that differ from the baseline need to be set —
# everything else inherits from the shared settings above.

def make_base_cfg(name: str, description: str, **kwargs) -> ExperimentConfig:
    """Helper: create a config inheriting all shared settings."""
    return ExperimentConfig(
        name=name,
        description=description,
        gemini_api_key=GEMINI_API_KEY,
        gemini_model=GEMINI_MODEL,
        strength_min=STRENGTH_MIN,
        max_iters=MAX_ITERS,
        ga_gens=GA_GENS,
        ga_pop=GA_POP,
        data_path=DATA_PATH,
        model_pkl=MODEL_PKL,
        ga_csv=GA_CSV,
        output_prefix=f"results_{name}",
        use_memory=False,   # memory OFF for all ablation experiments
        **kwargs,
    )


EXPERIMENTS = {

    # ── E0: Baseline (full prompt) ────────────────────────────
    "baseline": make_base_cfg(
        name="baseline",
        description="Full prompt: knowledge table + situation rules + static few-shot",
        use_knowledge_table=True,
        use_situation_rules=True,
        use_few_shot=True,
        rag_mode="static",
    ),

    # ── E1: No knowledge table ────────────────────────────────
    "no_knowledge": make_base_cfg(
        name="no_knowledge",
        description="Remove Material Effects table (no domain knowledge injection)",
        use_knowledge_table=False,
        use_situation_rules=True,
        use_few_shot=True,
        rag_mode="static",
    ),

    # ── E2: No situation rules ────────────────────────────────
    "no_rules": make_base_cfg(
        name="no_rules",
        description="Remove Situation A-D strategy rules",
        use_knowledge_table=True,
        use_situation_rules=False,
        use_few_shot=True,
        rag_mode="static",
    ),

    # ── E3: No few-shot examples ──────────────────────────────
    "no_fewshot": make_base_cfg(
        name="no_fewshot",
        description="Remove static few-shot examples",
        use_knowledge_table=True,
        use_situation_rules=True,
        use_few_shot=False,
        rag_mode="none",
    ),

    # ── E4: Pure zero-shot (remove everything) ────────────────
    "zero_shot": make_base_cfg(
        name="zero_shot",
        description="Zero-shot: only objective + bounds, no knowledge/rules/examples",
        use_knowledge_table=False,
        use_situation_rules=False,
        use_few_shot=False,
        rag_mode="none",
    ),

    # ── E5: Dynamic RAG (tabular format) ─────────────────────
    "dynamic_rag_tabular": make_base_cfg(
        name="dynamic_rag_tabular",
        description="Dynamic RAG: k-NN retrieval, tabular format (variable=value pairs)",
        use_knowledge_table=True,
        use_situation_rules=True,
        use_few_shot=True,
        rag_mode="dynamic",
        rag_k=5,
        rag_pool="feasible",
        rag_format="tabular",
    ),

    # ── E6: Dynamic RAG (natural language text format) ───────
    "dynamic_rag_text": make_base_cfg(
        name="dynamic_rag_text",
        description="Dynamic RAG: k-NN retrieval, natural language text format",
        use_knowledge_table=True,
        use_situation_rules=True,
        use_few_shot=True,
        rag_mode="dynamic",
        rag_k=5,
        rag_pool="feasible",
        rag_format="text",
    ),
}


# ─────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────

def run_one(cfg: ExperimentConfig, df, raw_b, der_b, meta,
            ga_ref, few_shot, skip_ga: bool = False) -> dict:
    """Run a single experiment and return its metrics dict."""
    print(f"\n{'#'*62}")
    print(f"# EXPERIMENT: {cfg.name}")
    print(f"# {cfg.description}")
    print(f"{'#'*62}")

    trajectory, run_summary, catboost_calls = run_llm(
        raw_b, der_b, meta, ga_ref, few_shot, cfg,
        memory=None, df=df,
    )

    metrics = compute_metrics(trajectory, ga_ref, raw_b, catboost_calls)
    save_results(trajectory, ga_ref, metrics, cfg, run_summary)

    print(f"\n  METRICS SUMMARY for {cfg.name}:")
    for k, v in metrics.items():
        print(f"    {k:<25}: {v}")

    return {"experiment": cfg.name, **metrics}


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--exp",      default=None,
                        help="Experiment name to run (default: all)")
    parser.add_argument("--skip-ga",  action="store_true",
                        help="Reuse saved GA reference CSV")
    parser.add_argument("--iters",    type=int, default=None,
                        help="Override MAX_ITERS for all experiments")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Number of repeated runs per experiment (for statistical analysis)")
    parser.add_argument("--strength", type=float, default=None,
                        help="Override STRENGTH_MIN for all experiments")
    args = parser.parse_args()

    # Override iters if specified
    if args.iters:
        for cfg in EXPERIMENTS.values():
            cfg.max_iters = args.iters
    if args.strength:
        for cfg in EXPERIMENTS.values():
            cfg.strength_min = args.strength
            cfg.output_prefix = f"results_{cfg.name}_s{int(args.strength)}"

    # Load shared resources once
    print("\n[Setup] Loading dataset and surrogate ...")
    df   = load_df(DATA_PATH)
    raw_b, der_b = get_bounds(df)
    meta = load_surrogate(MODEL_PKL)
    print(f"  Dataset: {len(df)} rows  |  "
          f"Feasible (28d>={STRENGTH_MIN}): "
          f"{df.dropna(subset=['28day'])[df['28day']>=STRENGTH_MIN].shape[0]} rows")

    # GA reference (shared across all experiments)
    if args.skip_ga and pd.io.common.file_exists(GA_CSV):
        print(f"\n[GA] Loading saved reference from '{GA_CSV}' ...")
        ga_ref = pd.read_csv(GA_CSV).iloc[0].to_dict()
        print(f"  GWP={ga_ref['gwp']:.2f}  28d={ga_ref['pred_28day']:.2f} MPa")
    else:
        # Use any config's GA settings (all share the same)
        first_cfg = next(iter(EXPERIMENTS.values()))
        ga_ref = run_ga(raw_b, der_b, meta, first_cfg)

    # Few-shot examples (shared across static experiments)
    few_shot = select_few_shot(df, STRENGTH_MIN, n=3)

    # Select experiments to run
    if args.exp:
        if args.exp not in EXPERIMENTS:
            print(f"Unknown experiment '{args.exp}'. Available: {list(EXPERIMENTS.keys())}")
            return
        to_run = {args.exp: EXPERIMENTS[args.exp]}
    else:
        to_run = EXPERIMENTS

    # Run experiments and collect summary
    all_metrics = []
    for exp_name, cfg in to_run.items():
        for rep in range(args.repeat):
            if args.repeat > 1:
                cfg.output_prefix = f"results_{cfg.name}_s{int(cfg.strength_min)}_r{rep + 1}"
                print(f"\n  >>> Repeat {rep + 1}/{args.repeat} for {cfg.name}")
            result = run_one(cfg, df, raw_b, der_b, meta,
                             ga_ref, few_shot, args.skip_ga)
            result["repeat"] = rep + 1
            result["strength_min"] = cfg.strength_min
            all_metrics.append(result)

    # Save combined comparison table
    if len(all_metrics) > 1:
        import numpy as np
        raw_df = pd.DataFrame(all_metrics)
        raw_path = f"ablation_raw_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        raw_df.to_csv(raw_path, index=False)

        # Compute mean ± std across repeats
        metric_cols = ["OGR", "QER", "MCE", "best_gwp", "gwp_gap",
                       "convergence_iter", "feasibility_rate"]
        stats_rows = []
        for (exp, s_min), group in raw_df.groupby(["experiment", "strength_min"]):
            row = {"experiment": exp, "strength_min": s_min, "n_repeats": len(group)}
            for col in metric_cols:
                if col in group.columns:
                    vals = group[col].dropna()
                    row[f"{col}_mean"] = round(float(vals.mean()), 4)
                    row[f"{col}_std"] = round(float(vals.std()), 4)
            stats_rows.append(row)

        stats_df = pd.DataFrame(stats_rows)
        stats_path = f"ablation_stats_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        stats_df.to_csv(stats_path, index=False)

        print(f"\n{'=' * 62}")
        print("  ABLATION STATS (mean ± std)")
        print(f"{'=' * 62}")
        for _, row in stats_df.iterrows():
            print(f"\n  {row['experiment']} (strength={row['strength_min']}, n={row['n_repeats']})")
            for col in metric_cols:
                if f"{col}_mean" in row:
                    print(f"    {col:<20}: {row[f'{col}_mean']:.4f} ± {row[f'{col}_std']:.4f}")

        print(f"\n  Raw data -> '{raw_path}'")
        print(f"  Stats    -> '{stats_path}'")


if __name__ == "__main__":
    main()

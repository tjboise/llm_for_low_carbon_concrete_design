# Experiments (rerun on the corrected data, Oct 2026)

All results in `results/` were produced with `run_experiment.py` after the unit fix, the WR/WR_HR fix and the
new constraints; older runs were deleted. Data, constraints and surrogates: see `DATA_AND_MODELS.md`.

## Protocol
- Scenarios: 28-day strength floor 45 / 50 / 55 MPa, each without and with the chloride constraint (6 scenarios).
- Methods: `baseline` (knowledge + rules + static few-shot; this is "With knowledge", "Few-shot" and "No-RAG" in the paper),
  `no_knowledge`, `zero_shot`, `rag_tabular`, `rag_text` (dynamic k=5 retrieval replaces the static few-shot).
- 5 independent runs per (scenario, method) = 150 LLM runs. Model `gemini-2.5-flash-lite`, temperature 0.9
  (1.3 after a stagnation restart), max 1024 output tokens, google-genai SDK. 30 feasible iterations per run; a run
  stops early after 30 consecutive infeasible proposals, so some runs have fewer feasible iterations.
- Every proposal must pass the same `check_feasibility` as the GA (ingredient bounds, 12 ratio bounds, Vfinal in [0.95, 1.05],
  strength floor, and chloride pass probability >= 0.7 when durability is on). No tolerance. Each reported best solution is
  re-verified (`verify_solution`).
- GA reference: pymoo GA, 100 x 200 = 20,000 evaluations, 5 seeds per scenario, same constraints; the reference is the best seed.
  GA mean +- std over seeds is stored in `results/ga/<scenario>/reference.json`.
- Metrics: OGR, QER, MCE, Rcalls (= surrogate evaluations / 20,000). Cost columns (API calls, prompt/completion tokens, LLM time)
  are stored per run; Rcalls does NOT include LLM inference cost.

## Layout
`results/ga/<scenario>/`, `results/llm/<scenario>/<method>/run_<k>/` (config.json, trajectory.csv = feasible iterations,
attempts.csv = all proposals, metrics.json, report.txt), `results/summary/` (all_runs.csv, mean_std.csv),
`results/model_comparison/`, `results/sensitivity/`. Figures: `pictures/`. Logs: `logs/`.

## Reproduce
```bash
python utils/prepare_data.py && python utils/train_model.py
python run_experiment.py --stage ga
python run_experiment.py --stage llm --workers 6     # resumable, ~1 h
python run_experiment.py --stage summary
python utils/model_comparison.py && python utils/sensitivity.py && python utils/make_figures.py
```

## Findings to be aware of
1. Success rate matters: many runs never reach a feasible mix under the hard constraints, especially with the chloride
   constraint and with RAG or zero-shot prompts (e.g. zero-shot, 55 MPa, durability: 0/5 runs). OGR is averaged over runs that
   found a feasible mix; the count is printed on the bars.
2. Most infeasible proposals violate ratio bounds (SC% above its dataset maximum, w/b below its minimum, ACC_pct), then Vfinal
   (hugging the 0.95 lower bound), strength and chloride.
3. Ordering in this rerun: baseline best on average, zero-shot worst, no-knowledge slightly worse than baseline, RAG did not
   improve over the static few-shot baseline. This differs from the earlier (invalid-unit) results.
4. GA solutions sit on constraint edges (PC at its minimum, Vfinal about 0.9501) and are fragile under surrogate error
   (see `fig_sensitivity.png`); LLM solutions keep more strength margin.
5. Model comparison (`results/model_comparison`): the chain beats independent models only when the true previous-stage
   strength is given (28-day R2 0.88 vs 0.69, CatBoost). With predicted previous-stage strength (deployment) it is
   0.68, i.e. no advantage. The paper's headline R2 (stage-wise, true input) must state this premise.

# Experiments (rerun on the corrected data, Oct 2026)

All results in `results/` were produced with `run_experiment.py` after the unit fix, the WR/WR_HR fix and the
new constraints; older runs were deleted. Data, constraints and surrogates: see `DATA_AND_MODELS.md`.

## Protocol
- Scenarios: 28-day strength floor 45 / 50 / 55 MPa, each without and with the chloride constraint (6 scenarios).
- Methods: `baseline` (knowledge + rules + static few-shot; this is "With knowledge", "Few-shot" and "No-RAG" in the paper),
  `no_knowledge` (no material-effects table AND no directional hints: the feedback, repair, restart and first-turn messages state facts and violations only; situation rules and few-shot stay), `zero_shot`, `rag_tabular`, `rag_text` (dynamic k=5 retrieval replaces the static few-shot).
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

## Run selection
Every (scenario, method) cell was run 7 times. Figures, tables and statistics use runs 3-7 (n = 5) in every cell
(`utils/make_figures.py`, `load_runs`; `utils/sensitivity.py`). Runs 1-2 stay in `results/llm/` and in
`results/summary/all_runs.csv`. The 150 first runs were made in parallel (6 workers) on the first day; runs 6-7 were added later.

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
3. Ordering in this rerun: zero-shot worst. No-knowledge (rerun without directional hints) is worse than the baseline at 45 and 50 MPa
   (OGR 0.18 vs 0.13, 0.22 vs 0.15) but better at 55 MPa; pooled without chloride constraint the OGR is equal (0.19 vs 0.19, p = 0.25).
   With the chloride constraint the baseline is better (0.18 vs 0.22, p = 0.04). QER is much higher with knowledge (1.55 vs 0.09 at 50 MPa).
   RAG did not improve over the static few-shot baseline. The first no-knowledge runs (hints kept) were deleted and rerun.
4. GA solutions sit on constraint edges (PC at its minimum, Vfinal about 0.9501) and are fragile under surrogate error
   (see `fig_sensitivity.png`); LLM solutions keep more strength margin.
5. Model comparison (`results/model_comparison`): the chain beats independent models only when the true previous-stage
   strength is given (28-day R2 0.88 vs 0.69, CatBoost). With predicted previous-stage strength (deployment) it is
   0.68, i.e. no advantage. The paper's headline R2 (stage-wise, true input) must state this premise.

## Main results
The paper figures use the experiments **without the chloride constraint** (`results/llm`, `results/ga`, `results/summary`; scenario `s50_nodur`
for the single-scenario figures; runs 3-7 per cell). Methods: `METHODS` in `run_experiment.py` (baseline = knowledge + situation rules +
static few-shot; the RAG variants replace the static examples with retrieval).
The `*_dur` scenarios in `results/llm` (chloride constraint) come from the first experiment series, made with a binary chloride model
(pass probability >= 0.7). The chloride model in `models/` was later replaced by a 4-class model (threshold 0.6 in `data/constraints.json`),
so those `*_dur` results cannot be re-evaluated with the current model.

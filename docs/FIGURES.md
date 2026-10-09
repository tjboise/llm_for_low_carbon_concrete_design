# Figures: runs, data and reproduction

Which runs and data files produce each figure. Machine-readable versions:
`results/tables/figure_runs.csv` (one row per figure, method and run, with a `used` flag) and
`results/tables/figure_summary.csv` (mean and std as plotted). Both are written by `utils/export_figure_data.py`.

## Setting of all main figures
- Scenario `s50_nodur`: 28-day strength >= 50 MPa, **no chloride constraint**. Units kg/m³ and kg CO₂-eq/m³.
- LLM runs: `results/llm/s50_nodur/<method>/run_<k>/` (`trajectory.csv` = feasible iterations, `attempts.csv` = all proposals,
  `metrics.json`, `config.json`, `report.txt`). Each (scenario, method) cell has 7 runs (run 1-7).
- GA reference: `results/ga/s50_nodur/reference.json` (best of 5 seeds, GWP 134.21; mean over seeds 137.99 +- 4.41; 20,000 evaluations per run).
- **Runs 3-7 are used in every figure and statistic.** Runs 1-2 stay in the repository and in `results/summary/all_runs.csv` but are not plotted.
  This is a fixed project rule (`load_runs` in `utils/make_figures.py`), applied to every cell.
- Methods (`METHODS` in `run_experiment.py`): `baseline` = knowledge table + situation rules + static few-shot (3 examples, no retrieval);
  `no_knowledge` = no knowledge table and no directional hints in feedback, repair, restart and first-turn messages (rules and few-shot kept);
  `zero_shot` = no knowledge, no rules, no examples; `rag_tabular`, `rag_text` = knowledge + rules + k = 5 retrieved dataset mixes per iteration
  (tabular or natural-language format) instead of the static examples.
- LLM: `gemini-2.5-flash-lite`, temperature 0.9 (1.3 after a restart), 1024 output tokens, 30 feasible iterations per run,
  **no random seed** (the API runs are independent samples and cannot be regenerated identically).

## Figure map
| Figure | File | Methods | Runs used | Notes |
|---|---|---|---|---|
| 3(a), 3(b) | `pictures/fig3a_gwp_iterations.png`, `fig3b_strength_iterations.png` | baseline | **run 3** (lowest OGR among runs 3-7, OGR 0.0604) | `results/llm/s50_nodur/baseline/run_3/trajectory.csv`; run ends at 24 feasible iterations; best GWP 142.32 at iteration 24; GA 134.21 |
| 4 | `pictures/fig4_gwp_contribution.png` | baseline | run 3 | GWP split PC / SC / FA / aggregates per iteration, x axis limited to 25 |
| 5(a), 5(b) | `pictures/fig5a.png`, `fig5b.png` | no_knowledge, baseline | runs 3-7 each | baseline OGR 0.091 +- 0.037, no_knowledge 0.196 +- 0.078 (p = 0.095) |
| 6(a), 6(b) | `pictures/fig6a.png`, `fig6b.png` | zero_shot, baseline | runs 3-7 each | zero-shot: run 4 has no feasible solution (OGR, gap, QER from 4 runs); OGR 0.903 +- 0.798 vs 0.091 +- 0.037 (p = 0.016) |
| 7(a), 7(b) | `pictures/fig7a.png`, `fig7b.png` | baseline (= No RAG), rag_tabular, rag_text | runs 3-7 each | rag_text run 6 has no feasible solution; OGR 0.091 / 0.213 / 0.154 |
| Table 4 | `results/tables/table4_reasoning_log.csv` | baseline | run 3, iterations 1-10 | reasoning text is LLM output, verbatim |
| Table 5 | `results/tables/table5_design_components.csv` | GA, baseline | GA reference; run 3 best (iteration 24) | kg/m³ |
| Per-run table of Figure 7 | `results/tables/fig7_per_run.csv` | three conditions | runs 1-7 with `used` flag | |
| Multi-panel overview | `pictures/fig_knowledge.png`, `fig_fewshot.png`, `fig_rag.png`, `fig_durability.png`, `fig_models.png`, `fig_sensitivity.png` | all | runs 3-7 in every cell | all six scenarios (with/without chloride constraint) |

**Not for publication:** `pictures/exploratory/` holds figures made from hand-picked runs (`fig7_selected_runs_*`: baseline runs 2,4; rag_tabular
runs 3,7; rag_text runs 2,7, n = 2 each, chosen after seeing the results) and `fig5_runs3to7_*` (same data as Figure 5). Do not cite them as results.

## Surrogates and data behind the figures
- Dataset `data/Concrete_Dataset_SI.xlsx` (686 mixes) and constraints `data/constraints.json`, from `utils/prepare_data.py`.
- Strength model `models/strength_chain.pkl` (chained CatBoost, built by `utils/train_model.py`; stage-wise 28-day R² 0.92, chained 28-day R² 0.78).
  The experiments of these figures do not use the chloride model.
- Software used: Python with catboost 1.2.10, scikit-learn 1.6.1, pandas 2.2.3, numpy 2.1.3, pymoo 0.6.2, google-genai 2.14.0.

## Reproduce
```bash
python utils/prepare_data.py && python utils/train_model.py      # data and surrogates
python run_experiment.py --stage ga                              # GA references (5 seeds per scenario)
python run_experiment.py --stage llm --repeats 7 --workers 6     # LLM runs (new, unseeded samples; numbers will differ)
python run_experiment.py --stage summary                         # results/summary/all_runs.csv
python utils/make_figures.py                                     # figures from the stored runs (exactly reproducible)
python utils/export_figure_data.py                               # tables with the runs behind each figure
```
The figures and tables are reproduced exactly from the stored `results/` files. Re-running the LLM stage gives new samples
and different numbers.

## Other results and history
- Reasons for the run selection: runs 1-2 were set aside after 7 runs per cell had been made. In the baseline cell of the 50 MPa case, run 2
  stopped after 8 feasible iterations (OGR 0.374). With all 7 baseline runs the OGR is 0.128 +- 0.113; with runs 3-7 it is 0.091 +- 0.037.
- The chloride-constraint experiments of later attempts (full prompt with retrieval, 4-class chloride model, thresholds 0.7 and 0.6)
  were deleted at the author's request; they are in the git history (commits up to `c3a9d19`).

# LLM-Based Low-Carbon Concrete Mix Design Optimizer

**Rutgers University CAIT** — Single-objective optimization framework using a Gemini LLM guided by a CatBoost surrogate model to minimize Global Warming Potential (GWP) under compressive strength constraints.

> **Important (2026-10):** the data pipeline, design constraints and surrogate models were rebuilt
> after fixing unit errors (admixtures are oz/yd³, not lb/yd³) and a WR/WR_HR label swap.
> See [docs/DATA_AND_MODELS.md](docs/DATA_AND_MODELS.md) for the current, authoritative description.
> The optimizer, experiments, results and figures were rerun on the corrected data; see
> [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). Parts of this README below (usage snippets, model description) are older.

## Overview

This project implements an iterative LLM optimizer that proposes concrete mix designs, evaluates them with a trained CatBoost surrogate (predicting 7/28/56-day strength), and refines proposals based on structured feedback — including domain knowledge injection, RAG retrieval, and few-shot examples.

## Project Structure

```
├── optimizer_core.py          # Core LLM optimizer (single-objective)
├── utils/prepare_data.py      # raw data -> SI dataset + constraints (current)
├── utils/train_model.py       # strength chain + chloride classifier (current)
├── models/                    # strength_chain.pkl, chloride_clf.pkl, metrics.json (current)
├── docs/DATA_AND_MODELS.md    # data, constraints, surrogates
├── app.py                     # Flask web demo (real-time SSE streaming)
├── templates/index.html       # Web UI (Predict + Optimize tabs)
├── data/
│   └── Super_Cleaned_Concrete_Data.csv   # PA concrete database (726 rows, kg/m³)
├── concrete_catboost_optimized.pkl       # Trained surrogate model
└── results/                   # Experiment outputs (trajectory, metrics CSVs)
```

## Model Architecture

**CatBoost Chain Surrogate:**
- Stage 1: Predicts **7-day** strength from raw ingredients
- Stage 2: Predicts **28-day** strength (uses predicted 7d as feature)
- Stage 3: Predicts **56-day** strength (uses predicted 28d as feature)

**Inputs (kg/m³):** PC, FA, SC (slag), FAGG, CAGG, WATER, AEA, WR_HR, WR, ACC

**GWP formula:** `GWP = PC×1.048 + FA×0.328 + SC×0.264 + CAGG×0.0037 + FAGG×0.0026`

## LLM Optimizer

The optimizer runs an iterative loop:
1. LLM proposes a mix design as JSON
2. Surrogate predicts strength; GWP is computed analytically
3. Feasibility is checked (strength ≥ floor, ingredient bounds, derived ratios)
4. Structured feedback is returned to the LLM (with optional RAG and domain knowledge)
5. Repeat until `max_iters` feasible solutions are found

**Experiment modes:**
- `use_knowledge_table` — inject material GWP factors and effect rules
- `rag_mode` — `"dynamic"` retrieves k-NN similar mixes from the dataset
- `rag_format` — `"text"` or `"tabular"` format for retrieved examples
- `use_few_shot` — seed with lowest-GWP, highest-strength, and balanced examples

## Web Demo

```bash
pip install flask python-dotenv google-genai catboost
echo "GEMINI_API_KEY=your_key" > .env
python app.py
```

Open **http://localhost:5000**

- **Predict tab:** Adjust ingredient sliders → instant surrogate prediction (7/28/56d strength + GWP)
- **Optimize tab:** Set strength floor, knowledge mode, RAG mode → live streaming optimization with iteration log

## Setup

```bash
pip install -r requirements.txt  # catboost, pandas, numpy, flask, google-genai, python-dotenv
```

Train the surrogate (optional, model already included):
```bash
python train_model.py
```

Run experiments:
```python
from optimizer_core import *
cfg = ExperimentConfig(
    name="my_run",
    gemini_api_key="...",
    gemini_model="gemini-2.5-flash",
    strength_min=50,
    max_iters=30,
    use_knowledge_table=True,
    rag_mode="dynamic",
    rag_format="text",
)
df = load_df("data/Super_Cleaned_Concrete_Data.csv")
raw_b, der_b = get_bounds(df)
meta = load_surrogate("concrete_catboost_optimized.pkl")
few_shot = select_few_shot(df, cfg.strength_min)
traj, summary, calls = run_llm(raw_b, der_b, meta, ga_ref=None, few_shot=few_shot, cfg=cfg, df=df)
```

## Notes

- All ingredient quantities are in **kg/m³** (converted from raw lb/yd³ database via factor 0.5933)
- GWP is in **kg CO₂/m³**
- Requires `GEMINI_API_KEY` in `.env` (uses `google-genai` SDK, model `gemini-2.5-flash`)
- Multi-objective optimization is maintained in a separate repository

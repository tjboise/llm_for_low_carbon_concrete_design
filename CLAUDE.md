# Guide for AI assistants

LLM-guided low-carbon concrete mix design (single objective: minimise GWP under strength and durability constraints).

Read first: `docs/DATA_AND_MODELS.md` — data pipeline, constraints and surrogate models. It is authoritative.

Rules that matter:
- All quantities are kg/m³; GWP is kg CO₂-eq/m³. Admixtures (AEA, WR_HR, WR, ACC) come from oz/yd³ (× 0.03708), the rest from lb/yd³ (× 0.5933). Never apply 0.5933 to admixtures.
- WR and WR_HR labels follow the chloride file (swapped in the raw strength file).
- Regenerate data and models only through `utils/prepare_data.py` then `utils/train_model.py`. Do not hand-edit `data/Concrete_Dataset_SI.xlsx` or `data/constraints.json`.
- Feasible mix = within `data/constraints.json` (ingredient bounds, ratio bounds, Vfinal in [0.95, 1.05]) + predicted 28-day strength ≥ floor + predicted 28-day RCPT < 1200 C (Port Authority "Low or better").
- Chloride classes are the Port Authority ones (<800 / 800–1200 / 1200–2000 / >2000 C), not ASTM bins.
- Strength inference is chained (predicted previous stage); report chained metrics, not stage-wise ones.
- `optimizer_core.py`, `app.py`, `run_experiment.py`, `results/`, `pictures/` are from the pre-fix data and not yet migrated; see section 4 of the doc.
- Never commit `.env` or API keys.

# Data, Design Constraints and Surrogate Models

Reference for the data pipeline, the constraints a generated mix must satisfy, and the two surrogate
models. Everything here is reproduced by two scripts:

```bash
python utils/prepare_data.py   # raw data -> data/Concrete_Dataset_SI.xlsx + data/constraints.json
python utils/train_model.py    # dataset  -> models/strength_chain.pkl, chloride_clf.pkl, metrics.json
```

All quantities are **kg/m³** (SI), GWP is **kg CO₂-eq/m³**, strength is **MPa**.

## 1. Data pipeline (`utils/prepare_data.py`)

| File | Role |
|---|---|
| `data/raw/Concrete_Data_raw_imperial.csv` | 756 mixes with 7/28/56-day strength. Binders, aggregates, water in **lb/yd³**; admixtures (AEA, WR_HR, WR, ACC) in **oz/yd³**. Has an SF (silica fume) column. |
| `data/classified_chloride_Classification.xlsx` | 1314 rapid chloride permeability (RCPT) tests, sheet `all_data`. Same imperial units. No strength. |
| `data/Concrete_Dataset_SI.xlsx` | **Output.** Sheets `mix_level`, `chloride_tests`, `readme`. |
| `data/constraints.json` | **Output.** Bounds for generated mixes (section 2). |

Steps, in order:

1. **WR / WR_HR swap.** The two raw files disagree on which column is WR and which is WR_HR. Matching
   mixes between them succeeds for 518 tests only after swapping, versus 45 without. The chloride file's labelling is
   used (WR_HR = high-range water reducer, large dosage; WR = normal water reducer). The strength
   file's two columns are swapped on load.
2. **Unit conversion to kg/m³.**
   - lb/yd³ × 0.5933 (= 0.45359237 / 0.764555) for PC, FA, SC, SF, FAGG, CAGG, WATER.
   - oz/yd³ × 0.03708 (= 0.028349523 / 0.764555, mass ounces) for AEA, WR_HR, WR, ACC.
   - Earlier versions of this project applied 0.5933 to admixtures as well, which overstated them about 16×. That
     was wrong. Every result produced before this fix is invalid.
3. **Strength consistency, then volume balance filter.** Mixes with 7d > 28d or 28d > 56d strength are removed first (7 mixes; same rule as the paper). `Vfinal = Σ(mass_i / density_i) + air`, `air = 0.07` if `AEA/PC ≥ 0.000244`
   (kg/kg, ≈0.39 oz/100 lb cement) else `0.03` (Pfeiffer et al. 2024, Eq. 19–20). Keep `0.95 ≤ Vfinal ≤ 1.05`.
   756 → drop 7 non-monotonic → **686 mixes**. Densities (kg/m³): PC 3150, FA 2200, SC 2900, SF 2200, FAGG 2630, CAGG 2710,
   WATER 1000, AEA 1010, WR_HR 1080, WR 1140, ACC 1340.
   An older curated set of 667 rows, used in the multi-target FLAME project, could not be reproduced by any simple
   rule. It is retired; this pipeline is the single source.
4. **GWP and derived ratios.** `GWP = 1.048·PC + 0.328·FA + 0.264·SC + 0.0026·FAGG + 0.0037·CAGG`.
   Admixtures, water and SF have factor 0. Derived: TOTAL_BINDER, w/b, b/a, SCM%, CAGG%, FAGG%, PC%, FA%, SC%,
   and AEA/WR_HR/WR/ACC as fractions of binder (`*_pct`).
5. **Chloride matching.** Each RCPT test is matched to a strength mix by all 11 ingredient quantities (tolerance 0.5,
   imperial units). Tests on special mixes are dropped: second cement (PC2 > 0), latex, lightweight aggregate, fibre.
   1314 → 502 tests → 459 after the cleaning and Vfinal filters (248 mixes).
6. **Chloride class** is recomputed from coulombs using the **Port Authority classes**, not ASTM bins:

   | Charge passed (C) | Class |
   |---|---|
   | < 800 | Very Low |
   | 800–1200 | Low |
   | 1200–2000 | Moderate |
   | > 2000 | High |

   `pass_rcpt = 1` if coulomb < **1200** (Low or better).

Known caveats: SF > 0 in 45 mixes, but SF is not a design variable and the surrogates ignore it. Dropped non-monotonic mixes are counted in the 756 → 686 step.

## 2. Design constraints (`data/constraints.json`)

A generated mix is feasible only if **all** of the following hold (checked after the surrogates are called):

| Constraint | Definition |
|---|---|
| Ingredient bounds | min/max of each of the 10 variables in the 686-mix dataset: PC, FA, SC, FAGG, CAGG, WATER, AEA, WR_HR, WR, ACC (key `raw`). |
| Ratio bounds | min/max of w/b, b/a, SCM%, CAGG%, FAGG%, PC%, FA%, SC%, and the four admixture `*_pct` (key `derived`). |
| Volume balance | `0.95 ≤ Vfinal ≤ 1.05` (key `physics.Vfinal`). Also Vagg and TOTAL_BINDER dataset ranges. |
| Strength | predicted 28-day strength ≥ the experiment's strength floor. |
| Durability | chloride classifier probability of passing RCPT at 28 days (coulomb < **1200 C**) ≥ **0.7** (`durability.min_pass_probability`). |

Why 1200 C: the Port Authority (NY/NJ) Low Carbon Concrete Pilot Program states a maximum of 1200 C at 28 days
for its structural categories (1700 C for marine). Taken from search summaries of the Task A and Task B reports. Verify the
exact category wording before citing it. Reviewer 2 argued GWP is misleading if durability is ignored; this constraint addresses that.

Current dataset ranges (kg/m³): PC 97–504, FA 0–162, SC 0–332, FAGG 498–1068, CAGG 564–1365, WATER 91–215,
AEA 0–1.48, WR_HR 0–7.8, WR 0–4.7, ACC 0–25. w/b 0.24–0.71, SCM% 0–0.76. See the JSON for all values.

## 3. Surrogate models (`utils/train_model.py`)

Both use the same 23 input features: the 10 raw ingredients, TOTAL_BINDER, w/b, b/a, SCM%, CAGG%, FAGG%, PC%, FA%,
SC%, and the 4 `*_pct` admixture fractions. One 80/20 split of mixes (seed 42) is shared by every model.

### Strength: chained CatBoost (`models/strength_chain.pkl`)
- Stage 1: 7-day ← features. Stage 2: 28-day ← features + 7-day. Stage 3: 56-day ← features + 28-day.
- **Training** uses the *true* previous-stage strength. **Inference** (`predict()` in the optimizer) feeds the *predicted*
  previous-stage strength, so errors propagate.
- Each stage is tuned by 5-fold grid search on the training rows that have the needed targets.
- Pickle keys: `models` (`"7day"`, `"28day"`, `"56day"`), `feature_names`, `unit="kg/m3"`.

Hold-out results (`models/metrics.json`):

| Stage | n_test | R² stage-wise (true input) | R² chained (predicted input) | MAE chained (MPa) |
|---|---|---|---|---|
| 7 d | 138 | 0.77 | 0.77 | 3.6 |
| 28 d | 138 | 0.92 | **0.78** | 4.2 |
| 56 d | 49 | 0.92 | 0.77 | 5.2 |

The headline metric chosen for the paper is the stage-wise 28-day R² = 0.92 (true 7-day strength as input); the paper must state that
premise. At deployment the optimizer uses chained inference (predicted 7-day input), where 28-day R² is 0.78, so keep both numbers.
A direct 28-day model from raw features gives R² 0.80 (MAE 4.1), i.e. no worse than the chain at inference.

### Chloride: CatBoost classifier (`models/chloride_clf.pkl`)
- Target: 28-day RCPT < 1200 C (binary), 154 mixes with a 28-day test, 49% pass.
- Pickle keys: `model`, `feature_names`, `unit`, `limit`, `age_days`. The final model is refit on all 154 mixes.
- Test AUC 0.88 and accuracy 0.77 on only 30 mixes (noisy). **5-fold CV on all rows: AUC 0.77, accuracy 0.71 at threshold 0.5.**
  Quote the CV figures. Threshold choice (CV, 78 failing mixes of 154): 0.5 → acc 0.71, 22 false passes; 0.6 → acc 0.72, 17; **0.7 → acc 0.76, precision 0.83, recall 0.64, 10 false passes** (chosen: highest accuracy and fewest failing mixes let through). Treat this model as a soft screen; compliance still needs a lab test.

## 4. Status

Data pipeline, constraints, surrogates, optimizer, GA reference, 150 LLM runs, model comparison, sensitivity analysis and
figures are done; see `EXPERIMENTS.md`. Open: update the manuscript text and tables, and add durability controls to the web demo UI.

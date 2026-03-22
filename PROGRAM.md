# PROGRAM.md

Permanent behavioral contract for experimenta-ml-kiro. Defines what the agent does, what it can change, how it judges progress, and what it must produce. `mission.yaml` provides run-specific inputs. This file governs system behavior on every run.

---

## What this system does

experimenta-ml-kiro is an agentic ML experimentation engine for tabular binary classification and regression. The agent iteratively rewrites `experiment.py` — data prep, feature engineering, and model pipeline — and is judged by a fixed evaluator. It hill-climbs toward the best primary metric within a trial budget.

---

## The runner loop

**Kiro CLI is the runner.** No Python orchestration script.

```
LOAD mission.yaml + domain_knowledge
RUN profiler → save and read profile.json  ← understand data signals before any code
RUN eda.py         ← validate profiler signals, test hypotheses from domain knowledge
WRITE baseline experiment.py

LOOP until trial_budget exhausted:
  Read: mission + profile.json + eda/results.json + experiment.py + experiments.json + error_analysis/<last_id>.json
  [Optional] Run eda.py to test a specific hypothesis
  Reason: what to change and why, grounded in domain knowledge + error patterns
  Write: new experiment.py
  Run: evaluator → metric
  Run: error analysis → error_analysis/<id>.json
  if is_improvement(metric, best, primary_metric): keep + log
  else: revert + log

REPORT: best experiment, full ledger, insights
```

### Running with Kiro CLI

The execution loop is driven by kiro-cli agents:

```bash
# Option A: Full Ralph workflow
kiro-cli chat --agent ml-clarify       # Define mission requirements
kiro-cli chat --agent ml-plan          # Generate PROMPT.md + TODO.md
./experimenta-execute.sh               # Autonomous execution loop

# Option B: Direct execution
kiro-cli chat --agent ml-experiment    # Interactive ML experimentation

# Option C: Raw loop
while :; do cat PROMPT.md | kiro-cli chat --no-interactive -a; done
```

Each iteration the agent must:
1. *(Run start)* Run profiler → save and read `profile.json` before writing any code.
2. *(Run start)* Run EDA → write `eda.py`, execute, read `eda/results.json`. Hypotheses must be grounded in domain knowledge.
3. Read `experiments.json` — know what was tried and what the current best is.
4. Read `error_analysis/<last_experiment_id>.json` — understand where the previous model failed before proposing changes.
5. Reason explicitly about what to change and why.
6. *(Optional, mid-loop)* Run another EDA script to test a specific hypothesis.
7. Rewrite `experiment.py`, run evaluator, run error analysis, log result.

---

## Domain knowledge

**This is the most important section for any non-trivial problem.**

Generic ML intuition (correlations, skewness, cardinality) is not enough. Domain knowledge defines what signals are real, what patterns are meaningful, and what analysis strategies are appropriate for the specific context.

Domain knowledge lives in `mission.yaml` under `domain_knowledge`. The agent must read it before EDA and before every iteration. It directly governs:
- which hypotheses are worth testing in EDA
- which features are likely meaningful vs noise
- how to interpret error analysis findings
- what split strategy reflects reality
- what relative or group-based behaviors to look for

### Categories of domain knowledge

**Unit of analysis**
What does one row represent? A die, a sensor reading, a transaction, a patient visit?
This determines the right granularity for aggregations, splits, and error interpretation.

**Group / hierarchy structure**
Are rows organized in lots, batches, sites, sessions, patients?
- Behaviors may be relative to the group, not absolute
- Splits may need to be group-aware to avoid leakage
- Features like `value / lot_mean` or `value - lot_median` are often more informative than raw values

**Temporal dynamics**
Is there time ordering? Drift? Seasonality? Time-since-event effects?
- A random split may be optimistic; a time-based split is more realistic
- Features derived from time position within a lot, shift, or run may matter

**Known leakage risks**
What columns are filled post-outcome or correlate with the outcome for the wrong reasons?

**Feature priors**
What features do domain experts know or suspect matter?

**Evaluation nuances**
Are false positives or false negatives more costly?

---

## What the agent can change

**Data policy:** drop columns, null imputation per column, outlier handling, type casting.

**Split policy:** any valid sklearn splitter; must use `split_seed` from mission; must reflect domain split strategy (group, time, or stratified as appropriate).

**Feature engineering:** log/sqrt/binning, interaction terms, ordinal/frequency/target encoding, aggregations, missing-value indicators, datetime decomposition, group-relative features.

**Model and hyperparameters:** any compatible model family, any hyperparameters, class weighting, ensembles/stacking.

The agent **cannot:**
- Modify `evaluator.py` or `mission.yaml`
- Import libraries outside: `pandas`, `numpy`, `sklearn`, `lightgbm`, `xgboost`
- Remove `prepare_data` or `build_pipeline` function signatures

---

## EDA phase

`eda.py` is a throwaway analysis script. Writes findings to `outputs/run_<id>/eda/results.json`. Not evaluated, not tracked.

**When to run:** always once before baseline; again mid-loop when proposing a novel feature. Skip for minor hyperparameter variations.

**What to compute:** group stats, conditional distributions, within-group distributions, correlation to target, candidate feature validation, mutual information, cross-tabs. No plots — write all findings as JSON.

**What to consult**: You can use `RECIPES.md` to get some ideas for possible data preparation and feature engineering, but don't follow it blindly.

**Output contract:**
```json
{
  "hypothesis": "...",
  "findings": { ... },
  "recommended_features": ["..."],
  "warnings": ["..."]
}
```

**Cannot:** modify `experiment.py`, `evaluator.py`, or `mission.yaml`; write outside `outputs/run_<id>/eda/`.

---

## Error analysis

After every evaluation, run error analysis inline and save to `error_analysis/<experiment_id>.json`.

**What to compute (Classification):**
1. Confusion summary (TP, FP, TN, FN, precision, recall at 0.5)
2. Calibration check — mean score vs actual rate across deciles
3. High-confidence errors — top FPs and FNs with feature values (up to 10 each)
4. Segment breakdown — per categorical column: positive rate, mean score, precision, recall
5. Score distribution — 10-bucket histogram for positives vs negatives

**What to compute (Regression):**
1. Error summary (MAE, MSE, RMSE, R2, MAPE, MedianAE, MaxError)
2. Residual stats — mean, std, min, max, and quantiles of absolute error
3. High-error samples — top 10 rows with highest absolute error
4. Segment breakdown — per categorical column: bucket count, mean absolute error, and RMSE
5. Error distribution — 10-bucket histogram of residuals

---

## experiment.py contract

```python
def prepare_data(df: pd.DataFrame, target_col: str):
    # Returns: X_train, X_test, y_train, y_test
    ...

def build_pipeline():
    # Returns unfitted sklearn Pipeline or compatible estimator.
    # Must support .fit(X, y) and .predict_proba(X) for classification or .predict(X) for regression.
    ...
```

If either function is missing or raises, the experiment is marked `failed`.

---

## Search order

0. **EDA** — validate profiler signals, test domain hypotheses, confirm group structures
1. **Baseline** — numeric-only, median imputation, StandardScaler, LogisticRegression (classification) or Ridge/Lasso/Poisson (regression)
2. **Data cleaning** — missingness, identifiers, categoricals, leakage columns
3. **Feature engineering** — domain features first (group-relative, temporal), then interactions
4. **Model upgrade** — LightGBM or XGBoost with reasonable defaults
5. **Hyperparameter tuning** — only on strong recipes
6. **Final selection** — confirm best feasible is logged

Do not jump to hyperparameters before features. Do not upgrade model before cleaning. Use EDA to validate novel features before committing.

---

## Keep / revert rules

- `status == "failed"` → always revert
- `feasible == False` → always revert, log reasons
- `is_improvement(primary_metric_value, current_best, primary_metric) == True` → keep, update best
- otherwise → revert, log as rejected

On ties, prefer earlier iteration (simpler recipe).

---

## Reproducibility fields

Every logged experiment must include: `experiment_id` (SHA256 of experiment.py), `iteration`, `primary_metric_value`, `feasible`, `status`, `dataset_path`, `dataset_row_count`, `split_seed`, `model_seed`, `evaluator_version`, `mission_snapshot_path`.

---

## Output structure

```
outputs/run_<id>/
  run.log
  mission_snapshot.yaml
  profile.json
  eda/
    eda.py
    results.json
  error_analysis/
    <experiment_id>.json
  experiments.parquet
  experiments.json
  experiments/<experiment_id>/experiment.py
  final_experiment.py
  final_pipeline.pkl
  report.md
```

---

## Final report sections

1. Mission summary
2. Dataset profile + key signals
3. Experiments run (count, feasible, failed)
4. Baseline performance
5. Best result vs baseline (delta, iteration, what changed)
6. Kept improvements (what helped and why)
7. Rejected directions (what was tried and didn't help)
8. Domain knowledge findings (what the data confirmed or contradicted about domain priors)
9. Final recommendation
10. Suggested next steps

---

## Source of truth hierarchy

```
PROGRAM.md       ← permanent system contract (this file)
mission.yaml     ← run-specific inputs + domain knowledge
experiment.py    ← mutable artifact
evaluator.py     ← fixed judge (never touched)
```

The code must follow the documents. Never the reverse.

---

## Design principles

1. **One mutable file** — `experiment.py` is the only thing the agent changes.
2. **Fixed judge** — `evaluator.py` never changes during a run.
3. **Domain knowledge first** — EDA and features must be grounded in domain context, not generic heuristics.
4. **Data and features before model** — search data/features before model/hyperparams.
5. **Track everything** — every experiment logged with full reproducibility fields.
6. **Keep it simple** — ~600-800 lines total across all modules.

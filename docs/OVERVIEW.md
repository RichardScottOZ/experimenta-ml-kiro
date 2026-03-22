# OVERVIEW.md

## What this project is

`experimenta-ml-kiro` is an AutoResearcher-style agentic ML experimentation engine for tabular binary classification and regression, ported to use **Kiro CLI** for autonomous execution.

An AI agent iteratively rewrites `experiment.py` — which contains data prep, feature engineering, and the model pipeline — and is judged each iteration by a fixed evaluator. The best version wins.

The loop is simple: **try → measure → keep/revert → repeat**.

---

## The core idea

In tabular ML, the hard part isn't model architecture. It's:
- cleaning data correctly
- choosing realistic splits
- creating useful feature transformations
- building aggregations that reflect the process

So the agent's canvas is `experiment.py`: a file it rewrites freely to improve data preparation, feature engineering, and the training pipeline.

The evaluator is fixed. It never changes. Every experiment is judged the same way.

---

## How the system works

### With Kiro CLI

```bash
# Phase 1: Define your mission
kiro-cli chat --agent ml-clarify

# Phase 2: Generate execution plan
kiro-cli chat --agent ml-plan

# Phase 3: Run the experimentation loop
./experimenta-execute.sh --max-iterations 50
```

Or the raw loop:

```bash
while :; do cat PROMPT.md | kiro-cli chat --no-interactive -a; done
```

### What happens internally

1. Load the mission (objective, constraints, metric, budget)
2. Profile the dataset (deterministic analysis, no LLM)
3. Write a baseline `experiment.py`
4. **Hill-climbing loop** until `trial_budget` is exhausted:
   - Agent reads: mission, profile, current `experiment.py`, metric history
   - Agent proposes a new `experiment.py`
   - Evaluator runs and scores it
   - If metric improved → keep (git commit)
   - If not → revert
5. Produce final artifacts

---

## The mutable file: `experiment.py`

The agent rewrites this file freely. It must define two functions:

```python
def prepare_data(df, target_col):
    """Returns X_train, X_test, y_train, y_test."""
    ...

def build_pipeline():
    """Returns an unfitted sklearn Pipeline."""
    ...
```

---

## Final outputs

| Artifact | Description |
|---|---|
| `final_experiment.py` | The winning experiment code |
| `final_pipeline.pkl` | Fitted pipeline from the winning experiment |
| `report.md` | Baseline vs best, what changed, final recommendation |

---

## What makes it valuable

- The agent explores data prep and feature engineering, not just hyperparameters
- Every experiment is reproducible
- Simple loop, simple artifacts, simple to understand
- Kiro CLI provides the autonomous execution engine

---

## Ported from

This project is a port of [jooaobrum/experimenta-ml](https://github.com/jooaobrum/experimenta-ml) to use Kiro CLI, following the patterns from [RichardScottOZ/ralph-kiro](https://github.com/RichardScottOZ/ralph-kiro).

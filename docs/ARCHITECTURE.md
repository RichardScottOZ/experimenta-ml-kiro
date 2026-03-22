# ARCHITECTURE.md

## Architectural goal

Build a small, agentic ML experimentation engine for tabular binary classification and regression that is driven by kiro-cli agents.

---

## Source of truth

1. `PROGRAM.md` — permanent behavioral contract
2. `mission.yaml` — run-specific inputs (dataset, target, metric, budget, constraints)
3. implementation — code that follows the contract

---

## The AutoResearcher mapping for tabular ML

| AutoResearcher | experimenta-ml-kiro | Role |
|---|---|---|
| `program.md` | `mission.yaml` | Human intent, constraints, metric. Read-only. |
| `prepare.py` | `profiler.py` + `evaluator.py` | Fixed infrastructure. Never mutated. |
| `train.py` | `experiment.py` | Data prep, features, model pipeline. **Mutable — agent rewrites this.** |
| val_bpb metric | `primary_metric` | Single number for keep/revert decisions. |

---

## Mutable vs fixed surfaces

### Fixed (never mutated during a run)

| File | Purpose |
|---|---|
| `mission.yaml` | Human intent and constraints |
| `profiler.py` | Deterministic dataset analysis |
| `evaluator.py` | Fixed evaluation contract |

### Ephemeral (agent writes and overwrites freely — not tracked)

| File | Purpose |
|---|---|
| `eda/eda.py` | Throwaway EDA script — agent writes to test hypotheses |
| `eda/results.json` | EDA findings — read by agent to inform next experiment |
| `error_analysis/<id>.json` | Per-experiment error diagnostics |

### Mutable (agent's canvas — versioned and tracked)

| File | Purpose |
|---|---|
| `experiment.py` | Data prep, feature engineering, model pipeline — agent rewrites freely |

---

## Core modules

```text
ml_agent/
  __init__.py     # package init
  mission.py      # load + validate mission.yaml
  profiler.py     # deterministic dataset profiling
  evaluator.py    # FIXED evaluation harness (never mutated)
  tracker.py      # experiment ledger
  recipe.py       # recipe ID + serialization (tracks experiment.py hash)
  utils.py        # make_run_dir, write_baseline_experiment
```

---

## Kiro CLI integration

```text
.kiro/
  agents/
    ml-clarify.json    # Mission requirements discovery agent
    ml-plan.json       # Planning agent (generates PROMPT.md + TODO.md)
    ml-experiment.json # Main ML experimentation execution agent
```

### Agent workflow

```
kiro-cli chat --agent ml-clarify    →  mission.yaml + clarify-session.md
kiro-cli chat --agent ml-plan       →  PROMPT.md + TODO.md
./experimenta-execute.sh            →  autonomous experimentation loop
```

Or directly:

```
kiro-cli chat --agent ml-experiment →  interactive experimentation
```

---

## The hill-climbing loop

```
baseline experiment.py
  (optionally preceded by EDA phase)
        │
        ▼
    evaluate → metric_0
        │
   ┌────┴────────────────────────────────────┐
   │  LOOP (trial_budget times)              │
   │                                         │
   │  agent reads:                           │
   │    - mission.yaml                       │
   │    - profile.json                       │
   │    - eda/results.json  (if exists)      │
   │    - current experiment.py              │
   │    - metric history                     │
   │                                         │
   │  agent writes:                          │
   │    - new experiment.py                  │
   │                                         │
   │  evaluate → metric_new                  │
   │                                         │
   │  if metric_new > metric_best:           │
   │      keep (git commit)                  │
   │  else:                                  │
   │      revert                             │
   └─────────────────────────────────────────┘
        │
        ▼
   best experiment.py → report
```

---

## Output directory layout

```text
outputs/
  run_<run_id>/
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
    experiments/
      <experiment_id>/
        experiment.py
    final_experiment.py
    final_pipeline.pkl
    report.md
```

---

## Reproducibility

Each experiment record captures:
- `experiment_id`: SHA256 of `experiment.py` content
- `split_seed`: from mission
- `model_seed`: from mission
- `evaluator_version`: hash of `evaluator.py`
- `mission_snapshot_path`: copy of mission at run start
- `dataset_path` + row count

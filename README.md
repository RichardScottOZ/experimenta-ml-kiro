# Experimenta ML for Kiro CLI

An agentic ML experimentation engine for tabular classification and regression, ported to use **Kiro CLI** for autonomous execution.

Ported from [jooaobrum/experimenta-ml](https://github.com/jooaobrum/experimenta-ml) using patterns from [RichardScottOZ/ralph-kiro](https://github.com/RichardScottOZ/ralph-kiro).

---

## The core idea

In practical tabular ML, most gains don't come from model tuning. They come from:

- cleaning data correctly
- choosing a realistic split
- building features that reflect the process
- aggregating the right groupings

So the agent doesn't search over model architectures. It searches over the full **recipe**: data policy, split strategy, feature engineering, model configuration.

The agent rewrites a single file — `experiment.py` — iteratively. A fixed evaluator judges every version the same way. The best feasible experiment wins.

---

## How it works

```
you write mission.yaml  →  agent reads it  →  agent profiles data
                                            →  agent runs EDA
                                            →  agent writes baseline
                                            →  hill-climbing loop
                                            →  final recipe + pipeline + report
```

The runner is **Kiro CLI**. There is no Python orchestration. You use kiro-cli agents and the execution loop script:

```bash
# Option A: Full workflow with kiro-cli agents
kiro-cli chat --agent ml-clarify       # Define mission requirements
kiro-cli chat --agent ml-plan          # Generate PROMPT.md + TODO.md
./experimenta-execute.sh               # Autonomous execution loop

# Option B: Direct interactive experimentation
kiro-cli chat --agent ml-experiment

# Option C: Raw execution loop
while :; do cat PROMPT.md | kiro-cli chat --no-interactive -a; done
```

The agent does the rest.

---

## Setup

```bash
git clone <this-repo>
cd experimenta-ml-kiro
pip install -e .
```

Dependencies: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `scipy`, `pyyaml`.

### Prerequisites

| Requirement | Description |
|-------------|-------------|
| **Python** | >= 3.10 |
| **Kiro CLI** | Installed and authenticated |
| **Git** | For state tracking |

### Install Kiro CLI

```bash
# macOS & Linux
curl -fsSL https://cli.kiro.dev/install | bash

# Verify installation
kiro-cli --version
```

---

## Running an experiment

### 1. Add your data

Put your training CSV or Parquet somewhere accessible, e.g. `data/myproject/train.csv`.

### 2. Write `mission.yaml`

```yaml
project_name: churn_prediction
task_type: binary_classification
target_column: churned
train_path: data/myproject/train.csv
primary_metric: roc_auc
trial_budget: 10
allowed_models:
  - lr
  - lgbm
  - xgb
split_seed: 42
model_seed: 0
hard_constraints:
  min_primary_metric: 0.75
  max_latency_ms: 200
domain_knowledge:
  unit_of_analysis: "One row = one customer subscription, observed monthly"
  group_structure:
    - "Customers belong to account_id groups."
    - "Split must be account-aware."
  temporal_dynamics:
    - "Data has a snapshot_month column."
    - "Use a time-based split to avoid leakage."
  known_leakage_risks:
    - "Column 'cancellation_reason' is filled after churn. Drop it."
  feature_priors:
    - "Engagement drop is historically the strongest predictor."
  evaluation_nuances:
    - "False negatives are more costly."
```

### 3. Run with Kiro CLI

#### Option A: Full Ralph workflow

```bash
# Phase 1: Clarify requirements (interactive)
kiro-cli chat --agent ml-clarify

# Phase 2: Generate execution plan
kiro-cli chat --agent ml-plan

# Phase 3: Autonomous execution
./experimenta-execute.sh --max-iterations 50
```

#### Option B: Direct experimentation

```bash
kiro-cli chat --agent ml-experiment
```

Then prompt:

> Read PROGRAM.md and mission.yaml. Run the full experimentation loop.

#### Option C: Raw loop

```bash
while :; do cat PROMPT.md | kiro-cli chat --no-interactive -a; done
```

---

## What the agent does

```
iteration 0   profiler → profile.json
              eda.py   → eda/results.json   (domain hypotheses validated)
              baseline experiment.py        (numeric-only, LR)

iteration 1   read error_analysis/0.json   (where did baseline fail?)
              propose cleaning + encoding
              run evaluator → keep/revert

iteration 2   run eda.py to test group feature hypothesis
              add group-relative features
              run evaluator → keep/revert

...

iteration N   hyperparameter refinement on best recipe so far
              run evaluator → keep/revert

final         report.md + final_experiment.py + final_pipeline.pkl
```

---

## Kiro CLI agents

| Agent | Command | Purpose |
|---|---|---|
| `ml-clarify` | `kiro-cli chat --agent ml-clarify` | Mission requirements discovery |
| `ml-plan` | `kiro-cli chat --agent ml-plan` | Generate PROMPT.md + TODO.md |
| `ml-experiment` | `kiro-cli chat --agent ml-experiment` | Run the experimentation loop |

### Execution script

```bash
# Safe execution with iteration limit
./experimenta-execute.sh --max-iterations 50

# Custom options
./experimenta-execute.sh \
  --max-iterations 30 \
  --completion-word DONE \
  --prompt-file PROMPT.md \
  --log-file execution.log
```

---

## The two contracts

### `experiment.py` — the mutable artifact

The agent rewrites this file freely. It must define exactly two functions:

```python
def prepare_data(df: pd.DataFrame, target_col: str):
    # Returns: X_train, X_test, y_train, y_test
    ...

def build_pipeline() -> Pipeline:
    # Returns an unfitted sklearn Pipeline or compatible estimator
    ...
```

### `evaluator.py` — the fixed judge

Never modified. Always computes the same metrics for every experiment:
- primary metric (from mission)
- supporting metrics (roc_auc, f1, precision, recall, etc.)
- feasibility check against hard_constraints
- latency on a 1000-row sample

---

## Output structure

```
outputs/run_<id>/
  mission_snapshot.yaml
  profile.json
  eda/
    eda.py
    results.json
  error_analysis/
    <experiment_id>.json
  experiments.json
  experiments/
    <experiment_id>/
      experiment.py
  final_experiment.py
  final_pipeline.pkl
  report.md
```

---

## Project structure

```
experimenta-ml-kiro/
├── .kiro/
│   └── agents/
│       ├── ml-clarify.json       # Mission requirements agent
│       ├── ml-plan.json          # Planning agent
│       └── ml-experiment.json    # ML experimentation agent
├── ml_agent/
│   ├── __init__.py
│   ├── evaluator.py              # Fixed judge (never modified)
│   ├── mission.py                # Load + validate mission.yaml
│   ├── profiler.py               # Dataset profiling
│   ├── recipe.py                 # Recipe ID + serialization
│   ├── tracker.py                # Experiment ledger
│   └── utils.py                  # Utilities
├── tests/
│   ├── test_evaluator.py
│   ├── test_mission.py
│   ├── test_profiler.py
│   ├── test_recipe.py
│   └── test_tracker.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── OVERVIEW.md
│   └── IDEATION.md
├── experiment.py                 # The mutable artifact
├── mission.yaml                  # Run-specific inputs
├── PROGRAM.md                    # Behavioral contract
├── RECIPES.md                    # Problem-pattern playbook
├── PROMPT.md                     # Kiro CLI execution instructions
├── TODO.md                       # Task checklist
├── experimenta-execute.sh        # Kiro CLI execution loop script
├── pyproject.toml
└── README.md
```

---

## Key files

| File | Role |
|---|---|
| `PROGRAM.md` | Permanent behavioral contract for the agent |
| `mission.yaml` | Run-specific inputs: dataset, metric, budget, domain knowledge |
| `experiment.py` | The mutable artifact the agent rewrites |
| `ml_agent/evaluator.py` | Fixed judge — never modified during a run |
| `RECIPES.md` | Problem-pattern playbook the agent can consult |
| `PROMPT.md` | Kiro CLI execution instructions |
| `TODO.md` | Task checklist for the execution loop |
| `experimenta-execute.sh` | Autonomous execution loop script |
| `.kiro/agents/*.json` | Kiro CLI agent configurations |

---

## Differences from Claude Code version

| Feature | Claude Code (original) | Kiro CLI (this port) |
|---|---|---|
| Runner | `claude --dangerously-skip-permissions` | `kiro-cli chat --agent ml-experiment` |
| Execution loop | Manual prompt in Claude Code | `./experimenta-execute.sh` or `while :; do ... done` |
| Agent config | Implicit (Claude Code reads PROGRAM.md) | Explicit `.kiro/agents/*.json` |
| Planning phase | N/A | `kiro-cli chat --agent ml-plan` → PROMPT.md + TODO.md |
| Clarification | Manual | `kiro-cli chat --agent ml-clarify` |
| Completion signal | Manual | Agent outputs "DONE" |
| Stuck detection | Manual | Agent outputs "STUCK" |

---

## Design principles

1. **One mutable file.** `experiment.py` is the only thing the agent changes.
2. **Fixed judge.** The evaluator never changes during a run.
3. **Domain knowledge first.** EDA and features must be grounded in domain context.
4. **Data and features before model.** Search data/features before model/hyperparams.
5. **Track everything.** Every experiment is logged with full reproducibility fields.
6. **Kiro CLI native.** Uses kiro-cli agents for autonomous execution.

---

## Scope (v1)

| Supported | Not supported |
|---|---|
| Tabular binary classification | Multi-class |
| Tabular regression | Deep learning |
| LR, LightGBM, XGBoost | Unrestricted library imports |
| Group, time, stratified splits | Cloud / distributed execution |
| Kiro CLI autonomous execution | Deployment automation |
| Reproducible tracking | |

---

## Sources

- **Original project**: [jooaobrum/experimenta-ml](https://github.com/jooaobrum/experimenta-ml)
- **Kiro CLI patterns**: [RichardScottOZ/ralph-kiro](https://github.com/RichardScottOZ/ralph-kiro)
- **Ralph Wiggum technique**: [ghuntley.com/ralph](https://ghuntley.com/ralph/)
- **Kiro CLI docs**: [kiro.dev/cli](https://kiro.dev/cli/)

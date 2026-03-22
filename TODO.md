# TODO — ML Experimentation

## Phase 0: Setup
- [ ] Read PROGRAM.md and mission.yaml
- [ ] Create output directory structure (outputs/run_<id>/)
- [ ] Snapshot mission.yaml to output directory
- [ ] **HARD STOP** - Verify setup is correct

## Phase 1: Data Understanding
- [ ] Run profiler → save profile.json
- [ ] Read profile.json — understand data signals
- [ ] Write eda.py to test domain knowledge hypotheses
- [ ] Run eda.py → save eda/results.json
- [ ] Read eda/results.json — note key findings
- [ ] **HARD STOP** - Review data signals before writing code

## Phase 2: Baseline
- [ ] Write baseline experiment.py (numeric-only, median imputation, simple model)
- [ ] Run evaluator → log baseline metric
- [ ] Run error analysis → save error_analysis/<id>.json
- [ ] **HARD STOP** - Verify baseline works end-to-end

## Phase 3: Data Cleaning (iterations)
- [ ] Read error analysis — identify cleaning opportunities
- [ ] Iteration: improve null handling, type casting, or column drops
- [ ] Run evaluator → keep if improved, revert if not
- [ ] Run error analysis

## Phase 4: Feature Engineering (iterations)
- [ ] Read error analysis + EDA findings
- [ ] Iteration: add domain-informed features (group-relative, temporal, interactions)
- [ ] Run evaluator → keep if improved, revert if not
- [ ] Run error analysis
- [ ] Iteration: add more features based on error patterns
- [ ] Run evaluator → keep if improved, revert if not

## Phase 5: Model Upgrade (iterations)
- [ ] Iteration: upgrade to LightGBM or XGBoost with reasonable defaults
- [ ] Run evaluator → keep if improved, revert if not
- [ ] Run error analysis

## Phase 6: Hyperparameter Tuning
- [ ] Iteration: tune hyperparameters on best recipe
- [ ] Run evaluator → keep if improved, revert if not
- [ ] **HARD STOP** - Review before finalizing

## Phase 7: Final Artifacts
- [ ] Confirm best feasible experiment is logged
- [ ] Copy best experiment.py to final_experiment.py
- [ ] Fit and save final_pipeline.pkl
- [ ] Write report.md with all required sections
- [ ] Verify all experiments are in experiments.json

---
## Completed
(Completed tasks will be moved here by the execution agent)

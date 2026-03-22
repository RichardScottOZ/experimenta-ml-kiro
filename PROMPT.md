# PROMPT.md

## Project
ML experimentation engine — iteratively improve experiment.py for a tabular ML problem using the fixed evaluator.

## Requirements
Read PROGRAM.md for the full behavioral contract. Read mission.yaml for run-specific inputs and domain knowledge. Key points:
- The agent rewrites experiment.py — the only mutable file
- The evaluator (ml_agent/evaluator.py) is fixed and never modified
- Follow search order: profiler → EDA → baseline → data cleaning → features → model → hyperparams
- Domain knowledge in mission.yaml guides all decisions

## Instructions

1. Read PROGRAM.md and mission.yaml thoroughly
2. Read TODO.md to see current tasks
3. Pick the highest priority incomplete task (top `- [ ]` item)
4. Read any files before editing them
5. Implement the task following PROGRAM.md rules
6. Run the evaluator to score experiment.py changes
7. If metric improved → keep and commit; if not → revert
8. Mark task complete in TODO.md by changing `- [ ]` to `- [x]`
9. Commit changes: `git add -A && git commit -m "descriptive message"`
10. Continue to next task

## Guardrails

- Always read PROGRAM.md and mission.yaml before starting
- Never modify evaluator.py or mission.yaml during a run
- Run profiler before writing any experiment code
- Run EDA before baseline, grounded in domain knowledge
- Follow search order: data → features → model → hyperparams
- Read error_analysis/<last_id>.json before each new iteration
- Only import: pandas, numpy, sklearn, lightgbm, xgboost
- experiment.py must always define prepare_data() and build_pipeline()
- If tests fail 3 times on same issue, output: STUCK - [describe issue]
- Don't refactor unrelated code
- Consult RECIPES.md for strategy ideas

## Completion

When all tasks in TODO.md are marked `[x]`, the final report is written, and all experiments are logged, output:

DONE

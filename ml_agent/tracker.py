import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ml_agent.evaluator import EvaluationResult


class Tracker:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self._rows: list[dict] = []
        (run_dir / "experiments").mkdir(exist_ok=True)

    def log_experiment(
        self,
        result: EvaluationResult,
        experiment_code: str,
        iteration: int,
        reproducibility: dict,
    ):
        row = {
            "experiment_id": result.experiment_id,
            "primary_metric_value": result.primary_metric_value,
            "feasible": result.feasible,
            "status": result.status,
            "iteration": iteration,
            "dataset_path": reproducibility.get("dataset_path"),
            "dataset_row_count": reproducibility.get("dataset_row_count"),
            "split_seed": reproducibility.get("split_seed"),
            "model_seed": reproducibility.get("model_seed"),
            "evaluator_version": reproducibility.get("evaluator_version"),
            "mission_snapshot_path": reproducibility.get("mission_snapshot_path"),
        }
        self._rows.append(row)

        exp_dir = self.run_dir / "experiments" / result.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "experiment.py").write_text(experiment_code)

        with open(self.run_dir / "experiments.json", "w") as f:
            json.dump(self._rows, f, indent=2)

        df = pd.DataFrame(self._rows)
        df.to_parquet(self.run_dir / "experiments.parquet", index=False)

    def log_event(self, event: str, **fields):
        entry = {"event": event, "ts": datetime.now(timezone.utc).isoformat(), **fields}
        with open(self.run_dir / "run.log", "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_all_results(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

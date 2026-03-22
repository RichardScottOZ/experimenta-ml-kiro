import json

import pandas as pd
import pytest

from ml_agent.evaluator import EvaluationResult
from ml_agent.tracker import Tracker
from ml_agent.utils import make_run_dir, write_baseline_experiment
from ml_agent.mission import Mission

EXPERIMENT_CODE = """\
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def prepare_data(df, target_col):
    y = df[target_col]
    X = df.drop(columns=[target_col]).select_dtypes(include=["number"])
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
"""

MISSION = Mission(
    project_name="test",
    task_type="binary_classification",
    target_column="target",
    train_path="data/train.csv",
    primary_metric="roc_auc",
    trial_budget=5,
    allowed_models=["lr"],
)

REPRODUCIBILITY = {
    "dataset_path": "data/train.csv",
    "dataset_row_count": 200,
    "split_seed": 42,
    "model_seed": 0,
    "evaluator_version": "1.0",
    "mission_snapshot_path": "outputs/run_abc/mission_snapshot.yaml",
}

RESULT = EvaluationResult(
    experiment_id="abc123",
    primary_metric_value=0.85,
    supporting_metrics={"roc_auc": 0.85, "f1": 0.80},
    segment_metrics={},
    feasible=True,
    violation_reasons=[],
    latency_ms=12.5,
    model_size_bytes=4096,
    status="feasible",
)


def test_log_experiment_writes_all_required_fields(tmp_path):
    tracker = Tracker(tmp_path)
    tracker.log_experiment(RESULT, EXPERIMENT_CODE, iteration=1, reproducibility=REPRODUCIBILITY)

    df = tracker.get_all_results()
    assert len(df) == 1
    row = df.iloc[0]

    assert row["experiment_id"] == "abc123"
    assert row["primary_metric_value"] == pytest.approx(0.85)
    assert row["feasible"] == True
    assert row["status"] == "feasible"
    assert row["iteration"] == 1
    assert row["dataset_path"] == "data/train.csv"
    assert row["dataset_row_count"] == 200
    assert row["split_seed"] == 42
    assert row["model_seed"] == 0
    assert row["evaluator_version"] == "1.0"
    assert row["mission_snapshot_path"] == "outputs/run_abc/mission_snapshot.yaml"


def test_experiments_parquet_readable(tmp_path):
    tracker = Tracker(tmp_path)
    tracker.log_experiment(RESULT, EXPERIMENT_CODE, iteration=1, reproducibility=REPRODUCIBILITY)

    df = pd.read_parquet(tmp_path / "experiments.parquet")
    assert len(df) >= 1
    assert "experiment_id" in df.columns


def test_run_log_jsonl_parseable(tmp_path):
    tracker = Tracker(tmp_path)
    tracker.log_event("start", iteration=1, note="test")
    tracker.log_event("end", iteration=1, metric=0.85)

    lines = (tmp_path / "run.log").read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert "event" in obj
        assert "ts" in obj


def test_experiment_code_saved_at_correct_path(tmp_path):
    tracker = Tracker(tmp_path)
    tracker.log_experiment(RESULT, EXPERIMENT_CODE, iteration=1, reproducibility=REPRODUCIBILITY)

    saved = tmp_path / "experiments" / "abc123" / "experiment.py"
    assert saved.exists()
    assert saved.read_text() == EXPERIMENT_CODE


def test_get_all_results_correct_shape(tmp_path):
    tracker = Tracker(tmp_path)
    result2 = EvaluationResult(
        experiment_id="def456",
        primary_metric_value=0.90,
        supporting_metrics={},
        segment_metrics={},
        feasible=True,
        violation_reasons=[],
        latency_ms=10.0,
        model_size_bytes=2048,
        status="feasible",
    )
    tracker.log_experiment(RESULT, EXPERIMENT_CODE, iteration=1, reproducibility=REPRODUCIBILITY)
    tracker.log_experiment(result2, EXPERIMENT_CODE, iteration=2, reproducibility=REPRODUCIBILITY)

    df = tracker.get_all_results()
    assert df.shape[0] == 2
    assert df.shape[1] == 11  # 11 tracked columns


def test_write_baseline_experiment(tmp_path):
    path = write_baseline_experiment(tmp_path, MISSION)
    assert path.exists()
    code = path.read_text()
    assert "prepare_data" in code
    assert "build_pipeline" in code
    assert "SimpleImputer" in code
    assert "LogisticRegression" in code
    assert "StandardScaler" in code

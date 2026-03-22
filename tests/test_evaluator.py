import numpy as np
import pandas as pd
import pytest

from ml_agent.evaluator import EvaluationResult, evaluate, is_improvement
from ml_agent.mission import Mission

VALID_EXPERIMENT = """\
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

REGRESSION_EXPERIMENT = """\
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def prepare_data(df, target_col):
    y = df[target_col]
    X = df.drop(columns=[target_col]).select_dtypes(include=["number"])
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge()),
    ])
"""


def _make_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feat1": rng.normal(size=n),
        "feat2": rng.uniform(size=n),
        "target": (rng.uniform(size=n) > 0.5).astype(int),
    })


def test_metrics_correct_on_synthetic_data(tmp_path):
    exp_path = tmp_path / "experiment.py"
    exp_path.write_text(VALID_EXPERIMENT)
    df = _make_df()
    result = evaluate(exp_path, df, MISSION)
    assert isinstance(result, EvaluationResult)
    assert result.status == "feasible"
    assert result.primary_metric_value > 0.0
    for metric in (
        "roc_auc", "f1", "precision", "recall", "accuracy",
        "balanced_accuracy", "average_precision", "brier_score",
        "log_loss", "ks_stat"
    ):
        assert metric in result.supporting_metrics
    assert 0.0 <= result.primary_metric_value <= 1.0


def test_infeasible_flagged_when_metric_below_threshold(tmp_path):
    exp_path = tmp_path / "experiment.py"
    exp_path.write_text(VALID_EXPERIMENT)
    df = _make_df()
    mission = Mission(
        project_name="test",
        task_type="binary_classification",
        target_column="target",
        train_path="data/train.csv",
        primary_metric="roc_auc",
        trial_budget=5,
        allowed_models=["lr"],
        hard_constraints={"min_primary_metric": 0.9999},
    )
    result = evaluate(exp_path, df, mission)
    assert result.feasible is False
    assert "metric_below_threshold" in result.violation_reasons
    assert result.status == "infeasible"


def test_latency_measured(tmp_path):
    exp_path = tmp_path / "experiment.py"
    exp_path.write_text(VALID_EXPERIMENT)
    df = _make_df()
    result = evaluate(exp_path, df, MISSION)
    assert isinstance(result.latency_ms, float)
    assert result.latency_ms > 0.0


def test_failed_experiment_syntax_error(tmp_path):
    exp_path = tmp_path / "experiment.py"
    exp_path.write_text("this is not valid python !!!")
    df = _make_df()
    result = evaluate(exp_path, df, MISSION)
    assert result.status == "failed"
    assert result.feasible is False
    assert result.primary_metric_value == 0.0


def test_failed_experiment_missing_function(tmp_path):
    exp_path = tmp_path / "experiment.py"
    exp_path.write_text("x = 1\n")
    df = _make_df()
    result = evaluate(exp_path, df, MISSION)
    assert result.status == "failed"
    assert result.feasible is False


def test_model_size_bytes_positive(tmp_path):
    exp_path = tmp_path / "experiment.py"
    exp_path.write_text(VALID_EXPERIMENT)
    df = _make_df()
    result = evaluate(exp_path, df, MISSION)
    assert isinstance(result.model_size_bytes, int)
    assert result.model_size_bytes > 0


def test_experiment_id_matches_recipe(tmp_path):
    from ml_agent.recipe import make_experiment_id
    exp_path = tmp_path / "experiment.py"
    exp_path.write_text(VALID_EXPERIMENT)
    df = _make_df()
    result = evaluate(exp_path, df, MISSION)
    assert result.experiment_id == make_experiment_id(VALID_EXPERIMENT)


def test_is_improvement():
    assert is_improvement(5.0, 10.0, "rmse") is True
    assert is_improvement(10.0, 5.0, "rmse") is False
    assert is_improvement(0.9, 0.8, "r2") is True
    assert is_improvement(0.8, 0.9, "r2") is False
    assert is_improvement(0.1, 0.2, "log_loss") is True
    assert is_improvement(0.8, 0.7, "roc_auc") is True


def test_regression_metrics_correct_on_synthetic_data(tmp_path):
    exp_path = tmp_path / "experiment_reg.py"
    exp_path.write_text(REGRESSION_EXPERIMENT)
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "feat1": rng.normal(size=n),
        "target": rng.normal(size=n),
    })
    mission = Mission(
        project_name="test_reg",
        task_type="regression",
        target_column="target",
        train_path="data/train.csv",
        primary_metric="rmse",
        trial_budget=5,
        allowed_models=["lr"],
    )
    result = evaluate(exp_path, df, mission)
    assert isinstance(result, EvaluationResult)
    assert result.status == "feasible"
    assert "rmse" in result.supporting_metrics
    assert "r2" in result.supporting_metrics
    assert result.primary_metric_value == result.supporting_metrics["rmse"]


def test_infeasible_flagged_when_metric_above_threshold(tmp_path):
    exp_path = tmp_path / "experiment_reg.py"
    exp_path.write_text(REGRESSION_EXPERIMENT)
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "feat1": rng.normal(size=n),
        "target": rng.normal(size=n),
    })
    mission = Mission(
        project_name="test_reg",
        task_type="regression",
        target_column="target",
        train_path="data/train.csv",
        primary_metric="rmse",
        trial_budget=5,
        allowed_models=["lr"],
        hard_constraints={"max_primary_metric": 0.0001},  # Impossible to achieve on random data
    )
    result = evaluate(exp_path, df, mission)
    assert result.feasible is False
    assert "metric_above_threshold" in result.violation_reasons
    assert result.status == "infeasible"

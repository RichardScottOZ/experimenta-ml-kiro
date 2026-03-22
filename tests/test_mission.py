import os
import tempfile
from pathlib import Path

import pytest
import yaml

from ml_agent.mission import Mission, MissionValidationError, load_mission, snapshot_mission


def write_mission(data: dict) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(data, f)
    f.close()
    return f.name


VALID = {
    "project_name": "test_project",
    "task_type": "binary_classification",
    "target_column": "label",
    "train_path": "data/train.csv",
    "primary_metric": "roc_auc",
    "trial_budget": 10,
    "allowed_models": ["lr", "lgbm"],
}


def test_valid_mission():
    path = write_mission(VALID)
    m = load_mission(path)
    assert isinstance(m, Mission)
    assert m.project_name == "test_project"
    assert m.task_type == "binary_classification"
    assert m.target_column == "label"
    assert m.trial_budget == 10
    assert m.allowed_models == ["lr", "lgbm"]
    os.unlink(path)


def test_missing_required_field():
    data = {k: v for k, v in VALID.items() if k != "target_column"}
    path = write_mission(data)
    with pytest.raises(MissionValidationError) as exc_info:
        load_mission(path)
    assert "target_column" in str(exc_info.value)
    os.unlink(path)


def test_multiple_missing_fields():
    data = {k: v for k, v in VALID.items() if k not in ("target_column", "train_path")}
    path = write_mission(data)
    with pytest.raises(MissionValidationError) as exc_info:
        load_mission(path)
    msg = str(exc_info.value)
    assert "target_column" in msg
    assert "train_path" in msg
    os.unlink(path)


def test_invalid_task_type():
    data = {**VALID, "task_type": "multiclass"}
    path = write_mission(data)
    with pytest.raises(MissionValidationError) as exc_info:
        load_mission(path)
    assert "task_type" in str(exc_info.value)
    os.unlink(path)


def test_invalid_trial_budget_zero():
    data = {**VALID, "trial_budget": 0}
    path = write_mission(data)
    with pytest.raises(MissionValidationError) as exc_info:
        load_mission(path)
    assert "trial_budget" in str(exc_info.value)
    os.unlink(path)


def test_invalid_trial_budget_negative():
    data = {**VALID, "trial_budget": -5}
    path = write_mission(data)
    with pytest.raises(MissionValidationError) as exc_info:
        load_mission(path)
    assert "trial_budget" in str(exc_info.value)
    os.unlink(path)


def test_invalid_allowed_models():
    data = {**VALID, "allowed_models": ["lr", "random_forest"]}
    path = write_mission(data)
    with pytest.raises(MissionValidationError) as exc_info:
        load_mission(path)
    assert "allowed_models" in str(exc_info.value)
    os.unlink(path)


def test_defaults_applied():
    path = write_mission(VALID)
    m = load_mission(path)
    assert m.split_seed == 42
    assert m.model_seed == 0
    assert m.segment_columns == []
    assert m.hard_constraints == {}
    assert m.test_path is None
    assert m.context is None
    os.unlink(path)


def test_snapshot_mission():
    path = write_mission(VALID)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        snap = snapshot_mission(path, run_dir)
        assert snap.exists()
        assert snap.name == "mission_snapshot.yaml"
        with open(snap) as f:
            content = yaml.safe_load(f)
        assert content["project_name"] == "test_project"
    os.unlink(path)


# --- Regression task_type tests ---

VALID_REGRESSION = {
    "project_name": "test_regression",
    "task_type": "regression",
    "target_column": "price",
    "train_path": "data/train.csv",
    "primary_metric": "rmse",
    "trial_budget": 5,
    "allowed_models": ["lr", "lgbm"],
}


def test_valid_regression_mission():
    path = write_mission(VALID_REGRESSION)
    m = load_mission(path)
    assert isinstance(m, Mission)
    assert m.task_type == "regression"
    assert m.primary_metric == "rmse"
    os.unlink(path)


def test_regression_with_classification_metric():
    data = {**VALID_REGRESSION, "primary_metric": "roc_auc"}
    path = write_mission(data)
    with pytest.raises(MissionValidationError) as exc_info:
        load_mission(path)
    assert "primary_metric" in str(exc_info.value)
    os.unlink(path)


def test_classification_with_regression_metric():
    data = {**VALID, "primary_metric": "rmse"}
    path = write_mission(data)
    with pytest.raises(MissionValidationError) as exc_info:
        load_mission(path)
    assert "primary_metric" in str(exc_info.value)
    os.unlink(path)


def test_unknown_task_type():
    data = {**VALID, "task_type": "time_series"}
    path = write_mission(data)
    with pytest.raises(MissionValidationError) as exc_info:
        load_mission(path)
    assert "task_type" in str(exc_info.value)
    os.unlink(path)

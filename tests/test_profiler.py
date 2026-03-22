import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_agent.mission import Mission
from ml_agent.profiler import DataLoadError, DataProfile, load_dataset, profile_dataset


MISSION = Mission(
    project_name="test",
    task_type="binary_classification",
    target_column="target",
    train_path="data/train.csv",
    primary_metric="roc_auc",
    trial_budget=5,
    allowed_models=["lr"],
)


def _make_run_dir():
    tmp = tempfile.mkdtemp()
    return Path(tmp)


# --- T2.1: Dataset loading ---

def test_load_csv(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    result = load_dataset(str(p))
    assert result.shape == (2, 2)


def test_load_parquet(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    p = tmp_path / "data.parquet"
    df.to_parquet(p, index=False)
    result = load_dataset(str(p))
    assert result.shape == (2, 2)


def test_load_bad_path():
    with pytest.raises(DataLoadError, match="not found"):
        load_dataset("/nonexistent/path/data.csv")


def test_load_unsupported_format(tmp_path):
    p = tmp_path / "data.txt"
    p.write_text("col1,col2\n1,2\n")
    with pytest.raises(DataLoadError, match="Unsupported"):
        load_dataset(str(p))


# --- T2.2: Profile computation ---

def _base_df():
    return pd.DataFrame({
        "num_col": [1.0, 2.0, 3.0, 4.0, 5.0],
        "cat_col": ["a", "b", "a", "b", "c"],
        "bool_col": [True, False, True, False, True],
        "target": [0, 1, 0, 1, 0],
    })


def test_profile_column_types():
    df = _base_df()
    run_dir = _make_run_dir()
    profile = profile_dataset(df, MISSION, Path(run_dir))
    assert profile.column_types["num_col"] == "numeric"
    assert profile.column_types["cat_col"] == "categorical"
    assert profile.column_types["bool_col"] == "boolean"


def test_profile_missingness():
    df = pd.DataFrame({
        "num_col": [1.0, None, 3.0, None, 5.0],
        "cat_col": ["a", "b", "a", "b", "c"],
        "target": [0, 1, 0, 1, 0],
    })
    run_dir = _make_run_dir()
    profile = profile_dataset(df, MISSION, Path(run_dir))
    assert abs(profile.missingness["num_col"] - 0.4) < 1e-9
    assert profile.missingness["cat_col"] == 0.0


def test_profile_skewness():
    # Exponential-like distribution has high skewness
    df = pd.DataFrame({
        "skewed_col": [1.0, 1.0, 1.0, 1.0, 100.0],
        "target": [0, 1, 0, 1, 0],
    })
    run_dir = _make_run_dir()
    profile = profile_dataset(df, MISSION, Path(run_dir))
    assert profile.skewness["skewed_col"] > 1.0


def test_profile_target_distribution():
    df = pd.DataFrame({
        "feat": [1, 2, 3, 4, 5, 6, 7, 8],
        "target": [0, 0, 0, 0, 0, 0, 1, 1],
    })
    run_dir = _make_run_dir()
    profile = profile_dataset(df, MISSION, Path(run_dir))
    assert profile.target_distribution["0"] == 6
    assert profile.target_distribution["1"] == 2
    assert abs(profile.target_distribution["imbalance_ratio"] - 3.0) < 1e-9


def test_profile_likely_ids():
    # Column with unique string per row → id-like
    n = 100
    df = pd.DataFrame({
        "id_col": [f"user_{i}" for i in range(n)],
        "num_col": list(range(n)),
        "target": [i % 2 for i in range(n)],
    })
    run_dir = _make_run_dir()
    profile = profile_dataset(df, MISSION, Path(run_dir))
    assert "id_col" in profile.likely_ids
    assert "num_col" not in profile.likely_ids


def test_profile_leakage_candidates():
    n = 50
    target = [i % 2 for i in range(n)]
    df = pd.DataFrame({
        "leak_col": [float(t) for t in target],  # perfect correlation
        "safe_col": list(range(n)),
        "target": target,
    })
    run_dir = _make_run_dir()
    profile = profile_dataset(df, MISSION, Path(run_dir))
    assert "leak_col" in profile.leakage_candidates
    assert "safe_col" not in profile.leakage_candidates


def test_profile_grouping_key_candidates():
    n = 100
    df = pd.DataFrame({
        "group_col": [f"group_{i % 5}" for i in range(n)],   # 5 unique groups → moderate
        "num_col": list(range(n)),
        "target": [i % 2 for i in range(n)],
    })
    run_dir = _make_run_dir()
    profile = profile_dataset(df, MISSION, Path(run_dir))
    assert "group_col" in profile.grouping_key_candidates
    assert "num_col" not in profile.grouping_key_candidates


def test_profile_json_saved():
    df = _base_df()
    run_dir = _make_run_dir()
    profile_dataset(df, MISSION, Path(run_dir))
    json_path = Path(run_dir) / "profile.json"
    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)
    assert "n_rows" in data
    assert "column_types" in data
    assert data["n_rows"] == 5


def test_profile_returns_dataprofile():
    df = _base_df()
    run_dir = _make_run_dir()
    profile = profile_dataset(df, MISSION, Path(run_dir))
    assert isinstance(profile, DataProfile)


def test_profile_continuous_target():
    # Synthetic regression mission
    regression_mission = Mission(
        project_name="test_reg",
        task_type="regression",
        target_column="target",
        train_path="data/train.csv",
        primary_metric="rmse",
        trial_budget=5,
        allowed_models=["lr"],
    )
    df = pd.DataFrame({
        "feat": [1, 2, 3, 4, 5],
        "target": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    run_dir = _make_run_dir()
    profile = profile_dataset(df, regression_mission, Path(run_dir))

    dist = profile.target_distribution
    assert dist["min"] == 10.0
    assert dist["max"] == 50.0
    assert dist["mean"] == 30.0
    assert dist["median"] == 30.0
    assert "skewness" in dist
    assert "kurtosis" in dist
    assert dist["pct_zeros"] == 0.0
    assert dist["pct_negative"] == 0.0


def test_profile_heavy_skew():
    regression_mission = Mission(
        project_name="test_skew",
        task_type="regression",
        target_column="target",
        train_path="data/train.csv",
        primary_metric="rmse",
        trial_budget=5,
        allowed_models=["lr"],
    )
    # Very skewed distribution: mostly small values, one huge outlier
    df = pd.DataFrame({
        "feat": list(range(100)),
        "target": [1.0] * 99 + [1000.0],
    })
    run_dir = _make_run_dir()
    profile = profile_dataset(df, regression_mission, Path(run_dir))
    assert profile.target_distribution["is_heavy_skew"] is True


def test_profile_classification_unchanged():
    # Ensure classification target distribution still has counts and imbalance_ratio
    df = pd.DataFrame({
        "feat": [1, 2, 3, 4, 5, 6],
        "target": [0, 0, 0, 0, 1, 1],
    })
    run_dir = _make_run_dir()
    profile = profile_dataset(df, MISSION, Path(run_dir))
    assert "0" in profile.target_distribution
    assert "1" in profile.target_distribution
    assert "imbalance_ratio" in profile.target_distribution
    assert "mean" not in profile.target_distribution

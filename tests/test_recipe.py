import pytest

from ml_agent.recipe import Recipe, RecipeError, diff_recipes, from_yaml, make_experiment_id, to_yaml

SAMPLE_CODE = """\
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def prepare_data(df, target_col):
    y = df[target_col]
    X = df.drop(columns=[target_col])
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_pipeline():
    return Pipeline([("model", LogisticRegression())])
"""


def make_sample_recipe() -> Recipe:
    return Recipe(
        experiment_id=make_experiment_id(SAMPLE_CODE),
        iteration=0,
        primary_metric_value=0.75,
        status="feasible",
    )


def test_recipe_id_stable_across_reserialization():
    recipe = make_sample_recipe()
    restored = from_yaml(to_yaml(recipe))
    assert restored.experiment_id == recipe.experiment_id


def test_round_trip_lossless():
    recipe = make_sample_recipe()
    restored = from_yaml(to_yaml(recipe))
    assert restored == recipe


def test_diff_returns_correct_changed_keys():
    a = make_sample_recipe()
    b = make_sample_recipe()
    b.status = "infeasible"
    b.primary_metric_value = 0.60
    diff = diff_recipes(a, b)
    assert "status" in diff
    assert "primary_metric_value" in diff
    assert "experiment_id" not in diff
    assert "iteration" not in diff


def test_diff_identical_recipes_empty():
    a = make_sample_recipe()
    b = make_sample_recipe()
    assert diff_recipes(a, b) == {}


def test_diff_old_new_values():
    a = make_sample_recipe()
    b = make_sample_recipe()
    b.status = "failed"
    diff = diff_recipes(a, b)
    assert diff["status"]["old"] == "feasible"
    assert diff["status"]["new"] == "failed"


def test_from_yaml_missing_fields_raises():
    with pytest.raises(RecipeError, match="Missing recipe fields"):
        from_yaml("experiment_id: abc\n")


def test_from_yaml_invalid_type_raises():
    with pytest.raises(RecipeError, match="expected a mapping"):
        from_yaml("- not\n- a\n- mapping\n")


def test_make_experiment_id_deterministic():
    assert make_experiment_id(SAMPLE_CODE) == make_experiment_id(SAMPLE_CODE)


def test_make_experiment_id_differs_for_different_code():
    other_code = SAMPLE_CODE + "\n# comment\n"
    assert make_experiment_id(SAMPLE_CODE) != make_experiment_id(other_code)

import hashlib
from dataclasses import dataclass

import yaml


class RecipeError(Exception):
    pass


@dataclass
class Recipe:
    experiment_id: str   # SHA256 of experiment.py content
    iteration: int
    primary_metric_value: float | None
    status: str          # "feasible" | "infeasible" | "failed" | "baseline"


def make_experiment_id(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def to_yaml(recipe: Recipe) -> str:
    data = {
        "experiment_id": recipe.experiment_id,
        "iteration": recipe.iteration,
        "primary_metric_value": recipe.primary_metric_value,
        "status": recipe.status,
    }
    return yaml.dump(data, default_flow_style=False, sort_keys=True)


def from_yaml(s: str) -> Recipe:
    raw = yaml.safe_load(s)
    if not isinstance(raw, dict):
        raise RecipeError("Invalid recipe YAML: expected a mapping.")
    required = {"experiment_id", "iteration", "primary_metric_value", "status"}
    missing = required - set(raw.keys())
    if missing:
        raise RecipeError(f"Missing recipe fields: {sorted(missing)}")
    return Recipe(
        experiment_id=raw["experiment_id"],
        iteration=raw["iteration"],
        primary_metric_value=raw["primary_metric_value"],
        status=raw["status"],
    )


def diff_recipes(a: Recipe, b: Recipe) -> dict:
    diffs = {}
    for key in ("experiment_id", "iteration", "primary_metric_value", "status"):
        val_a = getattr(a, key)
        val_b = getattr(b, key)
        if val_a != val_b:
            diffs[key] = {"old": val_a, "new": val_b}
    return diffs

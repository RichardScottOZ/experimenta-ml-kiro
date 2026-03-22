import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class MissionValidationError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


@dataclass
class Mission:
    project_name: str
    task_type: str
    target_column: str
    train_path: str
    primary_metric: str
    trial_budget: int
    allowed_models: list[str]
    test_path: str | None = None
    segment_columns: list[str] = field(default_factory=list)
    hard_constraints: dict = field(default_factory=dict)
    split_seed: int = 42
    model_seed: int = 0
    context: str | None = None

    @classmethod
    def from_yaml(cls, path: str) -> "Mission":
        return load_mission(path)


_REQUIRED = ["project_name", "task_type", "target_column", "train_path", "primary_metric", "trial_budget", "allowed_models"]
_ALLOWED_MODELS = {"lr", "lgbm", "xgb", "rf", "dt", "lasso", "svm", "poisson"}
_VALID_TASK_TYPES = {"binary_classification", "regression"}
_CLASSIFICATION_METRICS = {
    "roc_auc",
    "f1",
    "precision",
    "recall",
    "accuracy",
    "balanced_accuracy",
    "average_precision",
    "brier_score",
    "log_loss",
    "ks_stat",
}
_REGRESSION_METRICS = {
    "rmse",
    "mae",
    "mape",
    "r2",
    "median_ae",
    "max_error",
}
_VALID_METRICS = _CLASSIFICATION_METRICS | _REGRESSION_METRICS
_METRICS_BY_TASK = {
    "binary_classification": _CLASSIFICATION_METRICS,
    "regression": _REGRESSION_METRICS,
}


def load_mission(path: str) -> Mission:
    with open(path) as f:
        raw = yaml.safe_load(f)

    errors = []

    for key in _REQUIRED:
        if key not in raw or raw[key] is None:
            errors.append(f"Missing required field: '{key}'")

    if "task_type" in raw and raw["task_type"] not in _VALID_TASK_TYPES:
        errors.append(f"Invalid task_type: '{raw['task_type']}'. Must be one of {sorted(_VALID_TASK_TYPES)}.")

    if "trial_budget" in raw and raw["trial_budget"] is not None:
        if not isinstance(raw["trial_budget"], int) or raw["trial_budget"] < 1:
            errors.append(f"Invalid trial_budget: '{raw['trial_budget']}'. Must be an integer >= 1.")

    if "allowed_models" in raw and raw["allowed_models"] is not None:
        invalid = [m for m in raw["allowed_models"] if m not in _ALLOWED_MODELS]
        if not raw["allowed_models"]:
            errors.append("allowed_models must not be empty.")
        elif invalid:
            errors.append(f"Invalid allowed_models: {invalid}. Allowed: {sorted(_ALLOWED_MODELS)}.")

    task_type = raw.get("task_type")
    if "primary_metric" in raw and raw["primary_metric"] is not None and task_type in _METRICS_BY_TASK:
        allowed_metrics = _METRICS_BY_TASK[task_type]
        if raw["primary_metric"] not in allowed_metrics:
            errors.append(f"Invalid primary_metric: '{raw['primary_metric']}'. Allowed for {task_type}: {sorted(allowed_metrics)}.")

    if errors:
        raise MissionValidationError(errors)

    return Mission(
        project_name=raw["project_name"],
        task_type=raw["task_type"],
        target_column=raw["target_column"],
        train_path=raw["train_path"],
        primary_metric=raw["primary_metric"],
        trial_budget=raw["trial_budget"],
        allowed_models=raw["allowed_models"],
        test_path=raw.get("test_path"),
        segment_columns=raw.get("segment_columns", []),
        hard_constraints=raw.get("hard_constraints", {}),
        split_seed=raw.get("split_seed", 42),
        model_seed=raw.get("model_seed", 0),
        context=raw.get("context"),
    )


def snapshot_mission(source_path: str, run_dir: Path) -> Path:
    dest = run_dir / "mission_snapshot.yaml"
    shutil.copy2(source_path, dest)
    return dest

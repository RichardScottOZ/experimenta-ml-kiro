import importlib.util
import pickle
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
import numpy as np

from ml_agent.mission import Mission
from ml_agent.recipe import make_experiment_id


class EvaluationError(Exception):
    pass


@dataclass
class EvaluationResult:
    experiment_id: str
    primary_metric_value: float
    supporting_metrics: dict
    segment_metrics: dict
    feasible: bool
    violation_reasons: list
    latency_ms: float
    model_size_bytes: int
    status: str


def _failed_result(experiment_id, error: str = ""):
    reasons = [f"error: {error}"] if error else []
    return EvaluationResult(
        experiment_id=experiment_id,
        primary_metric_value=0.0,
        supporting_metrics={},
        segment_metrics={},
        feasible=False,
        violation_reasons=reasons,
        latency_ms=0.0,
        model_size_bytes=0,
        status="failed",
    )


_LOWER_IS_BETTER = {
    "rmse",
    "mae",
    "mape",
    "median_ae",
    "max_error",
    "log_loss",
    "brier_score",
}


def is_improvement(new_val, old_val, metric_name):
    if metric_name in _LOWER_IS_BETTER:
        return new_val < old_val
    return new_val > old_val


def _compute_classification_metrics(y_true, y_scores):
    y_pred = (y_scores >= 0.5).astype(int)
    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        ks = float(ks_2samp(pos_scores, neg_scores).statistic)
    else:
        ks = 0.0
    return {
        "roc_auc": float(roc_auc_score(y_true, y_scores)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "average_precision": float(average_precision_score(y_true, y_scores)),
        "brier_score": float(brier_score_loss(y_true, y_scores)),
        "log_loss": float(log_loss(y_true, y_scores, labels=[0, 1])),
        "ks_stat": ks,
    }


def _compute_regression_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "max_error": float(max_error(y_true, y_pred)),
    }


def evaluate(experiment_path, df, mission):
    code = experiment_path.read_text()
    experiment_id = make_experiment_id(code)

    sys.modules.pop("experiment", None)
    try:
        spec = importlib.util.spec_from_file_location("experiment", experiment_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        return _failed_result(experiment_id, traceback.format_exc())

    try:
        X_train, X_test, y_train, y_test = module.prepare_data(df, mission.target_column)
        pipeline = module.build_pipeline()
        pipeline.fit(X_train, y_train)
        if mission.task_type == "regression":
            y_out = pipeline.predict(X_test)
        else:
            y_out = pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        return _failed_result(experiment_id, traceback.format_exc())

    sample = X_test.sample(n=1000, replace=True, random_state=0)
    t0 = time.perf_counter()
    if mission.task_type == "regression":
        pipeline.predict(sample)
    else:
        pipeline.predict_proba(sample)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    model_size_bytes = len(pickle.dumps(pipeline))

    if mission.task_type == "regression":
        supporting_metrics = _compute_regression_metrics(y_test.values, y_out)
    else:
        supporting_metrics = _compute_classification_metrics(y_test.values, y_out)
    primary_metric_value = supporting_metrics[mission.primary_metric]

    segment_metrics = {}
    if mission.segment_columns:
        if y_test.index.isin(df.index).all():
            df_test = df.loc[y_test.index]
        else:
            df_test = None
        for col in mission.segment_columns:
            if df_test is None or col not in df_test.columns:
                continue
            segment_metrics[col] = {}
            for val in df_test[col].unique():
                mask = (df_test[col] == val).values
                y_seg = y_test.values[mask]
                s_seg = y_out[mask]
                if mission.task_type == "binary_classification" and len(set(y_seg)) < 2:
                    continue
                if len(y_seg) == 0:
                    continue
                try:
                    if mission.task_type == "regression":
                        segment_metrics[col][str(val)] = _compute_regression_metrics(y_seg, s_seg)
                    else:
                        segment_metrics[col][str(val)] = _compute_classification_metrics(y_seg, s_seg)
                except Exception:
                    continue

    feasible = True
    violation_reasons = []
    constraints = mission.hard_constraints

    if "min_primary_metric" in constraints:
        if primary_metric_value < constraints["min_primary_metric"]:
            feasible = False
            violation_reasons.append("metric_below_threshold")

    if "max_primary_metric" in constraints:
        if primary_metric_value > constraints["max_primary_metric"]:
            feasible = False
            violation_reasons.append("metric_above_threshold")

    if "max_latency_ms" in constraints:
        if latency_ms > constraints["max_latency_ms"]:
            feasible = False
            violation_reasons.append("latency_exceeded")

    if "max_model_size_mb" in constraints:
        if model_size_bytes > constraints["max_model_size_mb"] * 1024 * 1024:
            feasible = False
            violation_reasons.append("model_size_exceeded")

    status = "feasible" if feasible else "infeasible"

    return EvaluationResult(
        experiment_id=experiment_id,
        primary_metric_value=primary_metric_value,
        supporting_metrics=supporting_metrics,
        segment_metrics=segment_metrics,
        feasible=feasible,
        violation_reasons=violation_reasons,
        latency_ms=latency_ms,
        model_size_bytes=model_size_bytes,
        status=status,
    )

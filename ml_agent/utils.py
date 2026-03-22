import uuid
from pathlib import Path

from ml_agent.mission import Mission


def make_run_dir(base: str = "outputs") -> Path:
    run_id = uuid.uuid4().hex
    run_dir = Path(base) / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "recipes").mkdir()
    (run_dir / "pipelines").mkdir()
    return run_dir


_BASELINE_TEMPLATE = """\
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer


def prepare_data(df, target_col):
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Drop columns that are purely identifiers (unique string per row)
    n = len(X)
    id_cols = [c for c in X.columns if X[c].dtype == object and X[c].nunique() > 0.9 * n]
    X = X.drop(columns=id_cols)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def build_pipeline():
    # Determined at fit time via ColumnTransformer — handles any mix of numeric/categorical
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    from sklearn.impute import SimpleImputer

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    # Selector transformer that splits columns at fit time
    class AutoColumnTransformer(ColumnTransformer):
        def fit(self, X, y=None):
            self.num_cols_ = X.select_dtypes(include="number").columns.tolist()
            self.cat_cols_ = X.select_dtypes(exclude="number").columns.tolist()
            self.transformers = [
                ("num", numeric_pipe, self.num_cols_),
                ("cat", categorical_pipe, self.cat_cols_),
            ]
            return super().fit(X, y)

        def transform(self, X):
            return super().transform(X)

    return Pipeline([
        ("preprocessor", AutoColumnTransformer(remainder="drop")),
        ("model", LogisticRegression(max_iter=1000, random_state=0)),
    ])
"""


def write_baseline_experiment(run_dir: Path, mission: Mission) -> Path:
    dest = run_dir / "experiment.py"
    dest.write_text(_BASELINE_TEMPLATE)
    return dest

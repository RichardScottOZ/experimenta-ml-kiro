import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def prepare_data(df: pd.DataFrame, target_col: str):
    """
    Standard data preparation placeholder.
    Splits data into train and test sets.
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Simple random split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def build_pipeline():
    """
    Standard pipeline placeholder.
    Includes simple imputation, scaling, and a baseline model.
    """
    # Identify numeric columns (implementation should be dynamic based on X in actual use)
    # This is a template; the agent will rewrite this for specific datasets.

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=42)),
        ]
    )

    return pipeline

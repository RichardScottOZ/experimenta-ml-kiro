import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import pandas as pd

from ml_agent.mission import Mission


class DataLoadError(Exception):
    pass


@dataclass
class DataProfile:
    n_rows: int
    n_cols: int

    @property
    def row_count(self) -> int:
        return self.n_rows

    @property
    def col_count(self) -> int:
        return self.n_cols
    column_types: dict
    missingness: dict
    cardinality: dict
    skewness: dict
    target_distribution: dict
    likely_ids: list
    leakage_candidates: list
    grouping_key_candidates: list
    # EDA extensions — give the agent full visibility into the data
    column_samples: dict = field(default_factory=dict)       # up to 10 non-null representative values per column
    numeric_stats: dict = field(default_factory=dict)        # min, max, mean, median, std, q25, q75
    top_categories: dict = field(default_factory=dict)       # top-10 value counts for categoricals
    target_correlation: dict = field(default_factory=dict)   # point-biserial / eta correlation with target
    high_missingness_cols: list = field(default_factory=list)  # cols with >30% missing
    text_pattern_hints: dict = field(default_factory=dict)   # heuristic hints about string structure


def load_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise DataLoadError(f"File not found: {path}")
    ext = p.suffix.lower()
    try:
        if ext == ".csv":
            return pd.read_csv(p)
        elif ext == ".parquet":
            return pd.read_parquet(p)
        else:
            raise DataLoadError(f"Unsupported file format: '{ext}'. Expected .csv or .parquet.")
    except DataLoadError:
        raise
    except Exception as e:
        raise DataLoadError(f"Failed to load dataset from '{path}': {e}")


def _infer_column_type(series: pd.Series, n_rows: int) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        unique_vals = set(series.dropna().unique())
        if unique_vals <= {0, 1, True, False}:
            return "boolean"
        return "numeric"
    cardinality = series.nunique()
    if n_rows > 0 and (cardinality / n_rows) > 0.9:
        return "id-like"
    return "categorical"


def _text_pattern_hint(series: pd.Series) -> str:
    """Heuristic: detect common extractable patterns in string columns."""
    sample = series.dropna().astype(str).head(20)
    hints = []

    # comma-separated format (e.g. "Braund, Mr. Owen Harris")
    if sample.str.contains(r",\s+\w+\.", regex=True).mean() > 0.5:
        hints.append("looks like: 'Lastname, Title Firstname' — consider extracting Title via regex r'([A-Za-z]+)\\.'")

    # prefix letter + number (e.g. "C85", "B57")
    if sample.str.match(r"^[A-Za-z]\d+").mean() > 0.5:
        hints.append("looks like: prefix-letter + number (e.g. 'C85') — consider extracting first letter as category")

    # slash-separated (e.g. ticket "PC 17599", "A/5 21171")
    if sample.str.contains(r"[A-Za-z]/\d|[A-Za-z]/[A-Za-z]", regex=True).mean() > 0.3:
        hints.append("contains slash-separated codes — consider extracting prefix before slash as ticket type")

    # purely numeric string (should have been numeric)
    if sample.str.match(r"^\d+(\.\d+)?$").mean() > 0.8:
        hints.append("numeric values stored as strings — consider casting to float")

    # space-separated multi-word
    if sample.str.contains(r"\s").mean() > 0.7 and not hints:
        hints.append("multi-word string — may contain extractable tokens (first word, last word, word count)")

    return "; ".join(hints) if hints else ""


def _numeric_stats(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) == 0:
        return {}
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
        "q25": float(s.quantile(0.25)),
        "q75": float(s.quantile(0.75)),
    }


def _continuous_target_stats(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) == 0:
        return {}

    skew = float(s.skew())
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
        "skewness": skew,
        "kurtosis": float(s.kurt()),
        "q25": float(s.quantile(0.25)),
        "q75": float(s.quantile(0.75)),
        "pct_zeros": float((s == 0).mean()),
        "pct_negative": float((s < 0).mean()),
        "is_heavy_skew": abs(skew) > 1.5,
    }


def _target_correlation(series: pd.Series, target: pd.Series, col_type: str) -> float:
    """
    Numeric columns: Pearson correlation with target.
    Categorical/boolean: eta (correlation ratio) — variance explained by group means.
    Returns 0.0 on error.
    """
    try:
        if col_type in ("numeric", "boolean"):
            return float(abs(series.corr(target)))
        else:
            # eta correlation ratio
            groups = [target[series == val].dropna() for val in series.dropna().unique()]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) < 2:
                return 0.0
            grand_mean = target.mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
            ss_total = ((target - grand_mean) ** 2).sum()
            return float((ss_between / ss_total) ** 0.5) if ss_total > 0 else 0.0
    except Exception:
        return 0.0


def profile_dataset(df: pd.DataFrame, mission: Mission, run_dir: Path) -> DataProfile:
    target = mission.target_column
    feature_cols = [c for c in df.columns if c != target]
    n_rows, n_cols = df.shape

    column_types = {c: _infer_column_type(df[c], n_rows) for c in feature_cols}
    missingness = {c: float(df[c].isnull().mean()) for c in feature_cols}
    cardinality = {c: int(df[c].nunique()) for c in feature_cols}
    skewness = {
        c: float(df[c].skew())
        for c in feature_cols
        if column_types[c] == "numeric"
    }

    if mission.task_type == "regression":
        target_distribution = _continuous_target_stats(df[target])
    else:
        target_counts = df[target].value_counts().to_dict()
        target_counts = {str(k): int(v) for k, v in target_counts.items()}
        counts = list(target_counts.values())
        imbalance_ratio = float(max(counts) / min(counts)) if len(counts) >= 2 and min(counts) > 0 else 1.0
        target_distribution = {**target_counts, "imbalance_ratio": imbalance_ratio}

    likely_ids = [c for c, t in column_types.items() if t == "id-like"]

    leakage_candidates = []
    if pd.api.types.is_numeric_dtype(df[target]):
        for c in feature_cols:
            if column_types[c] in ("numeric", "boolean"):
                try:
                    corr = abs(df[c].corr(df[target]))
                    if corr > 0.95:
                        leakage_candidates.append(c)
                except Exception:
                    pass

    grouping_key_candidates = [
        c for c in feature_cols
        if column_types[c] == "categorical"
        and 2 < cardinality[c] < n_rows * 0.5
    ]

    # --- EDA extensions ---

    # Full column samples — up to 10 representative non-null values per column
    column_samples = {}
    for c in feature_cols:
        non_null = df[c].dropna()
        if len(non_null) == 0:
            column_samples[c] = []
            continue
        # for categoricals sample unique values; for numerics sample spread
        if column_types[c] in ("categorical", "id-like"):
            unique_vals = non_null.unique()
            sample = unique_vals[:10].tolist()
        else:
            sample = non_null.sample(min(10, len(non_null)), random_state=0).tolist()
        column_samples[c] = [str(v) if not isinstance(v, (int, float, bool)) else v for v in sample]

    # Numeric stats (min/max/mean/median/std/q25/q75)
    numeric_stats = {
        c: _numeric_stats(df[c])
        for c in feature_cols
        if column_types[c] == "numeric"
    }

    # Top-10 value counts for categorical/boolean columns
    top_categories = {}
    for c in feature_cols:
        if column_types[c] in ("categorical", "boolean", "id-like"):
            vc = df[c].value_counts().head(10).to_dict()
            top_categories[c] = {str(k): int(v) for k, v in vc.items()}

    # Correlation / association with target
    target_series = df[target].astype(float)
    target_correlation = {
        c: _target_correlation(df[c], target_series, column_types[c])
        for c in feature_cols
    }

    # High-missingness columns (>30%)
    high_missingness_cols = [c for c, m in missingness.items() if m > 0.30]

    # Text pattern hints for string/object/id-like columns
    text_pattern_hints = {}
    for c in feature_cols:
        if column_types[c] in ("categorical", "id-like") and df[c].dtype == object:
            hint = _text_pattern_hint(df[c])
            if hint:
                text_pattern_hints[c] = hint

    profile = DataProfile(
        n_rows=n_rows,
        n_cols=n_cols,
        column_types=column_types,
        missingness=missingness,
        cardinality=cardinality,
        skewness=skewness,
        target_distribution=target_distribution,
        likely_ids=likely_ids,
        leakage_candidates=leakage_candidates,
        grouping_key_candidates=grouping_key_candidates,
        column_samples=column_samples,
        numeric_stats=numeric_stats,
        top_categories=top_categories,
        target_correlation=target_correlation,
        high_missingness_cols=high_missingness_cols,
        text_pattern_hints=text_pattern_hints,
    )

    with open(run_dir / "profile.json", "w") as f:
        json.dump(asdict(profile), f, indent=2)

    return profile

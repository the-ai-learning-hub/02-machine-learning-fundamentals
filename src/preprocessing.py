"""Feature engineering and preprocessing pipelines shared by the supervised-learning notebooks.

Every transformer here learns its statistics in ``fit`` (on the training data only) and applies
them in ``transform``. That is what keeps the pipelines free of data leakage during
cross-validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import HOUSING_BINARY, HOUSING_NUMERIC

# ----------------------------------------------------------------------------- housing


def make_housing_preprocessor(scale: bool = True) -> ColumnTransformer:
    """One-hot encode the yes/no columns and (optionally) standardise the numeric ones."""
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric, HOUSING_NUMERIC),
            ("bin", OneHotEncoder(drop="if_binary", dtype=float), HOUSING_BINARY),
        ],
        verbose_feature_names_out=False,
    )


# ----------------------------------------------------------------------------- titanic

#: Columns that are only known *after* the sinking. Using them is data leakage.
TITANIC_LEAKY = ["boat", "body"]
#: Columns we drop for other reasons (identifiers, free text, mostly missing).
TITANIC_DROP = ["ticket", "home.dest"]

TITANIC_NUMERIC = ["age", "fare", "sibsp", "parch", "family_size"]
TITANIC_CATEGORICAL = ["pclass", "sex", "embarked", "title", "deck", "is_alone"]
TITANIC_FEATURES = TITANIC_NUMERIC + TITANIC_CATEGORICAL

_TITLE_MAP = {
    "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Lady": "Rare", "Dona": "Rare",
    "Countess": "Rare", "the Countess": "Rare", "Sir": "Rare", "Don": "Rare", "Jonkheer": "Rare",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare", "Capt": "Rare",
}


def extract_title(name: str) -> str:
    """``"Allen, Miss. Elisabeth Walton"`` → ``"Miss"``. Uncommon titles become ``"Rare"``."""
    raw = name.split(",", 1)[-1].split(".", 1)[0].strip()
    return _TITLE_MAP.get(raw, "Rare")


def add_titanic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``title``, ``family_size``, ``is_alone`` and ``deck``.

    These features only use information available at boarding time, row by row,
    so they can be computed before the train/test split without leaking anything.
    """
    out = df.copy()
    out["title"] = out["name"].map(extract_title)
    out["family_size"] = out["sibsp"] + out["parch"] + 1
    out["is_alone"] = (out["family_size"] == 1).map({True: "yes", False: "no"})
    out["deck"] = out["cabin"].fillna("U").astype(str).str[0]
    out["deck"] = out["deck"].replace({"T": "U"})  # a single passenger on the boat deck
    out["pclass"] = out["pclass"].astype(str)
    return out


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Fill missing ``target`` values with the median of ``target`` within each ``group``.

    The medians are learned in ``fit`` (training data only). Groups that are unseen or
    have no data fall back to the overall training median.
    """

    def __init__(self, target: str = "age", group: str = "title"):
        self.target = target
        self.group = group

    def fit(self, X: pd.DataFrame, y=None) -> "GroupMedianImputer":
        self.medians_ = X.groupby(self.group)[self.target].median().to_dict()
        self.global_median_ = float(X[self.target].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        fill = X[self.group].map(self.medians_).astype(float).fillna(self.global_median_)
        X[self.target] = X[self.target].fillna(fill)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def make_titanic_preprocessor() -> Pipeline:
    """Age imputation by title → scale numeric columns + one-hot encode categorical ones."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", dtype=float)),
        ]
    )
    columns = ColumnTransformer(
        [("num", numeric, TITANIC_NUMERIC), ("cat", categorical, TITANIC_CATEGORICAL)],
        verbose_feature_names_out=False,
    )
    return Pipeline([("age_by_title", GroupMedianImputer("age", "title")), ("columns", columns)])

"""Baseline model definitions for the Version 1 salary project."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

RANDOM_STATE = 42
BASELINE_RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def build_linear_regression_model() -> LinearRegression:
    """Return the baseline linear regression model."""
    return LinearRegression()


def build_random_forest_model() -> RandomForestRegressor:
    """Return the baseline random forest regressor."""
    return RandomForestRegressor(**BASELINE_RANDOM_FOREST_PARAMS)


def get_baseline_models() -> dict[str, object]:
    """Return all baseline models used in Version 1."""
    return {
        "linear_regression": build_linear_regression_model(),
        "random_forest": build_random_forest_model(),
    }

"""Baseline model definitions for the Version 1 salary project."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import KFold

RANDOM_STATE = 42
RIDGE_REGULARIZATION_ALPHAS = np.logspace(-4, 4, 81)
LASSO_REGULARIZATION_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
RIDGE_CV = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
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


def build_ridge_regression_model() -> RidgeCV:
    """Return the regularized ridge regression baseline."""
    return RidgeCV(
        alphas=RIDGE_REGULARIZATION_ALPHAS,
        cv=RIDGE_CV,
        scoring="neg_root_mean_squared_error",
    )


def build_lasso_regression_model() -> LassoCV:
    """Return the regularized lasso regression baseline."""
    return LassoCV(
        alphas=LASSO_REGULARIZATION_ALPHAS,
        cv=5,
        random_state=RANDOM_STATE,
        max_iter=20000,
    )


def build_random_forest_model() -> RandomForestRegressor:
    """Return the baseline random forest regressor."""
    return RandomForestRegressor(**BASELINE_RANDOM_FOREST_PARAMS)


def get_baseline_models() -> dict[str, object]:
    """Return all baseline models used in Version 1."""
    return {
        "linear_regression": build_linear_regression_model(),
        "ridge_regression": build_ridge_regression_model(),
        "lasso_regression": build_lasso_regression_model(),
        "random_forest": build_random_forest_model(),
    }

"""Starter training utilities for baseline regression models."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def get_baseline_models(random_state: int = 42) -> dict[str, object]:
    """Return the first regression models planned for the project."""
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(random_state=random_state),
    }


if __name__ == "__main__":
    models = get_baseline_models()
    print("Available models:", ", ".join(models))

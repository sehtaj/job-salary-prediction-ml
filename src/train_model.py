"""Stage 8 baseline model training workflow for the Version 1 project."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.model_registry import (
    BASELINE_RANDOM_FOREST_PARAMS,
    LASSO_REGULARIZATION_ALPHAS,
    RANDOM_FOREST_CV,
    RANDOM_FOREST_RANDOM_SEARCH_ITERATIONS,
    RANDOM_FOREST_SEARCH_SPACE,
    RANDOM_STATE,
    RIDGE_CV,
    RIDGE_REGULARIZATION_ALPHAS,
    get_baseline_models,
)
from src.preprocessing import build_preprocessor, get_feature_groups

X_TRAIN_PATH = Path("data/splits/X_train.csv")
Y_TRAIN_PATH = Path("data/splits/y_train.csv")
MODEL_DIR = Path("models")
TRAINING_REPORT_PATH = Path("results/training/training_report.md")


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the saved training split from Stage 7."""
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)["avg_salary"]
    return X_train, y_train


def build_training_pipelines(X_train: pd.DataFrame) -> dict[str, Pipeline]:
    """Create a reusable preprocessing + model pipeline for each baseline model."""
    numerical_features, categorical_features = get_feature_groups(X_train)
    preprocessor = build_preprocessor(numerical_features, categorical_features)

    models = get_baseline_models()
    return {
        model_name: Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        for model_name, model in models.items()
    }


def train_pipelines(pipelines: dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Fit each baseline pipeline on the training split."""
    for model_name, pipeline in pipelines.items():
        if model_name == "random_forest":
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=RANDOM_FOREST_SEARCH_SPACE,
                n_iter=RANDOM_FOREST_RANDOM_SEARCH_ITERATIONS,
                cv=RANDOM_FOREST_CV,
                scoring="neg_root_mean_squared_error",
                random_state=RANDOM_STATE,
                n_jobs=1,
                refit=True,
            )
            search.fit(X_train, y_train)
            pipelines[model_name] = search.best_estimator_
            continue

        pipeline.fit(X_train, y_train)


def save_trained_pipelines(pipelines: dict[str, Pipeline]) -> dict[str, Path]:
    """Persist fitted baseline pipelines under the models directory."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}

    for model_name, pipeline in pipelines.items():
        output_path = MODEL_DIR / f"{model_name}_pipeline.joblib"
        joblib.dump(pipeline, output_path)
        saved_paths[model_name] = output_path

    return saved_paths


def build_training_report(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    saved_paths: dict[str, Path],
    pipelines: dict[str, Pipeline],
) -> str:
    """Summarize the Stage 8 training setup and saved artifacts."""
    lines = [
        "# Training Report",
        "",
        "## Stage 8 Scope",
        "- This report documents baseline model training for Version 1.",
        "- Both models were trained inside sklearn Pipelines so preprocessing stays attached to the model artifact.",
        "",
        "## Training Data",
        f"- X_train shape: {X_train.shape}",
        f"- y_train shape: {y_train.shape}",
        f"- Random state: {RANDOM_STATE}",
        "",
        "## Models Trained",
        "- `LinearRegression` as the interpretable baseline",
        "- `RidgeCV` as the L2-regularized linear model",
        "- `LassoCV` as the L1-regularized linear model",
        "- `RandomForestRegressor` as the nonlinear tabular baseline",
        "",
        "## Ridge Search Configuration",
        f"- `alpha_count={len(RIDGE_REGULARIZATION_ALPHAS)}`",
        f"- `alpha_min={RIDGE_REGULARIZATION_ALPHAS[0]}`",
        f"- `alpha_max={RIDGE_REGULARIZATION_ALPHAS[-1]}`",
        f"- `cross_validation={RIDGE_CV.get_n_splits()}-fold shuffled CV`",
        f"- Ridge selected alpha: {pipelines['ridge_regression'].named_steps['model'].alpha_}",
        "",
        "## Lasso Search Configuration",
        f"- `alphas={LASSO_REGULARIZATION_ALPHAS}`",
        f"- Lasso selected alpha: {pipelines['lasso_regression'].named_steps['model'].alpha_}",
        "",
        "## Random Forest Configuration",
        f"- baseline `n_estimators={BASELINE_RANDOM_FOREST_PARAMS['n_estimators']}`",
        f"- baseline `max_depth={BASELINE_RANDOM_FOREST_PARAMS['max_depth']}`",
        f"- baseline `min_samples_split={BASELINE_RANDOM_FOREST_PARAMS['min_samples_split']}`",
        f"- baseline `min_samples_leaf={BASELINE_RANDOM_FOREST_PARAMS['min_samples_leaf']}`",
        f"- `random_state={BASELINE_RANDOM_FOREST_PARAMS['random_state']}`",
        f"- tuning search iterations: {RANDOM_FOREST_RANDOM_SEARCH_ITERATIONS}",
        f"- tuning cross-validation: {RANDOM_FOREST_CV.get_n_splits()}-fold shuffled CV",
        f"- tuned `n_estimators={pipelines['random_forest'].named_steps['model'].n_estimators}`",
        f"- tuned `max_depth={pipelines['random_forest'].named_steps['model'].max_depth}`",
        f"- tuned `min_samples_split={pipelines['random_forest'].named_steps['model'].min_samples_split}`",
        f"- tuned `min_samples_leaf={pipelines['random_forest'].named_steps['model'].min_samples_leaf}`",
        f"- tuned `max_features={pipelines['random_forest'].named_steps['model'].max_features}`",
        "",
        "## Saved Model Artifacts",
    ]

    for model_name, path in saved_paths.items():
        lines.append(f"- `{model_name}`: `{path}`")

    lines.extend(
        [
            "",
            "## Notes",
            "- The Linear Regression model provides a simple baseline for comparison.",
            "- Ridge now uses a denser logarithmic alpha grid with shuffled 10-fold cross-validation to make regularization tuning more stable.",
            "- Random Forest is tuned with a focused randomized search over the highest-impact tree hyperparameters rather than a brute-force grid.",
            "- Final model comparison happens in Stage 9 using the held-out test set.",
        ]
    )

    return "\n".join(lines)


def run_training_workflow() -> dict[str, object]:
    """Train and save the Version 1 baseline models."""
    X_train, y_train = load_training_data()
    pipelines = build_training_pipelines(X_train)
    train_pipelines(pipelines, X_train, y_train)
    saved_paths = save_trained_pipelines(pipelines)

    TRAINING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_REPORT_PATH.write_text(
        build_training_report(X_train, y_train, saved_paths, pipelines) + "\n",
        encoding="utf-8",
    )

    return {
        "X_train_shape": X_train.shape,
        "y_train_shape": y_train.shape,
        "saved_paths": saved_paths,
    }


if __name__ == "__main__":
    metadata = run_training_workflow()
    print(f"Saved training report to {TRAINING_REPORT_PATH}")
    for model_name, path in metadata["saved_paths"].items():
        print(f"Saved {model_name} pipeline to {path}")

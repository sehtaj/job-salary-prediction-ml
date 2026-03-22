"""Stage 9 model evaluation workflow for the Version 1 salary project."""

from __future__ import annotations

import os
from pathlib import Path

MPLCONFIGDIR = Path("results/evaluation/.mplconfig")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR.resolve()))

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from src.model_registry import get_baseline_models

X_TEST_PATH = Path("data/splits/X_test.csv")
Y_TEST_PATH = Path("data/splits/y_test.csv")
MODEL_PATHS = {
    model_name: Path(f"models/{model_name}_pipeline.joblib")
    for model_name in get_baseline_models().keys()
}

RESULTS_DIR = Path("results/evaluation")
METRICS_PATH = RESULTS_DIR / "model_metrics.csv"
PREDICTIONS_PATH = RESULTS_DIR / "test_set_predictions.csv"
REPORT_PATH = RESULTS_DIR / "evaluation_report.md"
COMPARISON_PLOT_PATH = RESULTS_DIR / "model_comparison.png"
BEST_MODEL_SCATTER_PATH = RESULTS_DIR / "best_model_actual_vs_predicted.png"


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the held-out test split from Stage 7."""
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)["avg_salary"]
    return X_test, y_test


def load_models() -> dict[str, object]:
    """Load all saved baseline pipelines."""
    return {model_name: joblib.load(path) for model_name, path in MODEL_PATHS.items()}


def evaluate_models(
    models: dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate predictions and evaluation metrics for each baseline model."""
    metric_rows: list[dict[str, float | str]] = []
    prediction_frame = pd.DataFrame({"actual_salary": y_test})

    for model_name, pipeline in models.items():
        predictions = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions)

        metric_rows.append(
            {
                "model": model_name,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
            }
        )
        prediction_frame[f"{model_name}_prediction"] = predictions

    metrics_df = pd.DataFrame(metric_rows).sort_values("rmse").reset_index(drop=True)
    return metrics_df, prediction_frame


def save_evaluation_outputs(metrics_df: pd.DataFrame, prediction_frame: pd.DataFrame) -> None:
    """Persist metrics and prediction outputs for later review."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_PATH, index=False)
    prediction_frame.to_csv(PREDICTIONS_PATH, index=False)


def plot_model_comparison(metrics_df: pd.DataFrame) -> None:
    """Create a simple side-by-side comparison chart for the evaluation metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [("mse", "MSE"), ("rmse", "RMSE"), ("r2", "R²")]

    for axis, (column, label) in zip(axes, metrics):
        axis.bar(metrics_df["model"], metrics_df[column], color=["#4c78a8", "#f58518"])
        axis.set_title(label)
        axis.set_xlabel("Model")
        axis.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(COMPARISON_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_best_model_predictions(metrics_df: pd.DataFrame, prediction_frame: pd.DataFrame) -> str:
    """Plot actual vs predicted salary for the best-performing model by RMSE."""
    best_model_name = str(metrics_df.iloc[0]["model"])
    prediction_column = f"{best_model_name}_prediction"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        prediction_frame["actual_salary"],
        prediction_frame[prediction_column],
        alpha=0.7,
        color="#2ca02c",
    )
    bounds = [
        min(prediction_frame["actual_salary"].min(), prediction_frame[prediction_column].min()),
        max(prediction_frame["actual_salary"].max(), prediction_frame[prediction_column].max()),
    ]
    ax.plot(bounds, bounds, linestyle="--", color="black")
    ax.set_title(f"Actual vs Predicted Salary: {best_model_name}")
    ax.set_xlabel("Actual avg_salary")
    ax.set_ylabel("Predicted avg_salary")
    fig.tight_layout()
    fig.savefig(BEST_MODEL_SCATTER_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return best_model_name


def build_evaluation_report(metrics_df: pd.DataFrame, best_model_name: str) -> str:
    """Create a markdown summary of the Stage 9 evaluation results."""
    best_row = metrics_df.iloc[0]

    lines = [
        "# Evaluation Report",
        "",
        "## Stage 9 Scope",
        "- This report compares the two Version 1 baseline models on the held-out test set.",
        "- The best model is selected primarily by the lowest RMSE, with MSE and R² also reported.",
        "",
        "## Model Metrics",
    ]

    for _, row in metrics_df.iterrows():
        lines.append(
            f"- `{row['model']}`: MSE={row['mse']:.4f}, RMSE={row['rmse']:.4f}, R²={row['r2']:.4f}"
        )

    regularized_metrics = metrics_df[metrics_df["model"].isin(["ridge_regression", "lasso_regression"])]
    best_regularized_model = (
        str(regularized_metrics.iloc[0]["model"]) if not regularized_metrics.empty else "not_available"
    )

    lines.extend(
        [
            "",
            "## Best Model",
            f"- Best-performing model: `{best_model_name}`",
            f"- Best RMSE: {best_row['rmse']:.4f}",
            f"- Best MSE: {best_row['mse']:.4f}",
            f"- Best R²: {best_row['r2']:.4f}",
            f"- Better regularized model between Ridge and Lasso: `{best_regularized_model}`",
            "",
            "## Saved Outputs",
            f"- Metrics table: `{METRICS_PATH}`",
            f"- Test predictions: `{PREDICTIONS_PATH}`",
            f"- Comparison plot: `{COMPARISON_PLOT_PATH}`",
            f"- Best-model scatter plot: `{BEST_MODEL_SCATTER_PATH}`",
        ]
    )

    return "\n".join(lines)


def run_evaluation_workflow() -> dict[str, object]:
    """Execute the full Stage 9 evaluation workflow."""
    X_test, y_test = load_test_data()
    models = load_models()
    metrics_df, prediction_frame = evaluate_models(models, X_test, y_test)
    save_evaluation_outputs(metrics_df, prediction_frame)
    plot_model_comparison(metrics_df)
    best_model_name = plot_best_model_predictions(metrics_df, prediction_frame)

    REPORT_PATH.write_text(
        build_evaluation_report(metrics_df, best_model_name) + "\n",
        encoding="utf-8",
    )

    return {
        "metrics": metrics_df,
        "best_model": best_model_name,
        "prediction_shape": prediction_frame.shape,
    }


if __name__ == "__main__":
    metadata = run_evaluation_workflow()
    print(f"Saved evaluation report to {REPORT_PATH}")
    print(f"Best model: {metadata['best_model']}")

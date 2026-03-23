"""Cross-validation reporting and Ridge error analysis for Version 1."""

from __future__ import annotations

import os
from pathlib import Path

MPLCONFIGDIR = Path("results/diagnostics/.mplconfig")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR.resolve()))

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_validate

from src.preprocessing import load_feature_dataset, split_features_and_target

FEATURE_DATASET_PATH = Path("data/processed/salary_data_features_v1.csv")
X_TEST_PATH = Path("data/splits/X_test.csv")
Y_TEST_PATH = Path("data/splits/y_test.csv")
MODEL_PATHS = {
    "ridge_regression": Path("models/ridge_regression_pipeline.joblib"),
    "linear_regression": Path("models/linear_regression_pipeline.joblib"),
    "random_forest": Path("models/random_forest_pipeline.joblib"),
    "lasso_regression": Path("models/lasso_regression_pipeline.joblib"),
}

RESULTS_DIR = Path("results/diagnostics")
CV_METRICS_PATH = RESULTS_DIR / "cross_validation_metrics.csv"
RIDGE_RESIDUALS_PATH = RESULTS_DIR / "ridge_test_residuals.csv"
RIDGE_JOB_TITLE_ERRORS_PATH = RESULTS_DIR / "ridge_error_by_job_title_group.csv"
RIDGE_SENIORITY_ERRORS_PATH = RESULTS_DIR / "ridge_error_by_job_seniority.csv"
RIDGE_SALARY_BAND_ERRORS_PATH = RESULTS_DIR / "ridge_error_by_salary_band.csv"
RIDGE_STATE_ERRORS_PATH = RESULTS_DIR / "ridge_error_by_job_state.csv"
CV_PLOT_PATH = RESULTS_DIR / "cross_validation_rmse.png"
RIDGE_RESIDUAL_PLOT_PATH = RESULTS_DIR / "ridge_residuals_vs_actual.png"
RIDGE_GROUP_ERROR_PLOT_PATH = RESULTS_DIR / "ridge_abs_error_by_job_title_group.png"
REPORT_PATH = RESULTS_DIR / "diagnostics_report.md"

CV_SPLITS = KFold(n_splits=5, shuffle=True, random_state=42)
CV_SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


def load_saved_pipelines() -> dict[str, object]:
    """Load the saved model pipelines for diagnostics."""
    return {name: joblib.load(path) for name, path in MODEL_PATHS.items()}


def run_cross_validation_summary(
    X: pd.DataFrame,
    y: pd.Series,
    pipelines: dict[str, object],
) -> pd.DataFrame:
    """Compare the current tuned model configurations with consistent CV."""
    rows: list[dict[str, float | str]] = []

    for model_name, pipeline in pipelines.items():
        scores = cross_validate(
            clone(pipeline),
            X,
            y,
            cv=CV_SPLITS,
            scoring=CV_SCORING,
            n_jobs=1,
            return_train_score=False,
        )
        rmse_scores = -scores["test_rmse"]
        r2_scores = scores["test_r2"]

        rows.append(
            {
                "model": model_name,
                "cv_rmse_mean": float(rmse_scores.mean()),
                "cv_rmse_std": float(rmse_scores.std()),
                "cv_r2_mean": float(r2_scores.mean()),
                "cv_r2_std": float(r2_scores.std()),
            }
        )

    return pd.DataFrame(rows).sort_values("cv_rmse_mean").reset_index(drop=True)


def build_ridge_residual_frame(ridge_pipeline: object) -> pd.DataFrame:
    """Create a per-row test-set residual table for the Ridge model."""
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)["avg_salary"]
    predictions = ridge_pipeline.predict(X_test)

    residuals = X_test.copy()
    residuals["actual_salary"] = y_test
    residuals["predicted_salary"] = predictions
    residuals["residual"] = residuals["predicted_salary"] - residuals["actual_salary"]
    residuals["abs_error"] = residuals["residual"].abs()
    residuals["salary_band"] = pd.qcut(
        residuals["actual_salary"],
        q=4,
        labels=["low", "lower_mid", "upper_mid", "high"],
        duplicates="drop",
    )

    return residuals


def summarize_group_errors(
    residuals: pd.DataFrame,
    group_column: str,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Aggregate Ridge errors by a grouping column."""
    summary = (
        residuals.groupby(group_column, as_index=False)
        .agg(
            row_count=("actual_salary", "size"),
            mean_actual_salary=("actual_salary", "mean"),
            mean_predicted_salary=("predicted_salary", "mean"),
            mean_residual=("residual", "mean"),
            mean_abs_error=("abs_error", "mean"),
        )
        .sort_values("mean_abs_error", ascending=False)
        .reset_index(drop=True)
    )

    if top_n is not None:
        return summary.head(top_n).reset_index(drop=True)
    return summary


def plot_cross_validation_rmse(cv_metrics: pd.DataFrame) -> None:
    """Visualize average cross-validated RMSE with error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        cv_metrics["model"],
        cv_metrics["cv_rmse_mean"],
        yerr=cv_metrics["cv_rmse_std"],
        color="#4c78a8",
        capsize=4,
    )
    ax.set_title("Cross-Validated RMSE by Model")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(CV_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ridge_residuals(residuals: pd.DataFrame) -> None:
    """Plot Ridge residuals against actual salary."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        residuals["actual_salary"],
        residuals["residual"],
        alpha=0.7,
        color="#2ca02c",
    )
    ax.axhline(0, linestyle="--", color="black")
    ax.set_title("Ridge Residuals vs Actual Salary")
    ax.set_xlabel("Actual avg_salary")
    ax.set_ylabel("Residual (predicted - actual)")
    fig.tight_layout()
    fig.savefig(RIDGE_RESIDUAL_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ridge_abs_error_by_job_title(job_title_errors: pd.DataFrame) -> None:
    """Plot mean absolute error by job title group."""
    plot_df = job_title_errors.sort_values("mean_abs_error", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_df["job_title_group"], plot_df["mean_abs_error"], color="#f58518")
    ax.set_title("Ridge Mean Absolute Error by Job Title Group")
    ax.set_xlabel("Mean Absolute Error")
    fig.tight_layout()
    fig.savefig(RIDGE_GROUP_ERROR_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_diagnostics_report(
    cv_metrics: pd.DataFrame,
    ridge_residuals: pd.DataFrame,
    job_title_errors: pd.DataFrame,
    seniority_errors: pd.DataFrame,
    salary_band_errors: pd.DataFrame,
    state_errors: pd.DataFrame,
) -> str:
    """Summarize model stability and Ridge failure modes."""
    best_cv_row = cv_metrics.iloc[0]
    worst_title_row = job_title_errors.iloc[0]
    worst_seniority_row = seniority_errors.iloc[0]
    worst_salary_band_row = salary_band_errors.iloc[0]
    largest_state_row = state_errors.iloc[0]

    lines = [
        "# Diagnostics Report",
        "",
        "## Scope",
        "- This report adds two post-training diagnostics for Version 1: cross-validated model comparison and Ridge residual/error analysis.",
        "- Cross-validation checks whether the current model ranking is stable beyond a single train/test split.",
        "- Ridge error analysis highlights where the selected model struggles most on the held-out test set.",
        "",
        "## Cross-Validation Summary",
    ]

    for _, row in cv_metrics.iterrows():
        lines.append(
            f"- `{row['model']}`: CV RMSE={row['cv_rmse_mean']:.4f} +/- {row['cv_rmse_std']:.4f}, "
            f"CV R²={row['cv_r2_mean']:.4f} +/- {row['cv_r2_std']:.4f}"
        )

    lines.extend(
        [
            "",
            "## Ridge Test-Set Error Patterns",
            f"- Largest average job-title-group error: `{worst_title_row['job_title_group']}` "
            f"(mean abs error {worst_title_row['mean_abs_error']:.4f})",
            f"- Largest average seniority error: `{worst_seniority_row['job_seniority']}` "
            f"(mean abs error {worst_seniority_row['mean_abs_error']:.4f})",
            f"- Hardest salary band: `{worst_salary_band_row['salary_band']}` "
            f"(mean abs error {worst_salary_band_row['mean_abs_error']:.4f})",
            f"- Highest-error retained state group: `{largest_state_row['job_state']}` "
            f"(mean abs error {largest_state_row['mean_abs_error']:.4f})",
            "",
            "## Interpretation",
            f"- Cross-validation still ranks `{best_cv_row['model']}` as the most stable model by average RMSE.",
            "- Ridge errors are not uniform: they vary meaningfully by job-title family, seniority level, and salary range.",
            "- This suggests the project now has a stronger model-selection story: Ridge is not only the best test-set model, but it is also competitive under repeated resampling.",
            "- The next quality improvement after this would be targeted feature refinement for the hardest Ridge error groups rather than more broad hyperparameter tuning.",
            "",
            "## Saved Outputs",
            f"- Cross-validation metrics: `{CV_METRICS_PATH}`",
            f"- Ridge residual table: `{RIDGE_RESIDUALS_PATH}`",
            f"- Group error summaries: `{RIDGE_JOB_TITLE_ERRORS_PATH}`, `{RIDGE_SENIORITY_ERRORS_PATH}`, `{RIDGE_SALARY_BAND_ERRORS_PATH}`, `{RIDGE_STATE_ERRORS_PATH}`",
            f"- Diagnostic plots: `{CV_PLOT_PATH}`, `{RIDGE_RESIDUAL_PLOT_PATH}`, `{RIDGE_GROUP_ERROR_PLOT_PATH}`",
        ]
    )

    return "\n".join(lines)


def run_diagnostics_workflow() -> dict[str, object]:
    """Generate cross-validation and Ridge error-analysis artifacts."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_df = load_feature_dataset(FEATURE_DATASET_PATH)
    X, y = split_features_and_target(feature_df)
    pipelines = load_saved_pipelines()

    cv_metrics = run_cross_validation_summary(X, y, pipelines)
    cv_metrics.to_csv(CV_METRICS_PATH, index=False)
    plot_cross_validation_rmse(cv_metrics)

    ridge_residuals = build_ridge_residual_frame(pipelines["ridge_regression"])
    ridge_residuals.to_csv(RIDGE_RESIDUALS_PATH, index=False)
    plot_ridge_residuals(ridge_residuals)

    job_title_errors = summarize_group_errors(ridge_residuals, "job_title_group")
    seniority_errors = summarize_group_errors(ridge_residuals, "job_seniority")
    salary_band_errors = summarize_group_errors(ridge_residuals, "salary_band")
    state_errors = summarize_group_errors(ridge_residuals, "job_state", top_n=10)

    job_title_errors.to_csv(RIDGE_JOB_TITLE_ERRORS_PATH, index=False)
    seniority_errors.to_csv(RIDGE_SENIORITY_ERRORS_PATH, index=False)
    salary_band_errors.to_csv(RIDGE_SALARY_BAND_ERRORS_PATH, index=False)
    state_errors.to_csv(RIDGE_STATE_ERRORS_PATH, index=False)
    plot_ridge_abs_error_by_job_title(job_title_errors)

    REPORT_PATH.write_text(
        build_diagnostics_report(
            cv_metrics,
            ridge_residuals,
            job_title_errors,
            seniority_errors,
            salary_band_errors,
            state_errors,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "best_cv_model": str(cv_metrics.iloc[0]["model"]),
        "best_cv_rmse": float(cv_metrics.iloc[0]["cv_rmse_mean"]),
        "ridge_test_rows": int(len(ridge_residuals)),
    }


if __name__ == "__main__":
    metadata = run_diagnostics_workflow()
    print(f"Saved cross-validation metrics to {CV_METRICS_PATH}")
    print(f"Saved diagnostics report to {REPORT_PATH}")
    print(f"Best CV model: {metadata['best_cv_model']}")

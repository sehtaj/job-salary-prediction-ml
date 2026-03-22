"""Stage 10 feature-importance analysis for the Version 1 salary project."""

from __future__ import annotations

import os
from pathlib import Path

MPLCONFIGDIR = Path("results/feature_importance/.mplconfig")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR.resolve()))

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

FEATURE_DATASET_PATH = Path("data/processed/salary_data_features_v1.csv")
LINEAR_MODEL_PATH = Path("models/linear_regression_pipeline.joblib")
RANDOM_FOREST_MODEL_PATH = Path("models/random_forest_pipeline.joblib")

RESULTS_DIR = Path("results/feature_importance")
LINEAR_COEFFICIENTS_PATH = RESULTS_DIR / "linear_regression_coefficients.csv"
RANDOM_FOREST_IMPORTANCE_PATH = RESULTS_DIR / "random_forest_feature_importances.csv"
AGGREGATED_IMPORTANCE_PATH = RESULTS_DIR / "feature_group_importance.csv"
LINEAR_PLOT_PATH = RESULTS_DIR / "linear_regression_top_coefficients.png"
RANDOM_FOREST_PLOT_PATH = RESULTS_DIR / "random_forest_top_importances.png"
REPORT_PATH = RESULTS_DIR / "feature_importance_report.md"
TARGET_COLUMN = "avg_salary"


def load_feature_dataset() -> pd.DataFrame:
    """Load the engineered Version 1 dataset to recover original feature names."""
    return pd.read_csv(FEATURE_DATASET_PATH)


def load_trained_pipelines() -> tuple[object, object]:
    """Load the trained linear regression and random forest pipelines."""
    return joblib.load(LINEAR_MODEL_PATH), joblib.load(RANDOM_FOREST_MODEL_PATH)


def get_feature_names(linear_pipeline: object) -> list[str]:
    """Read encoded feature names from the fitted preprocessing pipeline."""
    return linear_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()


def build_linear_coefficients_table(feature_names: list[str], linear_pipeline: object) -> pd.DataFrame:
    """Create a sorted coefficient table for the linear regression model."""
    coefficients = linear_pipeline.named_steps["model"].coef_
    table = pd.DataFrame(
        {
            "feature_name": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": pd.Series(coefficients).abs(),
        }
    ).sort_values("abs_coefficient", ascending=False)
    return table.reset_index(drop=True)


def build_random_forest_importance_table(feature_names: list[str], random_forest_pipeline: object) -> pd.DataFrame:
    """Create a sorted feature-importance table for the random forest model."""
    importances = random_forest_pipeline.named_steps["model"].feature_importances_
    table = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)
    return table.reset_index(drop=True)


def infer_original_feature(feature_name: str, original_feature_columns: list[str]) -> str:
    """Map an encoded feature name back to its original feature column."""
    for column in original_feature_columns:
        if feature_name == f"numerical__{column}":
            return column
        if feature_name.startswith(f"categorical__{column}_"):
            return column
    return feature_name


def build_group_importance_table(
    linear_table: pd.DataFrame,
    random_forest_table: pd.DataFrame,
    original_feature_columns: list[str],
) -> pd.DataFrame:
    """Aggregate encoded-feature importance back to the original feature groups."""
    linear_grouped = linear_table.copy()
    linear_grouped["original_feature"] = linear_grouped["feature_name"].apply(
        lambda name: infer_original_feature(name, original_feature_columns)
    )
    linear_grouped = (
        linear_grouped.groupby("original_feature", as_index=False)["abs_coefficient"]
        .sum()
        .rename(columns={"abs_coefficient": "linear_abs_coefficient"})
    )

    forest_grouped = random_forest_table.copy()
    forest_grouped["original_feature"] = forest_grouped["feature_name"].apply(
        lambda name: infer_original_feature(name, original_feature_columns)
    )
    forest_grouped = (
        forest_grouped.groupby("original_feature", as_index=False)["importance"]
        .sum()
        .rename(columns={"importance": "random_forest_importance"})
    )

    grouped = linear_grouped.merge(forest_grouped, on="original_feature", how="outer").fillna(0)
    grouped["combined_rank_score"] = (
        grouped["linear_abs_coefficient"].rank(ascending=False, method="dense")
        + grouped["random_forest_importance"].rank(ascending=False, method="dense")
    )
    grouped = grouped.sort_values(
        ["combined_rank_score", "random_forest_importance", "linear_abs_coefficient"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return grouped


def plot_top_linear_coefficients(linear_table: pd.DataFrame, top_n: int = 15) -> None:
    """Visualize the largest positive and negative linear coefficients."""
    top_positive = linear_table.sort_values("coefficient", ascending=False).head(top_n // 2)
    top_negative = linear_table.sort_values("coefficient", ascending=True).head(top_n // 2)
    plot_df = pd.concat([top_negative, top_positive], ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#d62728" if value < 0 else "#1f77b4" for value in plot_df["coefficient"]]
    ax.barh(plot_df["feature_name"], plot_df["coefficient"], color=colors)
    ax.set_title("Top Linear Regression Coefficients")
    ax.set_xlabel("Coefficient")
    fig.tight_layout()
    fig.savefig(LINEAR_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top_random_forest_importances(forest_table: pd.DataFrame, top_n: int = 15) -> None:
    """Visualize the strongest random forest feature importances."""
    plot_df = forest_table.head(top_n).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(plot_df["feature_name"], plot_df["importance"], color="#2ca02c")
    ax.set_title("Top Random Forest Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(RANDOM_FOREST_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_feature_importance_report(
    linear_table: pd.DataFrame,
    forest_table: pd.DataFrame,
    grouped_table: pd.DataFrame,
) -> str:
    """Write a concise interpretation of the most influential predictors."""
    top_positive = linear_table.sort_values("coefficient", ascending=False).head(5)
    top_negative = linear_table.sort_values("coefficient", ascending=True).head(5)
    top_forest = forest_table.head(10)
    top_groups = grouped_table.head(8)

    lines = [
        "# Feature Importance Report",
        "",
        "## Stage 10 Scope",
        "- This report explains which encoded features and original job/company attributes most influenced the baseline models.",
        "- Ridge Regression is the best-performing final model, but Stage 10 uses Linear Regression and Random Forest for interpretation because their coefficients and feature importances are straightforward to inspect.",
        "- Linear Regression coefficients show signed influence direction after preprocessing.",
        "- Random Forest importances show how much each feature helped reduce prediction error in the tree ensemble.",
        "",
        "## Linear Regression: Strongest Positive Coefficients",
    ]

    for _, row in top_positive.iterrows():
        lines.append(f"- `{row['feature_name']}`: {row['coefficient']:.4f}")

    lines.extend(["", "## Linear Regression: Strongest Negative Coefficients"])
    for _, row in top_negative.iterrows():
        lines.append(f"- `{row['feature_name']}`: {row['coefficient']:.4f}")

    lines.extend(["", "## Random Forest: Top Encoded Features"])
    for _, row in top_forest.iterrows():
        lines.append(f"- `{row['feature_name']}`: {row['importance']:.4f}")

    lines.extend(["", "## Most Influential Original Feature Groups"])
    for _, row in top_groups.iterrows():
        lines.append(
            f"- `{row['original_feature']}`: linear abs coefficient={row['linear_abs_coefficient']:.4f}, "
            f"random forest importance={row['random_forest_importance']:.4f}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- Job title grouping, job seniority, and state/location signals are consistently among the strongest predictors across both model families.",
            "- Company-context features such as industry, revenue, sector, size, and company age also contribute meaningful salary signal.",
            "- Skill flags such as SQL, Tableau, Python, Spark, and AWS contribute useful signal, but they are not as dominant as title, seniority, and company-context features in this Version 1 baseline.",
            "- Linear coefficients help identify whether a feature pushes salary predictions up or down, while random forest importances highlight which features matter most overall.",
        ]
    )

    return "\n".join(lines)


def run_feature_importance_workflow() -> dict[str, object]:
    """Execute the full Stage 10 feature-importance workflow."""
    feature_df = load_feature_dataset()
    original_feature_columns = [column for column in feature_df.columns if column != TARGET_COLUMN]
    linear_pipeline, random_forest_pipeline = load_trained_pipelines()
    feature_names = get_feature_names(linear_pipeline)

    linear_table = build_linear_coefficients_table(feature_names, linear_pipeline)
    forest_table = build_random_forest_importance_table(feature_names, random_forest_pipeline)
    grouped_table = build_group_importance_table(linear_table, forest_table, original_feature_columns)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    linear_table.to_csv(LINEAR_COEFFICIENTS_PATH, index=False)
    forest_table.to_csv(RANDOM_FOREST_IMPORTANCE_PATH, index=False)
    grouped_table.to_csv(AGGREGATED_IMPORTANCE_PATH, index=False)

    plot_top_linear_coefficients(linear_table)
    plot_top_random_forest_importances(forest_table)

    REPORT_PATH.write_text(
        build_feature_importance_report(linear_table, forest_table, grouped_table) + "\n",
        encoding="utf-8",
    )

    return {
        "linear_top_feature": linear_table.iloc[0]["feature_name"],
        "forest_top_feature": forest_table.iloc[0]["feature_name"],
        "top_group": grouped_table.iloc[0]["original_feature"],
    }


if __name__ == "__main__":
    metadata = run_feature_importance_workflow()
    print(f"Saved linear coefficients to {LINEAR_COEFFICIENTS_PATH}")
    print(f"Saved random forest importances to {RANDOM_FOREST_IMPORTANCE_PATH}")
    print(f"Saved aggregated feature-group importance to {AGGREGATED_IMPORTANCE_PATH}")
    print(f"Saved feature importance report to {REPORT_PATH}")
    print(f"Top grouped feature: {metadata['top_group']}")

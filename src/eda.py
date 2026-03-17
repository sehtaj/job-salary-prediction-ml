"""Exploratory data analysis utilities for the salary dataset."""

from __future__ import annotations

import os
from pathlib import Path

MPLCONFIGDIR = Path("results/eda/.mplconfig")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR.resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_loading import load_dataset

OUTPUT_DIR = Path("results/eda")
NUMERICAL_SUMMARY_PATH = OUTPUT_DIR / "numerical_summary.csv"
CORRELATION_PATH = OUTPUT_DIR / "correlation_matrix.csv"
AVG_SALARY_DIST_PATH = OUTPUT_DIR / "avg_salary_distribution.png"
NUMERICAL_REL_PATH = OUTPUT_DIR / "salary_vs_numerical_features.png"
CATEGORICAL_REL_PATH = OUTPUT_DIR / "salary_vs_categorical_features.png"
CORRELATION_HEATMAP_PATH = OUTPUT_DIR / "correlation_heatmap.png"
REPORT_PATH = OUTPUT_DIR / "eda_findings.md"

TOP_CATEGORICAL_COLUMNS = [
    "Job Title",
    "Location",
    "Industry",
    "Sector",
    "Revenue",
    "Size",
]
SKILL_COLUMNS = ["python_yn", "spark", "aws", "excel"]


def ensure_output_dir() -> None:
    """Create the directory used for EDA artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def summarize_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build and save summary statistics for all numerical columns."""
    numerical_summary = (
        df.select_dtypes(include="number")
        .describe()
        .T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    )
    numerical_summary.to_csv(NUMERICAL_SUMMARY_PATH)
    return numerical_summary


def summarize_categorical_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Capture top value counts for key categorical columns."""
    distributions: dict[str, pd.Series] = {}
    for column in TOP_CATEGORICAL_COLUMNS:
        distributions[column] = df[column].value_counts().head(10)
    return distributions


def plot_avg_salary_distribution(df: pd.DataFrame) -> None:
    """Visualize the target distribution with a histogram and boxplot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df["avg_salary"], bins=30, kde=True, ax=axes[0], color="#1f77b4")
    axes[0].set_title("Distribution of avg_salary")
    axes[0].set_xlabel("Average Salary (thousands USD)")

    sns.boxplot(x=df["avg_salary"], ax=axes[1], color="#9ecae1")
    axes[1].set_title("Boxplot of avg_salary")
    axes[1].set_xlabel("Average Salary (thousands USD)")

    fig.tight_layout()
    fig.savefig(AVG_SALARY_DIST_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_numerical_relationships(df: pd.DataFrame) -> pd.DataFrame:
    """Plot salary relationships for selected numerical features."""
    selected_columns = ["avg_salary", "Rating", "age", "min_salary", "max_salary"]
    correlation_matrix = df.select_dtypes(include="number").corr(numeric_only=True)
    correlation_matrix.to_csv(CORRELATION_PATH)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    features = ["Rating", "age", "min_salary", "max_salary"]

    for axis, feature in zip(axes.flat, features):
        sns.scatterplot(
            data=df,
            x=feature,
            y="avg_salary",
            alpha=0.6,
            s=35,
            ax=axis,
        )
        axis.set_title(f"avg_salary vs {feature}")

    fig.tight_layout()
    fig.savefig(NUMERICAL_REL_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        correlation_matrix.loc[selected_columns, selected_columns],
        annot=True,
        cmap="Blues",
        fmt=".2f",
        ax=ax,
    )
    ax.set_title("Correlation Heatmap for Key Numerical Features")
    fig.tight_layout()
    fig.savefig(CORRELATION_HEATMAP_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return correlation_matrix


def plot_categorical_relationships(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Plot salary relationships for the most useful categorical features."""
    title_salary = (
        df.groupby("Job Title")["avg_salary"]
        .agg(["count", "mean"])
        .query("count >= 5")
        .sort_values("mean", ascending=False)
        .head(10)
        .reset_index()
    )
    sector_salary = (
        df.groupby("Sector")["avg_salary"]
        .agg(["count", "mean"])
        .query("count >= 5")
        .sort_values("mean", ascending=False)
        .head(10)
        .reset_index()
    )
    skill_salary = (
        df.groupby(SKILL_COLUMNS)["avg_salary"]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    sns.barplot(data=title_salary, x="mean", y="Job Title", ax=axes[0], color="#2ca02c")
    axes[0].set_title("Top Job Titles by Mean avg_salary")
    axes[0].set_xlabel("Mean avg_salary")

    sns.barplot(data=sector_salary, x="mean", y="Sector", ax=axes[1], color="#ff7f0e")
    axes[1].set_title("Top Sectors by Mean avg_salary")
    axes[1].set_xlabel("Mean avg_salary")

    fig.tight_layout()
    fig.savefig(CATEGORICAL_REL_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "title_salary": title_salary,
        "sector_salary": sector_salary,
        "skill_salary": skill_salary,
    }


def identify_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Identify avg_salary outliers using the IQR rule."""
    q1 = df["avg_salary"].quantile(0.25)
    q3 = df["avg_salary"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df["avg_salary"] < lower_bound) | (df["avg_salary"] > upper_bound)]
    return outliers, lower_bound, upper_bound


def build_findings_report(
    df: pd.DataFrame,
    numerical_summary: pd.DataFrame,
    categorical_distributions: dict[str, pd.Series],
    correlation_matrix: pd.DataFrame,
    categorical_relationships: dict[str, pd.DataFrame],
    outliers: pd.DataFrame,
    lower_bound: float,
    upper_bound: float,
) -> str:
    """Create a markdown summary of the EDA results."""
    title_salary = categorical_relationships["title_salary"]
    sector_salary = categorical_relationships["sector_salary"]

    lines = [
        "# EDA Findings",
        "",
        "## Dataset Overview",
        f"- Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        f"- Target column present: {'avg_salary' in df.columns}",
        f"- `avg_salary` mean: {df['avg_salary'].mean():.2f}",
        f"- `avg_salary` median: {df['avg_salary'].median():.2f}",
        f"- `avg_salary` range: {df['avg_salary'].min():.2f} to {df['avg_salary'].max():.2f}",
        "",
        "## Numerical Feature Summary",
        f"- Highest correlation with `avg_salary`: `max_salary` ({correlation_matrix.loc['avg_salary', 'max_salary']:.2f})",
        f"- Next highest correlation with `avg_salary`: `min_salary` ({correlation_matrix.loc['avg_salary', 'min_salary']:.2f})",
        f"- Skill indicator most correlated with `avg_salary`: `python_yn` ({correlation_matrix.loc['avg_salary', 'python_yn']:.2f})",
        f"- Weak linear relationship from `Rating` to `avg_salary`: {correlation_matrix.loc['avg_salary', 'Rating']:.2f}",
        "",
        "## Categorical Distributions",
        f"- Most common job title: {categorical_distributions['Job Title'].index[0]} ({categorical_distributions['Job Title'].iloc[0]} rows)",
        f"- Most common location: {categorical_distributions['Location'].index[0]} ({categorical_distributions['Location'].iloc[0]} rows)",
        f"- Largest sector: {categorical_distributions['Sector'].index[0]} ({categorical_distributions['Sector'].iloc[0]} rows)",
        f"- Most common industry: {categorical_distributions['Industry'].index[0]} ({categorical_distributions['Industry'].iloc[0]} rows)",
        "",
        "## Salary Relationships",
        f"- Highest-paying frequent job title: {title_salary.iloc[0]['Job Title']} ({title_salary.iloc[0]['mean']:.2f})",
        f"- Highest-paying frequent sector: {sector_salary.iloc[0]['Sector']} ({sector_salary.iloc[0]['mean']:.2f})",
        f"- Mean salary with Python skill: {df.groupby('python_yn')['avg_salary'].mean().loc[1]:.2f}",
        f"- Mean salary without Python skill: {df.groupby('python_yn')['avg_salary'].mean().loc[0]:.2f}",
        f"- Mean salary with Spark skill: {df.groupby('spark')['avg_salary'].mean().loc[1]:.2f}",
        f"- Mean salary with AWS skill: {df.groupby('aws')['avg_salary'].mean().loc[1]:.2f}",
        "",
        "## Outliers and Unusual Values",
        f"- IQR bounds for `avg_salary`: {lower_bound:.2f} to {upper_bound:.2f}",
        f"- Outlier count: {len(outliers)}",
        f"- Highest observed `avg_salary`: {outliers['avg_salary'].max():.2f}" if not outliers.empty else "- No outliers detected",
        "",
        "## Notes for Modeling",
        "- `min_salary` and `max_salary` are extremely correlated with `avg_salary`, so they are likely target-leakage features if kept as predictors.",
        "- Binary skill indicators such as `python_yn`, `spark`, and `aws` show higher average salary and are likely useful predictors.",
        "- `hourly` roles have much lower `avg_salary` values than the rest of the dataset and deserve special handling during cleaning or modeling.",
    ]

    return "\n".join(lines)


def run_eda() -> None:
    """Generate Stage 3 EDA artifacts and summary files."""
    ensure_output_dir()
    sns.set_theme(style="whitegrid")

    df = load_dataset()
    numerical_summary = summarize_numerical_features(df)
    categorical_distributions = summarize_categorical_features(df)
    plot_avg_salary_distribution(df)
    correlation_matrix = plot_numerical_relationships(df)
    categorical_relationships = plot_categorical_relationships(df)
    outliers, lower_bound, upper_bound = identify_outliers(df)

    report = build_findings_report(
        df=df,
        numerical_summary=numerical_summary,
        categorical_distributions=categorical_distributions,
        correlation_matrix=correlation_matrix,
        categorical_relationships=categorical_relationships,
        outliers=outliers,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")

    print(f"Saved numerical summary to {NUMERICAL_SUMMARY_PATH}")
    print(f"Saved correlation matrix to {CORRELATION_PATH}")
    print(f"Saved plot to {AVG_SALARY_DIST_PATH}")
    print(f"Saved plot to {NUMERICAL_REL_PATH}")
    print(f"Saved plot to {CATEGORICAL_REL_PATH}")
    print(f"Saved plot to {CORRELATION_HEATMAP_PATH}")
    print(f"Saved findings report to {REPORT_PATH}")


if __name__ == "__main__":
    run_eda()

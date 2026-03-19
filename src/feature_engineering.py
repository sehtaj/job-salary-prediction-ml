"""Stage 5 feature engineering for the Version 1 salary baseline."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

CLEANED_DATASET_PATH = Path("data/processed/salary_data_cleaned.csv")
FEATURE_DATASET_PATH = Path("data/processed/salary_data_features_v1.csv")
FEATURE_REPORT_PATH = Path("results/feature_engineering/feature_report.md")

TARGET_COLUMN = "avg_salary"

KEEP_AS_IS_COLUMNS = [
    "Rating",
    "Size",
    "Industry",
    "Sector",
    "Revenue",
    "Type of ownership",
    "job_state",
    "hourly",
    "python_yn",
    "R_yn",
    "spark",
    "aws",
    "excel",
]

DROP_FROM_FEATURES = [
    "Salary Estimate",
    "Company Name",
    "Location",
    "Headquarters",
    "Founded",
    "Competitors",
    "company_txt",
    "same_state",
    "employer_provided",
    "min_salary",
    "max_salary",
]


def simplify_job_title(title: str) -> str:
    """Reduce raw job titles into a manageable set of baseline categories."""
    normalized = title.lower()

    if any(token in normalized for token in ["director", "manager", "head of", "vice president"]):
        return "leadership"
    if "machine learning" in normalized or "deep learning" in normalized:
        return "machine_learning"
    if "engineer" in normalized:
        return "data_engineer"
    if "analyst" in normalized:
        return "data_analyst"
    if "research scientist" in normalized:
        return "research_scientist"
    if "scientist" in normalized:
        return "data_scientist"
    return "other"


def extract_job_seniority(title: str) -> str:
    """Extract a coarse seniority level from the job title."""
    normalized = title.lower()

    if "director" in normalized or "vice president" in normalized or "head of" in normalized:
        return "director"
    if "principal" in normalized:
        return "principal"
    if "lead" in normalized:
        return "lead"
    if "senior" in normalized or normalized.startswith("sr") or " sr." in normalized or " sr " in normalized:
        return "senior"
    if any(token in normalized for token in ["junior", "jr", "associate", "intern"]):
        return "entry"
    return "mid"


def extract_skill_flag(descriptions: pd.Series, pattern: str) -> pd.Series:
    """Create a binary feature from a job-description regex pattern."""
    return (
        descriptions.fillna("")
        .str.lower()
        .str.contains(pattern, regex=True)
        .astype(int)
    )


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build the approved Version 1 feature set."""
    featured = df.copy()

    featured["company_age"] = featured["age"]
    featured["job_title_group"] = featured["Job Title"].apply(simplify_job_title)
    featured["job_seniority"] = featured["Job Title"].apply(extract_job_seniority)
    featured["sql_yn"] = extract_skill_flag(featured["Job Description"], r"\bsql\b")
    featured["tableau_yn"] = extract_skill_flag(featured["Job Description"], r"\btableau\b")

    final_columns = [
        TARGET_COLUMN,
        *KEEP_AS_IS_COLUMNS,
        "company_age",
        "job_title_group",
        "job_seniority",
        "sql_yn",
        "tableau_yn",
    ]
    feature_df = featured[final_columns].copy()

    metadata: dict[str, object] = {
        "input_shape": df.shape,
        "output_shape": feature_df.shape,
        "features": [column for column in final_columns if column != TARGET_COLUMN],
        "dropped_columns": DROP_FROM_FEATURES + ["age", "Job Title", "Job Description"],
        "engineered_columns": ["company_age", "job_title_group", "job_seniority", "sql_yn", "tableau_yn"],
        "skill_counts": {
            "python_yn": int(feature_df["python_yn"].sum()),
            "R_yn": int(feature_df["R_yn"].sum()),
            "spark": int(feature_df["spark"].sum()),
            "aws": int(feature_df["aws"].sum()),
            "excel": int(feature_df["excel"].sum()),
            "sql_yn": int(feature_df["sql_yn"].sum()),
            "tableau_yn": int(feature_df["tableau_yn"].sum()),
        },
    }
    return feature_df, metadata


def build_feature_report(metadata: dict[str, object], feature_df: pd.DataFrame) -> str:
    """Summarize the Stage 5 feature-engineering decisions."""
    lines = [
        "# Feature Engineering Report",
        "",
        "## Scope",
        "- This report documents the Version 1 baseline feature set for salary prediction using structured and lightly engineered features.",
        "- The goal is to create a strong tabular baseline before any advanced NLP workflow.",
        "",
        "## Dataset Shape",
        f"- Input shape: {metadata['input_shape'][0]} rows x {metadata['input_shape'][1]} columns",
        f"- Output shape: {metadata['output_shape'][0]} rows x {metadata['output_shape'][1]} columns",
        f"- Target retained in dataset: `{TARGET_COLUMN}`",
        "",
        "## Key Feature Decisions",
        "- Salary-derived predictors `min_salary`, `max_salary`, and `Salary Estimate` were excluded to avoid leakage.",
        "- `Location` was replaced by the already cleaned `job_state` field.",
        "- `age` was renamed to `company_age` for clarity.",
        "- `Job Title` was transformed into lower-cardinality features `job_title_group` and `job_seniority`.",
        "- Existing structured skill flags were retained, and new binary skill flags were extracted from `Job Description` using keyword matching.",
        "- Raw text and identity-heavy fields such as `Job Description`, `Company Name`, and `company_txt` were excluded from the Version 1 baseline feature set.",
        "",
        "## Final Predictor Columns",
    ]

    for column in metadata["features"]:
        lines.append(f"- `{column}`")

    lines.extend(
        [
            "",
            "## Dropped Columns",
        ]
    )
    for column in metadata["dropped_columns"]:
        lines.append(f"- `{column}`")

    lines.extend(
        [
            "",
            "## Engineered Feature Summary",
            f"- `job_title_group` categories: {sorted(feature_df['job_title_group'].unique().tolist())}",
            f"- `job_seniority` categories: {sorted(feature_df['job_seniority'].unique().tolist())}",
            f"- `sql_yn` positive rows: {metadata['skill_counts']['sql_yn']}",
            f"- `tableau_yn` positive rows: {metadata['skill_counts']['tableau_yn']}",
        ]
    )

    return "\n".join(lines)


def run_feature_engineering_pipeline() -> tuple[pd.DataFrame, dict[str, object]]:
    """Generate and save the Version 1 feature-engineered dataset and report."""
    cleaned_df = pd.read_csv(CLEANED_DATASET_PATH)
    feature_df, metadata = engineer_features(cleaned_df)

    FEATURE_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(FEATURE_DATASET_PATH, index=False)

    FEATURE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_REPORT_PATH.write_text(build_feature_report(metadata, feature_df) + "\n", encoding="utf-8")

    return feature_df, metadata


if __name__ == "__main__":
    feature_df, metadata = run_feature_engineering_pipeline()
    print(f"Saved feature-engineered dataset to {FEATURE_DATASET_PATH}")
    print(f"Saved feature report to {FEATURE_REPORT_PATH}")
    print(f"Output shape: {metadata['output_shape']}")

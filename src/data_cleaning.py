"""Stage 4 data cleaning pipeline for the salary dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loading import load_dataset

PROCESSED_DATASET_PATH = Path("data/processed/salary_data_cleaned.csv")
CLEANING_REPORT_PATH = Path("results/cleaning/cleaning_report.md")

TEXT_COLUMNS = [
    "Job Title",
    "Salary Estimate",
    "Job Description",
    "Company Name",
    "Location",
    "Headquarters",
    "Size",
    "Type of ownership",
    "Industry",
    "Sector",
    "Revenue",
    "Competitors",
    "company_txt",
    "job_state",
]

PLACEHOLDER_MAP = {
    "Headquarters": {"-1"},
    "Size": {"-1", "Unknown"},
    "Type of ownership": {"-1", "Unknown"},
    "Industry": {"-1"},
    "Sector": {"-1"},
    "Revenue": {"-1", "Unknown / Non-Applicable", "Unknown / Non Applicable"},
    "Competitors": {"-1", "Unknown / Non-Applicable", "Unknown / Non Applicable"},
}

NUMERICAL_INVALID_RULES = {
    "Rating": lambda s: s < 0,
    "Founded": lambda s: s < 0,
    "age": lambda s: s < 0,
}


def standardize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and normalize line breaks in string columns."""
    cleaned = df.copy()
    for column in TEXT_COLUMNS:
        cleaned[column] = (
            cleaned[column]
            .astype("string")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    cleaned["job_state"] = cleaned["job_state"].str.upper()
    return cleaned


def mark_placeholder_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert placeholder strings to missing values before imputation."""
    cleaned = df.copy()
    counts: dict[str, int] = {}

    for column, placeholders in PLACEHOLDER_MAP.items():
        mask = cleaned[column].isin(placeholders)
        counts[column] = int(mask.sum())
        cleaned.loc[mask, column] = pd.NA

    return cleaned, counts


def mark_invalid_numerical_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace invalid numerical placeholder values with missing values."""
    cleaned = df.copy()
    counts: dict[str, int] = {}

    for column, rule in NUMERICAL_INVALID_RULES.items():
        mask = rule(cleaned[column])
        counts[column] = int(mask.sum())
        cleaned.loc[mask, column] = pd.NA

    return cleaned, counts


def impute_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], dict[str, str]]:
    """Impute numerical values with medians and categorical values with Unknown."""
    cleaned = df.copy()
    numerical_fill_values: dict[str, float] = {}
    categorical_fill_values: dict[str, str] = {}

    numerical_columns = cleaned.select_dtypes(include="number").columns
    categorical_columns = cleaned.select_dtypes(exclude="number").columns

    for column in numerical_columns:
        if cleaned[column].isna().any():
            fill_value = float(cleaned[column].median())
            numerical_fill_values[column] = fill_value
            cleaned[column] = cleaned[column].fillna(fill_value)

    for column in categorical_columns:
        if cleaned[column].isna().any():
            categorical_fill_values[column] = "Unknown"
            cleaned[column] = cleaned[column].fillna("Unknown")

    return cleaned, numerical_fill_values, categorical_fill_values


def validate_salary_columns(df: pd.DataFrame) -> dict[str, int]:
    """Run sanity checks for the core salary fields."""
    return {
        "min_gt_max": int((df["min_salary"] > df["max_salary"]).sum()),
        "avg_outside_range": int(
            ((df["avg_salary"] < df["min_salary"]) | (df["avg_salary"] > df["max_salary"])).sum()
        ),
        "negative_salary_rows": int((df[["min_salary", "max_salary", "avg_salary"]] < 0).any(axis=1).sum()),
        "zero_salary_rows": int((df[["min_salary", "max_salary", "avg_salary"]] == 0).any(axis=1).sum()),
    }


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply Stage 4 cleaning steps and return the cleaned dataset plus audit metadata."""
    audit: dict[str, object] = {
        "original_shape": df.shape,
        "original_missing": df.isna().sum().sum(),
        "duplicate_rows_removed": int(df.duplicated().sum()),
    }

    cleaned = df.drop_duplicates().copy()
    cleaned = standardize_text_columns(cleaned)
    cleaned, placeholder_counts = mark_placeholder_values(cleaned)
    cleaned, invalid_numeric_counts = mark_invalid_numerical_values(cleaned)

    audit["placeholder_counts"] = placeholder_counts
    audit["invalid_numeric_counts"] = invalid_numeric_counts
    audit["missing_after_standardization"] = int(cleaned.isna().sum().sum())

    cleaned, numerical_fill_values, categorical_fill_values = impute_missing_values(cleaned)
    audit["numerical_fill_values"] = numerical_fill_values
    audit["categorical_fill_values"] = categorical_fill_values
    audit["final_missing"] = int(cleaned.isna().sum().sum())
    audit["final_shape"] = cleaned.shape
    audit["salary_validation"] = validate_salary_columns(cleaned)

    return cleaned, audit


def build_cleaning_report(audit: dict[str, object], cleaned_df: pd.DataFrame) -> str:
    """Generate a markdown summary of the Stage 4 cleaning decisions."""
    placeholder_counts = audit["placeholder_counts"]
    invalid_numeric_counts = audit["invalid_numeric_counts"]
    numerical_fill_values = audit["numerical_fill_values"]
    salary_validation = audit["salary_validation"]

    lines = [
        "# Data Cleaning Report",
        "",
        "## Dataset Shape",
        f"- Original shape: {audit['original_shape'][0]} rows x {audit['original_shape'][1]} columns",
        f"- Final shape: {audit['final_shape'][0]} rows x {audit['final_shape'][1]} columns",
        f"- Duplicate rows removed: {audit['duplicate_rows_removed']}",
        "",
        "## Missing and Placeholder Handling",
        f"- Raw `NaN` count before cleaning: {audit['original_missing']}",
        f"- Missing values introduced after converting placeholders: {audit['missing_after_standardization']}",
        f"- Final missing values after imputation: {audit['final_missing']}",
        "- Placeholder-like categorical values were standardized to missing first, then imputed as `Unknown`.",
        "",
        "## Placeholder Counts",
    ]

    for column, count in placeholder_counts.items():
        lines.append(f"- `{column}` placeholders standardized: {count}")

    lines.extend(
        [
            "",
            "## Numerical Fixes",
        ]
    )

    for column, count in invalid_numeric_counts.items():
        lines.append(f"- `{column}` invalid values replaced with missing: {count}")

    if numerical_fill_values:
        lines.append("")
        lines.append("## Numerical Imputation Values")
        for column, value in numerical_fill_values.items():
            lines.append(f"- `{column}` filled with median: {value:.2f}")

    lines.extend(
        [
            "",
            "## Salary Validation",
            f"- `min_salary > max_salary` rows: {salary_validation['min_gt_max']}",
            f"- `avg_salary` outside `[min_salary, max_salary]` rows: {salary_validation['avg_outside_range']}",
            f"- Negative salary rows: {salary_validation['negative_salary_rows']}",
            f"- Zero salary rows: {salary_validation['zero_salary_rows']}",
            "",
            "## Modeling Readiness",
            f"- Remaining missing values: {audit['final_missing']}",
            f"- Remaining duplicate rows: {int(cleaned_df.duplicated().sum())}",
            "- The cleaned dataset is ready for modeling from a Stage 4 cleaning perspective.",
        ]
    )

    return "\n".join(lines)


def save_cleaned_dataset(df: pd.DataFrame, output_path: Path = PROCESSED_DATASET_PATH) -> Path:
    """Save the cleaned dataset to the processed data directory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def run_cleaning_pipeline() -> tuple[pd.DataFrame, dict[str, object]]:
    """Run the full Stage 4 cleaning pipeline and persist outputs."""
    raw_df = load_dataset()
    cleaned_df, audit = clean_dataset(raw_df)
    save_cleaned_dataset(cleaned_df)
    CLEANING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLEANING_REPORT_PATH.write_text(build_cleaning_report(audit, cleaned_df) + "\n", encoding="utf-8")
    return cleaned_df, audit


if __name__ == "__main__":
    cleaned_df, audit = run_cleaning_pipeline()
    print(f"Saved cleaned dataset to {PROCESSED_DATASET_PATH}")
    print(f"Saved cleaning report to {CLEANING_REPORT_PATH}")
    print(f"Final shape: {audit['final_shape']}")
    print(f"Final missing values: {audit['final_missing']}")

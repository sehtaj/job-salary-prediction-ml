"""Stage 11 reusable salary prediction workflow for Version 1."""

from __future__ import annotations

import json
from numbers import Real
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.feature_engineering import extract_job_seniority, simplify_job_title

FEATURE_DATASET_PATH = Path("data/processed/salary_data_features_v1.csv")
BEST_MODEL_PATH = Path("models/ridge_regression_pipeline.joblib")

RESULTS_DIR = Path("results/predictions")
SAMPLE_INPUTS_PATH = RESULTS_DIR / "sample_prediction_inputs.json"
PREDICTIONS_PATH = RESULTS_DIR / "sample_predictions.csv"
REPORT_PATH = RESULTS_DIR / "prediction_report.md"

TARGET_COLUMN = "avg_salary"
PREDICTOR_COLUMNS = [
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
    "company_age",
    "job_title_group",
    "job_seniority",
    "sql_yn",
    "tableau_yn",
]

NUMERIC_COLUMNS = ["Rating", "company_age"]
BINARY_COLUMNS = ["hourly", "python_yn", "R_yn", "spark", "aws", "excel", "sql_yn", "tableau_yn"]
CATEGORICAL_COLUMNS = [
    "Size",
    "Industry",
    "Sector",
    "Revenue",
    "Type of ownership",
    "job_state",
    "job_title_group",
    "job_seniority",
]
SKILL_PATTERNS = {
    "sql_yn": r"\bsql\b",
    "tableau_yn": r"\btableau\b",
}


def load_reference_feature_dataset() -> pd.DataFrame:
    """Load the feature-engineered dataset to derive schema defaults."""
    return pd.read_csv(FEATURE_DATASET_PATH)


def load_prediction_pipeline() -> object:
    """Load the best-performing trained pipeline for inference."""
    return joblib.load(BEST_MODEL_PATH)


def build_default_feature_values(reference_df: pd.DataFrame) -> dict[str, Any]:
    """Create fallback values for incomplete user input."""
    defaults: dict[str, Any] = {}
    predictors = reference_df[PREDICTOR_COLUMNS]

    for column in PREDICTOR_COLUMNS:
        if column in NUMERIC_COLUMNS:
            defaults[column] = float(predictors[column].median())
        else:
            mode = predictors[column].mode(dropna=True)
            defaults[column] = mode.iloc[0] if not mode.empty else ""

    return defaults


def extract_skill_from_text(text: str, pattern: str) -> int:
    """Return a binary skill flag from free-form job-description text."""
    description = text or ""
    return int(pd.Series([description]).str.lower().str.contains(pattern, regex=True).iloc[0])


def coerce_numeric(value: Any, column_name: str) -> float:
    """Convert numeric-like user input into a float with a clear error."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{column_name}` must be numeric; received {value!r}.") from exc


def coerce_binary(value: Any, column_name: str) -> int:
    """Normalize binary user input to 0 or 1."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and value in (0, 1):
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        truthy = {"1", "true", "yes", "y"}
        falsy = {"0", "false", "no", "n"}
        if normalized in truthy:
            return 1
        if normalized in falsy:
            return 0

    raise ValueError(f"`{column_name}` must be a binary value (0/1, yes/no, true/false); received {value!r}.")


def prepare_prediction_features(
    user_input: dict[str, Any],
    defaults: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    """Turn raw user input into the exact predictor frame expected by the saved model."""
    warnings: list[str] = []
    prepared = defaults.copy()
    derived_columns: set[str] = set()

    for column in PREDICTOR_COLUMNS:
        if column in user_input:
            prepared[column] = user_input[column]

    if "company_age" not in user_input and "age" in user_input:
        prepared["company_age"] = user_input["age"]

    job_title = str(user_input.get("Job Title", "") or "")
    if "job_title_group" not in user_input:
        if job_title:
            prepared["job_title_group"] = simplify_job_title(job_title)
            derived_columns.add("job_title_group")
        else:
            warnings.append("Missing `Job Title`; used the dataset-default `job_title_group` value.")

    if "job_seniority" not in user_input:
        if job_title:
            prepared["job_seniority"] = extract_job_seniority(job_title)
            derived_columns.add("job_seniority")
        else:
            warnings.append("Missing `Job Title`; used the dataset-default `job_seniority` value.")

    job_description = str(user_input.get("Job Description", "") or "")
    for feature_name, pattern in SKILL_PATTERNS.items():
        if feature_name not in user_input:
            if job_description:
                prepared[feature_name] = extract_skill_from_text(job_description, pattern)
                derived_columns.add(feature_name)
            else:
                warnings.append(
                    f"Missing `Job Description`; used the dataset-default `{feature_name}` value."
                )

    missing_input_columns = [
        column
        for column in PREDICTOR_COLUMNS
        if column not in user_input and column not in derived_columns
    ]
    if missing_input_columns:
        warnings.append(
            "Filled missing predictor fields with dataset defaults: "
            + ", ".join(sorted(missing_input_columns))
        )

    for column in NUMERIC_COLUMNS:
        prepared[column] = coerce_numeric(prepared[column], column)

    for column in BINARY_COLUMNS:
        prepared[column] = coerce_binary(prepared[column], column)

    for column in CATEGORICAL_COLUMNS:
        prepared[column] = str(prepared[column]).strip()

    feature_frame = pd.DataFrame([{column: prepared[column] for column in PREDICTOR_COLUMNS}])
    return feature_frame, warnings


def predict_salary(
    user_input: dict[str, Any],
    model_pipeline: object,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """Generate a salary prediction and preserve validation feedback."""
    features, warnings = prepare_prediction_features(user_input, defaults)
    prediction = float(model_pipeline.predict(features)[0])

    return {
        "prediction": prediction,
        "prepared_features": features.iloc[0].to_dict(),
        "warnings": warnings,
    }


def build_sample_inputs() -> list[dict[str, Any]]:
    """Create representative sample inputs to validate the prediction workflow."""
    return [
        {
            "sample_name": "complete_sample",
            "Job Title": "Senior Data Scientist",
            "Job Description": "Build machine learning models with Python, SQL, Tableau, AWS, and Spark.",
            "Rating": 4.1,
            "Size": "1001 to 5000 employees",
            "Industry": "Biotech & Pharmaceuticals",
            "Sector": "Biotech & Pharmaceuticals",
            "Revenue": "$1 to $5 billion (USD)",
            "Type of ownership": "Company - Private",
            "job_state": "CA",
            "hourly": 0,
            "python_yn": 1,
            "R_yn": 0,
            "spark": 1,
            "aws": 1,
            "excel": 1,
            "company_age": 18,
        },
        {
            "sample_name": "incomplete_sample",
            "Job Title": "Data Analyst",
            "Rating": 3.6,
            "Industry": "Insurance Carriers",
            "job_state": "TX",
            "python_yn": "yes",
            "aws": 0,
        },
    ]


def build_prediction_report(predictions: pd.DataFrame) -> str:
    """Summarize Stage 11 behavior and sample-prediction validation."""
    lines = [
        "# Prediction Report",
        "",
        "## Stage 11 Scope",
        "- This report documents the reusable salary prediction workflow for Version 1.",
        "- The saved Ridge Regression pipeline is used for inference so preprocessing and prediction remain connected.",
        "- New inputs can provide either direct engineered fields or lightweight raw fields such as `Job Title` and `Job Description`.",
        "",
        "## Validation Behavior",
        "- Missing predictor fields are filled with dataset-derived defaults.",
        "- `Job Title` is transformed into `job_title_group` and `job_seniority` when provided.",
        "- `Job Description` is scanned for SQL and Tableau to derive `sql_yn` and `tableau_yn` when provided.",
        "- Binary fields accept 0/1, yes/no, and true/false style input.",
        "",
        "## Sample Prediction Results",
    ]

    for _, row in predictions.iterrows():
        lines.append(f"- `{row['sample_name']}` predicted salary: {row['predicted_avg_salary']:.2f}")
        if row["warnings"]:
            lines.append(f"- `{row['sample_name']}` warnings: {row['warnings']}")
        else:
            lines.append(f"- `{row['sample_name']}` warnings: none")

    lines.extend(
        [
            "",
            "## Notes",
            "- The prediction system uses the Stage 5 feature schema and the Stage 8 trained Ridge model.",
            "- Unknown categorical values are handled safely by the saved preprocessing pipeline through one-hot encoding with `handle_unknown='ignore'`.",
            "- This workflow is ready to be reused in a CLI, notebook, or lightweight app interface.",
        ]
    )

    return "\n".join(lines)


def run_prediction_workflow() -> dict[str, Any]:
    """Run Stage 11 end to end and save sample prediction outputs."""
    reference_df = load_reference_feature_dataset()
    defaults = build_default_feature_values(reference_df)
    model_pipeline = load_prediction_pipeline()

    sample_inputs = build_sample_inputs()
    prediction_rows: list[dict[str, Any]] = []

    for sample in sample_inputs:
        sample_name = sample["sample_name"]
        user_input = {key: value for key, value in sample.items() if key != "sample_name"}
        result = predict_salary(user_input, model_pipeline, defaults)
        prediction_rows.append(
            {
                "sample_name": sample_name,
                "predicted_avg_salary": result["prediction"],
                "warnings": " | ".join(result["warnings"]),
            }
        )

    predictions_df = pd.DataFrame(prediction_rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_INPUTS_PATH.write_text(json.dumps(sample_inputs, indent=2) + "\n", encoding="utf-8")
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    REPORT_PATH.write_text(build_prediction_report(predictions_df) + "\n", encoding="utf-8")

    return {
        "sample_count": len(sample_inputs),
        "prediction_path": PREDICTIONS_PATH,
        "report_path": REPORT_PATH,
    }


if __name__ == "__main__":
    metadata = run_prediction_workflow()
    print(f"Saved sample prediction inputs to {SAMPLE_INPUTS_PATH}")
    print(f"Saved sample predictions to {metadata['prediction_path']}")
    print(f"Saved prediction report to {metadata['report_path']}")

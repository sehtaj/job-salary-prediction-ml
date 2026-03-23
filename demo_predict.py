"""Minimal demo for showing how the salary predictor works."""

from __future__ import annotations

from src.predict_salary import (
    build_default_feature_values,
    load_prediction_pipeline,
    load_reference_feature_dataset,
    predict_salary,
)


def build_demo_input() -> dict[str, object]:
    """Return a polished demo input for a live walkthrough."""
    return {
        "Job Title": "Senior Data Scientist",
        "Job Description": "Build machine learning models with Python, SQL, Tableau, AWS, and Spark.",
        "Rating": 4.2,
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
        "company_age": 15,
    }


def main() -> None:
    """Run a single clean demo prediction."""
    reference_df = load_reference_feature_dataset()
    defaults = build_default_feature_values(reference_df)
    model = load_prediction_pipeline()
    demo_input = build_demo_input()
    result = predict_salary(demo_input, model, defaults)

    print("Salary Predictor Demo")
    print("---------------------")
    print("Example role:")
    print("- Job Title: Senior Data Scientist")
    print("- State: CA")
    print("- Industry: Biotech & Pharmaceuticals")
    print("- Skills: Python, SQL, Tableau, AWS, Spark")
    print("- Company Size: 1001 to 5000 employees")
    print("- Company Rating: 4.2")
    print("- Company Age: 15 years")
    print()
    print(f"Predicted avg_salary: {result['prediction']:.2f}")

    if result["warnings"]:
        print()
        print("Warnings:")
        for warning in result["warnings"]:
            print(f"- {warning}")


if __name__ == "__main__":
    main()

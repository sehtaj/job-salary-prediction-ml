# Prediction Report

## Stage 11 Scope
- This report documents the reusable salary prediction workflow for Version 1.
- The saved Ridge Regression pipeline is used for inference so preprocessing and prediction remain connected.
- New inputs can provide either direct engineered fields or lightweight raw fields such as `Job Title` and `Job Description`.

## Validation Behavior
- Missing predictor fields are filled with dataset-derived defaults.
- `Job Title` is transformed into `job_title_group` and `job_seniority` when provided.
- `Job Description` is scanned for SQL and Tableau to derive `sql_yn` and `tableau_yn` when provided.
- Binary fields accept 0/1, yes/no, and true/false style input.

## Sample Prediction Results
- `complete_sample` predicted salary: 157.64
- `complete_sample` warnings: none
- `incomplete_sample` predicted salary: 74.25
- `incomplete_sample` warnings: Missing `Job Description`; used the dataset-default `sql_yn` value. | Missing `Job Description`; used the dataset-default `tableau_yn` value. | Filled missing predictor fields with dataset defaults: R_yn, Revenue, Sector, Size, Type of ownership, company_age, excel, hourly, spark, sql_yn, tableau_yn

## Notes
- The prediction system uses the Stage 5 feature schema and the Stage 8 trained Ridge model.
- Unknown categorical values are handled safely by the saved preprocessing pipeline through one-hot encoding with `handle_unknown='ignore'`.
- This workflow is ready to be reused in a CLI, notebook, or lightweight app interface.

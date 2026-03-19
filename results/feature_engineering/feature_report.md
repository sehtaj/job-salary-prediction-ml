# Feature Engineering Report

## Scope
- This report documents the Version 1 baseline feature set for salary prediction using structured and lightly engineered features.
- The goal is to create a strong tabular baseline before any advanced NLP workflow.

## Dataset Shape
- Input shape: 467 rows x 28 columns
- Output shape: 467 rows x 19 columns
- Target retained in dataset: `avg_salary`

## Key Feature Decisions
- Salary-derived predictors `min_salary`, `max_salary`, and `Salary Estimate` were excluded to avoid leakage.
- `Location` was replaced by the already cleaned `job_state` field.
- `age` was renamed to `company_age` for clarity.
- `Job Title` was transformed into lower-cardinality features `job_title_group` and `job_seniority`.
- Existing structured skill flags were retained, and new binary skill flags were extracted from `Job Description` using keyword matching.
- Raw text and identity-heavy fields such as `Job Description`, `Company Name`, and `company_txt` were excluded from the Version 1 baseline feature set.

## Final Predictor Columns
- `Rating`
- `Size`
- `Industry`
- `Sector`
- `Revenue`
- `Type of ownership`
- `job_state`
- `hourly`
- `python_yn`
- `R_yn`
- `spark`
- `aws`
- `excel`
- `company_age`
- `job_title_group`
- `job_seniority`
- `sql_yn`
- `tableau_yn`

## Dropped Columns
- `Salary Estimate`
- `Company Name`
- `Location`
- `Headquarters`
- `Founded`
- `Competitors`
- `company_txt`
- `same_state`
- `employer_provided`
- `min_salary`
- `max_salary`
- `age`
- `Job Title`
- `Job Description`

## Engineered Feature Summary
- `job_title_group` categories: ['data_analyst', 'data_engineer', 'data_scientist', 'leadership', 'machine_learning', 'other', 'research_scientist']
- `job_seniority` categories: ['director', 'entry', 'lead', 'mid', 'principal', 'senior']
- `sql_yn` positive rows: 243
- `tableau_yn` positive rows: 99

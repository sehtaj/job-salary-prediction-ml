# Preprocessing Report

## Stage 6 Scope
- This report documents the preprocessing workflow for the Version 1 baseline models.
- Numerical features are passed through unchanged.
- Categorical features are one-hot encoded with `handle_unknown='ignore'` so train/test columns stay aligned.

## Feature Groups
### Numerical Features
- `Rating`
- `hourly`
- `python_yn`
- `R_yn`
- `spark`
- `aws`
- `excel`
- `company_age`
- `sql_yn`
- `tableau_yn`

### Categorical Features
- `Size`
- `Industry`
- `Sector`
- `Revenue`
- `Type of ownership`
- `job_state`
- `job_title_group`
- `job_seniority`

## Shape Verification
- X_train before encoding: (373, 18)
- X_test before encoding: (94, 18)
- X_train after encoding: (373, 171)
- X_test after encoding: (94, 171)
- y_train shape: (373,)
- y_test shape: (94,)
- Encoded feature count: 171

## Notes
- One-hot encoding was applied only to categorical columns.
- `handle_unknown='ignore'` ensures unseen categories in the test set do not break the pipeline.
- The encoded feature list was saved for later model interpretation and debugging.

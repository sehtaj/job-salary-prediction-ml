# Diagnostics Report

## Scope
- This report adds two post-training diagnostics for Version 1: cross-validated model comparison and Ridge residual/error analysis.
- Cross-validation checks whether the current model ranking is stable beyond a single train/test split.
- Ridge error analysis highlights where the selected model struggles most on the held-out test set.

## Cross-Validation Summary
- `ridge_regression`: CV RMSE=27.1953 +/- 2.9616, CV R²=0.4609 +/- 0.0426
- `random_forest`: CV RMSE=28.0535 +/- 3.5087, CV R²=0.4255 +/- 0.0736
- `lasso_regression`: CV RMSE=28.1626 +/- 3.3495, CV R²=0.4218 +/- 0.0601
- `linear_regression`: CV RMSE=30.6666 +/- 1.4473, CV R²=0.3076 +/- 0.0577

## Ridge Test-Set Error Patterns
- Largest average job-title-group error: `leadership` (mean abs error 61.7317)
- Largest average seniority error: `director` (mean abs error 72.9031)
- Hardest salary band: `high` (mean abs error 38.2832)
- Highest-error retained state group: `IL` (mean abs error 37.6739)

## Interpretation
- Cross-validation still ranks `ridge_regression` as the most stable model by average RMSE.
- Ridge errors are not uniform: they vary meaningfully by job-title family, seniority level, and salary range.
- This suggests the project now has a stronger model-selection story: Ridge is not only the best test-set model, but it is also competitive under repeated resampling.
- The next quality improvement after this would be targeted feature refinement for the hardest Ridge error groups rather than more broad hyperparameter tuning.

## Saved Outputs
- Cross-validation metrics: `results/diagnostics/cross_validation_metrics.csv`
- Ridge residual table: `results/diagnostics/ridge_test_residuals.csv`
- Group error summaries: `results/diagnostics/ridge_error_by_job_title_group.csv`, `results/diagnostics/ridge_error_by_job_seniority.csv`, `results/diagnostics/ridge_error_by_salary_band.csv`, `results/diagnostics/ridge_error_by_job_state.csv`
- Diagnostic plots: `results/diagnostics/cross_validation_rmse.png`, `results/diagnostics/ridge_residuals_vs_actual.png`, `results/diagnostics/ridge_abs_error_by_job_title_group.png`

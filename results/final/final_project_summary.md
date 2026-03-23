# Final Project Summary

## Project Status

Version 1 of the salary-prediction project is complete as an end-to-end tabular machine learning workflow.

The project includes:
- data loading
- exploratory data analysis
- data cleaning
- feature engineering
- preprocessing and encoding
- train/test split
- model training
- evaluation
- feature-importance analysis
- reusable salary prediction
- diagnostics and error analysis

## Best Model

- Selected model: `ridge_regression`
- MSE: `962.6841`
- RMSE: `31.0272`
- R²: `0.4181`

## Model Ranking

1. `ridge_regression`
2. `linear_regression`
3. `random_forest`
4. `lasso_regression`

## Key Findings

- Regularized linear modeling worked better than the tree-based baseline on this dataset.
- A denser Ridge alpha search with shuffled 10-fold cross-validation improved the final Ridge result slightly.
- Random Forest tuning improved the nonlinear baseline substantially, but it still did not beat Ridge.
- Job title grouping, job seniority, and job state were some of the strongest salary signals.
- Industry, revenue, size, sector, and company age also contributed meaningful signal.
- Skill flags helped, but title and company-context features had larger influence overall.
- The final prediction workflow reuses the saved preprocessing + model pipeline so inference stays consistent with training.
- Cross-validation and residual diagnostics supported Ridge as the most stable final model and showed that leadership and high-salary roles remain the hardest cases.

## Final Artifacts

Processed data:
- `data/processed/salary_data_cleaned.csv`
- `data/processed/salary_data_features_v1.csv`

Saved splits:
- `data/splits/X_train.csv`
- `data/splits/X_test.csv`
- `data/splits/y_train.csv`
- `data/splits/y_test.csv`

Saved models:
- `models/linear_regression_pipeline.joblib`
- `models/ridge_regression_pipeline.joblib`
- `models/lasso_regression_pipeline.joblib`
- `models/random_forest_pipeline.joblib`

Key reports:
- `results/evaluation/evaluation_report.md`
- `results/feature_importance/feature_importance_report.md`
- `results/predictions/prediction_report.md`
- `results/diagnostics/diagnostics_report.md`

## Reproducibility

The project is reproducible through the stage modules in `src/` and can be rerun from data loading through prediction using the commands documented in `README.md`.

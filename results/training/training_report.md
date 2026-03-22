# Training Report

## Stage 8 Scope
- This report documents baseline model training for Version 1.
- Both models were trained inside sklearn Pipelines so preprocessing stays attached to the model artifact.

## Training Data
- X_train shape: (373, 18)
- y_train shape: (373,)
- Random state: 42

## Models Trained
- `LinearRegression` as the interpretable baseline
- `RidgeCV` as the L2-regularized linear model
- `LassoCV` as the L1-regularized linear model
- `RandomForestRegressor` as the nonlinear tabular baseline

## Regularized Linear Model Search Space
- `alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]`
- Ridge selected alpha: 10.0
- Lasso selected alpha: 1.0

## Random Forest Configuration
- `n_estimators=300`
- `max_depth=None`
- `min_samples_split=2`
- `min_samples_leaf=1`
- `random_state=42`

## Saved Model Artifacts
- `linear_regression`: `models/linear_regression_pipeline.joblib`
- `ridge_regression`: `models/ridge_regression_pipeline.joblib`
- `lasso_regression`: `models/lasso_regression_pipeline.joblib`
- `random_forest`: `models/random_forest_pipeline.joblib`

## Notes
- The Linear Regression model provides a simple baseline for comparison.
- The Random Forest hyperparameters were chosen as a strong initial baseline and can be tuned later if evaluation suggests it.
- Final model comparison happens in Stage 9 using the held-out test set.

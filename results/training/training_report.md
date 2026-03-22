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
- `RandomForestRegressor` as the nonlinear tabular baseline

## Random Forest Configuration
- `n_estimators=300`
- `max_depth=None`
- `min_samples_split=2`
- `min_samples_leaf=1`
- `random_state=42`

## Saved Model Artifacts
- `linear_regression`: `models/linear_regression_pipeline.joblib`
- `random_forest`: `models/random_forest_pipeline.joblib`

## Notes
- The Linear Regression model provides a simple baseline for comparison.
- The Random Forest hyperparameters were chosen as a strong initial baseline and can be tuned later if evaluation suggests it.
- Final model comparison happens in Stage 9 using the held-out test set.

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

## Ridge Search Configuration
- `alpha_count=81`
- `alpha_min=0.0001`
- `alpha_max=10000.0`
- `cross_validation=10-fold shuffled CV`
- Ridge selected alpha: 7.943282347242821

## Lasso Search Configuration
- `alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]`
- Lasso selected alpha: 1.0

## Random Forest Configuration
- baseline `n_estimators=300`
- baseline `max_depth=None`
- baseline `min_samples_split=2`
- baseline `min_samples_leaf=1`
- `random_state=42`
- tuning search iterations: 30
- tuning cross-validation: 5-fold shuffled CV
- tuned `n_estimators=800`
- tuned `max_depth=None`
- tuned `min_samples_split=15`
- tuned `min_samples_leaf=1`
- tuned `max_features=0.5`

## Saved Model Artifacts
- `linear_regression`: `models/linear_regression_pipeline.joblib`
- `ridge_regression`: `models/ridge_regression_pipeline.joblib`
- `lasso_regression`: `models/lasso_regression_pipeline.joblib`
- `random_forest`: `models/random_forest_pipeline.joblib`

## Notes
- The Linear Regression model provides a simple baseline for comparison.
- Ridge now uses a denser logarithmic alpha grid with shuffled 10-fold cross-validation to make regularization tuning more stable.
- Random Forest is tuned with a focused randomized search over the highest-impact tree hyperparameters rather than a brute-force grid.
- Final model comparison happens in Stage 9 using the held-out test set.

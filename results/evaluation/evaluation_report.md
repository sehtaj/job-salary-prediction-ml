# Evaluation Report

## Stage 9 Scope
- This report compares the Version 1 baseline models on the held-out test set.
- The best model is selected primarily by the lowest RMSE, with MSE and R² also reported.

## Model Metrics
- `ridge_regression`: MSE=962.6841, RMSE=31.0272, R²=0.4181
- `linear_regression`: MSE=1026.3917, RMSE=32.0373, R²=0.3796
- `random_forest`: MSE=1109.7804, RMSE=33.3134, R²=0.3292
- `lasso_regression`: MSE=1113.6436, RMSE=33.3713, R²=0.3269

## Best Model
- Best-performing model: `ridge_regression`
- Best RMSE: 31.0272
- Best MSE: 962.6841
- Best R²: 0.4181
- Better regularized model between Ridge and Lasso: `ridge_regression`

## Saved Outputs
- Metrics table: `results/evaluation/model_metrics.csv`
- Test predictions: `results/evaluation/test_set_predictions.csv`
- Comparison plot: `results/evaluation/model_comparison.png`
- Best-model scatter plot: `results/evaluation/best_model_actual_vs_predicted.png`

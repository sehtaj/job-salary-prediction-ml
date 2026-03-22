# Salary Prediction Using Job and Company Features

This project predicts average salary (`avg_salary`) from structured job-posting and company features using machine learning. It is organized as an end-to-end Version 1 baseline that covers planning, data cleaning, feature engineering, model training, evaluation, feature-importance analysis, and a reusable prediction workflow.

## Problem Statement

The goal is to predict job salaries from tabular signals such as:
- company rating, size, industry, sector, revenue, and ownership type
- job location and title information
- binary skill indicators such as Python, R, Spark, AWS, Excel, SQL, and Tableau

Target variable:
- `avg_salary`

Primary dataset:
- `data/raw/salary_data_cleaned.csv`

## Final Model Set

The project compares four Version 1 baseline models:
- `Linear Regression`
- `Ridge Regression`
- `Lasso Regression`
- `Random Forest Regressor`

Best final model:
- `Ridge Regression`

Held-out test metrics for the best model:
- `MSE = 975.7049`
- `RMSE = 31.2363`
- `R² = 0.4102`

## Key Results

- `Ridge Regression` performed best on the test set and became the final selected Version 1 model.
- `Lasso Regression` underperformed Ridge on prediction accuracy, but it remains useful as a sparsity-oriented comparison model.
- `Random Forest` did not outperform the regularized linear models on this dataset.
- Feature-importance analysis showed that job title grouping, job seniority, state/location, industry, revenue, and other company-context features were the strongest drivers of salary prediction.
- Skill flags such as SQL, Tableau, Python, Spark, and AWS added signal, but they were not as dominant as title and company-context features.

## Project Structure

```text
job-salary-prediction-ml/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── models/
├── notebooks/
├── results/
│   ├── cleaning/
│   ├── eda/
│   ├── evaluation/
│   ├── feature_engineering/
│   ├── feature_importance/
│   ├── final/
│   ├── predictions/
│   ├── preprocessing/
│   ├── splits/
│   └── training/
└── src/
```

## Core Pipeline Files

- `src/data_loading.py`: load and inspect the raw dataset
- `src/eda.py`: exploratory data analysis and visual outputs
- `src/data_cleaning.py`: clean duplicates, placeholders, and missing values
- `src/feature_engineering.py`: build the Version 1 modeling dataset
- `src/preprocessing.py`: scale numeric features and one-hot encode categorical features
- `src/data_splitting.py`: create train/test splits
- `src/model_registry.py`: define the baseline models
- `src/train_model.py`: train and save model pipelines
- `src/evaluate_model.py`: evaluate all models on the held-out test set
- `src/feature_importance.py`: analyze coefficients and feature importances
- `src/predict_salary.py`: run the reusable salary prediction workflow

## Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## End-to-End Run Order

Run the full Version 1 pipeline in this order:

```bash
.venv/bin/python -m src.data_loading
.venv/bin/python -m src.eda
.venv/bin/python -m src.data_cleaning
.venv/bin/python -m src.feature_engineering
.venv/bin/python -m src.preprocessing
.venv/bin/python -m src.data_splitting
.venv/bin/python -m src.train_model
.venv/bin/python -m src.evaluate_model
.venv/bin/python -m src.feature_importance
.venv/bin/python -m src.predict_salary
```

## Machine Learning Pipeline

Data Loading  
-> Data Exploration  
-> Data Cleaning  
-> Feature Engineering  
-> Feature Encoding  
-> Train/Test Split  
-> Model Training  
-> Model Evaluation  
-> Feature Importance Analysis  
-> Salary Prediction

## Important Outputs

Model evaluation:
- `results/evaluation/model_metrics.csv`
- `results/evaluation/model_comparison.png`
- `results/evaluation/best_model_actual_vs_predicted.png`

Feature importance:
- `results/feature_importance/feature_importance_report.md`
- `results/feature_importance/linear_regression_top_coefficients.png`
- `results/feature_importance/random_forest_top_importances.png`

Prediction system:
- `results/predictions/sample_prediction_inputs.json`
- `results/predictions/sample_predictions.csv`
- `results/predictions/prediction_report.md`

Final summary:
- `results/final/final_project_summary.md`

## Sample Prediction Workflow

To validate the final inference system:

```bash
.venv/bin/python -m src.predict_salary
```

This loads the saved `Ridge Regression` pipeline, rebuilds any derived fields needed for inference, validates incomplete input, and writes sample predictions to `results/predictions/`.

## Version 1 Scope

This repository currently focuses on a strong tabular baseline only:
- structured features
- light feature engineering
- no TF-IDF or text-vectorization pipeline yet

The next natural extension is a Version 2 workflow that adds richer NLP features from job descriptions.

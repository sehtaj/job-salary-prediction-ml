# Predicting Job Salaries Using Machine Learning

This project uses machine learning to predict job salaries from job posting and company attributes in `data/raw/salary_data_cleaned.csv`.

## Project Objective

The objective is to build regression models that predict the target variable `avg_salary` using relevant features from the dataset.

## Dataset

- File: `data/raw/salary_data_cleaned.csv`
- Target variable: `avg_salary`

## Important Features

- `Job Title`
- `Rating`
- `Location`
- `Size`
- `Industry`
- `Sector`
- `Revenue`
- `company_age`
- `python_yn`
- `spark`
- `aws`
- `excel`
- `R_yn`

## Models Planned

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor

## Evaluation Metrics

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R2 Score

## Machine Learning Pipeline

Data Loading
-> Data Exploration
-> Data Cleaning
-> Feature Engineering
-> Feature Encoding
-> Train/Test Split
-> Model Training (Linear Regression, Ridge, Lasso, Random Forest)
-> Model Evaluation (MSE, RMSE, R2)
-> Feature Importance Analysis
-> Salary Prediction

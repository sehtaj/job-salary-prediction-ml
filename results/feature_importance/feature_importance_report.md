# Feature Importance Report

## Stage 10 Scope
- This report explains which encoded features and original job/company attributes most influenced the baseline models.
- Ridge Regression is the best-performing final model, but Stage 10 uses Linear Regression and Random Forest for interpretation because their coefficients and feature importances are straightforward to inspect.
- Linear Regression coefficients show signed influence direction after preprocessing.
- Random Forest importances show how much each feature helped reduce prediction error in the tree ensemble.

## Linear Regression: Strongest Positive Coefficients
- `categorical__job_state_LOS ANGELES`: 37.6403
- `categorical__Industry_Investment Banking & Asset Management`: 36.1559
- `categorical__Industry_Health Care Products Manufacturing`: 35.1401
- `categorical__Industry_Religious Organizations`: 34.7994
- `categorical__job_state_CA`: 30.6852

## Linear Regression: Strongest Negative Coefficients
- `categorical__job_state_ID`: -44.7275
- `categorical__job_seniority_entry`: -39.9568
- `categorical__Industry_Video Games`: -33.0263
- `categorical__Industry_Sporting Goods Stores`: -32.5587
- `categorical__Industry_Transportation Equipment Manufacturing`: -30.7518

## Random Forest: Top Encoded Features
- `categorical__job_title_group_data_analyst`: 0.1295
- `categorical__job_seniority_mid`: 0.1017
- `numerical__hourly`: 0.0863
- `categorical__job_state_CA`: 0.0848
- `numerical__Rating`: 0.0669
- `numerical__company_age`: 0.0532
- `categorical__job_title_group_data_scientist`: 0.0353
- `numerical__python_yn`: 0.0208
- `categorical__job_seniority_entry`: 0.0204
- `categorical__job_seniority_principal`: 0.0192

## Most Influential Original Feature Groups
- `job_state`: linear abs coefficient=451.1269, random forest importance=0.1456
- `Industry`: linear abs coefficient=753.2373, random forest importance=0.0949
- `job_seniority`: linear abs coefficient=128.6817, random forest importance=0.1672
- `job_title_group`: linear abs coefficient=59.6957, random forest importance=0.1913
- `Revenue`: linear abs coefficient=131.8515, random forest importance=0.0456
- `hourly`: linear abs coefficient=5.8443, random forest importance=0.0863
- `Sector`: linear abs coefficient=146.5432, random forest importance=0.0264
- `Size`: linear abs coefficient=72.6905, random forest importance=0.0311

## Interpretation
- Job title grouping, job seniority, and state/location signals are consistently among the strongest predictors across both model families.
- Company-context features such as industry, revenue, sector, size, and company age also contribute meaningful salary signal.
- Skill flags such as SQL, Tableau, Python, Spark, and AWS contribute useful signal, but they are not as dominant as title, seniority, and company-context features in this Version 1 baseline.
- Linear coefficients help identify whether a feature pushes salary predictions up or down, while random forest importances highlight which features matter most overall.

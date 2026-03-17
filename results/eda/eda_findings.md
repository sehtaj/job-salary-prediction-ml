# EDA Findings

## Dataset Overview
- Shape: 742 rows x 28 columns
- Target column present: True
- `avg_salary` mean: 100.63
- `avg_salary` median: 97.50
- `avg_salary` range: 13.50 to 254.00

## Numerical Feature Summary
- Highest correlation with `avg_salary`: `max_salary` (0.99)
- Next highest correlation with `avg_salary`: `min_salary` (0.98)
- Skill indicator most correlated with `avg_salary`: `python_yn` (0.33)
- Weak linear relationship from `Rating` to `avg_salary`: 0.01

## Categorical Distributions
- Most common job title: Data Scientist (131 rows)
- Most common location: New York, NY (55 rows)
- Largest sector: Information Technology (180 rows)
- Most common industry: Biotech & Pharmaceuticals (112 rows)

## Salary Relationships
- Highest-paying frequent job title: Principal Data Scientist (176.30)
- Highest-paying frequent sector: Media (116.67)
- Mean salary with Python skill: 112.65
- Mean salary without Python skill: 87.16
- Mean salary with Spark skill: 113.35
- Mean salary with AWS skill: 112.56

## Outliers and Unusual Values
- IQR bounds for `avg_salary`: 0.00 to 196.00
- Outlier count: 11
- Highest observed `avg_salary`: 254.00

## Notes for Modeling
- `min_salary` and `max_salary` are extremely correlated with `avg_salary`, so they are likely target-leakage features if kept as predictors.
- Binary skill indicators such as `python_yn`, `spark`, and `aws` show higher average salary and are likely useful predictors.
- `hourly` roles have much lower `avg_salary` values than the rest of the dataset and deserve special handling during cleaning or modeling.

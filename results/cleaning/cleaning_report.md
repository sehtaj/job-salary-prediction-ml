# Data Cleaning Report

## Dataset Shape
- Original shape: 742 rows x 28 columns
- Final shape: 467 rows x 28 columns
- Duplicate rows removed: 275

## Missing and Placeholder Handling
- Raw `NaN` count before cleaning: 0
- Missing values introduced after converting placeholders: 515
- Final missing values after imputation: 0
- Placeholder-like categorical values were standardized to missing first, then imputed as `Unknown`.

## Placeholder Counts
- `Headquarters` placeholders standardized: 1
- `Size` placeholders standardized: 6
- `Type of ownership` placeholders standardized: 2
- `Industry` placeholders standardized: 7
- `Sector` placeholders standardized: 7
- `Revenue` placeholders standardized: 134
- `Competitors` placeholders standardized: 285

## Numerical Fixes
- `Rating` invalid values replaced with missing: 7
- `Founded` invalid values replaced with missing: 33
- `age` invalid values replaced with missing: 33

## Numerical Imputation Values
- `Rating` filled with median: 3.75
- `Founded` filled with median: 1995.00
- `age` filled with median: 25.00

## Salary Validation
- `min_salary > max_salary` rows: 0
- `avg_salary` outside `[min_salary, max_salary]` rows: 0
- Negative salary rows: 0
- Zero salary rows: 0

## Modeling Readiness
- Remaining missing values: 0
- Remaining duplicate rows: 0
- The cleaned dataset is ready for modeling from a Stage 4 cleaning perspective.

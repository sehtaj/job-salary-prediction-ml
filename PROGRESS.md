# ML Project Progress Tracker

## Stage 0 — Project Planning
- [x] Define the project objective and target variable (`avg_salary`)
- [x] Confirm the dataset file name and location (`salary_data_cleaned.csv`)
- [x] Review all available columns and identify useful predictors
- [x] Decide which models to train first
- [x] Define success metrics: MSE, RMSE, and R2 score
- [x] Outline the end-to-end ML workflow in `README.md`

## Stage 1 — Project Setup
- [x] Create the project folder structure
- [x] Create the `data/`, `notebooks/`, `src/`, `models/`, and `results/` directories
- [x] Set up a Python virtual environment
- [x] Install required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
- [x] Create starter Python files in `src/`
- [x] Verify that the project runs without import errors

## Stage 2 — Data Loading
- [x] Place the dataset inside the appropriate data folder
- [x] Implement data loading logic in `src/data_loading.py`
- [x] Load the CSV file using pandas
- [x] Inspect dataset shape, column names, and data types
- [x] Preview the first few rows of the dataset
- [x] Check whether the target column `avg_salary` is present

## Stage 3 — Exploratory Data Analysis
- [x] Generate summary statistics for numerical features
- [x] Review categorical feature distributions
- [x] Visualize the distribution of `avg_salary`
- [x] Explore relationships between salary and key numerical features
- [x] Explore relationships between salary and categorical features
- [x] Check correlations among numerical columns
- [x] Identify possible outliers or unusual values
- [x] Document EDA findings in a notebook or markdown file

## Stage 4 — Data Cleaning
- [ ] Check for missing values in each column
- [ ] Decide how to handle missing numerical values
- [ ] Decide how to handle missing categorical values
- [ ] Remove or fix duplicate records if present
- [ ] Standardize inconsistent text values in categorical columns
- [ ] Validate salary-related columns (`min_salary`, `max_salary`, `avg_salary`)
- [ ] Confirm that cleaned data is ready for modeling
- [ ] Save the cleaned dataset to `data/processed/`

## Stage 5 — Feature Engineering
- [ ] Select the most relevant input features
- [ ] Confirm whether salary-derived columns should remain as predictors
- [ ] Create or refine company-related features such as `company_age`
- [ ] Use skill indicator columns (`python_yn`, `R_yn`, `spark`, `aws`, `excel`) effectively
- [ ] Consider extracting useful information from `Job Title`
- [ ] Consider extracting state or region information from `Location`
- [ ] Remove features that do not add modeling value
- [ ] Save the feature-engineered dataset

## Stage 6 — Feature Encoding
- [ ] Separate numerical and categorical features
- [ ] Identify columns that require encoding
- [ ] Apply one-hot encoding to categorical features
- [ ] Ensure encoded features align between training and testing data
- [ ] Build a preprocessing workflow for reuse
- [ ] Verify the final feature matrix shape

## Stage 7 — Train/Test Split
- [ ] Define feature matrix `X` and target vector `y`
- [ ] Split the dataset into training and testing sets
- [ ] Set a fixed `random_state` for reproducibility
- [ ] Verify the shapes of train and test datasets
- [ ] Store split datasets or pipeline objects if needed

## Stage 8 — Model Training
- [ ] Implement baseline training logic in `src/train_model.py`
- [ ] Train a Linear Regression model
- [ ] Train a Random Forest Regressor model
- [ ] Tune key Random Forest hyperparameters if needed
- [ ] Compare training workflows for both models
- [ ] Save trained model artifacts to `models/`

## Stage 9 — Model Evaluation
- [ ] Implement evaluation logic in `src/evaluate_model.py`
- [ ] Generate predictions on the test set
- [ ] Calculate MSE for each model
- [ ] Calculate RMSE for each model
- [ ] Calculate R2 score for each model
- [ ] Compare model performance side by side
- [ ] Identify the best-performing model
- [ ] Save evaluation results to `results/`

## Stage 10 — Feature Importance Analysis
- [ ] Extract coefficients from the Linear Regression model
- [ ] Extract feature importances from the Random Forest model
- [ ] Match importance values to feature names
- [ ] Visualize the most influential features
- [ ] Interpret which job attributes most affect salary predictions
- [ ] Document feature importance insights

## Stage 11 — Prediction System
- [ ] Build a reusable prediction pipeline
- [ ] Ensure preprocessing and model inference are connected
- [ ] Create a function to predict salary for new job inputs
- [ ] Validate predictions on sample input data
- [ ] Handle invalid or incomplete user input gracefully
- [ ] Save or expose the final prediction workflow for reuse

## Stage 12 — Final Project Outputs
- [ ] Finalize all source files in `src/`
- [ ] Clean up notebooks and remove unused code
- [ ] Update `README.md` with project overview, setup steps, and results
- [ ] Summarize key findings and model performance
- [ ] Organize saved models, processed data, and result files
- [ ] Review the full project structure for completeness
- [ ] Confirm the project can be reproduced end to end
- [ ] Mark completed items in this tracker as progress is made

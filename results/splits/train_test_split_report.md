# Train/Test Split Report

## Configuration
- Target column: `avg_salary`
- Test size: 0.2
- Random state: 42

## Dataset Shapes
- Full feature matrix X: (467, 18)
- Full target vector y: (467,)
- X_train: (373, 18)
- X_test: (94, 18)
- y_train: (373,)
- y_test: (94,)

## Notes
- The split is reproducible because a fixed random_state was used.
- Split CSV files were saved under `data/splits/` for reuse in model training and evaluation.

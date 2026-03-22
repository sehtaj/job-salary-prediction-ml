"""Stage 7 train/test split workflow for the Version 1 salary model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_DATASET_PATH = Path("data/processed/salary_data_features_v1.csv")
TARGET_COLUMN = "avg_salary"
RANDOM_STATE = 42
TEST_SIZE = 0.2

X_TRAIN_PATH = Path("data/splits/X_train.csv")
X_TEST_PATH = Path("data/splits/X_test.csv")
Y_TRAIN_PATH = Path("data/splits/y_train.csv")
Y_TEST_PATH = Path("data/splits/y_test.csv")
SPLIT_REPORT_PATH = Path("results/splits/train_test_split_report.md")


def load_feature_dataset(path: Path = FEATURE_DATASET_PATH) -> pd.DataFrame:
    """Load the Version 1 feature-engineered dataset."""
    return pd.read_csv(path)


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Define the predictor matrix X and target vector y."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def run_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into reproducible training and testing partitions."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def save_split_datasets(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Persist the split datasets for later modeling stages."""
    X_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(X_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    y_train.to_frame(name=TARGET_COLUMN).to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_frame(name=TARGET_COLUMN).to_csv(Y_TEST_PATH, index=False)


def build_split_report(
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    test_size: float,
    random_state: int,
) -> str:
    """Create a markdown summary of the split configuration and shapes."""
    lines = [
        "# Train/Test Split Report",
        "",
        "## Configuration",
        f"- Target column: `{TARGET_COLUMN}`",
        f"- Test size: {test_size}",
        f"- Random state: {random_state}",
        "",
        "## Dataset Shapes",
        f"- Full feature matrix X: {X.shape}",
        f"- Full target vector y: {y.shape}",
        f"- X_train: {X_train.shape}",
        f"- X_test: {X_test.shape}",
        f"- y_train: {y_train.shape}",
        f"- y_test: {y_test.shape}",
        "",
        "## Notes",
        "- The split is reproducible because a fixed random_state was used.",
        "- Split CSV files were saved under `data/splits/` for reuse in model training and evaluation.",
    ]
    return "\n".join(lines)


def run_split_workflow() -> dict[str, object]:
    """Execute the Stage 7 split workflow and save outputs."""
    df = load_feature_dataset()
    X, y = split_features_and_target(df)
    X_train, X_test, y_train, y_test = run_train_test_split(X, y)
    save_split_datasets(X_train, X_test, y_train, y_test)

    SPLIT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_REPORT_PATH.write_text(
        build_split_report(X, y, X_train, X_test, y_train, y_test, TEST_SIZE, RANDOM_STATE) + "\n",
        encoding="utf-8",
    )

    return {
        "X_shape": X.shape,
        "y_shape": y.shape,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_shape": y_train.shape,
        "y_test_shape": y_test.shape,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }


if __name__ == "__main__":
    metadata = run_split_workflow()
    print(f"Saved X_train to {X_TRAIN_PATH}")
    print(f"Saved X_test to {X_TEST_PATH}")
    print(f"Saved y_train to {Y_TRAIN_PATH}")
    print(f"Saved y_test to {Y_TEST_PATH}")
    print(f"Split report saved to {SPLIT_REPORT_PATH}")
    print(f"X_train shape: {metadata['X_train_shape']}")
    print(f"X_test shape: {metadata['X_test_shape']}")

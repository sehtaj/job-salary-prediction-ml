"""Helpers for locating and loading the salary dataset."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "salary_data_cleaned.csv"


def resolve_dataset_path(file_path: str | Path | None = None) -> Path:
    """Return an absolute path to the dataset file."""
    dataset_path = Path(file_path) if file_path is not None else DEFAULT_DATASET_PATH

    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    return dataset_path.resolve()


def load_dataset(file_path: str | Path | None = None) -> pd.DataFrame:
    """Load the salary dataset into a pandas DataFrame."""
    dataset_path = resolve_dataset_path(file_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if dataset_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, got: {dataset_path}")

    return pd.read_csv(dataset_path)


def inspect_dataset(file_path: str | Path | None = None) -> pd.DataFrame:
    """Load the dataset and print the first-pass inspection details."""
    df = load_dataset(file_path)

    print(f"Dataset shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print("\nFirst five rows:")
    print(df.head())
    print(f"\nTarget column 'avg_salary' present: {'avg_salary' in df.columns}")

    return df


if __name__ == "__main__":
    df = inspect_dataset()
    print(f"Loaded dataset from: {DEFAULT_DATASET_PATH}")

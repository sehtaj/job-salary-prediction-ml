"""Helpers for loading the salary dataset."""

from pathlib import Path

import pandas as pd


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset into a pandas DataFrame."""
    dataset_path = Path(file_path)
    return pd.read_csv(dataset_path)


if __name__ == "__main__":
    df = load_dataset("salary_data_cleaned.csv")
    print(df.head())

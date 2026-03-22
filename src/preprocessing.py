"""Stage 6 preprocessing and feature-encoding workflow for Version 1."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

FEATURE_DATASET_PATH = Path("data/processed/salary_data_features_v1.csv")
PREPROCESSING_REPORT_PATH = Path("results/preprocessing/preprocessing_report.md")
ENCODED_FEATURES_PATH = Path("results/preprocessing/encoded_feature_names.csv")
TARGET_COLUMN = "avg_salary"


def load_feature_dataset(path: Path = FEATURE_DATASET_PATH) -> pd.DataFrame:
    """Load the Version 1 feature-engineered dataset."""
    return pd.read_csv(path)


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictors from the salary target."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def get_feature_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify numerical features and categorical features for encoding."""
    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical_features = [column for column in X.columns if column not in categorical_features]
    return numerical_features, categorical_features


def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build a reusable preprocessing workflow for baseline models."""
    return ColumnTransformer(
        transformers=[
            ("numerical", "passthrough", numerical_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )


def validate_preprocessing_workflow(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, object]:
    """Run a dry-run split and transform to verify encoded feature alignment."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    encoded_feature_names = preprocessor.get_feature_names_out().tolist()
    pd.DataFrame({"feature_name": encoded_feature_names}).to_csv(ENCODED_FEATURES_PATH, index=False)

    return {
        "train_shape_before": X_train.shape,
        "test_shape_before": X_test.shape,
        "train_shape_after": X_train_encoded.shape,
        "test_shape_after": X_test_encoded.shape,
        "target_train_shape": y_train.shape,
        "target_test_shape": y_test.shape,
        "encoded_feature_count": len(encoded_feature_names),
        "encoded_feature_names": encoded_feature_names,
    }


def build_preprocessing_report(
    numerical_features: list[str],
    categorical_features: list[str],
    metadata: dict[str, object],
) -> str:
    """Create a markdown summary of Stage 6 preprocessing decisions."""
    lines = [
        "# Preprocessing Report",
        "",
        "## Stage 6 Scope",
        "- This report documents the preprocessing workflow for the Version 1 baseline models.",
        "- Numerical features are passed through unchanged.",
        "- Categorical features are one-hot encoded with `handle_unknown='ignore'` so train/test columns stay aligned.",
        "",
        "## Feature Groups",
        "### Numerical Features",
    ]

    for feature in numerical_features:
        lines.append(f"- `{feature}`")

    lines.extend(["", "### Categorical Features"])
    for feature in categorical_features:
        lines.append(f"- `{feature}`")

    lines.extend(
        [
            "",
            "## Shape Verification",
            f"- X_train before encoding: {metadata['train_shape_before']}",
            f"- X_test before encoding: {metadata['test_shape_before']}",
            f"- X_train after encoding: {metadata['train_shape_after']}",
            f"- X_test after encoding: {metadata['test_shape_after']}",
            f"- y_train shape: {metadata['target_train_shape']}",
            f"- y_test shape: {metadata['target_test_shape']}",
            f"- Encoded feature count: {metadata['encoded_feature_count']}",
            "",
            "## Notes",
            "- One-hot encoding was applied only to categorical columns.",
            "- `handle_unknown='ignore'` ensures unseen categories in the test set do not break the pipeline.",
            "- The encoded feature list was saved for later model interpretation and debugging.",
        ]
    )

    return "\n".join(lines)


def run_preprocessing_workflow() -> dict[str, object]:
    """Execute the Stage 6 preprocessing workflow and save its report."""
    df = load_feature_dataset()
    X, y = split_features_and_target(df)
    numerical_features, categorical_features = get_feature_groups(X)
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    metadata = validate_preprocessing_workflow(X, y, preprocessor)

    PREPROCESSING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREPROCESSING_REPORT_PATH.write_text(
        build_preprocessing_report(numerical_features, categorical_features, metadata) + "\n",
        encoding="utf-8",
    )

    metadata["numerical_features"] = numerical_features
    metadata["categorical_features"] = categorical_features
    return metadata


if __name__ == "__main__":
    metadata = run_preprocessing_workflow()
    print(f"Saved preprocessing report to {PREPROCESSING_REPORT_PATH}")
    print(f"Saved encoded feature names to {ENCODED_FEATURES_PATH}")
    print(f"Train encoded shape: {metadata['train_shape_after']}")
    print(f"Test encoded shape: {metadata['test_shape_after']}")

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(path: str | None = None) -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parents[1]
    csv_path = Path(path) if path else base_dir / "data" / "medical_insurance.csv"
    df = pd.read_csv(csv_path)
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    if 'sex' in df.columns:
        df['sex'] = df['sex'].map({'female': 0, 'male': 1})
    if 'smoker' in df.columns:
        df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
    if 'region' in df.columns:
        df = pd.get_dummies(df, columns=['region'], drop_first=True)
    target_col = 'annual_medical_cost' if 'annual_medical_cost' in df.columns else 'charges'
    X = df.drop(target_col, axis=1)
    # Drop identifier columns if present
    for id_col in ('person_id', 'id', 'index'):
        if id_col in X.columns:
            X = X.drop(columns=[id_col])
    # Encode remaining categorical columns
    X = pd.get_dummies(X, drop_first=True)
    y = df[target_col]
    return X, y


def prepare_split(test_size: float = 0.2, random_state: int = 42, path: str | None = None):
    df = load_data(path)
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_split()
    print("Prepared data splits:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

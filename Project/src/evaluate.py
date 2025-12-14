#!/usr/bin/env python3
from pathlib import Path
import sys
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir / "src"))
from data_processing import prepare_split


def evaluate(model_path: Path | None = None):
    if model_path is None:
        model_path = base_dir / "models" / "insurance_model.pkl"
    model = joblib.load(model_path)
    X_train, X_test, y_train, y_test = prepare_split()
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    print(f"Model: {model_path}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


if __name__ == "__main__":
    evaluate()

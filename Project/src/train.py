#!/usr/bin/env python3
from pathlib import Path
import sys
import joblib
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir / "src"))
from data_processing import prepare_split


def train_and_save(model_path: Path | None = None, n_estimators: int = 100, random_state: int = 42):
    X_train, X_test, y_train, y_test = prepare_split()
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    if model_path is None:
        model_path = base_dir / "models" / "insurance_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train insurance charges regression model")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()
    train_and_save(model_path=Path(args.model_path) if args.model_path else None, n_estimators=args.n_estimators)

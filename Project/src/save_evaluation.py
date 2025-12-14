from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def save_evaluation(model_path: Path, test_csv: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    model = joblib.load(model_path)
    df = pd.read_csv(test_csv)
    target = 'annual_medical_cost' if 'annual_medical_cost' in df.columns else 'charges'
    X_test = df.drop(columns=[target])
    y_test = df[target]
    # Ensure same preprocessing as training (dummies) — safe fallback: align columns
    X_test = pd.get_dummies(X_test, drop_first=True)
    # Align columns with model training (if model was trained on different columns)
    # If mismatch, add missing cols with zeros
    try:
        preds = model.predict(X_test)
    except Exception:
        # Try aligning columns using model.feature_names_in_ if available
        if hasattr(model, 'feature_names_in_'):
            cols = list(model.feature_names_in_)
            for c in cols:
                if c not in X_test.columns:
                    X_test[c] = 0
            X_test = X_test[cols]
            preds = model.predict(X_test)
        else:
            raise

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    metrics = {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    # residual plot
    residuals = y_test - preds
    plt.figure(figsize=(8,5))
    plt.scatter(preds, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    plt.tight_layout()
    plot_path = out_dir / 'residuals.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved metrics to {out_dir / 'metrics.json'} and plot to {plot_path}")


if __name__ == '__main__':
    base = Path(__file__).resolve().parents[1]
    model_path = base / 'models' / 'insurance_model.pkl'
    test_csv = base / 'result' / 'processed_test.csv'
    out_dir = base / 'result'
    save_evaluation(model_path, test_csv, out_dir)

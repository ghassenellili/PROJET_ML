# Insurance Charges ML Project

This project implements a minimal ML pipeline for the `medical_insurance.csv` dataset.

Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

2. Train the model:

```bash
python src/train.py --n_estimators 100
```

3. Evaluate the saved model:

```bash
python src/evaluate.py
```

Files added

- `src/data_processing.py`: load and preprocess dataset, produce train/test splits
- `src/train.py`: train RandomForest and save to `models/insurance_model.pkl`
- `src/evaluate.py`: load model and report MAE/RMSE/R2
- `requirements.txt`: minimal dependencies
- `notebooks/EDA_instructions.md`: quick EDA steps

Results

- `result/metrics.json`: evaluation metrics (MAE, RMSE, R2)
- `result/residuals.png`: residual diagnostic plot

End-to-end

- `notebooks/00_run_pipeline.ipynb`: runs preprocessing, training, and evaluation end-to-end and saves outputs to `result/`.

Next steps

- Run `python src/train.py` to train and create `models/insurance_model.pkl`.

- If you want, I can run training here (requires permission).
⚠️ Les modèles entraînés sont gérés avec Git LFS.
Assurez-vous d’avoir Git LFS installé avant de cloner le projet.

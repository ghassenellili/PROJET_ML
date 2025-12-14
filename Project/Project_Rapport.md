**Title**: Project Report — Annual Medical Cost Prediction

**Executive Summary**
- **Goal**: Build and evaluate an end-to-end pipeline to predict annual medical costs for insured individuals using demographic, clinical and policy features.
- **Dataset**: `data/medical_insurance.csv`, containing demographic attributes, clinical indicators and prior utilization metrics.
- **Key outcome**: a trained supervised model with preprocessing pipeline and evaluation metrics saved in `result/metrics.json`.

**Context & Objectives**
- **Context**: Estimate annual healthcare expenditures to support pricing and risk management decisions.
- **Specific objectives**: data cleaning and feature engineering, train/test split, model training and comparison, and persistence of metrics and processed datasets.

**Data**
- **Source**: [data/medical_insurance.csv](data/medical_insurance.csv)
- **Features**: demographic (age, sex, region), clinical (BMI, blood pressure, comorbidities), policy details (plan_type, deductible) and the target `annual_medical_cost` (or `charges`).
- **Quality**: rows with missing values are removed during preprocessing (see the data preparation notebook).

**Data Preparation**
- **Code / Notebooks**: preprocessing pipeline in [notebooks/02_data_preparation.ipynb](notebooks/02_data_preparation.ipynb) and helpers in [src/data_processing.py](src/data_processing.py).
- **Steps**: load data, handle missing values, encode categorical variables (`sex`, `smoker`, plus one-hot for others), drop identifier columns, and run basic quality checks (dtypes, descriptive stats).
- **Train/test split**: performed with `test_size=0.2`, `random_state=42` in the preparation notebook.

**Exploratory Data Analysis (EDA)**
- **Notebook**: [notebooks/01_EDA.ipynb](notebooks/01_EDA.ipynb)
- **Key findings**: target distribution is skewed with high-cost outliers; notable correlations between `age`, `chronic_count`, `smoker` and costs; clinical variables like `bmi` and `hba1c` show predictive signal.

**Modeling**
- **Files**: training logic in [src/train.py](src/train.py) and modeling experiments in [notebooks/03_modeling.ipynb](notebooks/03_modeling.ipynb).
- **Approach**: evaluate regression baselines (regularized linear models), tree-based models and boosting methods; use cross-validation for model selection.
- **Features**: categorical encoding and numeric scaling applied as appropriate per model.

**Evaluation**
- **Scripts**: metrics computed and saved via [src/evaluate.py](src/evaluate.py) and [src/save_evaluation.py](src/save_evaluation.py).
- **Notebook**: [notebooks/04_evaluation.ipynb](notebooks/04_evaluation.ipynb)
- **Metrics**: RMSE, MAE and R² are produced (see [result/metrics.json](result/metrics.json)).

**Results**
- **Main metrics**: refer to [result/metrics.json](result/metrics.json) for numeric results.
- **Observations**: MAE indicates usable average error for segmentation; RMSE is sensitive to high-cost outliers — consider a log transformation of the target or separate modeling for heavy utilizers.

**Interpretation & Business Impact**
- **Influential features**: `age`, `chronic_count`, `smoker`, `bmi` and prior utilization metrics (visits, claim amounts) are strong predictors.
- **Use cases**: predicted costs can be used for premium estimation, identifying high-risk members, and budgeting.

**Limitations**
- **Data**: possible sampling bias, missingness not at random, and encoding inconsistencies across categorical features.
- **Models**: sensitivity to outliers and risk of overfitting when feature dimensionality increases without proper regularization.

**Reproducibility — How to run locally**
- **Install dependencies**: `python -m pip install -r requirements.txt`.
- **Run preparation**: open and run [notebooks/02_data_preparation.ipynb] or execute `python src/data_processing.py` if an entry point exists.
- **Train models**: `python src/train.py` from the `Project/` folder will train models and save metrics to `result/`.
- **Evaluate**: run `python src/evaluate.py` or execute [notebooks/04_evaluation.ipynb].

**Key files**
- **Raw data**: [data/medical_insurance.csv](data/medical_insurance.csv)
- **Notebooks**: [notebooks/01_EDA.ipynb](notebooks/01_EDA.ipynb), [notebooks/02_data_preparation.ipynb](notebooks/02_data_preparation.ipynb), [notebooks/03_modeling.ipynb](notebooks/03_modeling.ipynb), [notebooks/04_evaluation.ipynb](notebooks/04_evaluation.ipynb)
- **Code**: [src/data_processing.py](src/data_processing.py), [src/train.py](src/train.py), [src/evaluate.py](src/evaluate.py), [src/save_evaluation.py](src/save_evaluation.py)
- **Outputs**: [result/metrics.json](result/metrics.json), [result/processed_train.csv](result/processed_train.csv), [result/processed_test.csv](result/processed_test.csv)

**Recommended next steps**
- **Robustness**: try a log transformation of the target and robust regression approaches to mitigate outlier impact.
- **Interpretability**: apply SHAP or LIME to explain individual predictions and validate feature effects.
- **Productionization**: serialize the preprocessing pipeline and model, add unit tests and CI, and prepare a lightweight inference API.

---
Report generated automatically. Would you like me to add a visual summary (key plots) or create a git commit for these changes?



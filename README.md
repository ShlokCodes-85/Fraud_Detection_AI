# Fraud Detection AI

A small fraud/anomaly detection project that trains and evaluates multiple classifiers on a transactions dataset, applies SMOTE to handle class imbalance, saves the best model and scaler, and provides demo scripts for production and an interactive Streamlit UI.

## Highlights

- Trains and compares: `LogisticRegression`, `RandomForest`, `SVM`, and `XGBoost`
- Uses SMOTE for balancing training data and `StandardScaler` for feature scaling
- Saves best model artifact(s) and scaler as `.joblib`
- Includes a Jupyter notebook for exploration and training: `fraud_detection.ipynb`
- Demo scripts: `demo_production.py` and `demo_streamlit.py`
- Sample outputs: `demo_top_risky.csv`

## Repository Structure

- `fraud_detection.ipynb` — Main notebook: EDA, preprocessing, training, evaluation, and model export
- `fraud_anomaly.csv` — Primary dataset (transactions + `is_anomaly` label)
- `demo_production.py` — Minimal production example to load model + scaler and score inputs
- `demo_streamlit.py` — Streamlit app for interactive demo
- `best_model_LogisticRegression.joblib` — Example saved Logistic Regression model (created by notebook)
- `best_model_XGBoost.joblib` — Example saved XGBoost model (created by notebook)
- `scaler.joblib` — Saved `StandardScaler` instance
- `demo_top_risky.csv` — Sample output listing top risky transactions
- `README.md` — (this file)
- `LICENSE` — Project license
- `README.md` — Project README (this file)

## Requirements

Recommended Python environment: 3.8+

Suggested libraries (install via pip):

- pandas
- numpy
- scikit-learn
- imbalanced-learn
- xgboost
- matplotlib
- seaborn
- joblib
- streamlit (if using the Streamlit demo)
- jupyter (to run the notebook)

Example install:

```powershell
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn joblib jupyter
# For Streamlit demo:
python -m pip install streamlit
```

## Quick Start

1. Clone / copy project files to your local machine and open the project folder.

2. Prepare the environment and install dependencies (see Requirements).

3. Inspect the dataset:

```python
# in a python session or notebook
import pandas as pd
df = pd.read_csv("fraud_anomaly.csv")
df.head()
```

4. Run the Jupyter notebook

- Open `fraud_detection.ipynb` in Jupyter or VS Code and run cells from top to bottom.
- The notebook:

	- Loads `fraud_anomaly.csv`
	- Performs EDA and feature engineering (adds cyclical hour features)
	- Splits data, scales with `StandardScaler`, balances using `SMOTE`
	- Trains models and computes `precision`, `recall`, `f1`, `roc_auc`
	- Plots confusion matrices and ROC curves
	- Determines best model by F1-score and also reports ROC-AUC. The chosen model is saved as `best_model_<ModelName>.joblib` and the scaler as `scaler.joblib`.

5. Run the production demo

```powershell
python demo_production.py
```
- `demo_production.py` expects to find saved artifacts (`best_model_<ModelName>.joblib` and `scaler.joblib`) or will load sample data from `fraud_anomaly.csv` for demonstration.
- The script demonstrates how to load the model and scaler and score new samples.

6. Run the Streamlit demo (interactive)

```powershell
streamlit run demo_streamlit.py
```
- Opens a local web UI to input transaction features and get a fraud risk prediction.
- Make sure `best_model_*.joblib` and `scaler.joblib` are in the working directory.

## Data & Features

The notebook uses these primary features:

- `transaction_amt` — Transaction amount
- `merchant_risk` — Merchant risk score
- `velocity` — Transaction velocity metric
- `transaction_hour` — Hour of the transaction (converted to cyclical: `hour_sin`, `hour_cos`)
- `is_anomaly` — Target binary label (0 = normal, 1 = anomaly/fraud)

Notes:

- The notebook converts `transaction_hour` into `hour_sin` and `hour_cos` for cyclic encoding.
- Features are scaled with `StandardScaler` prior to modeling.
- Training uses SMOTE to handle class imbalance on the training set.

## Model Training / Reproducibility

- The models are defined and trained inside `fraud_detection.ipynb`.
- Default models:
	- `LogisticRegression` (class_weight='balanced')
	- `RandomForestClassifier` (class_weight='balanced')
	- `SVC` with `probability=True` (class_weight='balanced')
	- `xgboost.XGBClassifier` (tuned with `n_estimators`, `learning_rate`, `max_depth`)
- After training, evaluation metrics stored in a `metrics_df` with columns: `precision`, `recall`, `f1`, `roc_auc`, `tn`, `fp`, `fn`, `tp`.
- The notebook selects the best model by F1-score and also reports ROC-AUC. The chosen model is saved as `best_model_<ModelName>.joblib` and the scaler as `scaler.joblib`.

To retrain from scratch:

- Open `fraud_detection.ipynb`, optionally change hyperparameters, re-run all cells, and re-save artifacts.
- Verify the saved `.joblib` files are created in the project root.

## Evaluation Notes & Business Guidance

- Because the dataset is imbalanced, rely more on metrics such as `recall` (catching fraud) or `precision` (minimizing false positives) depending on the business cost matrix.
- The notebook prints both the model with highest F1 and highest ROC-AUC; choose the final model based on operating point and expected costs.
- ROC curves and confusion matrices are plotted for visual inspection.
- Consider calibrating classifier thresholds if you need a target precision/recall tradeoff.

## Output Artifacts

- `best_model_<ModelName>.joblib` — Model object saved with `joblib.dump()`
- `scaler.joblib` — `StandardScaler` saved for consistent preprocessing
- `demo_top_risky.csv` — Example output listing top risky transactions by model score

## Tips for Productionization

- Use the saved `scaler.joblib` to ensure consistent preprocessing in production pipelines.
- Wrap model loading and scoring in a small API (FastAPI/Flask) for serving predictions.
- Add input validation and logging for predictions (record features, timestamp, prediction probability).
- Consider model monitoring (drift detection) and periodic retraining as transactional distribution changes.

## Troubleshooting

- If XGBoost raises warnings about label encoding, ensure `use_label_encoder=False` and `eval_metric='logloss'` are set (noted in the notebook).
- If `streamlit` fails to start, ensure the package is installed and run `streamlit run demo_streamlit.py` from the project directory.
- For dependency issues, create a fresh virtual environment:

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
(If `requirements.txt` is not present, install the packages listed in the Requirements section.)

## Author
Shlok Jain
# demo_production.py
# Loads data, trains (if needed), selects best model, saves artifacts,
# and writes demo outputs (metrics, ROC plot, confusion matrix, top risky CSV).

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, classification_report

# Config
DATA_PATH = 'fraud_anomaly.csv'
MODEL_PATH = None  # will be set to best_model_<name>.joblib
SCALER_PATH = 'scaler.joblib'
ROC_PNG = 'demo_roc.png'
CM_PNG = 'demo_confusion_matrix.png'
TOP_RISKY_CSV = 'demo_top_risky.csv'

print('Starting demo production run')

# Load data
df = pd.read_csv(DATA_PATH)
print('Loaded data shape:', df.shape)

# Feature engineering
if 'hour_sin' not in df.columns:
    df['hour_sin'] = np.sin(2 * np.pi * df['transaction_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['transaction_hour'] / 24)

features = ['transaction_amt', 'merchant_risk', 'velocity', 'hour_sin', 'hour_cos']
X = df[features]
y = df['is_anomaly']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print('Train/Test sizes:', X_train.shape, X_test.shape)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Balance with SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
print('Balanced train shape:', X_train_bal.shape)

# Define models (smaller for demo speed)
models = {
    'LogisticRegression': LogisticRegression(max_iter=200, class_weight='balanced', random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1),
}

# Train models
trained = {}
for name, model in models.items():
    print('Training', name)
    model.fit(X_train_bal, y_train_bal)
    trained[name] = model

# Evaluate
results = []
for name, model in trained.items():
    y_pred = model.predict(X_test_scaled)
    # some models (SVC without prob) might not have predict_proba
    try:
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    except Exception:
        # fallback: use decision_function then scale
        try:
            scores = model.decision_function(X_test_scaled)
            y_proba = (scores - scores.min()) / (scores.max() - scores.min())
        except Exception:
            y_proba = np.zeros_like(y_pred, dtype=float)

    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba.sum() > 0 else 0.0
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    results.append({
        'model': name, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': auc,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    })

metrics_df = pd.DataFrame(results).set_index('model')
print('\nMetrics:')
print(metrics_df[['precision','recall','f1','roc_auc']])

# Pick best by F1
best_by_f1 = metrics_df['f1'].idxmax()
MODEL_PATH = f'best_model_{best_by_f1}.joblib'
print('\nBest by F1:', best_by_f1)

# Save best model and scaler
joblib.dump(trained[best_by_f1], MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print('Saved model:', MODEL_PATH)
print('Saved scaler:', SCALER_PATH)

# Generate ROC plot
best_model = trained[best_by_f1]
try:
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
except Exception:
    try:
        scores = best_model.decision_function(X_test_scaled)
        y_proba = (scores - scores.min()) / (scores.max() - scores.min())
    except Exception:
        y_proba = np.zeros_like(y_test, dtype=float)

fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba) if y_proba.sum() > 0 else 0.0
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig(ROC_PNG)
print('Saved ROC plot to', ROC_PNG)
plt.close()

# Confusion matrix
y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(CM_PNG)
print('Saved confusion matrix to', CM_PNG)
plt.close()

# Top risky transactions
test_df = X_test.copy().reset_index(drop=True)
# recompute probabilities for rows order
try:
    proba_all = best_model.predict_proba(X_test_scaled)[:,1]
except Exception:
    proba_all = np.zeros(len(test_df))

test_df['score'] = proba_all
test_df['true_label'] = y_test.reset_index(drop=True)
top_risky = test_df.sort_values('score', ascending=False).head(20)

top_risky.to_csv(TOP_RISKY_CSV, index=False)
print('Saved top risky transactions to', TOP_RISKY_CSV)

print('\nDemo run complete.')

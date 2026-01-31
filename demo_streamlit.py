# demo_streamlit.py
# Minimal Streamlit dashboard to score a transaction and view demo outputs
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time

st.set_page_config(page_title='Fraud Detection Demo', layout='centered')

st.title('Fraud Detection — Demo Dashboard')

# Load artifacts
try:
    model_files = [f for f in ['best_model_LogisticRegression.joblib','best_model_RandomForest.joblib','best_model_XGBoost.joblib'] if __import__('os').path.exists(f)]
    if model_files:
        model = joblib.load(model_files[0])
    else:
        model = None
except Exception:
    model = None

try:
    scaler = joblib.load('scaler.joblib')
except Exception:
    scaler = None

st.sidebar.header('Input Transaction')
transaction_amt = st.sidebar.number_input('Transaction amount', value=100.0)
merchant_risk = st.sidebar.number_input('Merchant risk (0-1)', value=0.5)
velocity = st.sidebar.number_input('Velocity', value=1.0)
transaction_hour = st.sidebar.slider('Transaction hour', 0, 23, 12)

# Live controls
auto_refresh = st.sidebar.checkbox('Auto-refresh', value=False)
refresh_interval = st.sidebar.slider('Refresh interval (s)', 1, 10, 2)
do_score = st.sidebar.button('Score Transaction')

def compute_probability(amount, merisk, vel, hour):
    if model is None or scaler is None:
        return float(min(1.0, max(0.0, merisk * 0.6 + (amount / 1000.0) * 0.2 + (vel / 10.0) * 0.2)))
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    x = np.array([[amount, merisk, vel, hour_sin, hour_cos]])
    try:
        x_s = scaler.transform(x)
    except Exception:
        x_s = x
    try:
        prob = model.predict_proba(x_s)[:, 1][0]
    except Exception:
        try:
            score = model.decision_function(x_s)[0]
            prob = 1.0 / (1.0 + np.exp(-score))
        except Exception:
            prob = 0.0
    return float(np.clip(prob, 0.0, 1.0))

# compute once per-run
prob = compute_probability(transaction_amt, merchant_risk, velocity, transaction_hour) if do_score or not auto_refresh else compute_probability(transaction_amt, merchant_risk, velocity, transaction_hour)

# If auto-refresh requested, sleep then rerun to simulate live updates
if auto_refresh:
    time.sleep(refresh_interval)
    st.experimental_rerun()

st.markdown('---')
st.header('Live Fraud Probability')

# Center the circular progress bar
col1, col2, col3 = st.columns([1, 2, 1])

def circular_html(pct: float, size: int = 220):
        # pct in [0,1]
        percent = int(pct * 100)
        degree = int(pct * 360)
        # choose color based on pct (green->yellow->red)
        if pct < 0.5:
                # interpolate green to yellow
                color = '#2ecc71'
        elif pct < 0.8:
                color = '#f1c40f'
        else:
                color = '#e74c3c'
        html = f'''
        <div style="min-height:calc(100vh - 220px);display:flex;align-items:center;justify-content:center;padding:10px;">
            <div style="width:{size}px;height:{size}px;border-radius:50%;display:flex;align-items:center;justify-content:center;position:relative;font-family:Arial,Helvetica,sans-serif;">
                <div style="position:absolute;inset:0;border-radius:50%;background:conic-gradient({color} {degree}deg, #e6e6e6 {degree}deg);display:flex;align-items:center;justify-content:center;">
                </div>
                <div style="position:absolute;inset:12% 12% 12% 12%;border-radius:50%;background:#ffffff;display:flex;align-items:center;justify-content:center;flex-direction:column;">
                    <div style="font-size:32px;font-weight:800;color:#111;">{percent}%</div>
                    <div style="font-size:12px;color:#666;margin-top:6px;">Fraud probability</div>
                </div>
            </div>
        </div>
        '''
        return html

with col2:
    st.components.v1.html(circular_html(prob, size=260), height=320)
    st.markdown(f"**Probability:** {prob:.4f}  \n**Inputs:** amount={transaction_amt}, merchant_risk={merchant_risk}, velocity={velocity}, hour={transaction_hour}")

st.markdown('---')
st.header('Live artifacts (auto-updating)')

import os

# Prefer live anomaly CSV if present
data_file = 'fraud_anomaly.csv' if os.path.exists('fraud_anomaly.csv') else ('demo_top_risky.csv' if os.path.exists('demo_top_risky.csv') else None)
if data_file:
    st.subheader(f'Top risky transactions (from {data_file})')
    try:
        df_top = pd.read_csv(data_file)
        # if model exists, add live scored probability column
        if model is not None and scaler is not None:
            def score_row(r):
                try:
                    amt = float(r.get('amount', r.get('TransactionAmt', transaction_amt)))
                except Exception:
                    amt = transaction_amt
                mer = float(r.get('merchant_risk', merchant_risk)) if 'merchant_risk' in r else merchant_risk
                vel = float(r.get('velocity', velocity)) if 'velocity' in r else velocity
                hr = int(r.get('hour', transaction_hour)) if 'hour' in r else transaction_hour
                return compute_probability(amt, mer, vel, hr)
            # apply safely
            try:
                df_top['live_prob'] = df_top.apply(lambda row: score_row(row.to_dict()), axis=1)
            except Exception:
                pass
        st.dataframe(df_top.head(10))
    except Exception as e:
        st.error('Could not read data file: ' + str(e))
else:
    st.info('No live CSV artifact found (expected fraud_anomaly.csv or demo_top_risky.csv).')

# Show images if present (re-read each run so they change with updates)
if os.path.exists('demo_roc.png'):
    st.subheader('ROC Curve (live)')
    st.image('demo_roc.png')
if os.path.exists('demo_confusion_matrix.png'):
    st.subheader('Confusion Matrix (live)')
    st.image('demo_confusion_matrix.png')

st.write('Artifacts are re-read on each refresh so they reflect live data changes.')

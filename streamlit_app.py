"""
Streamlit app that loads trained models and predicts next-day close for BTC and ETH.
Run:
  streamlit run streamlit_app.py
"""
import streamlit as st
import joblib
import yfinance as yf
import pandas as pd
from features import prepare_features_for_prediction

st.set_page_config(page_title="Crypto Price Predictor", layout="centered")

st.title("₿ Crypto Price Predictor (Random Forest)")
st.write("Predicting next-day close price for BTC and ETH using lag + SMA features.")
st.write("---")

@st.cache_data(ttl=300)
def get_history(ticker: str, period: str = "120d"):
    # fetch enough history so rolling windows and lags are valid
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    return df

# Load models
model_load_error = False
try:
    rf_btc = joblib.load("rf_btc_predictor.pkl")
except Exception as e:
    st.error("Could not load rf_btc_predictor.pkl. Make sure you trained and saved the model (train_models.py).")
    st.exception(e)
    model_load_error = True

try:
    rf_eth = joblib.load("rf_eth_predictor.pkl")
except Exception as e:
    st.error("Could not load rf_eth_predictor.pkl. Make sure you trained and saved the model (train_models.py).")
    st.exception(e)
    model_load_error = True

if model_load_error:
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.header("Bitcoin (BTC)")
    btc_hist = get_history("BTC-USD", period="120d")
    st.line_chart(btc_hist['Close'].dropna().rename("BTC Close"))
    try:
        btc_feat = prepare_features_for_prediction(btc_hist)
        btc_pred = float(rf_btc.predict(btc_feat)[0])
        current = float(btc_hist['Close'].iloc[-1])
        delta = btc_pred - current
        pct = (delta / current) * 100.0
        st.metric(label="Predicted Close (Tomorrow)", value=f"${btc_pred:,.2f}", delta=f"${delta:,.2f} ({pct:.2f}%)")
        st.caption(f"Current BTC price: ${current:,.2f}")
    except Exception as e:
        st.error("Failed to prepare features for BTC prediction. Need more history or models are incompatible.")
        st.exception(e)

with col2:
    st.header("Ethereum (ETH)")
    eth_hist = get_history("ETH-USD", period="120d")
    st.line_chart(eth_hist['Close'].dropna().rename("ETH Close"))
    try:
        eth_feat = prepare_features_for_prediction(eth_hist)
        eth_pred = float(rf_eth.predict(eth_feat)[0])
        current_e = float(eth_hist['Close'].iloc[-1])
        delta_e = eth_pred - current_e
        pct_e = (delta_e / current_e) * 100.0
        st.metric(label="Predicted Close (Tomorrow)", value=f"${eth_pred:,.2f}", delta=f"${delta_e:,.2f} ({pct_e:.2f}%)")
        st.caption(f"Current ETH price: ${current_e:,.2f}")
    except Exception as e:
        st.error("Failed to prepare features for ETH prediction. Need more history or models are incompatible.")
        st.exception(e)

st.write("---")
st.markdown("Notes:")
st.markdown("- The model predicts the next day's 'Close' value using lagged closes and simple moving averages (7, 30).")
st.markdown("- If you see NaN / insufficient history errors, increase the `period` in get_history to fetch more days.")
st.markdown("- The model currently is a baseline Random Forest. Consider feature engineering (returns, volumes), hyperparameter tuning, and cross-validation for better results.")
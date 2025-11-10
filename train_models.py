"""
Training script for BTC and ETH RandomForest regressors.

Saves:
  - rf_btc_predictor.pkl
  - rf_eth_predictor.pkl

Usage:
  python train_models.py
"""
import joblib
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from features import create_and_split_data

def download_data(ticker: str, start: str = "2020-01-01"):
    df = yf.download(ticker, start=start, progress=False)
    if 'Close' not in df.columns:
        raise RuntimeError(f"No Close column after downloading {ticker}")
    return df

def train_and_save(ticker: str, out_filename: str):
    print(f"Downloading {ticker} ...")
    df = download_data(ticker)
    X_train, X_test, y_train, y_test, _ = create_and_split_data(df)

    print(f"Training RandomForest for {ticker} ...")
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, max_depth=12)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"{ticker}  R2: {r2_score(y_test, preds):.4f}  MSE: {mean_squared_error(y_test, preds):.4f}")

    joblib.dump(model, out_filename)
    print(f"Saved model to {out_filename}")

if __name__ == "__main__":
    train_and_save("BTC-USD", "rf_btc_predictor.pkl")
    train_and_save("ETH-USD", "rf_eth_predictor.pkl")
"""
Feature engineering helpers used by training and by the Streamlit app.

Produces the exact same columns for training and for prediction:
  - Lag_1d, Lag_7d, Lag_30d
  - SMA_7, SMA_30

Functions:
  - create_features(df, lag_days=[1,7,30]) -> DataFrame of features + Target
  - create_and_split_data(df, lag_days) -> X_train, X_test, y_train, y_test, X_full
  - prepare_features_for_prediction(df, lag_days) -> single-row DataFrame ready for model.predict
"""
import pandas as pd
from typing import Tuple, List


def create_features(df: pd.DataFrame, lag_days: List[int] = [1, 7, 30]) -> pd.DataFrame:
    df_copy = df[['Close']].copy()
    # Target is next-day close
    df_copy['Target'] = df_copy['Close'].shift(-1)

    for day in lag_days:
        df_copy[f'Lag_{day}d'] = df_copy['Close'].shift(day)

    df_copy['SMA_7'] = df_copy['Close'].rolling(window=7).mean()
    df_copy['SMA_30'] = df_copy['Close'].rolling(window=30).mean()

    return df_copy


def create_and_split_data(df: pd.DataFrame, lag_days: List[int] = [1, 7, 30], train_frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    df_feat = create_features(df, lag_days=lag_days)
    # Drop rows with NaNs (initial rows because of lag/rolling and last row because Target is NaN)
    df_feat = df_feat.dropna().copy()

    feature_cols = [f'Lag_{d}d' for d in lag_days] + ['SMA_7', 'SMA_30']
    X = df_feat[feature_cols]
    y = df_feat['Target']

    split_point = int(len(X) * train_frac)
    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]

    return X_train, X_test, y_train, y_test, X


def prepare_features_for_prediction(df: pd.DataFrame, lag_days: List[int] = [1, 7, 30]) -> pd.DataFrame:
    """
    Given recent historical price DataFrame (with a 'Close' column),
    compute the same features as in training and return a single-row DataFrame
    with the correct column order.

    The input df should contain enough history (>= 30+ rows). If the last row
    contains NaNs in the feature columns, the caller should fetch more history.
    """
    df_feat = create_features(df, lag_days=lag_days)
    # We expect the features for the "current" day are in the last row (Target is tomorrow)
    last_row = df_feat.iloc[-1:]

    feature_cols = [f'Lag_{d}d' for d in lag_days] + ['SMA_7', 'SMA_30']
    features = last_row[feature_cols]

    # If any NaNs present, raise informative error (caller can fetch more history)
    if features.isnull().any(axis=None):
        raise ValueError("Insufficient history to compute features for prediction (NaNs present). "
                         "Fetch more historical rows (e.g., 60-120 days).")

    # Return a single-row DataFrame with same ordering as training X
    return features.reset_index(drop=True)
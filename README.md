```markdown
# crypto-predictor

This repository contains code to train RandomForest regressors that predict the next-day Close price for BTC and ETH, plus a Streamlit UI to show predictions.

Files added:
- features.py (feature engineering helpers)
- train_models.py (training script)
- streamlit_app.py (Streamlit app to show predictions)
- requirements.txt

How to use
1. Create a Python environment and install dependencies:
   pip install -r requirements.txt

2. Train models (this will download data and save two model files):
   python train_models.py

3. Run Streamlit UI:
   streamlit run streamlit_app.py

Notes
- If you already have `rf_btc_predictor.pkl` and `rf_eth_predictor.pkl` in the repo root, the Streamlit app will load them directly and you can skip step 2.
- If you encounter NaN errors when predicting, fetch longer history in the streamlit app (the helper fetches 120 days by default).
- The current model is a baseline; for better performance consider:
  - more features (volume, returns, indicators)
  - target transforms (log, returns)
  - hyperparameter tuning and cross-validation
```
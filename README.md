# Real-Time Stock Price Pipeline & ML Predictor

> Live demo: https://cgm-glucose-predictor-ml.streamlit.app

End-to-end data pipeline that fetches live stock prices, stores in SQLite, engineers features, and predicts next-day price direction using Random Forest.

## What it does
- Fetches daily OHLCV data from Alpha Vantage API
- Stores in SQLite database
- Engineers 17 features: lags, rolling stats, momentum, volume ratios
- Predicts next-day price direction (UP/DOWN) with confidence score
- Live dashboard with price chart, volume, and feature importances

## Resume Bullet Points
- Built end-to-end real-time data pipeline ingesting live stock prices from Alpha Vantage API, storing in SQLite, and serving ML predictions via a deployed Streamlit dashboard
- Engineered 17 financial features including lag variables, rolling statistics, momentum indicators, and volume ratios from daily OHLCV data
- Deployed cloud-hosted ML pipeline predicting next-day stock price direction using Random Forest classification with automated data refresh

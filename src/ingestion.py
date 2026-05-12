import os
import requests
import pandas as pd
from datetime import datetime

try:
    import streamlit as st
    API_KEY = st.secrets.get("ALPHA_VANTAGE_KEY", os.environ.get("ALPHA_VANTAGE_KEY"))
except:
    API_KEY = os.environ.get("ALPHA_VANTAGE_KEY")

BASE_URL = "https://www.alphavantage.co/query"


def fetch_daily(symbol="AAPL"):
    print(f"Fetching {symbol} at {datetime.now().strftime('%H:%M:%S')}...")
    params = {
        "function":   "TIME_SERIES_DAILY",
        "symbol":     symbol,
        "apikey":     API_KEY,
        "outputsize": "compact"
    }
    response = requests.get(BASE_URL, params=params)
    data     = response.json()
    key = "Time Series (Daily)"
    if key not in data:
        print(f"Error: {data}")
        return None
    records = []
    for ts, values in data[key].items():
        records.append({
            "timestamp": pd.to_datetime(ts),
            "open":      float(values["1. open"]),
            "high":      float(values["2. high"]),
            "low":       float(values["3. low"]),
            "close":     float(values["4. close"]),
            "volume":    int(values["5. volume"]),
        })
    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    print(f"Fetched {len(df)} rows for {symbol}")
    return df


def fetch_quote(symbol="AAPL"):
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol":   symbol,
        "apikey":   API_KEY,
    }
    response = requests.get(BASE_URL, params=params)
    data     = response.json()
    quote    = data.get("Global Quote", {})
    if not quote:
        print(f"Error fetching quote: {data}")
        return None
    return {
        "symbol":     symbol,
        "price":      float(quote["05. price"]),
        "change":     float(quote["09. change"]),
        "change_pct": quote["10. change percent"],
        "volume":     int(quote["06. volume"]),
    }

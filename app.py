import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, 'src')
from database import get_prices, create_tables, insert_prices
from features import build_features, get_feature_columns
from model import train, predict
from ingestion import fetch_daily, fetch_quote

st.set_page_config(
    page_title="Stock Price Pipeline",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Real-Time Stock Pipeline & Predictor")
st.markdown("Fetches daily stock data, stores in SQLite, engineers features, and predicts next-day price direction.")
st.divider()

# Sidebar
st.sidebar.header("Settings")
symbol = st.sidebar.selectbox(
    "Select Stock",
    ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
    index=0
)

if st.sidebar.button("🔄 Fetch Latest Data"):
    with st.spinner(f"Fetching {symbol} data..."):
        create_tables()
        df_new = fetch_daily(symbol)
        if df_new is not None:
            from database import insert_prices
            insert_prices(df_new, symbol)
            st.sidebar.success(f"Data updated for {symbol}")
        else:
            st.sidebar.error("Fetch failed — check API key or rate limit")

# Load data
# Auto-fetch if database is empty
create_tables()
df_raw = get_prices(symbol, limit=200)
if df_raw.empty:
    df_new = fetch_daily(symbol)
    if df_new is not None:
        insert_prices(df_new, symbol)
        df_raw = get_prices(symbol, limit=200)

df_feat      = build_features(df_raw)
feature_cols = get_feature_columns(df_feat)

# Train model
model_obj, scaler_obj, metrics = train(df_feat, feature_cols)
result  = predict(df_feat, feature_cols)
current = df_feat["close"].iloc[-1]
prev    = df_feat["close"].iloc[-2]
change  = current - prev
change_pct = (change / prev) * 100

# Metrics row
st.subheader(f"{symbol} — Current Snapshot")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Price",    f"${current:.2f}", f"{change:+.2f}")
col2.metric("Change %",         f"{change_pct:+.2f}%")
col3.metric("Prediction",       result["direction"])
col4.metric("Confidence",       f"{result['confidence']}%")
col5.metric("Model Accuracy",   f"{metrics['accuracy']*100:.1f}%")

st.divider()

# Alert
if "UP" in result["direction"]:
    st.success(f"Model predicts {symbol} will go UP tomorrow with {result['confidence']}% confidence")
else:
    st.error(f"Model predicts {symbol} will go DOWN tomorrow with {result['confidence']}% confidence")

st.divider()

# Price chart
st.subheader("Price History")
fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                          gridspec_kw={"height_ratios": [3, 1]})

axes[0].plot(df_feat["timestamp"], df_feat["close"],
             color="#60a5fa", lw=1.5, label="Close Price")
axes[0].fill_between(df_feat["timestamp"], df_feat["close"],
                     alpha=0.1, color="#60a5fa")
axes[0].plot(df_feat["timestamp"], df_feat["roll_mean_20"],
             color="#f472b6", lw=1, ls="--", label="20-day MA")
axes[0].set_ylabel("Price ($)")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_title(f"{symbol} Daily Close Price")

axes[1].bar(df_feat["timestamp"], df_feat["volume"],
            color="#34d399", alpha=0.7)
axes[1].set_ylabel("Volume")
axes[1].grid(alpha=0.3)
axes[1].set_title("Daily Volume")

plt.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()

# Probability chart
st.subheader("Tomorrow's Direction Probability")
fig2, ax = plt.subplots(figsize=(6, 3))
bars = ax.barh(["DOWN 📉", "UP 📈"],
               [result["down_prob"], result["up_prob"]],
               color=["#ef4444", "#22c55e"], alpha=0.85)
ax.set_xlim(0, 100)
ax.set_xlabel("Probability (%)")
for bar, val in zip(bars, [result["down_prob"], result["up_prob"]]):
    ax.text(val + 1, bar.get_y() + bar.get_height()/2,
            f"{val}%", va="center", fontsize=12)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.divider()

# Feature importance
st.subheader("Feature Importances")
imp_df = pd.DataFrame({
    "feature":    feature_cols,
    "importance": model_obj.feature_importances_
}).sort_values("importance", ascending=False).head(10)

fig3, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(imp_df)))[::-1]
ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1],
        color=colors, edgecolor="none", alpha=0.9)
ax.set_xlabel("Importance Score")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
st.pyplot(fig3)
plt.close()

st.divider()

# Raw data table
st.subheader("Raw Data (Last 10 rows)")
st.dataframe(df_raw.tail(10).reset_index(drop=True))

st.markdown("Built with Alpha Vantage · SQLite · Random Forest · Streamlit")


import pandas as pd
import numpy as np


def build_features(df):
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # Price-based features
    df["returns"]       = df["close"].pct_change()
    df["hl_spread"]     = df["high"] - df["low"]
    df["oc_spread"]     = df["close"] - df["open"]

    # Lag features
    for lag in [1, 2, 3, 5]:
        df[f"lag_{lag}"] = df["close"].shift(lag)

    # Rolling statistics
    for w in [5, 10, 20]:
        df[f"roll_mean_{w}"] = df["close"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"]  = df["close"].shift(1).rolling(w).std()

    # Momentum
    df["momentum_5"]  = df["close"] - df["close"].shift(5)
    df["momentum_10"] = df["close"] - df["close"].shift(10)

    # Volume features
    df["volume_roll_mean"] = df["volume"].shift(1).rolling(5).mean()
    df["volume_ratio"]     = df["volume"] / df["volume_roll_mean"]

    # Target: next day closing price
    df["target"] = df["close"].shift(-1)

    df = df.dropna().reset_index(drop=True)
    return df


def get_feature_columns(df):
    exclude = {"timestamp", "symbol", "open", "high", "low",
               "close", "volume", "id", "target"}
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from database import get_prices

    df = get_prices("AAPL", limit=200)
    df_feat = build_features(df)
    feature_cols = get_feature_columns(df_feat)

    print(f"Features built: {len(feature_cols)}")
    print(f"Feature names: {feature_cols}")
    print(f"Shape: {df_feat.shape}")
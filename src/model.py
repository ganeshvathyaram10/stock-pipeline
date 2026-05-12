import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


MODEL_PATH  = "models/stock_model.pkl"
SCALER_PATH = "models/stock_scaler.pkl"


def train(df_feat, feature_cols):
    df = df_feat.copy()

    # Target: 1 if next day close is higher, 0 if lower
    df["target"] = (df["target"] > df["close"]).astype(int)

    X = df[feature_cols].values
    y = df["target"].values

    # Chronological split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2%}")
    print(classification_report(y_test, y_pred,
          target_names=["DOWN", "UP"]))

    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return model, scaler, {"accuracy": round(acc, 4)}


def predict(df_feat, feature_cols):
    if not os.path.exists(MODEL_PATH):
        print("No model found — training first")
        train(df_feat, feature_cols)

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    latest = df_feat.iloc[-1][feature_cols].values.reshape(1, -1)
    scaled = scaler.transform(latest)

    direction = model.predict(scaled)[0]
    proba     = model.predict_proba(scaled)[0]
    confidence = max(proba) * 100

    return {
        "direction":  "UP 📈" if direction == 1 else "DOWN 📉",
        "confidence": round(confidence, 1),
        "up_prob":    round(proba[1] * 100, 1),
        "down_prob":  round(proba[0] * 100, 1),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from database import get_prices
    from features import build_features, get_feature_columns

    df           = get_prices("AAPL", limit=200)
    df_feat      = build_features(df)
    feature_cols = get_feature_columns(df_feat)

    model, scaler, metrics = train(df_feat, feature_cols)
    result  = predict(df_feat, feature_cols)
    current = df_feat["close"].iloc[-1]

    print(f"\nCurrent price: ${current:.2f}")
    print(f"Prediction:    {result['direction']}")
    print(f"Confidence:    {result['confidence']}%")
    print(f"UP prob:       {result['up_prob']}%")
    print(f"DOWN prob:     {result['down_prob']}%")
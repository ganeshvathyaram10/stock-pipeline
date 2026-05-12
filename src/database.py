import sqlite3
import pandas as pd
import os

# Works both locally and on Streamlit Cloud
if os.path.exists("/mount/src"):
    # Running on Streamlit Cloud
    DB_PATH = "/tmp/stocks.db"
else:
    # Running locally
    DB_PATH = "data/stocks.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol    TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open      REAL,
            high      REAL,
            low       REAL,
            close     REAL,
            volume    INTEGER,
            UNIQUE(symbol, timestamp)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT NOT NULL,
            predicted_at    TEXT NOT NULL,
            predicted_for   TEXT NOT NULL,
            predicted_price REAL,
            actual_price    REAL
        )
    """)

    conn.commit()
    conn.close()


def insert_prices(df, symbol):
    conn = get_connection()
    inserted = 0
    for _, row in df.iterrows():
        try:
            conn.execute("""
                INSERT OR IGNORE INTO price_history
                (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, str(row["timestamp"]), row["open"],
                  row["high"], row["low"], row["close"], row["volume"]))
            inserted += 1
        except Exception as e:
            print(f"Insert error: {e}")
    conn.commit()
    conn.close()
    return inserted


def get_prices(symbol, limit=200):
    conn = get_connection()
    df = pd.read_sql("""
        SELECT * FROM price_history
        WHERE symbol = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, conn, params=(symbol, limit))
    conn.close()
    return df.sort_values("timestamp").reset_index(drop=True)


def insert_prediction(symbol, predicted_at, predicted_for, predicted_price):
    conn = get_connection()
    conn.execute("""
        INSERT INTO predictions
        (symbol, predicted_at, predicted_for, predicted_price)
        VALUES (?, ?, ?, ?)
    """, (symbol, predicted_at, predicted_for, predicted_price))
    conn.commit()
    conn.close()


def get_predictions(symbol, limit=50):
    conn = get_connection()
    df = pd.read_sql("""
        SELECT * FROM predictions
        WHERE symbol = ?
        ORDER BY predicted_at DESC
        LIMIT ?
    """, conn, params=(symbol, limit))
    conn.close()
    return df


if __name__ == "__main__":
    create_tables()
    print("Database initialized at", DB_PATH)
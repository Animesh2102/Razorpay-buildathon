"""
Aegis - Feature engineering.

Hard rule: every feature for transaction T must be computable using only
information available strictly BEFORE T happened (T's own user history up
to but not including T). This is what makes the later evaluation honest --
a model that peeks at "future" transactions of the same user would report
inflated precision/recall that would collapse in production.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"


def add_time_features(df):
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_odd_hour"] = df["hour_of_day"].isin([1, 2, 3, 4, 5]).astype(int)
    df["log_amount"] = np.log1p(df["amount"])
    return df


def add_velocity_features(df):
    """Rolling txn counts in the trailing 1h / 24h window, per user."""
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True).copy()
    df["_ones"] = 1
    ts_indexed = df.set_index("timestamp")

    roll_1h = ts_indexed.groupby("user_id")["_ones"].rolling("1h").sum().reset_index(level=0, drop=True)
    roll_24h = ts_indexed.groupby("user_id")["_ones"].rolling("24h").sum().reset_index(level=0, drop=True)

    df["txn_count_last_1h"] = roll_1h.values - 1
    df["txn_count_last_24h"] = roll_24h.values - 1
    df = df.drop(columns=["_ones"])
    return df


def add_user_history_features(df, global_avg, global_std):
    """Stateful, leak-free per-user running stats. Computed via a single
    forward pass per user so 'history' only ever means 'earlier rows'."""
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    n = len(df)

    hist_count = np.zeros(n)
    hist_avg = np.zeros(n)
    hist_std = np.zeros(n)
    is_new_device = np.zeros(n, dtype=int)
    is_new_country = np.zeros(n, dtype=int)
    minutes_since_last = np.zeros(n)

    for user_id, g in df.groupby("user_id", sort=False):
        seen_devices, seen_countries = set(), set()
        amounts = []
        last_ts = None
        for row in g.itertuples():
            i = row.Index
            hist_count[i] = len(amounts)
            if len(amounts) >= 3:
                hist_avg[i] = np.mean(amounts)
                hist_std[i] = np.std(amounts) + 1e-6
            else:
                hist_avg[i] = global_avg
                hist_std[i] = global_std
            is_new_device[i] = 0 if row.device_id in seen_devices else 1
            is_new_country[i] = 0 if row.country in seen_countries else 1
            minutes_since_last[i] = 999999.0 if last_ts is None else (row.timestamp - last_ts).total_seconds() / 60.0

            seen_devices.add(row.device_id)
            seen_countries.add(row.country)
            amounts.append(row.amount)
            last_ts = row.timestamp

    df["user_hist_count"] = hist_count
    df["user_hist_avg_amount"] = hist_avg
    df["user_hist_std_amount"] = hist_std
    df["is_new_device"] = is_new_device
    df["is_new_country"] = is_new_country
    df["minutes_since_last_txn"] = minutes_since_last
    df["amount_zscore"] = (df["amount"] - df["user_hist_avg_amount"]) / df["user_hist_std_amount"]
    df["is_first_txn_for_user"] = (df["user_hist_count"] == 0).astype(int)
    return df


FEATURE_COLUMNS = [
    "amount", "log_amount", "amount_zscore",
    "hour_of_day", "day_of_week", "is_odd_hour",
    "txn_count_last_1h", "txn_count_last_24h", "minutes_since_last_txn",
    "user_hist_count", "is_new_device", "is_new_country", "is_first_txn_for_user",
]
CATEGORICAL_COLUMNS = ["merchant_category"]


def build_features(df, global_avg=None, global_std=None):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if global_avg is None:
        global_avg = df["amount"].mean()
    if global_std is None:
        global_std = df["amount"].std()

    df = add_time_features(df)
    df = add_velocity_features(df)
    df = add_user_history_features(df, global_avg, global_std)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, prefix="cat")
    cat_cols = [c for c in df.columns if c.startswith("cat_")]

    feature_cols = FEATURE_COLUMNS + cat_cols
    return df, feature_cols, global_avg, global_std


if __name__ == "__main__":
    raw = pd.read_csv(DATA_DIR / "transactions.csv")
    feat_df, feature_cols, g_avg, g_std = build_features(raw)
    feat_df.to_csv(DATA_DIR / "features.csv", index=False)
    print(f"Built {len(feature_cols)} features on {len(feat_df):,} transactions")
    print(f"Feature columns: {feature_cols}")
    print("\nSample fraud row feature values:")
    print(feat_df[feat_df.is_fraud == 1][["fraud_type"] + FEATURE_COLUMNS].head(5).to_string())

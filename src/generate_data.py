"""
Aegis - Synthetic transaction data generator.

Why synthetic data:
Real fraud data is sensitive, imbalanced, and not something a hackathon
entrant has access to. Instead of scraping something questionable, we
simulate a realistic digital-payments population with individual behavioral
baselines, then inject THREE distinct, well-documented fraud patterns at a
known rate. This gives us clean ground truth to measure against -- and we
are upfront about this tradeoff in the README (synthetic data can't capture
every real-world nuance, e.g. adversarial adaptation over time).

Fraud patterns injected (all under one class of loss: transaction fraud):
  1. account_takeover  - new device + new country + amount far above the
                          user's historical average, arriving in a burst.
  2. card_testing       - many small transactions in rapid succession from
                          a new device (attacker probing a stolen card).
  3. velocity_abuse     - a fast burst of escalating-amount transactions
                          across multiple merchant categories.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

N_USERS = 1800
N_DAYS = 75
FRAUD_USER_RATE = 0.05          # share of users who experience a fraud incident
                                 # (kept high enough that the held-out test slice
                                 # still has ~70-90 fraud cases for stable P/R estimates)
START_TIME = datetime(2026, 4, 1)

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "travel", "dining", "utilities",
    "fashion", "entertainment", "pharmacy", "fuel", "digital_goods",
]
COUNTRIES = ["IN", "US", "AE", "SG", "GB", "NG", "RU", "BR"]
HOME_COUNTRY_WEIGHTS = [0.82, 0.05, 0.03, 0.03, 0.02, 0.02, 0.015, 0.015]


def make_user_profiles(n_users):
    users = []
    for uid in range(n_users):
        home_country = rng.choice(COUNTRIES, p=HOME_COUNTRY_WEIGHTS)
        avg_amount = rng.lognormal(mean=6.5, sigma=0.8)          # ~ INR 600-3000 typical
        std_amount = avg_amount * rng.uniform(0.25, 0.6)
        peak_hour = rng.choice(range(7, 23))
        n_cats = rng.integers(2, 5)
        cat_prefs = rng.choice(MERCHANT_CATEGORIES, size=n_cats, replace=False)
        cat_weights = rng.dirichlet(np.ones(n_cats))
        primary_device = f"dev_{uid}_{rng.integers(10000,99999)}"
        txns_per_day_lambda = rng.uniform(0.15, 1.4)
        users.append(dict(
            user_id=uid, home_country=home_country, avg_amount=avg_amount,
            std_amount=std_amount, peak_hour=peak_hour,
            cat_prefs=list(cat_prefs), cat_weights=list(cat_weights),
            primary_device=primary_device, txns_per_day_lambda=txns_per_day_lambda,
        ))
    return pd.DataFrame(users)


def gen_normal_txns(user, n_days):
    rows = []
    for day in range(n_days):
        n_today = rng.poisson(user.txns_per_day_lambda)
        for _ in range(n_today):
            hour = int(np.clip(rng.normal(user.peak_hour, 2.5), 0, 23))
            minute, second = rng.integers(0, 60), rng.integers(0, 60)
            ts = START_TIME + timedelta(days=day, hours=hour, minutes=int(minute), seconds=int(second))
            amount = max(20, rng.lognormal(np.log(user.avg_amount), 0.45))
            cat = rng.choice(user.cat_prefs, p=user.cat_weights)
            device = user.primary_device if rng.random() > 0.05 else f"dev_{user.user_id}_alt"
            country = user.home_country if rng.random() > 0.03 else rng.choice(COUNTRIES)
            rows.append(dict(
                user_id=user.user_id, timestamp=ts, amount=round(amount, 2),
                merchant_category=cat, device_id=device, country=country,
                is_fraud=0, fraud_type="none",
            ))
    return rows


def gen_fraud_incident(user, n_days):
    """Inject one fraud burst for this user at a random point after day 10
    (so behavioral history exists for the feature engineering step)."""
    fraud_type = rng.choice(["account_takeover", "card_testing", "velocity_abuse"])
    incident_day = rng.integers(10, n_days - 1)
    incident_start = START_TIME + timedelta(
        days=int(incident_day), hours=int(rng.integers(0, 24)), minutes=int(rng.integers(0, 60))
    )
    rows = []
    new_device = f"dev_fraud_{user.user_id}_{rng.integers(10000,99999)}"

    if fraud_type == "account_takeover":
        n_txns = rng.integers(1, 3)
        foreign_country = rng.choice([c for c in COUNTRIES if c != user.home_country])
        for i in range(n_txns):
            ts = incident_start + timedelta(minutes=int(rng.integers(0, 40)) * i)
            amount = user.avg_amount * rng.uniform(3.5, 9)
            rows.append(dict(
                user_id=user.user_id, timestamp=ts, amount=round(amount, 2),
                merchant_category=rng.choice(MERCHANT_CATEGORIES),
                device_id=new_device, country=foreign_country,
                is_fraud=1, fraud_type=fraud_type,
            ))

    elif fraud_type == "card_testing":
        n_txns = rng.integers(6, 15)
        for i in range(n_txns):
            ts = incident_start + timedelta(seconds=int(rng.integers(20, 240)) * i)
            amount = rng.uniform(10, 60)  # small "is this card alive" probes
            rows.append(dict(
                user_id=user.user_id, timestamp=ts, amount=round(amount, 2),
                merchant_category=rng.choice(["digital_goods", "entertainment"]),
                device_id=new_device, country=user.home_country,
                is_fraud=1, fraud_type=fraud_type,
            ))

    else:  # velocity_abuse
        n_txns = rng.integers(4, 9)
        for i in range(n_txns):
            ts = incident_start + timedelta(minutes=int(rng.integers(2, 25)) * i)
            amount = user.avg_amount * rng.uniform(1.5, 3.0) * (1 + i * 0.35)
            rows.append(dict(
                user_id=user.user_id, timestamp=ts, amount=round(amount, 2),
                merchant_category=rng.choice(MERCHANT_CATEGORIES),
                device_id=new_device if rng.random() > 0.4 else user.primary_device,
                country=user.home_country,
                is_fraud=1, fraud_type=fraud_type,
            ))
    return rows


def main():
    print(f"Generating {N_USERS} user profiles over {N_DAYS} days...")
    users = make_user_profiles(N_USERS)

    all_rows = []
    for _, user in users.iterrows():
        all_rows.extend(gen_normal_txns(user, N_DAYS))

    fraud_users = users.sample(frac=FRAUD_USER_RATE, random_state=RNG_SEED)
    for _, user in fraud_users.iterrows():
        all_rows.extend(gen_fraud_incident(user, N_DAYS))

    df = pd.DataFrame(all_rows).sort_values("timestamp").reset_index(drop=True)
    df["transaction_id"] = [f"txn_{i:07d}" for i in range(len(df))]
    df = df[["transaction_id", "user_id", "timestamp", "amount",
             "merchant_category", "device_id", "country", "is_fraud", "fraud_type"]]

    df.to_csv(DATA_DIR / "transactions.csv", index=False)

    print(f"\nTotal transactions: {len(df):,}")
    print(f"Fraud transactions: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
    print("\nFraud type breakdown:")
    print(df[df.is_fraud == 1]["fraud_type"].value_counts())
    print(f"\nDate range: {df.timestamp.min()} -> {df.timestamp.max()}")
    print(f"\nSaved to {DATA_DIR / 'transactions.csv'}")


if __name__ == "__main__":
    main()

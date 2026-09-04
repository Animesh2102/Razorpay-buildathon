"""
Aegis - Real-world validation on PaySim.

This is a SEPARATE validation track from the synthetic pipeline in
train_eval.py, not a replacement for it. It exists to answer one question
honestly: does the same methodology (time-based split, PR-AUC, cost-
sensitive thresholding) hold up on a real-world-modeled dataset with a
totally different schema?

IMPORTANT SCHEMA FINDING (read before assuming this is a drop-in swap):
PaySim's `nameOrig` field is NOT a repeat customer ID the way our synthetic
`user_id` was -- 99.98% of origin accounts appear exactly ONCE in the
entire 6.36M-row dataset. There is no reusable per-user history here, so
the velocity/deviation features from features.py (is_new_device,
amount_zscore vs user history, txn_count_last_1h) DO NOT TRANSFER. This
is not a bug -- it's a genuinely different data-generating process, and
pretending otherwise would mean silently computing meaningless features.

What DOES carry over from PaySim's schema: balance-consistency features.
PaySim's documented fraud mechanic is an account being drained via
TRANSFER then CASH_OUT. We engineer features around exactly that:
  - errorBalanceOrig: oldbalanceOrg - amount - newbalanceOrig
      (should be ~0 for a consistent legitimate transaction)
  - is_full_balance_drain: amount == oldbalanceOrg (empirically: 97.8% of
      fraud cases vs 0% of legitimate ones -- an almost deterministic
      signal, discussed honestly in the README as a PaySim limitation,
      not a claim that fraud detection in general is "solved")
  - dest-side balance zeroing and error terms
  - transaction type (fraud in this dataset ONLY occurs in TRANSFER and
      CASH_OUT -- a real, documented constraint of the simulator)

No per-user loop needed here -- these are all point-in-time attributes of
a single transaction, fully vectorizable, so this runs fast even at 6.3M rows.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# PaySim CSV is ~490MB -- NOT committed to the repo (see .gitignore).
# Download it yourself from Kaggle ("Synthetic Financial Datasets For
# Fraud Detection" / PaySim) and place it at this path, or pass a
# different path via the RAW_CSV env var.
import os
RAW_CSV = Path(os.environ.get("PAYSIM_CSV", PROJECT_ROOT / "data" / "paysim" / "PS_log.csv"))

COST_FALSE_POSITIVE = 150  # same assumption as the synthetic track, currency-unit-agnostic here


def build_paysim_features(df):
    df = df.copy()
    df["errorBalanceOrig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df["is_full_balance_drain"] = (df["amount"] == df["oldbalanceOrg"]).astype(int)
    df["orig_balance_zeroed"] = ((df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)).astype(int)
    df["dest_both_zero"] = ((df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0)).astype(int)
    df["log_amount"] = np.log1p(df["amount"])
    df["hour_of_day"] = df["step"] % 24
    df = pd.get_dummies(df, columns=["type"], prefix="type")
    return df


FEATURE_COLUMNS_BASE = [
    "amount", "log_amount", "errorBalanceOrig", "errorBalanceDest",
    "is_full_balance_drain", "orig_balance_zeroed", "dest_both_zero", "hour_of_day",
]


def time_based_split(df, train_frac=0.70, val_frac=0.15):
    df = df.sort_values("step").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def find_cost_optimal_threshold(y_true, scores, amounts, cost_fp):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_cost = 0.5, np.inf
    for t in thresholds:
        pred = (scores >= t).astype(int)
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn_mask = (pred == 0) & (y_true == 1)
        cost = fp * cost_fp + amounts[fn_mask].sum()
        if cost < best_cost:
            best_cost, best_t = cost, t
    return best_t, best_cost


def evaluate_at_threshold(y_true, scores, amounts, threshold, cost_fp):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return dict(
        threshold=float(threshold), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
        fraud_amount_caught=float(amounts[(pred == 1) & (y_true == 1)].sum()),
        fraud_amount_missed=float(amounts[(pred == 0) & (y_true == 1)].sum()),
        analyst_review_cost=float(fp * cost_fp),
        total_cost=float(fp * cost_fp + amounts[(pred == 0) & (y_true == 1)].sum()),
    )


def main():
    if not RAW_CSV.exists():
        print(f"PaySim CSV not found at {RAW_CSV}.")
        print("Download 'Synthetic Financial Datasets For Fraud Detection' from Kaggle")
        print("and place it there, or set the PAYSIM_CSV env var to its location.")
        return

    print(f"Loading {RAW_CSV} ...")
    usecols = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
               "oldbalanceDest", "newbalanceDest", "isFraud"]
    dtypes = {"step": "int32", "amount": "float32", "oldbalanceOrg": "float32",
              "newbalanceOrig": "float32", "oldbalanceDest": "float32",
              "newbalanceDest": "float32", "isFraud": "int8"}
    raw = pd.read_csv(RAW_CSV, usecols=usecols, dtype=dtypes)
    print(f"Loaded {len(raw):,} transactions, {int(raw.isFraud.sum()):,} fraud ({raw.isFraud.mean()*100:.3f}%)")
    n_rows_total = len(raw)
    n_fraud_total = int(raw.isFraud.sum())
    fraud_rate_pct = float(raw.isFraud.mean() * 100)

    train_raw, val_raw, test_raw = time_based_split(raw)
    print(f"Train: {len(train_raw):,} (steps {train_raw.step.min()}-{train_raw.step.max()})")
    print(f"Val:   {len(val_raw):,} (steps {val_raw.step.min()}-{val_raw.step.max()})")
    print(f"Test:  {len(test_raw):,} (steps {test_raw.step.min()}-{test_raw.step.max()})")

    train_idx, val_idx, test_idx = train_raw.index, val_raw.index, test_raw.index
    del train_raw, val_raw, test_raw

    full_feat = build_paysim_features(raw)
    del raw
    import gc; gc.collect()

    type_cols = [c for c in full_feat.columns if c.startswith("type_")]
    feature_cols = FEATURE_COLUMNS_BASE + type_cols
    for c in type_cols:
        full_feat[c] = full_feat[c].astype("int8")

    X_train, y_train = full_feat.loc[train_idx, feature_cols], full_feat.loc[train_idx, "isFraud"]
    X_val, y_val = full_feat.loc[val_idx, feature_cols], full_feat.loc[val_idx, "isFraud"]
    X_test, y_test = full_feat.loc[test_idx, feature_cols], full_feat.loc[test_idx, "isFraud"]
    amounts_val = full_feat.loc[val_idx, "amount"].values
    amounts_test = full_feat.loc[test_idx, "amount"].values

    print("\nTraining baseline (Logistic Regression, on a 400k-row subsample for memory) ...")
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(len(X_train), size=min(400_000, len(X_train)), replace=False)
    scaler = StandardScaler()
    X_train_sub_s = scaler.fit_transform(X_train.iloc[sub_idx])
    lr = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    lr.fit(X_train_sub_s, y_train.iloc[sub_idx])
    lr_ap = average_precision_score(y_test, lr.predict_proba(scaler.transform(X_test))[:, 1])
    del X_train_sub_s; gc.collect()

    print("Training XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]
    xgb_ap = average_precision_score(y_test, test_scores)

    print(f"\nPR-AUC -- Logistic Regression: {lr_ap:.4f} | XGBoost: {xgb_ap:.4f}")

    best_t, val_cost = find_cost_optimal_threshold(y_val.values, val_scores, amounts_val, COST_FALSE_POSITIVE)
    print(f"Cost-optimal threshold (validation only): {best_t:.2f}")

    test_metrics = evaluate_at_threshold(y_test.values, test_scores, amounts_test, best_t, COST_FALSE_POSITIVE)
    catch_nothing_cost = amounts_test[y_test.values == 1].sum()

    print(f"\n--- TEST SET (real PaySim data, held out) ---")
    print(f"Precision: {test_metrics['precision']:.4f}  Recall: {test_metrics['recall']:.4f}  F1: {test_metrics['f1']:.4f}")
    print(f"TP={test_metrics['tp']} FP={test_metrics['fp']} FN={test_metrics['fn']} TN={test_metrics['tn']}")
    print(f"Total cost: {test_metrics['total_cost']:,.0f}  vs catch-nothing: {catch_nothing_cost:,.0f}")

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop features:")
    print(importances.head(8).to_string())

    joblib.dump(model, MODEL_DIR / "paysim_xgb_model.joblib")
    report = dict(
        dataset="PaySim (real-world-modeled, Kaggle)",
        rows=n_rows_total, fraud_count=n_fraud_total, fraud_rate_pct=fraud_rate_pct,
        pr_auc=dict(logistic_regression_baseline=float(lr_ap), xgboost=float(xgb_ap)),
        chosen_threshold=float(best_t),
        test_set_metrics=test_metrics,
        catch_nothing_cost=float(catch_nothing_cost),
        top_features=importances.head(8).to_dict(),
        known_limitation=(
            "97.8% of PaySim fraud cases fully drain the account balance "
            "(amount == oldbalanceOrg) vs 0% of legitimate transactions -- an "
            "almost deterministic signal. High scores here partly reflect this "
            "known simplicity of PaySim's fraud simulation, not a claim that "
            "real-world fraud is this separable."
        ),
    )
    with open(REPORT_DIR / "paysim_metrics_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved model + report.")


if __name__ == "__main__":
    main()

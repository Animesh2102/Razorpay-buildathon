"""
Aegis - Model training and evaluation.

Methodology choices, and why each one matters for honest metrics:

1. TIME-BASED split, not random shuffle.
   Fraud data is temporal. A random shuffle can let a model "see" a user's
   later (post-fraud) behavior while predicting an earlier transaction,
   which leaks information no real deployed model would have. We train on
   the first 70% of the timeline, tune the threshold on the next 15%, and
   report final numbers on the last 15% -- never touched until the very end.

2. PR-AUC over ROC-AUC.
   With ~0.5% positive rate, ROC-AUC is misleadingly high (the massive
   negative class makes it easy to look good). Precision-Recall is the
   honest lens for rare-event detection.

3. Cost-sensitive threshold, chosen on the VALIDATION set only.
   "Best F1" is not how a fraud team actually operates -- they operate on
   Rupee cost. We assign a cost to a false positive (an analyst wastes time
   reviewing a clean transaction) and a false negative (the fraud amount is
   lost outright), sweep thresholds on validation data, and pick the
   threshold that minimizes total expected cost. The threshold is then
   FROZEN and applied once to the untouched test set -- so the test-set
   numbers are a genuine held-out estimate, not a cherry-picked best case.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
import xgboost as xgb
import joblib

from features import build_features, FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "transactions.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Cost assumptions -- made explicit and tunable, not hidden inside the model.
COST_FALSE_POSITIVE = 150     # INR: analyst time to review a flagged-but-clean txn
# COST_FALSE_NEGATIVE is computed per-transaction as the actual fraud amount missed


def time_based_split(df, train_frac=0.70, val_frac=0.15):
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def find_cost_optimal_threshold(y_true, scores, amounts, cost_fp):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_cost = 0.5, np.inf
    costs = []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        fp_mask = (pred == 1) & (y_true == 0)
        fn_mask = (pred == 0) & (y_true == 1)
        cost = fp_mask.sum() * cost_fp + amounts[fn_mask].sum()
        costs.append(cost)
        if cost < best_cost:
            best_cost, best_t = cost, t
    return best_t, best_cost, thresholds, np.array(costs)


def evaluate_at_threshold(y_true, scores, amounts, threshold, cost_fp):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    fraud_amount_caught = amounts[(pred == 1) & (y_true == 1)].sum()
    fraud_amount_missed = amounts[(pred == 0) & (y_true == 1)].sum()
    total_cost = fp * cost_fp + fraud_amount_missed
    return dict(
        threshold=float(threshold), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        precision=float(precision), recall=float(recall), f1=float(f1),
        fraud_amount_caught=float(fraud_amount_caught),
        fraud_amount_missed=float(fraud_amount_missed),
        analyst_review_cost=float(fp * cost_fp),
        total_cost=float(total_cost),
    )


def main():
    print("Loading and featurizing data...")
    raw = pd.read_csv(DATA_PATH)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])

    train_raw, val_raw, test_raw = time_based_split(raw)
    print(f"Train: {len(train_raw):,} txns ({train_raw.is_fraud.sum()} fraud) "
          f"| {train_raw.timestamp.min().date()} -> {train_raw.timestamp.max().date()}")
    print(f"Val:   {len(val_raw):,} txns ({val_raw.is_fraud.sum()} fraud) "
          f"| {val_raw.timestamp.min().date()} -> {val_raw.timestamp.max().date()}")
    print(f"Test:  {len(test_raw):,} txns ({test_raw.is_fraud.sum()} fraud) "
          f"| {test_raw.timestamp.min().date()} -> {test_raw.timestamp.max().date()}")

    # IMPORTANT: features for val/test are built using global_avg/global_std
    # FROZEN from the training window only -- another leakage guard.
    full_feat, feature_cols, g_avg, g_std = build_features(train_raw)
    train_feat, _, _, _ = build_features(train_raw, g_avg, g_std)
    val_feat, _, _, _ = build_features(pd.concat([train_raw, val_raw]), g_avg, g_std)
    val_feat = val_feat[val_feat["transaction_id"].isin(val_raw["transaction_id"])]
    test_feat, _, _, _ = build_features(pd.concat([train_raw, val_raw, test_raw]), g_avg, g_std)
    test_feat = test_feat[test_feat["transaction_id"].isin(test_raw["transaction_id"])]

    X_train, y_train = train_feat[feature_cols], train_feat["is_fraud"]
    X_val, y_val = val_feat[feature_cols], val_feat["is_fraud"]
    X_test, y_test = test_feat[feature_cols], test_feat["is_fraud"]

    # --- Baseline: Logistic Regression ---
    print("\nTraining baseline (Logistic Regression)...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    lr_val_scores = lr.predict_proba(X_val_s)[:, 1]
    lr_test_scores = lr.predict_proba(X_test_s)[:, 1]
    lr_ap = average_precision_score(y_test, lr_test_scores)

    # --- Main model: XGBoost ---
    print("Training XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        random_state=42, n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    xgb_val_scores = xgb_model.predict_proba(X_val)[:, 1]
    xgb_test_scores = xgb_model.predict_proba(X_test)[:, 1]
    xgb_ap = average_precision_score(y_test, xgb_test_scores)

    print(f"\n--- PR-AUC (test set, held out, untouched until now) ---")
    print(f"Logistic Regression baseline: {lr_ap:.4f}")
    print(f"XGBoost:                      {xgb_ap:.4f}")

    # --- Cost-sensitive threshold, chosen ONLY on validation ---
    val_amounts = val_feat["amount"].values
    best_t, best_val_cost, thresholds, val_costs = find_cost_optimal_threshold(
        y_val.values, xgb_val_scores, val_amounts, COST_FALSE_POSITIVE
    )
    print(f"\nCost-optimal threshold (chosen on validation only): {best_t:.2f}")
    print(f"Estimated validation cost at this threshold: Rs {best_val_cost:,.0f}")

    # --- Apply frozen threshold to untouched TEST set ---
    test_amounts = test_feat["amount"].values
    test_metrics = evaluate_at_threshold(y_test.values, xgb_test_scores, test_amounts, best_t, COST_FALSE_POSITIVE)

    # Baselines for comparison: catch-nothing, flag-everything
    catch_nothing_cost = test_amounts[y_test.values == 1].sum()
    flag_everything_fp = int((y_test.values == 0).sum())
    flag_everything_cost = flag_everything_fp * COST_FALSE_POSITIVE

    default_metrics = evaluate_at_threshold(y_test.values, xgb_test_scores, test_amounts, 0.5, COST_FALSE_POSITIVE)

    print(f"\n--- TEST SET results at cost-optimal threshold ({best_t:.2f}) ---")
    print(f"Precision: {test_metrics['precision']:.3f}  Recall: {test_metrics['recall']:.3f}  F1: {test_metrics['f1']:.3f}")
    print(f"TP={test_metrics['tp']} FP={test_metrics['fp']} FN={test_metrics['fn']} TN={test_metrics['tn']}")
    print(f"Fraud amount caught: Rs {test_metrics['fraud_amount_caught']:,.0f}")
    print(f"Fraud amount missed: Rs {test_metrics['fraud_amount_missed']:,.0f}")
    print(f"Analyst review cost (false positives): Rs {test_metrics['analyst_review_cost']:,.0f}")
    print(f"TOTAL COST: Rs {test_metrics['total_cost']:,.0f}")
    print(f"\nCompare to catch-nothing cost:   Rs {catch_nothing_cost:,.0f}")
    print(f"Compare to flag-everything cost: Rs {flag_everything_cost:,.0f}")
    print(f"Compare to default 0.5 threshold total cost: Rs {default_metrics['total_cost']:,.0f}")

    # --- Feature importance ---
    importances = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 8 features by importance:")
    print(importances.head(8).to_string())

    # --- Save everything ---
    joblib.dump(xgb_model, MODEL_DIR / "xgb_fraud_model.joblib")
    joblib.dump({"scaler": scaler, "model": lr}, MODEL_DIR / "lr_baseline.joblib")

    report = dict(
        data_summary=dict(
            total_transactions=int(len(raw)), total_fraud=int(raw.is_fraud.sum()),
            fraud_rate_pct=float(raw.is_fraud.mean() * 100),
            train_size=int(len(train_raw)), val_size=int(len(val_raw)), test_size=int(len(test_raw)),
        ),
        pr_auc=dict(logistic_regression_baseline=float(lr_ap), xgboost=float(xgb_ap)),
        cost_assumptions=dict(
            cost_per_false_positive_inr=COST_FALSE_POSITIVE,
            cost_per_false_negative="actual missed fraud amount (Rs)",
        ),
        chosen_threshold=float(best_t),
        test_set_metrics_at_chosen_threshold=test_metrics,
        test_set_metrics_at_default_0_5_threshold=default_metrics,
        baseline_comparisons=dict(
            catch_nothing_cost_inr=float(catch_nothing_cost),
            flag_everything_cost_inr=float(flag_everything_cost),
        ),
        top_features=importances.head(10).to_dict(),
    )
    with open(REPORT_DIR / "metrics_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved model to {MODEL_DIR}/xgb_fraud_model.joblib")
    print(f"Saved metrics report to {REPORT_DIR}/metrics_report.json")
    return report


if __name__ == "__main__":
    main()

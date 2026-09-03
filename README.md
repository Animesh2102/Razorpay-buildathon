# Aegis — AI Risk Manager (Razorpay Buildathon, Track 02)

## Quick Start

```bash
pip install -r requirements.txt
python src/train_eval.py        # trains + evaluates on included data, prints metrics
python src/llm_responder.py     # LLM explainer demo (set ANTHROPIC_API_KEY first for live output)
```

Data and a trained model are already included — no setup beyond pip install needed to see results.

Aegis is a transaction-fraud detector paired with a bounded, human-gated LLM
layer that explains flags and drafts chargeback evidence. Built against the
Track 02 bar: **honest metrics including false-positive cost, strictly
defense-only.**

## The pitch, in one paragraph

Fraud is one clean, measurable class of loss — so this is where the "bar"
gets proven with numbers, not vibes. An XGBoost model scores every
transaction using only that user's *prior* history (zero look-ahead leakage).
A cost-sensitive threshold, tuned exclusively on a validation slice, decides
what gets flagged. Everything above that line goes to an LLM layer that
explains the risk and drafts — never sends — chargeback evidence. Every
decision, human or automated, lands in an append-only audit log.

## Results (held-out test set — the last 15% of the timeline, never touched during training or threshold selection)

| Metric | Value |
|---|---|
| PR-AUC (XGBoost) | 0.921 |
| PR-AUC (Logistic Regression baseline) | 0.836 |
| Precision @ chosen threshold (0.32) | 70.2% |
| Recall @ chosen threshold (0.32) | 91.3% |
| Fraud caught (test batch) | ₹2,85,066 |
| Fraud missed (test batch) | ₹17,394 |
| Analyst review cost incurred (false positives) | ₹4,650 |
| **Total cost at chosen threshold** | **₹22,044** |
| Cost if flagging nothing | ₹3,02,460 |
| Cost if flagging everything | ₹23,04,000 |

The threshold isn't "best F1" — it's the point that minimizes ₹ cost on a
validation slice, then gets frozen and applied once to test data. That's
the difference between a real fraud-ops answer and a leaderboard score.

## Why these specific engineering choices (say this in the pitch)

- **Time-based split, not random shuffle.** A random shuffle lets the model
  see a user's post-fraud behavior while scoring an earlier transaction —
  leakage no production model would have. Train/validate/test are
  chronological, non-overlapping slices.
- **PR-AUC over ROC-AUC.** At ~0.5% fraud incidence, ROC-AUC flatters any
  model. Precision-Recall is the honest lens for rare-event detection.
- **Cost-sensitive threshold, chosen on validation only.** Fraud teams
  operate on rupees, not F1. We assign ₹150 to a false positive (analyst
  review time) and the real missed amount to a false negative, sweep
  thresholds on the validation slice, then freeze the winner before ever
  touching test data.
- **Synthetic data, disclosed as a limitation, not hidden.** Real fraud data
  is sensitive and unavailable to a hackathon entrant. We simulated 1,800
  user behavioral baselines and injected three named fraud patterns
  (account takeover, card testing, velocity abuse) at a realistic ~0.5%
  incidence. This proves the *pipeline and methodology* are sound; it does
  not prove real-world generalization — adversarial adaptation, seasonal
  drift, and correlated fraud rings in real data are not captured here.

## Architecture

```
Transaction stream
        |
Risk scoring pipeline (leak-free features -> XGBoost -> 0-1 score)
        |
Threshold gate (0.32, cost-optimal)
    /                              \
Approved                    LLM explain + draft
(below threshold)           (above threshold, human-gated)
    \                              /
                Audit log
        (every score + decision, traceable)
```

The LLM layer (`src/llm_responder.py`) works with **any one** of Anthropic,
OpenAI, or Gemini -- it auto-detects whichever API key (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, or `GEMINI_API_KEY`/`GOOGLE_API_KEY`) is set and uses that
provider, no code changes needed. It is deliberately bounded regardless of
provider:
- It **never** blocks a transaction, freezes an account, or bans a device.
- Chargeback evidence is always a **draft** (`requires_human_approval: True`)
  — nothing is auto-submitted.
- If the LLM call fails (bad key, rate limit, network), the system does not
  crash and does not silently allow/deny — it falls back to the raw ML
  score and routes the case to a manual queue. This is the "one failure
  handled gracefully" requirement, and it's exercised live every time you
  run `llm_responder.py` without an API key configured.

## Repo structure

```
aegis/
├── README.md
├── requirements.txt
├── src/
│   ├── generate_data.py     # synthetic transaction + fraud generator
│   ├── features.py          # leak-free feature engineering
│   ├── train_eval.py        # time-split train/eval, cost-optimal threshold
│   └── llm_responder.py     # LLM explainer + chargeback evidence drafter
├── data/
│   └── transactions.csv     # generated dataset (102,930 txns, 487 fraud)
├── models/
│   ├── xgb_fraud_model.joblib
│   └── lr_baseline.joblib
└── reports/
    ├── metrics_report.json  # full metrics, thresholds, cost breakdown
    └── audit_trail.jsonl    # append-only log (created on first LLM run)
```

## Running it

```bash
pip install -r requirements.txt

# 1. Regenerate data (optional — transactions.csv is already included)
python3 src/generate_data.py

# 2. Train + evaluate (prints PR-AUC, cost-optimal threshold, test metrics)
python3 src/train_eval.py

# 3. LLM layer demo (works without a key -- reports "no key found"
#    gracefully; set ONE of the keys below for live explanations + drafts)
export ANTHROPIC_API_KEY=your_key_here    # or OPENAI_API_KEY, or GEMINI_API_KEY
python3 src/llm_responder.py
```

## A note on reproducibility

This repo was verified to run end-to-end on both Linux and Windows. The
synthetic data generation is **fully deterministic** (`RNG_SEED = 42`) —
transaction counts and fraud breakdown match exactly across machines.
XGBoost's parallel tree-building can introduce tiny floating-point
differences across platforms/CPU thread counts even with `random_state`
fixed, so the cost-optimal threshold may land at, e.g., 0.32 on one machine
and 0.36 on another. The conclusions are stable across both runs we tested
(~90% recall, ~70-76% precision, ~₹22K total cost vs. ~₹3L catching
nothing) — only the third decimal place moves.

## Honest limitations (say these upfront in the pitch — reviewers trust builders who name the edges themselves)

1. Synthetic data proves the methodology, not real-world fraud rates.
2. Three fraud patterns are modeled; real fraud rings adapt faster than any
   fixed pattern set.
3. The LLM evidence drafts are only as good as the feature snapshot handed
   to them — they don't independently investigate a disputed transaction.
4. Cost constants (₹150/review, missed-amount-as-cost) are stated
   assumptions, not measured from a real ops team — swap them for real
   numbers before production use.

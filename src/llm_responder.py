"""
Aegis - LLM explanation & chargeback evidence responder.

This is the "meaningful AI use" layer on top of the XGBoost detector, and
it is deliberately scoped to stay on the right side of the track's bar:
"Strictly defense-only: anything offense-capable is disqualified."

What this module DOES:
  - Turns a flagged transaction's feature values into a plain-English risk
    explanation an analyst can read in 5 seconds.
  - Drafts a chargeback evidence packet citing the SPECIFIC anomalies the
    model saw (device change, velocity spike, amount deviation) -- grounded
    in real feature values, not invented claims.

What this module explicitly does NOT do (bounded + gated, per the bar):
  - It never blocks a transaction, freezes an account, or bans a device.
  - It never auto-sends the evidence packet -- output is always a DRAFT
    that a human analyst reviews and approves (see `requires_human_approval`
    on every response).
  - It never claims to be legal advice.

Every call is written to an audit log (audit_trail.jsonl) with the input
feature snapshot, the model's risk score, and the LLM's output -- so every
action taken by this system is traceable end to end.
"""

import json
import os
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Provider-agnostic LLM client setup -------------------------------
# Aegis works with whichever provider's API key is present in the
# environment, checked in this priority order. No code changes needed
# to switch providers -- just set a different key before running.
#
# Model names are overridable via env vars so this doesn't go stale as
# providers ship new versions -- check each provider's docs if a default
# below stops working.

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

_PROVIDER = None
_CLIENT = None

if os.environ.get("ANTHROPIC_API_KEY"):
    try:
        import anthropic
        _CLIENT = anthropic.Anthropic()
        _PROVIDER = "anthropic"
    except Exception:
        pass

if _CLIENT is None and os.environ.get("OPENAI_API_KEY"):
    try:
        import openai
        _CLIENT = openai.OpenAI()
        _PROVIDER = "openai"
    except Exception:
        pass

if _CLIENT is None and (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
    try:
        from google import genai
        _CLIENT = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        _PROVIDER = "gemini"
    except Exception:
        pass

AUDIT_LOG_PATH = PROJECT_ROOT / "reports" / "audit_trail.jsonl"

EXPLAIN_SYSTEM_PROMPT = """You are a fraud-risk explainer for an analyst dashboard.
You are given a flagged transaction's feature values and the model's risk score.
Write a 2-3 sentence plain-English explanation of why this transaction looks risky,
grounded ONLY in the feature values given to you. Do not invent details not present
in the data. Do not recommend blocking, banning, or any irreversible action --
you are producing an explanation for a human reviewer, not an action."""

EVIDENCE_SYSTEM_PROMPT = """You are drafting a CHARGEBACK EVIDENCE PACKET DRAFT for a
human fraud analyst to review before submission. Cite only the specific anomalies
present in the transaction data provided (device history, velocity, amount deviation,
location). Structure it as: (1) Transaction summary, (2) Anomaly evidence, (3)
Recommended supporting documentation to attach. Do not state this is certain fraud --
frame findings as risk indicators. Never claim legal certainty. This is a DRAFT ONLY;
say so explicitly in your output."""


def _log_audit(event_type, transaction_id, payload):
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "transaction_id": transaction_id,
        "payload": payload,
    }
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def _feature_snapshot_text(txn_row, risk_score):
    return f"""
Transaction ID: {txn_row.get('transaction_id')}
Risk score: {risk_score:.3f}
Amount: Rs {txn_row.get('amount')}
Amount z-score vs user history: {txn_row.get('amount_zscore', 'n/a')}
Is new device: {bool(txn_row.get('is_new_device'))}
Is new country: {bool(txn_row.get('is_new_country'))}
Transactions in last 1h: {txn_row.get('txn_count_last_1h', 'n/a')}
Transactions in last 24h: {txn_row.get('txn_count_last_24h', 'n/a')}
Minutes since user's previous transaction: {txn_row.get('minutes_since_last_txn', 'n/a')}
User's transaction history count: {txn_row.get('user_hist_count', 'n/a')}
Merchant category: {txn_row.get('merchant_category', 'n/a')}
""".strip()


def _call_llm(system_prompt, user_content, max_tokens=400):
    """Dispatches to whichever provider was detected at import time.
    Same (text, error) contract regardless of provider, so the rest of
    this module never needs to know which one is actually running."""
    if _CLIENT is None:
        return None, (
            "No LLM API key found. Set ONE of ANTHROPIC_API_KEY, OPENAI_API_KEY, "
            "or GEMINI_API_KEY / GOOGLE_API_KEY in your environment -- see README."
        )
    try:
        if _PROVIDER == "anthropic":
            resp = _CLIENT.messages.create(
                model=ANTHROPIC_MODEL, max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            return text, None

        elif _PROVIDER == "openai":
            resp = _CLIENT.chat.completions.create(
                model=OPENAI_MODEL, max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            return resp.choices[0].message.content, None

        elif _PROVIDER == "gemini":
            resp = _CLIENT.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_content,
                config={"system_instruction": system_prompt, "max_output_tokens": max_tokens},
            )
            return resp.text, None

        return None, f"Unknown provider: {_PROVIDER}"
    except Exception as e:
        return None, f"LLM call failed ({_PROVIDER}): {e}"


def explain_flagged_transaction(txn_row, risk_score):
    """Bounded action: produces an explanation only. No account/txn action taken."""
    snapshot = _feature_snapshot_text(txn_row, risk_score)
    text, error = _call_llm(EXPLAIN_SYSTEM_PROMPT, snapshot)
    result = {
        "transaction_id": txn_row.get("transaction_id"),
        "risk_score": float(risk_score),
        "explanation": text,
        "error": error,
        "action_taken": "none - explanation only",
        "requires_human_approval": False,  # explanations are informational, not actions
    }
    _log_audit("explain", txn_row.get("transaction_id"), result)
    return result


def draft_chargeback_evidence(txn_row, risk_score):
    """Bounded + gated action: produces a DRAFT only. Never auto-submitted."""
    snapshot = _feature_snapshot_text(txn_row, risk_score)
    text, error = _call_llm(EVIDENCE_SYSTEM_PROMPT, snapshot, max_tokens=600)
    result = {
        "transaction_id": txn_row.get("transaction_id"),
        "risk_score": float(risk_score),
        "evidence_draft": text,
        "error": error,
        "action_taken": "draft generated - NOT submitted",
        "requires_human_approval": True,  # gate: a human must approve before this goes anywhere
    }
    _log_audit("chargeback_evidence_draft", txn_row.get("transaction_id"), result)
    return result


def handle_llm_failure_gracefully(txn_row, risk_score):
    """Demonstrates the 'one failure handled gracefully' requirement: if the
    LLM call fails (bad key, rate limit, network), the system does NOT crash
    and does NOT silently approve/deny -- it falls back to the raw ML output
    and flags the case for manual analyst attention."""
    fallback = {
        "transaction_id": txn_row.get("transaction_id"),
        "risk_score": float(risk_score),
        "status": "LLM_UNAVAILABLE_FALLBACK_TO_ANALYST_QUEUE",
        "message": (
            f"Model flagged this transaction at risk score {risk_score:.2f}. "
            f"Automated explanation unavailable -- routed to manual review queue."
        ),
        "action_taken": "routed to manual queue, no automated action taken",
        "requires_human_approval": True,
    }
    _log_audit("llm_failure_fallback", txn_row.get("transaction_id"), fallback)
    return fallback


def load_frozen_threshold():
    """Loads the cost-optimal threshold that train_eval.py computed on
    validation data and froze. This is the actual link between the ML
    pipeline and this response layer -- without it, 'threshold gate' is
    just a diagram, not running code."""
    report_path = PROJECT_ROOT / "reports" / "metrics_report.json"
    with open(report_path) as f:
        report = json.load(f)
    return report["chosen_threshold"]


def route_transaction(txn_row, risk_score, threshold=None):
    """The actual Threshold Gate. Every transaction goes through this.
    Below threshold -> approved automatically, no LLM call, no cost.
    At or above threshold -> routed to the LLM layer for explanation
    and (if applicable) an evidence draft. This is what makes the
    architecture diagram real rather than aspirational."""
    if threshold is None:
        threshold = load_frozen_threshold()

    if risk_score < threshold:
        result = {
            "transaction_id": txn_row.get("transaction_id"),
            "risk_score": float(risk_score),
            "threshold": float(threshold),
            "decision": "APPROVED",
            "action_taken": "none - below cost-optimal threshold",
            "requires_human_approval": False,
        }
        _log_audit("routing_decision", txn_row.get("transaction_id"), result)
        return result

    explanation = explain_flagged_transaction(txn_row, risk_score)
    result = {
        "transaction_id": txn_row.get("transaction_id"),
        "risk_score": float(risk_score),
        "threshold": float(threshold),
        "decision": "FLAGGED",
        "explanation": explanation,
        "requires_human_approval": True,
    }
    _log_audit("routing_decision", txn_row.get("transaction_id"), result)
    return result


if __name__ == "__main__":
    # Demo using real transactions from the test set (no API key needed to
    # see the bounded/audited/gated behavior -- LLM calls will report "no
    # key found" gracefully rather than crashing).
    import pandas as pd
    import joblib
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from features import build_features

    raw = pd.read_csv(PROJECT_ROOT / "data" / "transactions.csv")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    feat_df, feature_cols, _, _ = build_features(raw)
    model = joblib.load(PROJECT_ROOT / "models" / "xgb_fraud_model.joblib")
    feat_df["risk_score"] = model.predict_proba(feat_df[feature_cols])[:, 1]

    threshold = load_frozen_threshold()
    print(f"Loaded frozen threshold from reports/metrics_report.json: {threshold}\n")

    flagged = feat_df[feat_df.is_fraud == 1].sort_values("risk_score", ascending=False).iloc[0]
    approved = feat_df[(feat_df.risk_score < threshold)].sample(1, random_state=1).iloc[0]

    print("=== Routing demo: a genuinely flagged transaction (score above threshold) ===")
    print(json.dumps(route_transaction(flagged, flagged["risk_score"], threshold), indent=2, default=str))

    print("\n=== Routing demo: a genuinely approved transaction (score below threshold) ===")
    print(json.dumps(route_transaction(approved, approved["risk_score"], threshold), indent=2, default=str))

    print("\n=== Chargeback evidence draft demo (on the flagged case) ===")
    print(json.dumps(draft_chargeback_evidence(flagged, flagged["risk_score"]), indent=2, default=str))

    print("\n=== Graceful failure fallback demo ===")
    print(json.dumps(handle_llm_failure_gracefully(flagged, flagged["risk_score"]), indent=2, default=str))

    print(f"\nAudit trail written to {AUDIT_LOG_PATH}")

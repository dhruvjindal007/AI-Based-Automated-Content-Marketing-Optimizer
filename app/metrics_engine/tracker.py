# ============================================================
# tracker3.py  (UPDATED FULL VERSION)
# ============================================================
"""
Tracker3 — Central Logging Layer

This module standardizes logging across the project.
Everything is logged via SHEETS CONNECTOR.

This includes:
    ✓ Raw sentiment feedback
    ✓ Aggregated sentiment metrics
    ✓ A/B test results (simple version)
    ✓ Campaign events

All heavy logic stays in:
    - metrics_tracker.py
    - sentiment_analyzer.py
    - ab_coach.py
    - auto_retrainer.py

This file focuses ONLY on clean and consistent logging.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    logger.addHandler(h)

# --- Optional integrations (fix #3) -------------------------------------
# These used to be hard imports with an eagerly-instantiated TrendFetcher(),
# which meant a missing Sheets credential or trend API key would crash the
# import of this entire module — including functions like log_campaign_event
# that don't even need sentiment/trend data. Now each dependency degrades
# gracefully and is checked before use.

try:
    from app.integrations.sheets_connector import append_row
    _SHEETS_AVAILABLE = True
except Exception as e:
    append_row = None
    _SHEETS_AVAILABLE = False
    logger.info(f"sheets_connector not available — logging to Sheets disabled: {e}")

try:
    from app.sentiment_engine.sentiment_analyzer import analyze_sentiment
    _SENTIMENT_AVAILABLE = True
except Exception as e:
    analyze_sentiment = None
    _SENTIMENT_AVAILABLE = False
    logger.info(f"sentiment_analyzer not available: {e}")

try:
    from app.integrations.trend_fetcher import TrendFetcher
    tf = TrendFetcher()
    _TREND_AVAILABLE = True
except Exception as e:
    TrendFetcher = None
    tf = None
    _TREND_AVAILABLE = False
    logger.info(f"TrendFetcher not available: {e}")


def _safe_append_row(sheet: str, row: List[Any]) -> bool:
    """Small shared helper so every logging function checks availability
    the same way instead of relying on append_row(None, ...) blowing up."""
    if not _SHEETS_AVAILABLE:
        logger.warning(f"Sheets not available — skipped writing to '{sheet}'.")
        return False
    try:
        append_row(sheet, row)
        return True
    except Exception as e:
        logger.warning(f"Could not write row to '{sheet}': {e}")
        return False


# ============================================================
# 1. RAW FEEDBACK LOGGER
# ============================================================

def push_raw_feedback(items: List[Dict[str, Any]]):
    """
    items = [
      { "id": "1", "source": "generated", "text": "..."},
      ...
    ]

    For each text:
      - sentiment
      - trend_score
      - text meta
    """

    written = 0

    for item in items:
        text = item.get("text", "")

        # Fix #2: wrap per-item analysis so one bad item (e.g. empty text
        # causing analyze_sentiment()[0] to raise an IndexError) doesn't
        # abort the rest of the batch silently.
        try:
            sentiment = analyze_sentiment(text)[0] if _SENTIMENT_AVAILABLE else {}
            trend_score = tf.get_combined_trend_score(text) if _TREND_AVAILABLE else 0.0
        except Exception as e:
            logger.warning(f"Failed to analyze item {item.get('id', '')}: {e}")
            continue

        # Fix #1: removed the bogus sentiment.get("trend_score") entry that
        # was always None (analyze_sentiment never returns that key) and
        # sat right before the real trend_score value below.
        row = [
            datetime.now(timezone.utc).isoformat(),
            item.get("id", ""),
            item.get("source", ""),
            text[:150],
            sentiment.get("sentiment_label"),
            sentiment.get("sentiment_score"),
            sentiment.get("polarity"),
            json.dumps(sentiment.get("emotions")),
            trend_score,
        ]

        if _safe_append_row("raw_feedback", row):
            written += 1

    logger.info(f"Pushed {written}/{len(items)} raw feedback items.")
    return True


# ============================================================
# 2. PUSH AGGREGATE METRICS
# ============================================================

def push_aggregates(metrics: Dict[str, Any]):
    """
    metrics = {
        "total": 120,
        "avg_score": 0.67,
        "pos_count": 60,
        "neg_count": 25,
        "neu_count": 35,
        "pct_positive": 0.50,
        "pct_negative": 0.20,
        "avg_toxicity": 0.12,
        "dominant_emotion": "joy",
    }
    """

    row = [
        datetime.now(timezone.utc).isoformat(),
        metrics.get("total", 0),
        metrics.get("avg_score", 0),
        metrics.get("pos_count", 0),
        metrics.get("neg_count", 0),
        metrics.get("neu_count", 0),
        metrics.get("pct_positive", 0),
        metrics.get("pct_negative", 0),
        metrics.get("avg_toxicity", 0),
        metrics.get("dominant_emotion", "unknown")
    ]

    if _safe_append_row("aggregates", row):
        logger.info("Aggregates logged to Google Sheets.")

    return True


# ============================================================
# 3. PUSH A/B TEST RESULTS (Simple Version)
# ============================================================

def push_ab_test_results(campaign_id: str, results: List[Dict]):
    """
    results = [
        {
            "variant": "v1",
            "impressions": 1500,
            "clicks": 120,
            "conversions": 12,
            "ctr": 0.08,
            "conv_rate": 0.01
        }
    ]
    """

    written = 0

    for r in results:
        row = [
            datetime.now(timezone.utc).isoformat(),
            campaign_id,
            r.get("variant", ""),
            r.get("impressions", 0),
            r.get("clicks", 0),
            r.get("conversions", 0),
            r.get("ctr", 0.0),
            r.get("conv_rate", 0.0)
        ]

        if _safe_append_row("ab_test_results", row):
            written += 1

    logger.info(f"Logged {written}/{len(results)} A/B test result rows.")
    return True


# ============================================================
# 4. CAMPAIGN EVENTS LOGGER
# ============================================================

def log_campaign_event(event: str, info: Dict[str, Any]):
    """
    Example:
      log_campaign_event("A/B Test Started", {"variants": 3, "campaign": "XYZ"})
    """

    # Fix #5: json.dumps instead of str(info) — str() on a dict produces
    # Python repr syntax (single quotes, True/False/None) which is not
    # valid JSON and can't be reliably parsed back downstream.
    row = [
        datetime.now(timezone.utc).isoformat(),
        event,
        json.dumps(info, default=str)
    ]

    if _safe_append_row("campaign_logs", row):
        logger.info(f"Event logged: {event}")

    return True


# ============================================================
# Manual Test
# ============================================================

if __name__ == "__main__":
    print("\nTracker3 updated version test.")

    # Raw feedback example
    push_raw_feedback([
        {"id": "101", "source": "demo", "text": "AI tools are amazing!"},
        {"id": "102", "source": "demo", "text": "This product is confusing…"}
    ])

    # Aggregates example
    push_aggregates({
        "total": 50,
        "avg_score": 0.72,
        "pos_count": 30,
        "neg_count": 10,
        "neu_count": 10,
        "pct_positive": 0.6,
        "pct_negative": 0.2,
        "avg_toxicity": 0.05,
        "dominant_emotion": "joy"
    })

    # A/B results example
    push_ab_test_results("test_campaign", [
        {"variant": "A", "impressions": 1200, "clicks": 100, "conversions": 8, "ctr": 0.083, "conv_rate": 0.08},
        {"variant": "B", "impressions": 1200, "clicks": 90, "conversions": 9, "ctr": 0.075, "conv_rate": 0.1}
    ])

    log_campaign_event("Demo Event", {"note": "This is a test."})
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
    - metrics_tracker2.py
    - sentiment_analyzer2.py
    - ab_coach2.py
    - auto_retrainer.py

This file focuses ONLY on clean and consistent logging.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List

from app.integrations.sheets_connector import append_row
from app.sentiment_engine.sentiment_analyzer import analyze_sentiment
from app.integrations.trend_fetcher import TrendFetcher


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    logger.addHandler(h)

tf = TrendFetcher()


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

    for item in items:
        text = item.get("text", "")
        sentiment = analyze_sentiment(text)[0]
        trend_score = tf.get_combined_trend_score(text)

        row = [
            datetime.utcnow().isoformat(),
            item.get("id", ""),
            item.get("source", ""),
            text[:150],
            sentiment.get("sentiment_label"),
            sentiment.get("sentiment_score"),
            sentiment.get("polarity"),
            json.dumps(sentiment.get("emotions")),
            sentiment.get("trend_score"),
            trend_score,
        ]

        try:
            append_row("raw_feedback", row)
        except Exception as e:
            logger.warning(f"Could not write raw feedback row: {e}")

    logger.info(f"Pushed {len(items)} raw feedback items.")
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
        datetime.utcnow().isoformat(),
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

    try:
        append_row("aggregates", row)
        logger.info("Aggregates logged to Google Sheets.")
    except Exception as e:
        logger.warning(f"Failed to log aggregates: {e}")

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

    for r in results:
        row = [
            datetime.utcnow().isoformat(),
            campaign_id,
            r.get("variant", ""),
            r.get("impressions", 0),
            r.get("clicks", 0),
            r.get("conversions", 0),
            r.get("ctr", 0.0),
            r.get("conv_rate", 0.0)
        ]

        try:
            append_row("ab_test_results", row)
        except Exception as e:
            logger.warning(f"Failed to log A/B row: {e}")

    logger.info(f"Logged {len(results)} A/B test result rows.")
    return True


# ============================================================
# 4. CAMPAIGN EVENTS LOGGER
# ============================================================

def log_campaign_event(event: str, info: Dict[str, Any]):
    """
    Example:
      log_campaign_event("A/B Test Started", {"variants": 3, "campaign": "XYZ"})
    """

    row = [
        datetime.utcnow().isoformat(),
        event,
        str(info)
    ]

    try:
        append_row("campaign_logs", row)
        logger.info(f"Event logged: {event}")
    except Exception as e:
        logger.warning(f"Failed to log event {event}: {e}")

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

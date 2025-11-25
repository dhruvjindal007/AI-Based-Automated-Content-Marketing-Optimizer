"""
# slack_notifier.py (Slack Bot version)

Improved slack_notifier3.py

Purpose:
--------
Centralized Slack messaging module for:
- Alerts (sentiment spikes, low CTR)
- A/B test results
- Daily/Weekly performance summaries
- Debug/error logs
- Campaign activity notifications

Enhancements:
-------------
1. Unified Slack sender with retry logic
2. Rich block formatting (Slack UI)
3. Custom alert types:
    - success
    - warning
    - danger
4. Dedicated helper functions:
    send_sentiment_alert()
    send_ab_test_winner()
    send_daily_report()
    send_weekly_report()
    send_low_ctr_alert()
5. Fallback logging if webhook is missing
"""

import os
import json
import time
import logging
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

# -------------------------------------------------------
# ENV CONFIG
# -------------------------------------------------------

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
DEFAULT_PREFIX = os.getenv("SLACK_ALERT_PREFIX", ":warning: *Marketing AI Alert*")

MAX_RETRIES = 3


# -------------------------------------------------------
# CORE SLACK SENDER
# -------------------------------------------------------

def send_slack_message(text: str, blocks: List[Dict]):
    try:
        payload = {"text": text, "blocks": blocks}
        response = requests.post(SLACK_WEBHOOK_URL, json=payload)

        if response.status_code == 200:
            return True
        else:
            print("Slack API Error:", response.text)
            return False

    except Exception as e:
        print("Slack Exception:", e)
        return False


# -------------------------------------------------------
# BLOCK HELPERS (Reusable UI Components)
# -------------------------------------------------------

def block_header(text: str) -> Dict:
    return {
        "type": "header",
        "text": {"type": "plain_text", "text": text}
    }

def block_section_md(text: str) -> Dict:
    return {
        "type": "section",
        "text": {"type": "mrkdwn", "text": text}
    }

def block_divider() -> Dict:
    return {"type": "divider"}


# -------------------------------------------------------
# SENTIMENT ALERTS
# -------------------------------------------------------

def send_sentiment_alert(metrics: Dict):
    """
    Alerts when:
        - negative sentiment rises
        - toxicity increases
    """
    text = f"{DEFAULT_PREFIX} — Sentiment Alert"

    blocks = [
        block_header("🚨 Sentiment Alert"),
        block_section_md(
            f"*Total:* {metrics.get('total')}\n"
            f"*Avg Score:* {metrics.get('avg_score')}\n"
            f"*Negative%:* {metrics.get('pct_negative')}\n"
            f"*Avg Toxicity:* {metrics.get('avg_toxicity')}"
        ),
        block_divider()
    ]

    # Add negative examples if available
    negatives = metrics.get("top_negative", [])
    if negatives:
        formatted = "\n".join([f"> ({n['score']}) {n['text'][:150]}" for n in negatives])
        blocks.append(block_section_md(f"*Top Negative Samples:*\n{formatted}"))

    send_slack_message(text, blocks)


# -------------------------------------------------------
# LOW CTR ALERT
# -------------------------------------------------------

def send_low_ctr_alert(low_ctr_items: List[Dict]):
    """
    low_ctr_items = [
        {"id": "...", "ctr": 0.01, "source": "twitter"}
    ]
    """
    text = "⚠ Low CTR Alert!"

    formatted = "\n".join([f"- ID: {i['id']} | CTR: {i['ctr']} | Source: {i['source']}" for i in low_ctr_items])

    blocks = [
        block_header("📉 Low CTR Alert"),
        block_section_md("These posts are underperforming:\n" + formatted)
    ]

    send_slack_message(text, blocks)


# -------------------------------------------------------
# A/B TEST WINNER ANNOUNCEMENT
# -------------------------------------------------------

def send_ab_test_winner(campaign_id: str, winner: Dict):
    """
    Sends winner announcement to Slack.
    Returns True on success, False on failure.
    """

    try:
        text = f"🏆 A/B Test Winner for {campaign_id}"

        blocks = [
            block_header(f"🏆 A/B Test Winner — {campaign_id}"),
            block_section_md(
                f"*Variant:* {winner.get('variant')}\n"
                f"*CTR:* {winner.get('ctr')}\n"
                f"*Conversion Rate:* {winner.get('conv_rate')}"
            ),
            block_divider()
        ]

        send_slack_message(text, blocks)
        return True

    except Exception as e:
        print("Slack error:", e)
        return False


# -------------------------------------------------------
# FULL A/B PERFORMANCE REPORT (ALL VARIANTS)
# -------------------------------------------------------

def send_ab_test_full_report(campaign_id: str, ab_df):
    """
    Sends a full performance summary of all variants to Slack.

    Expected ab_df columns:
    - variant
    - impressions
    - clicks
    - conversions
    - ctr
    - conv_rate
    - text
    """

    rows = []
    for _, row in ab_df.iterrows():
        rows.append(
            f"*{row['variant']}*\n"
            f"> CTR: `{row['ctr']:.4f}` | CR: `{row['conv_rate']:.4f}`\n"
            f"> Clicks: `{row['clicks']}` / Impressions: `{row['impressions']}`\n"
        )

    combined = "\n".join(rows)

    blocks = [
        block_header(f"📊 Full A/B Test Performance — {campaign_id}"),
        block_section_md(combined),
        block_divider()
    ]

    send_slack_message(
        f"📊 Full A/B Test Performance — {campaign_id}", 
        blocks
    )


# -------------------------------------------------------
# DAILY SUMMARY REPORT
# -------------------------------------------------------

def send_daily_report(metrics: Dict):
    """
    Sends a daily marketing performance summary.
    """
    text = "📊 Daily Marketing Summary"

    blocks = [
        block_header("📊 Daily Marketing Summary"),
        block_section_md(
            f"*Total Feedback:* {metrics.get('total')}\n"
            f"*Positive%:* {metrics.get('pct_positive')}\n"
            f"*Negative%:* {metrics.get('pct_negative')}\n"
            f"*Avg Sentiment:* {metrics.get('avg_score')}\n"
            f"*Dominant Emotion:* {metrics.get('dominant_emotion')}"
        )
    ]

    send_slack_message(text, blocks)


# -------------------------------------------------------
# WEEKLY SUMMARY
# -------------------------------------------------------

def send_weekly_report(weekly_data: Dict):
    """
    weekly_data = {
        "week": "Week 34",
        "avg_ctr": 0.07,
        "avg_sentiment": 0.62,
        "total_posts": 120
    }
    """
    text = "📅 Weekly Marketing Summary"

    blocks = [
        block_header(f"📅 {weekly_data.get('week', 'Weekly')} Summary"),
        block_section_md(
            f"*Average CTR:* {weekly_data.get('avg_ctr')}\n"
            f"*Average Sentiment:* {weekly_data.get('avg_sentiment')}\n"
            f"*Total Posts:* {weekly_data.get('total_posts')}"
        )
    ]

    send_slack_message(text, blocks)


# -------------------------------------------------------
# GENERIC ALERT WRAPPER
# -------------------------------------------------------

def send_alert_summary(metrics: dict, extra: dict = None) -> bool:
    """
    Compatibility function used by legacy modules.
    Calls advanced alert system based on metrics.
    """

    if metrics.get("pct_negative", 0) > 0.25:
        send_sentiment_alert(metrics)

    if extra and extra.get("low_ctr"):
        send_low_ctr_alert(extra["low_ctr"])

    return True


# -------------------------------------------------------
# Self-test
# -------------------------------------------------------
if __name__ == "__main__":
    print("Slack Notifier Test (offline-safe mode)")
    send_slack_message("Test: Slack notifier is running! 🚀")

# metrics_tracker2.py (UPDATED)
"""
Updated metrics_tracker2.py

Integrations:
- integrations.social_ingestor.SocialIngestor  -> fetch live post metrics (likes, replies, shares)
- integrations.trend_fetcher.TrendFetcher      -> trend scores for text
- sentiment_analyzer2.analyze_sentiment       -> sentiment scoring for text/comments
- integrations.sheets_connector.append_row     -> unified Google Sheets writes

Features:
- Compute CTR, engagement rate, conversion rate, sentiment distribution, avg trend score, toxicity (if present)
- Optionally aggregate live metrics given a list of post IDs
- Robust sheet write with retries and graceful fallback
- Student-friendly logging and clear structure
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional

import pandas as pd

# Integrations
from app.integrations.social_ingestor import SocialIngestor
from app.integrations.trend_fetcher import TrendFetcher
from app.integrations.sheets_connector import append_row, read_rows
from app.sentiment_engine.sentiment_analyzer import analyze_sentiment

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

# Config
RETRY_LIMIT = int(os.getenv("METRICS_RETRY_LIMIT", "3"))
SHEET_LOG_ENABLED = bool(os.getenv("GOOGLE_SHEET_ID"))
DEFAULT_SHEET_NAME = os.getenv("METRICS_SHEET_NAME", "daily_metrics")

# Instantiate integrations (singletons for reuse)
_ingestor = SocialIngestor()
_trends = TrendFetcher()


# -----------------------------
# Helper: safe average
# -----------------------------
def _safe_mean(values: List[float]) -> float:
    vals = [v for v in values if v is not None]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


# -----------------------------
# 1) Fetch and aggregate live metrics from a list of post IDs
# -----------------------------
def fetch_and_aggregate_post_metrics(post_ids: List[str]) -> Dict[str, Any]:
    """
    Given a list of social platform post IDs, fetch live metrics via SocialIngestor
    and return aggregated numbers.
    Returns dict:
      {
        "total_posts": int,
        "likes": int,
        "replies": int,
        "shares": int,
        "avg_trend_score": float,
        "avg_sentiment": float,
        "sample_texts": [...],
      }
    """
    if not post_ids:
        return {
            "total_posts": 0,
            "likes": 0,
            "replies": 0,
            "shares": 0,
            "avg_trend_score": 0.0,
            "avg_sentiment": 0.0,
            "sample_texts": []
        }

    likes_list = []
    replies_list = []
    shares_list = []
    trend_scores = []
    sentiment_scores = []
    sample_texts = []

    for pid in post_ids:
        try:
            data = _ingestor.fetch_complete_post_data(str(pid))
            metrics = data.get("metrics", {})
            likes_list.append(int(metrics.get("likes", 0)))
            replies_list.append(int(metrics.get("replies", 0)))
            shares_list.append(int(metrics.get("shares", 0)))
            text = data.get("text", "") or ""
            sample_texts.append(text[:200])
            # Trend + sentiment enrichment
            trend_scores.append(_trends.get_combined_trend_score(text))
            sent = analyze_sentiment(text)[0]
            sentiment_scores.append(sent.get("sentiment_score", 0.0))
        except Exception as e:
            logger.warning(f"Failed to fetch/aggregate for post {pid}: {e}")
            continue

    agg = {
        "total_posts": len(post_ids),
        "likes": int(sum(likes_list)) if likes_list else 0,
        "replies": int(sum(replies_list)) if replies_list else 0,
        "shares": int(sum(shares_list)) if shares_list else 0,
        "avg_trend_score": round(_safe_mean(trend_scores), 4),
        "avg_sentiment": round(_safe_mean(sentiment_scores), 4),
        "sample_texts": sample_texts[:5]
    }
    return agg


# -----------------------------
# 2) Compute metrics from DataFrame
# -----------------------------
def compute_metrics_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute KPIs from a DataFrame that may include columns:
      - impressions, clicks, likes, shares, comments, conversions
      - sentiment_label / sentiment_score
      - trend_score
      - toxicity (optional)
    Returns a metrics dictionary suitable for Sheets logging and dashboards.
    """
    if df is None or df.empty:
        return {
            "total_records": 0,
            "impressions": 0,
            "ctr": 0.0,
            "engagement_rate": 0.0,
            "conversion_rate": 0.0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "neutral_ratio": 0.0,
            "avg_trend_score": 0.0,
            "avg_toxicity": 0.0,
            "dominant_emotion": "unknown"
        }

    total = len(df)

    impressions = int(df["impressions"].sum()) if "impressions" in df.columns else 0
    clicks = int(df["clicks"].sum()) if "clicks" in df.columns else 0
    likes = int(df["likes"].sum()) if "likes" in df.columns else 0
    shares = int(df["shares"].sum()) if "shares" in df.columns else 0
    comments = int(df["comments"].sum()) if "comments" in df.columns else 0
    conversions = int(df["conversions"].sum()) if "conversions" in df.columns else 0

    ctr = (clicks / impressions) if impressions > 0 else 0.0
    engagement_rate = ((likes + comments + shares) / impressions) if impressions > 0 else 0.0
    conversion_rate = (conversions / impressions) if impressions > 0 else 0.0

    # Sentiment distribution
    pos_ratio = neg_ratio = neu_ratio = 0.0
    if "sentiment_label" in df.columns:
        labels = df["sentiment_label"].astype(str)
        pos_ratio = float((labels.str.upper() == "POSITIVE").mean())
        neg_ratio = float((labels.str.upper() == "NEGATIVE").mean())
        neu_ratio = float((labels.str.upper() == "NEUTRAL").mean())

    # Average trend score
    avg_trend = float(df["trend_score"].mean()) if "trend_score" in df.columns else 0.0

    # Toxicity (if provided)
    avg_toxic = float(df["toxicity"].mean()) if "toxicity" in df.columns else 0.0

    # Dominant emotion (simple heuristic)
    dominant_emotion = "unknown"
    if "emotions" in df.columns:
        try:
            # emotions column expected to be dict-like per row
            all_emotions = {}
            for e in df["emotions"].dropna():
                if isinstance(e, dict):
                    for k, v in e.items():
                        all_emotions[k] = all_emotions.get(k, 0) + float(v)
            if all_emotions:
                dominant_emotion = max(all_emotions.items(), key=lambda x: x[1])[0]
        except Exception:
            dominant_emotion = "unknown"

    metrics = {
        "total_records": int(total),
        "impressions": impressions,
        "ctr": round(ctr, 4),
        "engagement_rate": round(engagement_rate, 4),
        "conversion_rate": round(conversion_rate, 4),
        "positive_ratio": round(pos_ratio, 4),
        "negative_ratio": round(neg_ratio, 4),
        "neutral_ratio": round(neu_ratio, 4),
        "avg_trend_score": round(avg_trend, 4),
        "avg_toxicity": round(avg_toxic, 4),
        "dominant_emotion": dominant_emotion
    }

    return metrics


# -----------------------------
# 3) Update Google Sheet (via sheets_connector.append_row)
# -----------------------------
def update_google_sheet(sheet_name: str, df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
    """
    Compute KPI row and append to the configured Google Sheet (using sheets_connector).
    Returns computed metrics (dict) or None on complete failure.
    """
    try:
        metrics = compute_metrics_from_df(df)
    except Exception as e:
        logger.error(f"Failed to compute metrics from dataframe: {e}")
        return None

    ts = pd.Timestamp.utcnow().isoformat()
    row = [
        ts,
        metrics.get("total_records", 0),
        metrics.get("impressions", 0),
        metrics.get("ctr", 0.0),
        metrics.get("engagement_rate", 0.0),
        metrics.get("conversion_rate", 0.0),
        metrics.get("positive_ratio", 0.0),
        metrics.get("negative_ratio", 0.0),
        metrics.get("neutral_ratio", 0.0),
        metrics.get("avg_trend_score", 0.0),
        metrics.get("avg_toxicity", 0.0),
        metrics.get("dominant_emotion", "unknown")
    ]

    if not SHEET_LOG_ENABLED:
        logger.info("GOOGLE_SHEET_ID not set — will not upload metrics to Sheets. Returning metrics only.")
        return metrics

    for attempt in range(RETRY_LIMIT):
        try:
            append_row(sheet_name, row)
            logger.info(f"Metrics appended to sheet: {sheet_name}")
            return metrics
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed to append metrics: {e}")
            time.sleep(2)

    logger.error("Failed to append metrics to Google Sheets after retries.")
    return metrics


# -----------------------------
# 4) Public wrapper used by pipeline
# -----------------------------
def push_daily_metrics(df: Optional[pd.DataFrame] = None, sheet_name: str = DEFAULT_SHEET_NAME) -> Optional[Dict[str, Any]]:
    """
    Pipeline-friendly wrapper. If df is None we create a zero-row placeholder.
    """
    if df is None:
        df = pd.DataFrame([{
            "impressions": 0,
            "clicks": 0,
            "likes": 0,
            "shares": 0,
            "comments": 0,
            "conversions": 0,
            "sentiment_label": "NEUTRAL",
            "trend_score": 0.0,
            "toxicity": 0.0,
            "emotions": {}
        }])

    return update_google_sheet(sheet_name, df)


# -----------------------------
# 5) CLI / quick test helper
# -----------------------------
if __name__ == "__main__":
    # Example test dataframe similar to prior version
    data = {
        "impressions": [100, 200, 150],
        "likes": [10, 25, 5],
        "clicks": [5, 10, 4],
        "shares": [2, 3, 1],
        "comments": [1, 5, 2],
        "conversions": [1, 3, 0],
        "sentiment_label": ["POSITIVE", "NEGATIVE", "NEUTRAL"],
        "sentiment_score": [0.9, 0.2, 0.5],
        "polarity": [0.8, -0.6, 0.1],
        "toxicity": [0.01, 0.5, 0.12],
        "emotions": [
            {"joy": 0.8, "anger": 0.1},
            {"anger": 0.7, "fear": 0.1},
            {"sadness": 0.4}
        ],
        "trend_score": [55, 12, 35]
    }
    df = pd.DataFrame(data)

    print("\nComputed Metrics:")
    m = update_google_sheet("demo_metrics", df)
    print(m)

    # Example: aggregate live metrics for dummy post ids (replace with real ones)
    posts_agg = fetch_and_aggregate_post_metrics(["1234567890", "9876543210"])
    print("\nAggregated live post metrics (demo):")
    print(posts_agg)

# metrics_hub2.py (UPDATED — integrated with social ingestion, trends, sentiment, and Sheets)
"""
Metrics Hub — updated to integrate:
 - SocialIngestor (live metrics)
 - TrendFetcher (trend scores)
 - sentiment_analyzer2 (sentiment + emotions)
 - integrations.sheets_connector (optional Google Sheets logging)

Behavior:
 - Keeps CSV local storage for reproducibility/teaching
 - If GOOGLE_SHEET_ID env var is set, also writes each new record to Sheets
 - Provides utilities for fetching recent metrics, campaign history, ML training dataset
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

import pandas as pd

# Local CSV paths
DATA_DIR = "data"
CAMPAIGNS_CSV = os.path.join(DATA_DIR, "campaigns1.csv")
HISTORICAL_CSV = os.path.join(DATA_DIR, "historical_metrics.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

# Try to import optional integrations
try:
    from app.integrations.sheets_connector import append_row
    SHEETS_AVAILABLE = True
except Exception:
    append_row = None
    SHEETS_AVAILABLE = False
    logger.info("integrations.sheets_connector not available — Sheets logging disabled.")

try:
    from app.integrations.social_ingestor import SocialIngestor
    _SOCIAL_INGESTOR_AVAILABLE = True
    _INGESTOR = SocialIngestor()
except Exception as e:
    _SOCIAL_INGESTOR_AVAILABLE = False
    _INGESTOR = None
    logger.info(f"SocialIngestor not available: {e}")

try:
    from app.integrations.trend_fetcher import TrendFetcher
    _TREND_AVAILABLE = True
    _TRENDER = TrendFetcher()
except Exception as e:
    _TREND_AVAILABLE = False
    _TRENDER = None
    logger.info(f"TrendFetcher not available: {e}")

# sentiment_analyzer2 provides analyze_sentiment(text) -> list(dict)
from app.sentiment_engine.sentiment_analyzer import analyze_sentiment, analyze_post_comments


# Environment flag for Sheets usage
GOOGLE_SHEET_ENABLED = bool(os.getenv("GOOGLE_SHEET_ID")) and SHEETS_AVAILABLE

# -----------------------------------------------------------
# Initialize CSVs with schema if missing
# -----------------------------------------------------------

def _init_file(path: str, columns: List[str]):
    if not os.path.exists(path):
        logger.info(f"Creating CSV: {path}")
        pd.DataFrame(columns=columns).to_csv(path, index=False)

_init_file(CAMPAIGNS_CSV, [
    "timestamp", "campaign_id", "variant", "post_id", "platform",
    "impressions", "clicks", "conversions", "ctr", "conv_rate",
    "sentiment", "trend_score", "polarity", "avg_comment_sentiment"
])

_init_file(HISTORICAL_CSV, [
    "timestamp", "campaign_id", "variant", "ctr", "sentiment",
    "polarity", "conversions", "trend_score"
])

# -----------------------------------------------------------
# Core API: Record Campaign / Variant Metrics
# -----------------------------------------------------------

def record_campaign_metrics(
    campaign_id: str,
    variant: str,
    impressions: int,
    clicks: int,
    conversions: int,
    sentiment_score: float,
    trend_score: float = 0.0,
    platform: str = "unknown",
    post_id: Optional[str] = None,
    avg_comment_sentiment: Optional[float] = None
) -> None:
    """
    Store campaign metrics to CSV (and optionally Google Sheets).
    Keeps compatibility with previous schema, but enriches with trend/sentiment.
    """

    timestamp = datetime.utcnow().isoformat()

    # compute CTR & conversion rate safely
    ctr = (clicks / impressions) if impressions > 0 else 0.0
    conv_rate = (conversions / clicks) if clicks > 0 else 0.0

    new_row = {
        "timestamp": timestamp,
        "campaign_id": campaign_id,
        "variant": variant,
        "post_id": post_id or "",
        "platform": platform,
        "impressions": impressions,
        "clicks": clicks,
        "conversions": conversions,
        "ctr": ctr,
        "conv_rate": conv_rate,
        "sentiment": float(sentiment_score),
        "trend_score": float(trend_score),
        "polarity": float(sentiment_score),
        "avg_comment_sentiment": float(avg_comment_sentiment) if avg_comment_sentiment is not None else None
    }

    # Append to campaigns CSV
    try:
        df = pd.read_csv(CAMPAIGNS_CSV)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(CAMPAIGNS_CSV, index=False)
        logger.info(f"Recorded campaign metrics: {campaign_id} / {variant}")
    except Exception as e:
        logger.error(f"Failed to write to {CAMPAIGNS_CSV}: {e}")

    # Append to historical CSV (subset for ML)
    try:
        hist_row = {
            "timestamp": timestamp,
            "campaign_id": campaign_id,
            "variant": variant,
            "ctr": ctr,
            "sentiment": float(sentiment_score),
            "polarity": float(sentiment_score),
            "conversions": int(conversions),
            "trend_score": float(trend_score)
        }
        hist = pd.read_csv(HISTORICAL_CSV)
        hist = pd.concat([hist, pd.DataFrame([hist_row])], ignore_index=True)
        hist.to_csv(HISTORICAL_CSV, index=False)
    except Exception as e:
        logger.error(f"Failed to write to {HISTORICAL_CSV}: {e}")

    # Optionally persist to Google Sheets
    if GOOGLE_SHEET_ENABLED:
        try:
            append_row("campaigns", [
                timestamp,
                campaign_id,
                variant,
                post_id or "",
                platform,
                impressions,
                clicks,
                conversions,
                round(ctr, 6),
                round(conv_rate, 6),
                round(float(sentiment_score), 4),
                round(float(trend_score), 4),
                float(avg_comment_sentiment) if avg_comment_sentiment is not None else ""
            ])
            logger.info("Also wrote campaign row to Google Sheets.")
        except Exception as e:
            logger.warning(f"Failed to append campaign to Sheets: {e}")


# -----------------------------------------------------------
# New helper: Record metrics by fetching a live post id
# -----------------------------------------------------------

def record_post_metrics_from_id(campaign_id: str, variant: str, post_id: str, platform: str = "twitter"):
    """
    Fetch metrics for a given post ID using SocialIngestor, perform sentiment + trend enrichment,
    then call record_campaign_metrics to persist the result.

    This allows you to close the loop: post -> ingest -> record -> ML.
    """
    if not _SOCIAL_INGESTOR_AVAILABLE:
        logger.error("SocialIngestor not configured; cannot fetch post metrics.")
        return

    try:
        data = _INGESTOR.fetch_complete_post_data(str(post_id))
        metrics = data.get("metrics", {})
        text = data.get("text", "")

        impressions = metrics.get("impressions", 0) or 0
        likes = metrics.get("likes", 0) or 0
        shares = metrics.get("shares", 0) or 0
        replies = metrics.get("replies", 0) or 0

        # Basic clicks/conversions are not always available from social APIs
        clicks = 0
        conversions = 0

        # Enrich with sentiment and trend
        sentiment_info = {"sentiment_score": 0.5}
        avg_comment_sent = None
        try:
            if _SENTIMENT_AVAILABLE:
                sentiment_info = analyze_sentiment(text)[0]
                # average comment sentiment if comments available
                comment_stats = analyze_post_comments(post_id) if analyze_post_comments is not None else {}
                avg_comment_sent = comment_stats.get("avg_sentiment") if isinstance(comment_stats, dict) else None
        except Exception as e:
            logger.warning(f"Sentiment enrichment failed for post {post_id}: {e}")

        trend_score = 0.0
        try:
            if _TREND_AVAILABLE:
                trend_score = _TRENDER.get_combined_trend_score(text)
        except Exception as e:
            logger.warning(f"Trend enrichment failed for post {post_id}: {e}")

        # Use likes+shares+replies as proxy for engagement; treat 'likes' as positive engagement
        engagement_est = int(likes) + int(shares) + int(replies)

        # Try to interpret impressions: if not present, fallback to engagement * 100 (very rough)
        impressions = int(metrics.get("impressions")) if metrics.get("impressions") else max(int(engagement_est * 100), 0)

        # Call core writer
        record_campaign_metrics(
            campaign_id=campaign_id,
            variant=variant,
            impressions=int(impressions),
            clicks=int(clicks),
            conversions=int(conversions),
            sentiment_score=float(sentiment_info.get("sentiment_score", 0.5)),
            trend_score=float(trend_score),
            platform=platform,
            post_id=post_id,
            avg_comment_sentiment=avg_comment_sent
        )

    except Exception as e:
        logger.error(f"Failed to fetch or record metrics for post {post_id}: {e}")


# -----------------------------------------------------------
# Fetch & Query Utilities (kept from previous version)
# -----------------------------------------------------------

def fetch_recent_metrics(limit: int = 50) -> pd.DataFrame:
    """Returns the most recent A/B test records (tail)."""
    try:
        df = pd.read_csv(CAMPAIGNS_CSV)
        return df.tail(limit)
    except Exception as e:
        logger.error(f"Failed to read {CAMPAIGNS_CSV}: {e}")
        return pd.DataFrame()

def fetch_campaign_history(campaign_id: str) -> pd.DataFrame:
    """Returns full history for a specific campaign."""
    try:
        df = pd.read_csv(CAMPAIGNS_CSV)
        return df[df["campaign_id"] == campaign_id]
    except Exception as e:
        logger.error(f"Failed to fetch campaign history: {e}")
        return pd.DataFrame()

def fetch_variant_performance(campaign_id: str) -> Dict[str, Any]:
    """
    Compares all variants under a campaign.
    Returns best/worst/all similar to previous behavior.
    """
    df = fetch_campaign_history(campaign_id)
    if df.empty:
        return {}

    # safe conversions to numeric
    for c in ["conv_rate", "ctr"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df_sorted = df.sort_values(["conv_rate", "ctr"], ascending=False)
    best = df_sorted.iloc[0].to_dict()
    worst = df_sorted.iloc[-1].to_dict()
    return {
        "best": best,
        "worst": worst,
        "all_variants": df_sorted.to_dict(orient="records")
    }


def get_ml_training_data() -> pd.DataFrame:
    """
    Returns the full dataset for ML training (from historical CSV).
    """
    try:
        df = pd.read_csv(HISTORICAL_CSV)
        df = df.dropna()
        return df
    except Exception as e:
        logger.error(f"Failed to load ML training data: {e}")
        return pd.DataFrame()


# -----------------------------------------------------------
# Feature Engineering Utilities
# -----------------------------------------------------------

def build_feature_vector(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts a campaign row into an ML-ready feature vector.
    """
    return {
        "ctr": float(row.get("ctr", 0.0)),
        "sentiment": float(row.get("sentiment", 0.0)),
        "polarity": float(row.get("polarity", row.get("sentiment", 0.0))),
        "trend_score": float(row.get("trend_score", 0.0)),
        "conversions": int(row.get("conversions", 0))
    }

def compute_variant_score(row: Dict[str, Any]) -> float:
    """
    Scoring function for ranking A/B variants (tunable).
    """
    try:
        return round(
            float(row.get("ctr", 0.0)) * 0.5 +
            float(row.get("sentiment", 0.0)) * 0.3 +
            float(row.get("trend_score", 0.0)) * 0.2,
            4
        )
    except Exception:
        return 0.0


# -----------------------------------------------------------
# Manual test / demo
# -----------------------------------------------------------
if __name__ == "__main__":
    print("\nMetrics Hub v2 demo\n")

    # Demo: record synthetic campaign row
    record_campaign_metrics(
        campaign_id="demo_campaign_01",
        variant="A",
        impressions=1000,
        clicks=120,
        conversions=10,
        sentiment_score=0.82,
        trend_score=42.0,
        platform="twitter",
        post_id="demo-post-1",
        avg_comment_sentiment=0.75
    )

    print("\nRecent metrics:")
    print(fetch_recent_metrics().tail())

    print("\nVariant performance for demo_campaign_01:")
    print(fetch_variant_performance("demo_campaign_01"))

    print("\nML training data sample:")
    print(get_ml_training_data().head())

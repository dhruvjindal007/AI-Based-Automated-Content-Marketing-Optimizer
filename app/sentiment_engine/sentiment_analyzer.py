# sentiment_analyzer2.py (UPDATED FULL VERSION)
"""
Advanced sentiment_analyzer2.py

Upgraded with:
---------------------------------
1. Real social comment ingestion via SocialIngestor
2. Trend awareness using TrendFetcher
3. Google Sheets logging for sentiment results
4. Unified output for pipeline integration (generator → optimizer → metrics)
5. Strong fallbacks (HF → TextBlob)
6. Student-friendly readable structure
"""

import logging
from typing import List, Union, Dict

from textblob import TextBlob

# HuggingFace pipeline
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except:
    HF_AVAILABLE = False

# Language detection
try:
    from langdetect import detect
    LANG_AVAILABLE = True
except:
    LANG_AVAILABLE = False

# New Integrations
from app.integrations.social_ingestor import SocialIngestor
from app.integrations.trend_fetcher import TrendFetcher
from app.integrations.sheets_connector import append_row

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Lazy-loaded HF models
_senti_model = None
_emotion_model = None


# ------------------------------------------------------
# Initialize Models
# ------------------------------------------------------
def _init_sentiment_model():
    return pipeline("sentiment-analysis")

def _init_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )


# ------------------------------------------------------
# Utilities
# ------------------------------------------------------
def detect_language(text: str) -> str:
    if not LANG_AVAILABLE:
        return "unknown"
    try:
        return detect(text)
    except Exception:
        return "unknown"


def fallback_sentiment(text: str) -> Dict:
    polarity = TextBlob(text).sentiment.polarity
    if polarity >= 0.05:
        label = "POSITIVE"
    elif polarity <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return {
        "label": label,
        "score": abs(polarity),
        "polarity": polarity
    }


def simplify_emotion_output(raw_output: List[Dict]) -> Dict:
    return {x["label"]: float(x["score"]) for x in raw_output}


# ------------------------------------------------------
# NEW FEATURE: Analyze sentiment of *live social comments*
# ------------------------------------------------------
def analyze_post_comments(post_id: str) -> Dict:
    """
    Fetches comments using SocialIngestor → scores them →
    returns aggregated sentiment & toxicity.
    """
    ingestor = SocialIngestor()
    comments = ingestor.fetch_post_comments(post_id)

    if not comments:
        return {
            "post_id": post_id,
            "avg_sentiment": 0.5,
            "avg_polarity": 0.0,
            "avg_toxicity": 0.0,
            "labels": {},
            "samples": []
        }

    results = analyze_sentiment(comments)
    labels = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

    for r in results:
        labels[r["sentiment_label"]] += 1

    avg_sent = sum(r["sentiment_score"] for r in results) / len(results)
    avg_pol = sum(r["polarity"] for r in results) / len(results)

    # Toxicity (from emotions if available)
    avg_toxic = 0.0
    for r in results:
        if "anger" in r["emotions"]:
            avg_toxic += r["emotions"].get("anger", 0)
    avg_toxic /= len(results)

    # Log to Google Sheets
    try:
        append_row("comment_sentiment", [
            post_id,
            avg_sent,
            avg_pol,
            avg_toxic,
            labels
        ])
    except:
        pass

    return {
        "post_id": post_id,
        "avg_sentiment": round(avg_sent, 4),
        "avg_polarity": round(avg_pol, 4),
        "avg_toxicity": round(avg_toxic, 4),
        "labels": labels,
        "samples": results
    }


# ------------------------------------------------------
# MASTER FUNCTION — sentiment + emotion + trend awareness
# ------------------------------------------------------
def analyze_sentiment(texts: Union[str, List[str]]) -> List[Dict]:
    """
    Returns list of:
    {
        "text": ...,
        "sentiment_label": ...,
        "sentiment_score": ...,
        "polarity": ...,
        "emotions": {joy: 0.2, ...},
        "language": ...,
        "trend_score": ...   <-- NEW
    }
    """

    if isinstance(texts, str):
        texts = [texts]

    global _senti_model, _emotion_model

    # Load models once
    if HF_AVAILABLE:
        if _senti_model is None:
            _senti_model = _init_sentiment_model()
        if _emotion_model is None:
            _emotion_model = _init_emotion_model()

    trend_engine = TrendFetcher()
    results = []

    for text in texts:
        lang = detect_language(text)

        # SENTIMENT
        if HF_AVAILABLE:
            try:
                pred = _senti_model(text)[0]
                label = pred["label"].upper()
                score = float(pred["score"])
                polarity = TextBlob(text).sentiment.polarity
            except:
                s = fallback_sentiment(text)
                label, score, polarity = s["label"], s["score"], s["polarity"]
        else:
            s = fallback_sentiment(text)
            label, score, polarity = s["label"], s["score"], s["polarity"]

        if label.startswith("NEG"):
            norm_score = 1 - score
        else:
            norm_score = score

        # EMOTION
        emotions = {}
        if HF_AVAILABLE:
            try:
                emo_raw = _emotion_model(text)[0]
                emotions = simplify_emotion_output(emo_raw)
            except:
                emotions = {}

        # TREND SCORE (NEW)
        trend_score = trend_engine.get_combined_trend_score(text)

        entry = {
            "text": text,
            "sentiment_label": label,
            "sentiment_score": round(norm_score, 4),
            "polarity": polarity,
            "emotions": emotions,
            "language": lang,
            "trend_score": trend_score   # <-- integrated
        }

        # Save to Google Sheets
        try:
            append_row("sentiment_results", [
                text[:80] + "...",
                label,
                norm_score,
                polarity,
                trend_score
            ])
        except:
            pass

        results.append(entry)

    return results


# ------------------------------------------------------
# DataFrame helper
# ------------------------------------------------------
def analyze_from_dataframe(df, text_column: str):
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    out = analyze_sentiment(df[text_column].tolist())

    df["sentiment_label"] = [r["sentiment_label"] for r in out]
    df["sentiment_score"] = [r["sentiment_score"] for r in out]
    df["polarity"] = [r["polarity"] for r in out]
    df["emotions"] = [r["emotions"] for r in out]
    df["language"] = [r["language"] for r in out]
    df["trend_score"] = [r["trend_score"] for r in out]

    return df


# ------------------------------------------------------
# Test Run
# ------------------------------------------------------
if __name__ == "__main__":
    sample = [
        "I absolutely love this AI tool!",
        "This is frustrating and disappointing.",
        "Not sure if this is good or bad 😂"
    ]

    out = analyze_sentiment(sample)
    for r in out:
        print(r)

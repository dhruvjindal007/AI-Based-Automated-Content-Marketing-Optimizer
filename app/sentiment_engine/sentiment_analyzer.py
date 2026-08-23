# sentiment_analyzer.py (UPDATED FULL VERSION)
"""
Advanced sentiment_analyzer.py

Upgraded with:
---------------------------------
1. Real social comment ingestion via SocialIngestor
2. Trend awareness using TrendFetcher
3. Google Sheets logging for sentiment results
4. Unified output for pipeline integration (generator -> optimizer -> metrics)
5. Strong fallbacks (HF -> TextBlob)
6. Student-friendly readable structure
"""

import json
import logging
from typing import List, Union, Dict

from textblob import TextBlob

# HuggingFace pipeline
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# Language detection
try:
    from langdetect import detect
    LANG_AVAILABLE = True
except Exception:
    LANG_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(_handler)

# ------------------------------------------------------
# Optional integrations (Fix #5: soft-degrade instead of hard-crash)
# ------------------------------------------------------
try:
    from app.integrations.social_ingestor import SocialIngestor
    SOCIAL_AVAILABLE = True
except Exception as e:
    SOCIAL_AVAILABLE = False
    logger.warning("SocialIngestor unavailable, comment ingestion disabled: %s", e)

try:
    from app.integrations.trend_fetcher import TrendFetcher
    TREND_AVAILABLE = True
except Exception as e:
    TREND_AVAILABLE = False
    logger.warning("TrendFetcher unavailable, trend_score will be None: %s", e)

try:
    from app.integrations.sheets_connector import append_row
    SHEETS_AVAILABLE = True
except Exception as e:
    SHEETS_AVAILABLE = False
    logger.warning("Sheets connector unavailable, logging to Sheets disabled: %s", e)

# Lazy-loaded HF models
_senti_model = None
_emotion_model = None

# Fix #6: module-level singletons instead of per-call instantiation
_trend_engine = TrendFetcher() if TREND_AVAILABLE else None
_ingestor = SocialIngestor() if SOCIAL_AVAILABLE else None

VALID_LABELS = ("POSITIVE", "NEGATIVE", "NEUTRAL")


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
    except Exception as e:
        logger.warning("Language detection failed: %s", e)
        return "unknown"


def fallback_sentiment(text: str) -> Dict:
    """
    TextBlob-based fallback. Fix #4: score is normalized onto the same
    0-1 "positivity" scale as the HF path (polarity in [-1, 1] -> [0, 1]),
    rather than using abs(polarity) as if it were a confidence value.
    A neutral text (polarity ~ 0) now lands near 0.5, not near 0.
    """
    polarity = TextBlob(text).sentiment.polarity
    if polarity >= 0.05:
        label = "POSITIVE"
    elif polarity <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    norm_score = (polarity + 1) / 2  # -1..1 -> 0..1

    return {
        "label": label,
        "score": norm_score,
        "polarity": polarity
    }


def simplify_emotion_output(raw_output: List[Dict]) -> Dict:
    return {x["label"]: float(x["score"]) for x in raw_output}


def _safe_append_row(sheet: str, row: List) -> None:
    """Fix #2: log failures instead of swallowing them silently."""
    if not SHEETS_AVAILABLE:
        return
    try:
        append_row(sheet, row)
    except Exception as e:
        logger.warning("Failed to append row to sheet '%s': %s", sheet, e)


def _normalize_comment(c) -> str:
    """Fix #9: SocialIngestor may return dicts (author/timestamp/text) rather
    than plain strings. Normalize defensively instead of str(dict)-ing it."""
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        for key in ("text", "comment", "body", "content"):
            if key in c and isinstance(c[key], str):
                return c[key]
        logger.warning("Comment dict had no recognizable text field, coercing to str: %s", c)
        return str(c)
    return str(c)


# ------------------------------------------------------
# NEW FEATURE: Analyze sentiment of *live social comments*
# ------------------------------------------------------
def analyze_post_comments(post_id: str) -> Dict:
    """
    Fetches comments using SocialIngestor -> scores them ->
    returns aggregated sentiment & toxicity.
    """
    if not SOCIAL_AVAILABLE or _ingestor is None:
        logger.warning("analyze_post_comments called but SocialIngestor is unavailable.")
        return {
            "post_id": post_id,
            "avg_sentiment": 0.5,
            "avg_polarity": 0.0,
            "avg_toxicity": 0.0,
            "labels": {},
            "samples": []
        }

    raw_comments = _ingestor.fetch_post_comments(post_id)

    if not raw_comments:
        return {
            "post_id": post_id,
            "avg_sentiment": 0.5,
            "avg_polarity": 0.0,
            "avg_toxicity": 0.0,
            "labels": {},
            "samples": []
        }

    comments = [_normalize_comment(c) for c in raw_comments]
    results = analyze_sentiment(comments)

    # Fix #10: don't KeyError on an unexpected label; bucket it instead
    labels = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for r in results:
        lbl = r["sentiment_label"]
        if lbl not in labels:
            logger.warning("Unexpected sentiment_label '%s', bucketing as NEUTRAL", lbl)
            lbl = "NEUTRAL"
        labels[lbl] += 1

    avg_sent = sum(r["sentiment_score"] for r in results) / len(results)
    avg_pol = sum(r["polarity"] for r in results) / len(results)

    avg_toxic = 0.0
    for r in results:
        avg_toxic += r["emotions"].get("anger", 0)
    avg_toxic /= len(results)

    # Fix #7: dicts must be serialized before writing to a Sheets row
    _safe_append_row("comment_sentiment", [
        post_id,
        avg_sent,
        avg_pol,
        avg_toxic,
        json.dumps(labels)
    ])

    return {
        "post_id": post_id,
        "avg_sentiment": round(avg_sent, 4),
        "avg_polarity": round(avg_pol, 4),
        "avg_toxicity": round(avg_toxic, 4),
        "labels": labels,
        "samples": results
    }


# ------------------------------------------------------
# MASTER FUNCTION - sentiment + emotion + trend awareness
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
        "trend_score": ...   # None if TrendFetcher is unavailable
    }

    Fix #1: this is the single place trend_score is computed. Downstream
    callers (e.g. tracker3.push_raw_feedback) should read
    result["trend_score"] instead of calling TrendFetcher again themselves.
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

    results = []

    for text in texts:
        lang = detect_language(text)

        # SENTIMENT
        used_hf = False
        if HF_AVAILABLE:
            try:
                pred = _senti_model(text)[0]
                label = pred["label"].upper()
                score = float(pred["score"])
                polarity = TextBlob(text).sentiment.polarity
                used_hf = True
            except Exception as e:
                logger.warning("HF sentiment model failed, falling back to TextBlob: %s", e)
                s = fallback_sentiment(text)
                label, score, polarity = s["label"], s["score"], s["polarity"]
        else:
            s = fallback_sentiment(text)
            label, score, polarity = s["label"], s["score"], s["polarity"]

        # Fix #4: the (1 - score) inversion only makes sense for a genuine
        # HF softmax confidence. fallback_sentiment() already returns a
        # normalized 0-1 positivity score, so it passes through unchanged.
        if used_hf:
            norm_score = (1 - score) if label.startswith("NEG") else score
        else:
            norm_score = score

        # EMOTION
        emotions = {}
        if HF_AVAILABLE:
            try:
                emo_raw = _emotion_model(text)[0]
                emotions = simplify_emotion_output(emo_raw)
            except Exception as e:
                logger.warning("HF emotion model failed: %s", e)
                emotions = {}

        # TREND SCORE
        trend_score = None
        if TREND_AVAILABLE and _trend_engine is not None:
            try:
                trend_score = _trend_engine.get_combined_trend_score(text)
            except Exception as e:
                logger.warning("TrendFetcher failed for text: %s", e)

        entry = {
            "text": text,
            "sentiment_label": label,
            "sentiment_score": round(norm_score, 4),
            "polarity": polarity,
            "emotions": emotions,
            "language": lang,
            "trend_score": trend_score
        }

        # Fix #8: only append "..." when actually truncated
        preview = text if len(text) <= 80 else text[:80] + "..."

        _safe_append_row("sentiment_results", [
            preview,
            label,
            norm_score,
            polarity,
            trend_score
        ])

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
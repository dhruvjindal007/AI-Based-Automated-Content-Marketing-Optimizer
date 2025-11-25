# trend_fetcher.py
# -------------------------------------------------------------
# Open-Source / Free Trend Data Fetching Module
# -------------------------------------------------------------
# This module provides trend data WITHOUT using paid APIs.
# All data sources are 100% free, open-source, or freemium.
#
# Trend Sources Used:
# 1. Google Trends (PyTrends - Free)
# 2. Reddit Trending Topics (PRAW - Free with basic API keys)
# 3. Keyword Extraction using spaCy (Open-source)
# -------------------------------------------------------------

import os
import logging
from typing import List, Dict, Any

# ---------- Google Trends (Free) ----------
from pytrends.request import TrendReq

# ---------- Reddit Trending (Free) ----------
import praw

# ---------- Keyword Extraction (Open Source) ----------
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


class TrendFetcher:
    """
    TrendFetcher collects trending signals from:
      - Google Trends
      - Reddit Discussions
    and generates a unified trend score.
    """

    def __init__(self):
        logging.info("Initializing TrendFetcher...")

        # Google Trends client
        try:
            self.pytrends = TrendReq(hl='en-US', tz=330)
            logging.info("PyTrends connected successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize PyTrends: {e}")
            self.pytrends = None

        # Reddit client (Free Tier)
        try:
            self.reddit = praw.Reddit(
                client_id=os.getenv("REDDIT_CLIENT_ID"),
                client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                user_agent="TrendFetcher/1.0 (EducationProject)"
            )
            logging.info("Reddit API connected successfully.")
        except Exception as e:
            logging.warning(f"Reddit API not configured: {e}")
            self.reddit = None

        # Load spaCy model for keyword extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            # fallback: install model instruction
            logging.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None

    # -------------------------------------------------------------
    #   1. Extract Keywords from Text
    # -------------------------------------------------------------
    def extract_keywords(self, text: str) -> List[str]:
        if not self.nlp:
            return []

        doc = self.nlp(text.lower())
        keywords = []

        for token in doc:
            if token.is_stop or token.is_punct:
                continue
            if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
                if token.text not in STOP_WORDS:
                    keywords.append(token.text)

        return list(set(keywords))[:5]  # limit to top 5

    # -------------------------------------------------------------
    #   2. Google Trends Score (Free)
    # -------------------------------------------------------------
    def fetch_google_trend_score(self, keyword: str) -> int:
        if not self.pytrends:
            return 0

        try:
            self.pytrends.build_payload([keyword])
            df = self.pytrends.interest_over_time()

            if df.empty:
                return 0

            # latest trend value
            return int(df[keyword].iloc[-1])
        except Exception:
            return 0

    # -------------------------------------------------------------
    #   3. Google Rising Related Queries (Free)
    # -------------------------------------------------------------
    def fetch_google_rising_queries(self, keyword: str) -> List[str]:
        if not self.pytrends:
            return []

        try:
            self.pytrends.build_payload([keyword])
            related = self.pytrends.related_queries()

            if keyword not in related or "rising" not in related[keyword]:
                return []

            df_rising = related[keyword]["rising"]
            return list(df_rising["query"].head(5))
        except Exception:
            return []

    # -------------------------------------------------------------
    #   4. Global Trending Searches (Free)
    # -------------------------------------------------------------
    def fetch_google_global_trends(self) -> List[str]:
        if not self.pytrends:
            return []
        try:
            df = self.pytrends.trending_searches()
            return list(df[0].head(10))
        except Exception:
            return []

    # -------------------------------------------------------------
    #   5. Reddit Hot Topics (Free)
    # -------------------------------------------------------------
    def fetch_reddit_trending(self, subreddit: str = "all") -> List[str]:
        if not self.reddit:
            return []
        try:
            posts = self.reddit.subreddit(subreddit).hot(limit=10)
            return [post.title for post in posts]
        except Exception:
            return []

    # -------------------------------------------------------------
    #   6. Normalize to 0 - 100 Trend Score
    # -------------------------------------------------------------
    def normalize_score(self, value: int) -> int:
        if not value:
            return 0
        try:
            return min(100, int(value))
        except Exception:
            return 0

    # -------------------------------------------------------------
    #   7. Combined Trend Score for Any Content
    # -------------------------------------------------------------
    def get_combined_trend_score(self, text: str) -> int:
        keywords = self.extract_keywords(text)
        if not keywords:
            return 0

        scores = []
        for kw in keywords:
            score = self.fetch_google_trend_score(kw)
            scores.append(score)

        if not scores:
            return 0

        avg_score = sum(scores) / len(scores)
        return self.normalize_score(avg_score)

    # -------------------------------------------------------------
    #   8. Fetch Suggestions for Content Optimization
    # -------------------------------------------------------------
    def get_trend_insights(self, text: str) -> Dict[str, Any]:
        keywords = self.extract_keywords(text)
        insights = {}

        for kw in keywords:
            insights[kw] = {
                "trend_score": self.fetch_google_trend_score(kw),
                "rising_queries": self.fetch_google_rising_queries(kw)
            }
        return insights


# -------------------------------------------------------------
# Example usage (for your students)
# -------------------------------------------------------------
if __name__ == "__main__":
    tf = TrendFetcher()
    sample_text = "AI marketing automation for small business growth"

    print("Keywords:", tf.extract_keywords(sample_text))
    print("Trend Score:", tf.get_combined_trend_score(sample_text))
    print("Insights:", tf.get_trend_insights(sample_text))
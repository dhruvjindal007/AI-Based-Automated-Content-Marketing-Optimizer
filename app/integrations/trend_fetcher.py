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
import time
import logging
from typing import List, Dict, Any, Optional

# ---------- Google Trends (Free) ----------
from pytrends.request import TrendReq

# ---------- Reddit Trending (Free) ----------
import praw

# ---------- Keyword Extraction (Open Source) ----------
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TrendFetcher")

# Weighting for the combined score when both sources are available.
# Must sum to 1.0.
GOOGLE_WEIGHT = 0.7
REDDIT_WEIGHT = 0.3


class TrendFetcher:
    """
    TrendFetcher collects trending signals from:
      - Google Trends (keyword interest, 0-100 scale)
      - Reddit (whether keywords show up in current hot posts)
    and blends them into a single 0-100 trend score.

    If Reddit credentials aren't configured, the combined score
    transparently degrades to Google-Trends-only (and says so via
    the `sources_used` field returned by get_combined_trend_score_detailed).
    """

    def __init__(self, max_retries: int = 3, backoff_seconds: float = 2.0):
        logger.info("Initializing TrendFetcher...")
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

        # Cache of ACTUAL per-keyword Google Trends results (score +
        # rising queries), keyed by keyword. NOTE: TrendReq is a single
        # stateful session -- build_payload() overwrites one shared
        # internal config each call, it does NOT remember per-keyword
        # state. So we must cache the real fetched data, not just a
        # "build_payload was called" flag -- caching the flag alone
        # caused stale/wrong results when a keyword was looked up again
        # after a *different* keyword's payload had since been built.
        self._google_cache: Dict[str, Dict[str, Any]] = {}

        # Google Trends client
        try:
            self.pytrends = TrendReq(hl="en-US", tz=330)
            logger.info("PyTrends connected successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize PyTrends: {e}")
            self.pytrends = None

        # Reddit client (Free Tier)
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        if not client_id or not client_secret:
            logger.warning(
                "REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set. "
                "Reddit features disabled -> combined score will fall back "
                "to Google Trends only."
            )
            self.reddit = None
        else:
            try:
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent="TrendFetcher/1.0 (EducationProject)",
                )
                logger.info("Reddit API connected successfully.")
            except Exception as e:
                logger.warning(f"Reddit API not configured: {e}")
                self.reddit = None

        # Load spaCy model for keyword extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            logger.warning(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm"
            )
            self.nlp = None

    # -------------------------------------------------------------
    #   Internal: retry wrapper for flaky PyTrends calls
    # -------------------------------------------------------------
    def _with_retry(self, fn, *args, **kwargs) -> Optional[Any]:
        """Call fn(*args, **kwargs), retrying with exponential backoff on
        failure. Returns None if all attempts fail (caller decides fallback)."""
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_err = e
                wait = self.backoff_seconds * attempt
                logger.warning(
                    f"{fn.__name__} failed (attempt {attempt}/{self.max_retries}): {e}. "
                    f"Retrying in {wait:.1f}s..."
                )
                time.sleep(wait)
        logger.error(f"{fn.__name__} failed after {self.max_retries} attempts: {last_err}")
        return None

    def _fetch_google_data(self, keyword: str) -> Dict[str, Any]:
        """
        Builds the payload for `keyword`, fetches BOTH interest_over_time
        and related_queries while that payload is live, and caches the
        actual results. This must build+fetch together in one go because
        TrendReq holds only one payload config at a time -- if you build
        for keyword A, then later build for keyword B, then come back and
        try to read data "for A" without rebuilding, you'll silently get
        B's data. Caching the real results (not a build_payload flag)
        avoids that trap entirely: once we have A's real numbers cached,
        we never need to touch the shared session state for A again.
        """
        if keyword in self._google_cache:
            return self._google_cache[keyword]

        result = {"trend_score": 0, "rising_queries": []}
        if not self.pytrends:
            self._google_cache[keyword] = result
            return result

        built = False
        for attempt in range(1, self.max_retries + 1):
            try:
                self.pytrends.build_payload([keyword])
                built = True
                break
            except Exception as e:
                wait = self.backoff_seconds * attempt
                logger.warning(
                    f"build_payload failed for '{keyword}' "
                    f"(attempt {attempt}/{self.max_retries}): {e}. Retrying in {wait:.1f}s..."
                )
                time.sleep(wait)

        if not built:
            logger.error(f"build_payload failed for '{keyword}' after {self.max_retries} attempts.")
            self._google_cache[keyword] = result
            return result

        # Fetch interest_over_time while this payload is the live one
        df = self._with_retry(self.pytrends.interest_over_time)
        if df is not None and not df.empty and keyword in df.columns:
            try:
                result["trend_score"] = int(df[keyword].iloc[-1])
            except Exception as e:
                logger.error(f"Failed to parse trend score for '{keyword}': {e}")

        # Fetch related_queries while this SAME payload is still live --
        # this is what the old code got wrong: it called build_payload
        # again in a separate method, doubling requests for no reason
        # since the payload from interest_over_time was still valid.
        related = self._with_retry(self.pytrends.related_queries)
        if related and keyword in related and related[keyword].get("rising") is not None:
            try:
                df_rising = related[keyword]["rising"]
                result["rising_queries"] = list(df_rising["query"].head(5))
            except Exception as e:
                logger.error(f"Failed to parse rising queries for '{keyword}': {e}")

        self._google_cache[keyword] = result
        return result

    # -------------------------------------------------------------
    #   1. Extract Keywords from Text
    # -------------------------------------------------------------
    def extract_keywords(self, text: str) -> List[str]:
        """Extract up to 5 keywords, preserving first-seen order
        (original used set() here, which randomizes results)."""
        if not self.nlp:
            return []

        doc = self.nlp(text.lower())
        keywords: List[str] = []
        seen = set()

        for token in doc:
            if token.is_stop or token.is_punct:
                continue
            if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
                if token.text not in STOP_WORDS and token.text not in seen:
                    keywords.append(token.text)
                    seen.add(token.text)
            if len(keywords) == 5:
                break

        return keywords

    # -------------------------------------------------------------
    #   2. Google Trends Score (Free)
    # -------------------------------------------------------------
    def fetch_google_trend_score(self, keyword: str) -> int:
        return self._fetch_google_data(keyword)["trend_score"]

    # -------------------------------------------------------------
    #   3. Google Rising Related Queries (Free)
    # -------------------------------------------------------------
    def fetch_google_rising_queries(self, keyword: str) -> List[str]:
        return self._fetch_google_data(keyword)["rising_queries"]

    # -------------------------------------------------------------
    #   4. Global Trending Searches (Free)
    # -------------------------------------------------------------
    def fetch_google_global_trends(self) -> List[str]:
        if not self.pytrends:
            return []

        df = self._with_retry(self.pytrends.trending_searches)
        if df is None or df.empty:
            return []

        try:
            return list(df[0].head(10))
        except Exception as e:
            logger.error(f"Failed to parse global trends: {e}")
            return []

    # -------------------------------------------------------------
    #   5. Reddit Hot Topics (Free)
    # -------------------------------------------------------------
    def fetch_reddit_trending(self, subreddit: str = "all", limit: int = 25) -> List[str]:
        if not self.reddit:
            return []
        try:
            posts = self.reddit.subreddit(subreddit).hot(limit=limit)
            return [post.title for post in posts]
        except Exception as e:
            logger.error(f"Reddit fetch failed for r/{subreddit}: {e}")
            return []

    # -------------------------------------------------------------
    #   6. Reddit-based keyword score (NEW - this is what makes
    #      Reddit an actual input to the combined score, not just
    #      a dead-end method)
    # -------------------------------------------------------------
    def fetch_reddit_keyword_score(
        self, keywords: List[str], subreddit: str = "all", limit: int = 25
    ) -> int:
        """
        Score 0-100 based on what fraction of current hot post titles
        in `subreddit` mention at least one of `keywords`.

        This is a simple presence/frequency signal, not a real popularity
        metric like Google Trends' index -- treat it as "is this topic
        being talked about on Reddit right now", not "how big is it".
        """
        if not keywords:
            return 0

        titles = self.fetch_reddit_trending(subreddit=subreddit, limit=limit)
        if not titles:
            return 0

        lowered_keywords = [kw.lower() for kw in keywords]
        matches = 0
        for title in titles:
            title_lower = title.lower()
            if any(kw in title_lower for kw in lowered_keywords):
                matches += 1

        return self.clamp_score((matches / len(titles)) * 100)

    # -------------------------------------------------------------
    #   7. Clamp score to 0 - 100
    # -------------------------------------------------------------
    def clamp_score(self, value: Optional[float]) -> int:
        """Clamps/guards a numeric score into the 0-100 range."""
        if value is None:
            return 0
        try:
            return max(0, min(100, int(round(value))))
        except (TypeError, ValueError):
            return 0

    # -------------------------------------------------------------
    #   8. Combined Trend Score for Any Content
    #      (NOW actually combines Google Trends + Reddit)
    # -------------------------------------------------------------
    def get_combined_trend_score(self, text: str, subreddit: str = "all") -> int:
        """
        Returns a single 0-100 score.

        If Reddit is configured: GOOGLE_WEIGHT * google_avg + REDDIT_WEIGHT * reddit_score
        If Reddit is NOT configured: falls back to Google Trends only,
        and this fallback is logged (not silently blended as if Reddit
        contributed).

        Use get_combined_trend_score_detailed() if you need to know which
        sources actually contributed, e.g. for a UI badge or a report.
        """
        return self.get_combined_trend_score_detailed(text, subreddit)["score"]

    def get_combined_trend_score_detailed(
        self, text: str, subreddit: str = "all"
    ) -> Dict[str, Any]:
        keywords = self.extract_keywords(text)
        if not keywords:
            return {"score": 0, "sources_used": [], "keywords": [], "google_avg": 0, "reddit_score": 0}

        google_scores = [self.fetch_google_trend_score(kw) for kw in keywords]
        google_scores = [s for s in google_scores if s]  # drop failed lookups
        google_avg = sum(google_scores) / len(google_scores) if google_scores else 0

        sources_used = []
        if google_scores:
            sources_used.append("google_trends")

        if self.reddit:
            reddit_score = self.fetch_reddit_keyword_score(keywords, subreddit=subreddit)
            sources_used.append("reddit")
        else:
            reddit_score = 0
            logger.info(
                "Reddit not configured -- combined score is Google Trends only "
                "for this call."
            )

        if "google_trends" in sources_used and "reddit" in sources_used:
            final_score = (GOOGLE_WEIGHT * google_avg) + (REDDIT_WEIGHT * reddit_score)
        elif "google_trends" in sources_used:
            final_score = google_avg
        elif "reddit" in sources_used:
            final_score = reddit_score
        else:
            final_score = 0

        return {
            "score": self.clamp_score(final_score),
            "sources_used": sources_used,
            "keywords": keywords,
            "google_avg": self.clamp_score(google_avg),
            "reddit_score": reddit_score,
        }

    # -------------------------------------------------------------
    #   9. Fetch Suggestions for Content Optimization
    # -------------------------------------------------------------
    def get_trend_insights(self, text: str) -> Dict[str, Any]:
        keywords = self.extract_keywords(text)
        insights = {}

        for kw in keywords:
            insights[kw] = {
                "trend_score": self.fetch_google_trend_score(kw),
                "rising_queries": self.fetch_google_rising_queries(kw),
            }
        return insights


# -------------------------------------------------------------
# Example usage (for your students)
# -------------------------------------------------------------
if __name__ == "__main__":
    tf = TrendFetcher()
    sample_text = "AI marketing automation for small business growth"

    print("Keywords:", tf.extract_keywords(sample_text))

    detailed = tf.get_combined_trend_score_detailed(sample_text)
    print("Combined Trend Score:", detailed["score"])
    print("Sources actually used:", detailed["sources_used"])
    print("Google avg:", detailed["google_avg"], "| Reddit score:", detailed["reddit_score"])

    print("Insights:", tf.get_trend_insights(sample_text))
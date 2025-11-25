# trend_based_optimizer3.py (UPDATED FULL VERSION)
import logging
from datetime import datetime

from app.integrations.trend_fetcher import TrendFetcher
from app.integrations.sheets_connector import append_row

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)


class TrendBasedOptimizer:
    """
    Trend-Based Optimizer
    ----------------------
    Enhances generated marketing content using real trend signals from:

        • Google Trends (PyTrends)
        • Reddit Hot Topics
        • Keyword Extraction (spaCy)

    Provides:
        - Trend score (0–100)
        - Trend insights (rising queries, related topics)
        - Optimized version of the content
        - Writes trend scoring rows to Google Sheets
    """

    def __init__(self):
        logger.info("Initializing TrendBasedOptimizer with TrendFetcher...")
        self.fetcher = TrendFetcher()

    # ----------------------------------------------------------------------
    # Generate Trend Score + Insights for a given content text
    # ----------------------------------------------------------------------
    def analyze_trends(self, text: str):
        """
        Returns:
            trend_score: int 0–100
            insights: dict of keyword → rising queries & score
        """
        trend_score = self.fetcher.get_combined_trend_score(text)
        insights = self.fetcher.get_trend_insights(text)

        return trend_score, insights

    # ----------------------------------------------------------------------
    # Core Optimization Logic
    # ----------------------------------------------------------------------
    def optimize_content(self, original_text: str) -> dict:
        """
        Takes the generated content and boosts it with trend-awareness.

        Returns dictionary:
            {
                "original": "...",
                "optimized": "...",
                "trend_score": 0-100,
                "insights": {...}
            }
        """

        trend_score, insights = self.analyze_trends(original_text)

        # Build optimization suggestions using extracted trends
        trending_keywords = list(insights.keys())
        rising_phrases = []
        for kw, detail in insights.items():
            rising = detail.get("rising_queries", [])
            if rising:
                rising_phrases.extend(rising)

        # --------------------------------------------------------
        # Apply optimization (Simple, explainable for students)
        # --------------------------------------------------------
        optimized_text = original_text

        # If no strong trends, minimal changes
        if trend_score < 20:
            optimized_text += "\n\n(✨ Tip: Consider adding more trending topics for better reach.)"

        else:
            # Inject trending keywords
            if trending_keywords:
                optimized_text += f"\n\n🔥 Trending Now: {', '.join(trending_keywords[:3])}"

            # Add rising search terms for extra SEO strength
            if rising_phrases:
                optimized_text += f"\n📈 People are searching for: {', '.join(rising_phrases[:5])}"

            # If score is high, boost call-to-action
            if trend_score > 60:
                optimized_text += "\n🚀 Ride the trend wave — publish this soon for maximum impact!"

        # Log results to Google Sheets
        try:
            append_row("trend_scores", [
                datetime.utcnow().isoformat(),
                original_text[:60] + "...",
                trend_score,
                ", ".join(trending_keywords[:5])
            ])
        except Exception as e:
            logger.warning(f"Could not write trend data to sheets: {e}")

        return {
            "original": original_text,
            "optimized": optimized_text,
            "trend_score": trend_score,
            "insights": insights
        }

    # ----------------------------------------------------------------------
    # Public Entry Point
    # ----------------------------------------------------------------------
    def run(self, text: str):
        """
        Simple wrapper for pipeline usage.
        """
        return self.optimize_content(text)


# ----------------------------------------------------------------------
# Developer Test (Keep for your students)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = "AI marketing automation tool to boost business growth"

    optimizer = TrendBasedOptimizer()
    output = optimizer.run(sample_text)

    print("Original:", output["original"])
    print("\nOptimized:", output["optimized"])
    print("\nTrend Score:", output["trend_score"])
    print("\nInsights:", output["insights"])

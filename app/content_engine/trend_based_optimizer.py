"""
trend_based_optimizer3.py
--------------------------
Enhances generated marketing content using real trend signals from:

    * Google Trends (PyTrends)
    * Reddit Hot Topics
    * Keyword Extraction (spaCy)

Provides:
    - Trend score (0-100)
    - Trend insights (rising queries, related topics)
    - An optimized version of the content
    - Optional logging of trend-scoring rows to Google Sheets
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.integrations.trend_fetcher import TrendFetcher
from app.integrations.sheets_connector import append_row

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Only attach a handler if the host application hasn't configured logging
    # itself -- avoids duplicate log lines when this module is imported
    # multiple times or the app already has root logging configured.
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class TrendAnalysisError(Exception):
    """Raised when the underlying trend fetcher fails to produce a score."""


@dataclass
class OptimizationResult:
    """Structured result of an optimization run (safer than a bare dict)."""

    original: str
    optimized: str
    trend_score: int
    insights: Dict[str, Any] = field(default_factory=dict)
    trending_keywords: List[str] = field(default_factory=list)
    rising_phrases: List[str] = field(default_factory=list)
    sheet_logged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # ------------------------------------------------------------
    # Dict-style access for backward compatibility with existing
    # call sites (e.g. content_generator.py's opt["optimized"]) that
    # predate this class and haven't been migrated to attribute
    # access yet. Remove once every caller uses opt.optimized instead.
    # ------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class TrendBasedOptimizer:
    """
    Boosts generated marketing copy with trend-awareness signals.

    Parameters
    ----------
    fetcher:
        Optional pre-built ``TrendFetcher``. Mainly useful for tests, so a
        mock/fake fetcher can be injected instead of hitting real APIs.
    log_to_sheets:
        If False, skips the Google Sheets write entirely (handy for local
        dev/tests where credentials aren't available).
    sheet_name:
        Target sheet/tab for the trend-score log rows.
    """

    LOW_TREND_THRESHOLD = 20
    HIGH_TREND_THRESHOLD = 60
    MAX_KEYWORDS_SHOWN = 3
    MAX_RISING_PHRASES_SHOWN = 5

    def __init__(
        self,
        fetcher: Optional[TrendFetcher] = None,
        log_to_sheets: bool = True,
        sheet_name: str = "trend_scores",
    ) -> None:
        logger.info("Initializing TrendBasedOptimizer...")
        self.fetcher = fetcher or TrendFetcher()
        self.log_to_sheets = log_to_sheets
        self.sheet_name = sheet_name

    # ----------------------------------------------------------------------
    # Trend analysis
    # ----------------------------------------------------------------------
    def analyze_trends(self, text: str) -> tuple[int, Dict[str, Any]]:
        """
        Returns (trend_score, insights).

        trend_score is clamped to the 0-100 range; insights is a dict of
        keyword -> {"rising_queries": [...], ...}.
        """
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")

        try:
            raw_score = self.fetcher.get_combined_trend_score(text)
            insights = self.fetcher.get_trend_insights(text) or {}
        except Exception as exc:
            raise TrendAnalysisError(f"Trend analysis failed: {exc}") from exc

        trend_score = max(0, min(100, int(raw_score or 0)))
        return trend_score, insights

    # ----------------------------------------------------------------------
    # Optimization text construction (pure, easily unit-testable)
    # ----------------------------------------------------------------------
    def _extract_signals(
        self, insights: Dict[str, Any]
    ) -> tuple[List[str], List[str]]:
        """Pull deduplicated, order-preserving keywords and rising phrases."""
        keywords: List[str] = []
        phrases: List[str] = []
        seen_kw, seen_phrase = set(), set()

        for kw, detail in insights.items():
            if kw not in seen_kw:
                keywords.append(kw)
                seen_kw.add(kw)

            for rising in (detail or {}).get("rising_queries", []) or []:
                if rising not in seen_phrase:
                    phrases.append(rising)
                    seen_phrase.add(rising)

        return keywords, phrases

    def _build_optimized_text(
        self,
        original_text: str,
        trend_score: int,
        keywords: List[str],
        phrases: List[str],
    ) -> str:
        lines = [original_text]

        if trend_score < self.LOW_TREND_THRESHOLD:
            lines.append(
                "\n(Tip: consider weaving in more trending topics for better reach.)"
            )
            return "".join(lines)

        if keywords:
            shown = ", ".join(keywords[: self.MAX_KEYWORDS_SHOWN])
            lines.append(f"\n\nTrending now: {shown}")

        if phrases:
            shown = ", ".join(phrases[: self.MAX_RISING_PHRASES_SHOWN])
            lines.append(f"\nPeople are searching for: {shown}")

        if trend_score > self.HIGH_TREND_THRESHOLD:
            lines.append("\nRide the trend wave -- publish soon for maximum impact.")

        return "".join(lines)

    # ----------------------------------------------------------------------
    # Sheets logging (isolated so failures never break optimization)
    # ----------------------------------------------------------------------
    def _log_to_sheets(
        self, original_text: str, trend_score: int, keywords: List[str]
    ) -> bool:
        if not self.log_to_sheets:
            return False

        preview = original_text[:60] + ("..." if len(original_text) > 60 else "")
        try:
            append_row(
                self.sheet_name,
                [
                    datetime.now(timezone.utc).isoformat(),
                    preview,
                    trend_score,
                    ", ".join(keywords[:5]),
                ],
            )
            return True
        except Exception as exc:
            logger.warning("Could not write trend data to sheets: %s", exc)
            return False

    # ----------------------------------------------------------------------
    # Core Optimization Logic
    # ----------------------------------------------------------------------
    def optimize_content(self, original_text: str) -> OptimizationResult:
        """
        Takes generated content and boosts it with trend-awareness.

        Raises
        ------
        ValueError
            If original_text is empty/whitespace.
        TrendAnalysisError
            If the underlying trend fetcher fails.
        """
        trend_score, insights = self.analyze_trends(original_text)
        keywords, phrases = self._extract_signals(insights)

        optimized_text = self._build_optimized_text(
            original_text, trend_score, keywords, phrases
        )

        sheet_logged = self._log_to_sheets(original_text, trend_score, keywords)

        return OptimizationResult(
            original=original_text,
            optimized=optimized_text,
            trend_score=trend_score,
            insights=insights,
            trending_keywords=keywords,
            rising_phrases=phrases,
            sheet_logged=sheet_logged,
        )

    # ----------------------------------------------------------------------
    # Public Entry Point
    # ----------------------------------------------------------------------
    def run(self, text: str) -> OptimizationResult:
        """Simple wrapper for pipeline usage."""
        return self.optimize_content(text)


# ----------------------------------------------------------------------
# Developer / manual smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = "AI marketing automation tool to boost business growth"

    optimizer = TrendBasedOptimizer()
    result = optimizer.run(sample_text)

    print("Original:", result.original)
    print("\nOptimized:", result.optimized)
    print("\nTrend Score:", result.trend_score)
    print("\nInsights:", result.insights)
    print("\nLogged to sheet:", result.sheet_logged)
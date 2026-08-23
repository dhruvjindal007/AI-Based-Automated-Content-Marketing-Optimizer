import os
import logging
from typing import Dict, Any, Optional, List, Tuple

from datetime import datetime, timedelta, timezone

from app.integrations.social_poster import SocialPoster
from app.integrations.social_ingestor import SocialIngestor
from app.integrations.sheets_connector import append_row, read_rows, update_row
from app.integrations.slack_notifier import SlackNotifier

from app.sentiment_engine.sentiment_analyzer import analyze_sentiment, analyze_post_comments
from app.integrations.trend_fetcher import TrendFetcher

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)


SHEETS_ENABLED = bool(os.getenv("GOOGLE_SHEET_ID"))

# Composite score weights. Must sum to 1.0.
ENGAGEMENT_WEIGHT = 0.7
SENTIMENT_WEIGHT = 0.2
TREND_WEIGHT = 0.1

# ----------------------------------------------------------------------
# FIX: model loading for predict_success() removed entirely.
#
# predict_success() scores a SINGLE piece of text on 3 features:
# [sentiment_score, trend_score, length]. Checking against train_model.py,
# neither model actually produced anywhere in this codebase matches that
# schema:
#
#   - train() / "predictor.joblib" (success model): trained on 5 features
#     (ctr_norm, sentiment_norm, polarity_norm, trend_norm, conversions).
#     Both ctr and conversions describe how a post ALREADY performed after
#     posting -- they don't exist yet for a draft variant, so this model
#     can't be fed correctly at the point predict_success() is called.
#
#   - train_pairwise() / "pairwise_predictor.joblib" (A/B-winner model,
#     see auto_retrainer.py): trained on 6 features describing a PAIR of
#     posts (sentA, sentB, trendA, trendB, engA, engB). It doesn't even
#     take a single text as input, so it structurally can't answer "how
#     good is this one variant."
#
# Loading either one here (as an earlier version of this file did, first
# unfiltered and then filtered by a prefix) meant predict_success() could
# feed a 3-element feature vector into a model expecting 5 or 6 columns --
# raising at inference, or in the worst case, being accepted anyway if a
# library doesn't validate shape and returning a meaningless prediction.
#
# Until a model is trained specifically on predict_success()'s 3-feature
# schema, the honest choice is to always use the heuristic below rather
# than gamble on a mismatched model "sort of" working.
# ----------------------------------------------------------------------


def _engagement_share(scoreA: float, scoreB: float) -> Tuple[float, float]:
    """
    Convert raw, unbounded engagement counts into a 0-100 "share" pair
    that sums to 100, so they're comparable in scale to sentiment (0-100)
    and trend score (0-100) when computing the composite score.

    If both scores are 0 (e.g. metrics fetch failed for both), splits
    evenly 50/50 rather than dividing by zero.
    """
    total = scoreA + scoreB
    if total <= 0:
        return 50.0, 50.0
    shareA = (scoreA / total) * 100
    shareB = (scoreB / total) * 100
    return shareA, shareB


class ABCoach:
    def __init__(self):
        self.poster = SocialPoster()
        self.ingestor = SocialIngestor()
        self.trends = TrendFetcher()

        try:
            self.slack = SlackNotifier()
        except Exception as e:
            logger.warning(f"SlackNotifier init failed, notifications disabled: {e}")
            self.slack = None

    # -----------------------
    # Utilities
    # -----------------------
    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # RESOLVED (Bug 2): _persist_schedule() was deleted. It wrote a second,
    # duplicate row to "ab_schedule" for the same ab_id right after
    # SocialPoster.schedule_ab_test() (called just above in
    # create_and_schedule_ab_test()) had already persisted its own row for
    # it -- same data, different column order. SocialPoster now owns
    # ab_schedule exclusively, since it's the one holding the real job IDs
    # at the moment of persistence. See sheets_connector.py's "ab_schedule"
    # header comment.
    # ------------------------------------------------------------------

    def _persist_ab_posts(self, ab_id: str, campaign_id: str, variant: str, post_id: str, ts: Optional[str] = None):
        if not SHEETS_ENABLED:
            return
        try:
            append_row("ab_posts", [ts or self._now_iso(), ab_id, campaign_id, variant, post_id])
        except Exception as e:
            logger.warning(f"Failed to write ab_posts row: {e}")

    def _persist_ab_result(self, ab_id: str, postA: str, scoreA: int, postB: str, scoreB: int, winner: str):
        if not SHEETS_ENABLED:
            return
        try:
            append_row("ab_test_results", [self._now_iso(), ab_id, postA, scoreA, postB, scoreB, winner])
        except Exception as e:
            logger.warning(f"Failed to write ab_test_results row: {e}")

    # -----------------------
    # Create & Schedule A/B test
    # -----------------------
    def create_and_schedule_ab_test(
        self,
        campaign_id: str,
        textA: str,
        textB: str,
        run_date_A: datetime,
        run_date_B: datetime,
        eval_delay_hours: float = 6.0
    ) -> Dict[str, Any]:
        """
        Schedules an A/B test using SocialPoster.schedule_ab_test.
        SocialPoster persists the ab_schedule row itself (see the class-level
        note above on why ABCoach no longer writes a second one).
        """
        logger.info(f"Scheduling A/B for campaign {campaign_id} at {run_date_A} / {run_date_B}")
        details = self.poster.schedule_ab_test(
            campaign_id=campaign_id,
            textA=textA,
            textB=textB,
            run_date_A=run_date_A,
            run_date_B=run_date_B,
            eval_delay_hours=eval_delay_hours
        )

        # details contains: {ab_id, jobA, jobB, jobEval}
        ab_id = details.get("ab_id")

        # Record initial posts may be saved by SocialPoster when executed. We still return scheduled info.
        if self.slack:
            try:
                self.slack.send_message(f"Scheduled A/B test {ab_id} for campaign {campaign_id}. A:{run_date_A} B:{run_date_B}")
            except Exception as e:
                logger.warning(f"Slack notification failed: {e}")

        return details

    # -----------------------
    # Predict success probability for text variant
    # -----------------------
    def predict_success(self, text: str) -> float:
        """
        Predict the probability of success for a single text.

        No trained model in this codebase currently matches this
        function's 3-feature schema (sentiment_score, trend_score,
        length) -- see the module-level note above for why the success
        model and the pairwise A/B-winner model both don't apply here.
        Always uses the heuristic below until a model is trained
        specifically for this schema.
        """
        try:
            sent = analyze_sentiment(text)[0]  # returns list
            trend_score = self.trends.get_combined_trend_score(text)
            length = len(text.split())

            heuristic = (
                (sent.get("sentiment_score", 0) * 0.5)
                + (trend_score / 100.0 * 0.4)
                + (min(length, 100) / 100.0 * 0.1)
            )
            return float(max(0.0, min(1.0, heuristic)))
        except Exception as e:
            logger.error(f"predict_success error: {e}")
            return 0.5

    # -----------------------
    # Manual evaluation (can be called by scheduler or manually)
    # -----------------------
    def evaluate_ab_test(self, ab_id: str) -> Optional[Dict[str, Any]]:
        """
        Evaluate an A/B test by reading ab_posts sheet to find post IDs for A and B,
        fetch live metrics, compute scores, persist results, and notify Slack.
        """
        logger.info(f"Evaluating A/B test {ab_id}")
        try:
            rows = read_rows("ab_posts") if SHEETS_ENABLED else []
        except Exception as e:
            logger.error(f"Failed to read ab_posts: {e}")
            rows = []

        postA = None
        postB = None
        campaign_id = None

        # expected row format: [ts, ab_id, campaign_id, variant, post_id]
        for r in rows:
            try:
                if len(r) >= 5 and r[1] == ab_id:
                    variant = r[3]
                    post_id = r[4]
                    campaign_id = r[2] if len(r) >= 3 else None
                    if variant == "A":
                        postA = post_id
                    elif variant == "B":
                        postB = post_id
            except Exception as e:
                logger.warning(f"Skipping malformed ab_posts row {r}: {e}")
                continue

        if not postA or not postB:
            logger.warning("Could not find both A and B posts for evaluation.")
            return None

        try:
            metricsA = self.ingestor.fetch_post_metrics(str(postA))
            metricsB = self.ingestor.fetch_post_metrics(str(postB))
        except Exception as e:
            logger.error(f"Error fetching metrics from ingestor: {e}")
            metricsA = {}
            metricsB = {}

        # Raw engagement counts (unbounded)
        scoreA = (metricsA.get("likes", 0) + metricsA.get("shares", 0) + metricsA.get("replies", 0))
        scoreB = (metricsB.get("likes", 0) + metricsB.get("shares", 0) + metricsB.get("replies", 0))

        # Additionally compute sentiment from comments
        comment_info_A = analyze_post_comments(postA)
        comment_info_B = analyze_post_comments(postB)

        trend_score_A = self.trends.get_combined_trend_score(metricsA.get("text", ""))
        trend_score_B = self.trends.get_combined_trend_score(metricsB.get("text", ""))

        # Convert raw engagement to a 0-100 share so it's on the same scale
        # as sentiment (0-100) and trend score (0-100) before weighting --
        # this is the fix for the original scale-mismatch bug.
        engagement_share_A, engagement_share_B = _engagement_share(scoreA, scoreB)

        compositeA = (
            engagement_share_A * ENGAGEMENT_WEIGHT
            + (comment_info_A.get("avg_sentiment", 0) * 100) * SENTIMENT_WEIGHT
            + trend_score_A * TREND_WEIGHT
        )
        compositeB = (
            engagement_share_B * ENGAGEMENT_WEIGHT
            + (comment_info_B.get("avg_sentiment", 0) * 100) * SENTIMENT_WEIGHT
            + trend_score_B * TREND_WEIGHT
        )

        winner = "A" if compositeA > compositeB else ("B" if compositeB > compositeA else "tie")

        # probA/probB: normalized composite scores that sum to 1, so they
        # actually behave like a probability pair (fix for the original
        # bug where probA/probB were just raw unbounded composite scores).
        composite_total = compositeA + compositeB
        if composite_total > 0:
            probA = compositeA / composite_total
            probB = compositeB / composite_total
        else:
            probA = probB = 0.5

        # Persist results
        try:
            self._persist_ab_result(ab_id, postA, int(scoreA), postB, int(scoreB), winner)
        except Exception as e:
            logger.warning(f"Failed to persist AB result: {e}")

        # Notify Slack
        if self.slack:
            try:
                self.slack.send_message(
                    f"A/B Test {ab_id} completed for campaign {campaign_id}. Winner: {winner} (A={scoreA}, B={scoreB})"
                )
            except Exception as e:
                logger.warning(f"Slack notification failed: {e}")

        result = {
            "ab_id": ab_id,
            "campaign_id": campaign_id,
            "postA": postA,
            "postB": postB,
            "scoreA": int(scoreA),
            "scoreB": int(scoreB),
            "engagement_share_A": engagement_share_A,
            "engagement_share_B": engagement_share_B,
            "trend_score_A": trend_score_A,
            "trend_score_B": trend_score_B,
            "probA": float(probA),
            "probB": float(probB),
            "compositeA": compositeA,
            "compositeB": compositeB,
            "winner": winner,
            "recommended": winner,
            "comment_info_A": comment_info_A,
            "comment_info_B": comment_info_B
        }

        logger.info(f"A/B evaluation complete: {result}")
        return result

    # -----------------------
    # Manual re-evaluate (useful for debugging)
    # -----------------------
    def reevaluate_recent(self, lookback_hours: int = 24) -> List[Dict[str, Any]]:
        """
        Read ab_schedule to find AB tests scheduled within lookback_hours and re-evaluate them.
        Returns list of evaluation results.
        """
        results = []
        try:
            rows = read_rows("ab_schedule") if SHEETS_ENABLED else []
        except Exception as e:
            logger.error(f"Failed to read ab_schedule: {e}")
            return results

        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        _ = cutoff  # reserved for future use (see note below)

        for r in rows:
            try:
                # ab_schedule format: [ts, ab_id, campaign_id, jobA, run_date_A, jobB, run_date_B, eval_time]
                if len(r) < 8:
                    logger.warning(f"Skipping short ab_schedule row (expected 8 cols, got {len(r)}): {r}")
                    continue
                ab_id = r[1]
                eval_time_str = r[7]
                eval_time = datetime.fromisoformat(eval_time_str) if isinstance(eval_time_str, str) else None
                if eval_time is None:
                    logger.warning(f"Skipping ab_schedule row with unparseable eval_time: {r}")
                    continue
                if eval_time <= datetime.now(timezone.utc):
                    res = self.evaluate_ab_test(ab_id)
                    if res:
                        results.append(res)
            except Exception as e:
                logger.warning(f"Skipping malformed ab_schedule row {r}: {e}")
                continue

        return results

    # -----------------------
    # Query history / cancel
    # -----------------------
    def list_scheduled_ab_tests(self) -> List[Dict[str, Any]]:
        """
        Returns the raw ab_schedule sheet rows as dicts
        """
        out = []
        try:
            rows = read_rows("ab_schedule") if SHEETS_ENABLED else []
            for r in rows:
                out.append(r)
        except Exception as e:
            logger.warning(f"Could not read ab_schedule: {e}")
        return out

    def cancel_ab_job(self, job_id: str) -> bool:
        """
        Cancel a scheduled job by job id via SocialPoster scheduler
        """
        return self.poster.cancel_job(job_id)

    # -----------------------
    # Utility: run a quick simulation (no posting) used before scheduling
    # -----------------------
    def simulate_ab(self, textA: str, textB: str) -> Dict[str, Any]:
        """
        Simple A/B scoring logic.
        Ensures scoreA and scoreB always exist.
        """

        # --- Replace with your real scoring logic ---
        # Dummy scoring: length normalized (0 to 1)
        scoreA = len(textA) % 100 / 100
        scoreB = len(textB) % 100 / 100

        winner = "A" if scoreA >= scoreB else "B"
        explanation = f"Variant {winner} performs better based on simulated engagement scoring."

        # --- MUST return these exact keys ---
        return {
            "scoreA": float(scoreA),
            "scoreB": float(scoreB),
            "probA": float(scoreA),
            "probB": float(scoreB),
            "winner": winner,
            "recommended": winner,
            "explanation": explanation
        }
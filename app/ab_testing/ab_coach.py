# ab_coach.py
"""
A/B Testing
"""

import os
import time
import joblib
import logging
from typing import Dict, Any, Optional, Tuple, List

from datetime import datetime, timedelta

from app.integrations.social_poster import SocialPoster
from app.integrations.social_ingestor import SocialIngestor
from app.integrations.sheets_connector import append_row, read_rows, update_row, find_row
from app.integrations.slack_notifier import SlackNotifier

from app.sentiment_engine.sentiment_analyzer import analyze_sentiment, analyze_post_comments
from app.integrations.trend_fetcher import TrendFetcher

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)


MODEL_DIR = os.getenv("MODEL_DIR", "models")
SHEETS_ENABLED = bool(os.getenv("GOOGLE_SHEET_ID"))


def _load_latest_model() -> Optional[Any]:
    """Load the most recent model artifact from MODEL_DIR using joblib."""
    try:
        files = []
        if not os.path.exists(MODEL_DIR):
            return None
        for f in os.listdir(MODEL_DIR):
            if f.endswith(".pkl") or f.endswith(".joblib"):
                files.append(os.path.join(MODEL_DIR, f))
        if not files:
            return None
        latest = max(files, key=os.path.getmtime)
        logger.info(f"Loading model: {latest}")
        return joblib.load(latest)
    except Exception as e:
        logger.warning(f"Could not load model: {e}")
        return None


class ABCoach:
    def __init__(self):
        self.poster = SocialPoster()
        self.ingestor = SocialIngestor()
        self.trends = TrendFetcher()
        self.slack = SlackNotifier() if 'SlackNotifier' in globals() and SlackNotifier is not None else None
        self.model = _load_latest_model()

    # -----------------------
    # Utilities
    # -----------------------
    @staticmethod
    def _now_iso() -> str:
        return datetime.utcnow().isoformat()

    def _persist_schedule(self, ab_id: str, campaign_id: str, jobA: str, jobB: str, runA: str, runB: str, eval_time: str):
        if not SHEETS_ENABLED:
            logger.debug("Sheets disabled: skipping persist_schedule")
            return
        try:
            append_row("ab_schedule", [self._now_iso(), ab_id, campaign_id, jobA, jobB, runA, runB, eval_time])
        except Exception as e:
            logger.warning(f"Failed to write ab_schedule row: {e}")

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
        Persists schedule to Sheets and optionally returns scheduled job ids.
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
        jobA = details.get("jobA")
        jobB = details.get("jobB")
        jobEval = details.get("jobEval")

        eval_time = (run_date_B + timedelta(hours=eval_delay_hours)).isoformat()

        # Persist schedule
        self._persist_schedule(ab_id, campaign_id, jobA, jobB, run_date_A.isoformat(), run_date_B.isoformat(), eval_time)

        # Record initial posts may be saved by SocialPoster when executed. We still return scheduled info.
        if self.slack:
            try:
                self.slack.send_message(f"Scheduled A/B test {ab_id} for campaign {campaign_id}. A:{run_date_A} B:{run_date_B}")
            except Exception:
                pass

        return details

    # -----------------------
    # Predict success probability for text variant
    # -----------------------
    def predict_success(self, text: str) -> float:
        """
        Predict the probability of success for a single text using the latest saved model.
        Model expected to accept features: sentiment_score, trend_score, length, maybe embeddings.
        If no model available, return a heuristic score (0..1).
        """
        try:
            # Basic features
            sent = analyze_sentiment(text)[0]  # returns list
            trend_score = self.trends.get_combined_trend_score(text)
            length = len(text.split())

            # Build X in a compatible shape for model
            features = [[sent.get("sentiment_score", 0), trend_score, length]]
            if self.model:
                try:
                    proba = None
                    # If model has predict_proba
                    if hasattr(self.model, "predict_proba"):
                        proba = self.model.predict_proba(features)[0]
                        # assume binary classifier with [prob_neg, prob_pos] or similar: pick max positive
                        if proba.shape and proba.shape[0] >= 1:
                            # choose last column as positive class probability if shape >1
                            prob = float(proba[-1]) if len(proba) > 1 else float(proba[0])
                        else:
                            prob = float(proba)
                    else:
                        # fallback to predict (0/1) and map to 0.75/0.25
                        pred = self.model.predict(features)[0]
                        prob = 0.75 if pred == 1 else 0.25
                    return max(0.0, min(1.0, prob))
                except Exception as e:
                    logger.warning(f"Model prediction failed: {e}")
            # Heuristic fallback
            heuristic = (sent.get("sentiment_score", 0) * 0.5) + (trend_score / 100.0 * 0.4) + (min(length, 100) / 100.0 * 0.1)
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
            except Exception:
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

        # Compute simple engagement-based scores (customizable)
        scoreA = (metricsA.get("likes", 0) + metricsA.get("shares", 0) + metricsA.get("replies", 0))
        scoreB = (metricsB.get("likes", 0) + metricsB.get("shares", 0) + metricsB.get("replies", 0))

        # Additionally compute sentiment from comments
        comment_info_A = analyze_post_comments(postA)
        comment_info_B = analyze_post_comments(postB)

        # Combine into composite score (weights can be tuned)
        compositeA = scoreA * 0.7 + (comment_info_A.get("avg_sentiment", 0) * 100) * 0.2 + (self.trends.get_combined_trend_score(metricsA.get("text", "") ) ) * 0.1
        compositeB = scoreB * 0.7 + (comment_info_B.get("avg_sentiment", 0) * 100) * 0.2 + (self.trends.get_combined_trend_score(metricsB.get("text", "") ) ) * 0.1

        winner = "A" if compositeA > compositeB else ("B" if compositeB > compositeA else "tie")

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
            except Exception:
                pass

        result = {
            "ab_id": ab_id,
            "campaign_id": campaign_id,
            "postA": postA,
            "postB": postB,
            "scoreA": int(scoreA),
            "scoreB": int(scoreB),
            "probA": float(compositeA),
            "probB": float(compositeB),
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

        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)

        for r in rows:
            try:
                # ab_schedule format: [ts, ab_id, campaign_id, jobA, jobB, runA, runB, eval_time]
                if len(r) < 8:
                    continue
                ab_id = r[1]
                eval_time_str = r[7]
                eval_time = datetime.fromisoformat(eval_time_str) if isinstance(eval_time_str, str) else None
                if eval_time and eval_time <= datetime.utcnow():
                    res = self.evaluate_ab_test(ab_id)
                    if res:
                        results.append(res)
            except Exception:
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
    def simulate_ab(self, textA, textB):
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


# End of ABCoach

"""
auto_retrainer.py
------------------
Automatically retrains the ranking/ML model using:

    * A/B test results (labels)
    * Engagement metrics
    * Sentiment scores
    * Trend scores

This is what makes the system self-learning: it pulls labelled A/B
results from Sheets, builds features from sentiment/trend/engagement
signals, retrains when there's enough new data, versions the model,
and notifies Slack.
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import joblib
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

# Integrations
from app.integrations.sheets_connector import read_rows, append_row
from app.integrations.social_ingestor import SocialIngestor
from app.integrations.slack_notifier import SlackNotifier

# AI Components
from app.sentiment_engine.sentiment_analyzer import analyze_sentiment
from app.integrations.trend_fetcher import TrendFetcher

# train_pairwise (not train!) is the A/B-winner model matching the
# 6-column pairwise feature set built in preprocess_data() below.
# train_model.train() is a different model (single-post conversion
# "success" prediction, 5-column feature set) with an incompatible
# signature and schema -- calling it here would raise at runtime.
from app.ml_engine.train_model import train_pairwise

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

REQUIRED_AB_COLUMNS = {"postA", "scoreA", "postB", "scoreB", "winner"}
FEATURE_COLUMNS = ["sentA", "sentB", "trendA", "trendB", "engA", "engB"]

# FIX (aligned with train_model.py): that module already defines the
# intended naming convention for this model via
# PairwiseTrainConfig.latest_model_path -> "models/pairwise_predictor.joblib",
# mirroring how train()'s TrainConfig saves "models/predictor.joblib".
# train_pairwise() itself deliberately does NOT save to disk (see its
# docstring: "AutoRetrainer's own save_model/versioning handles
# persistence"), so this file is responsible for actually writing files
# that match that convention. The previous version of this file invented
# its own unrelated scheme (pairwise_model_<ts>.pkl) that matched nothing
# else in the codebase.
PAIRWISE_MODEL_BASENAME = "pairwise_predictor"


@dataclass
class RetrainConfig:
    """Tunable knobs, split out of the class so they're easy to override in tests."""

    model_dir: str = "models"
    min_samples_to_retrain: int = 20
    sheet_ab_results: str = "ab_test_results"
    sheet_model_versions: str = "model_versions"
    autostart_scheduler: bool = True

    @property
    def latest_model_path(self) -> str:
        return os.path.join(self.model_dir, f"{PAIRWISE_MODEL_BASENAME}.joblib")


class AutoRetrainer:
    """
    Orchestrates the full retraining cycle. All external dependencies
    (ingestor, sentiment fn, trend fetcher, slack, scheduler) are
    injectable so this class can be unit-tested with fakes instead of
    hitting real APIs.
    """

    def __init__(
        self,
        config: Optional[RetrainConfig] = None,
        ingestor: Optional[SocialIngestor] = None,
        sentiment_fn: Optional[Callable[[str], list]] = None,
        trend_fetcher: Optional[TrendFetcher] = None,
        slack: Optional[SlackNotifier] = None,
        scheduler: Optional[BackgroundScheduler] = None,
    ) -> None:
        self.config = config or RetrainConfig()

        self.ingestor = ingestor or SocialIngestor()
        self.sentiment_fn = sentiment_fn or analyze_sentiment
        self.trends = trend_fetcher or TrendFetcher()
        self.slack = slack or SlackNotifier()

        os.makedirs(self.config.model_dir, exist_ok=True)

        self.scheduler = scheduler or BackgroundScheduler()
        if self.config.autostart_scheduler and not self.scheduler.running:
            self.scheduler.start()

        logger.info("AutoRetrainer initialized.")

    # --------------------------------------------------------------
    # 1. LOAD TRAINING DATA FROM GOOGLE SHEETS
    # --------------------------------------------------------------
    def load_training_data(self) -> pd.DataFrame:
        try:
            rows = read_rows(self.config.sheet_ab_results)
        except Exception as exc:
            logger.error("Error loading Sheets data: %s", exc)
            return pd.DataFrame()

        if not rows or len(rows) < 2:
            logger.warning("No A/B training data available in Sheets.")
            return pd.DataFrame()

        header, *body = rows
        df = pd.DataFrame(body, columns=header) if header else pd.DataFrame(rows)

        if not REQUIRED_AB_COLUMNS.issubset(df.columns):
            missing = REQUIRED_AB_COLUMNS - set(df.columns)
            logger.warning("A/B data missing required columns: %s", missing)
            return pd.DataFrame()

        return df

    # --------------------------------------------------------------
    # 2. PREPROCESSING PIPELINE
    # --------------------------------------------------------------
    def _score_post(self, post_id: str) -> Optional[dict[str, Any]]:
        """Fetch metrics/text/sentiment/trend for a single post. Returns
        None (rather than raising) if any step fails, so one bad row
        doesn't kill the whole preprocessing pass."""
        try:
            metrics = self.ingestor.fetch_post_metrics(str(post_id))
            text = metrics.get("text", "") or ""

            sentiment_result = self.sentiment_fn(text)
            sentiment_score = 0.0
            if sentiment_result:
                sentiment_score = sentiment_result[0].get("sentiment_score", 0)

            trend_score = self.trends.get_combined_trend_score(text)

            engagement = (
                metrics.get("likes", 0)
                + metrics.get("shares", 0)
                + metrics.get("replies", 0)
            )

            return {
                "text": text,
                "sentiment": sentiment_score,
                "trend": trend_score,
                "engagement": engagement,
            }
        except Exception as exc:
            logger.warning("Skipping post %s -- scoring failed: %s", post_id, exc)
            return None

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        logger.info("Preprocessing %d training rows...", len(df))

        processed_rows = []
        skipped = 0

        for _, row in df.iterrows():
            post_a, post_b, winner = row.get("postA"), row.get("postB"), row.get("winner")

            scored_a = self._score_post(post_a)
            scored_b = self._score_post(post_b)

            if scored_a is None or scored_b is None:
                skipped += 1
                continue

            label = 1 if str(winner).strip().lower() == "a" else 0

            processed_rows.append(
                {
                    "textA": scored_a["text"],
                    "textB": scored_b["text"],
                    "sentA": scored_a["sentiment"],
                    "sentB": scored_b["sentiment"],
                    "trendA": scored_a["trend"],
                    "trendB": scored_b["trend"],
                    "engA": scored_a["engagement"],
                    "engB": scored_b["engagement"],
                    "label": label,
                }
            )

        if skipped:
            logger.warning("Skipped %d rows during preprocessing.", skipped)

        return pd.DataFrame(processed_rows)

    # --------------------------------------------------------------
    # 3. CHECK IF WE HAVE ENOUGH NEW DATA
    # --------------------------------------------------------------
    def check_threshold(self, df: pd.DataFrame) -> bool:
        threshold = self.config.min_samples_to_retrain

        if len(df) >= threshold:
            logger.info("Enough training samples (%d). Ready to retrain.", len(df))
            return True

        logger.info("Not enough data for retraining: %d/%d", len(df), threshold)
        return False

    # --------------------------------------------------------------
    # 4. RETRAIN MODEL
    # --------------------------------------------------------------
    def retrain(self, df: pd.DataFrame) -> Optional[Any]:
        if df.empty:
            logger.warning("Cannot retrain -- empty dataset.")
            return None

        logger.info("Retraining ML model on %d samples...", len(df))

        missing_cols = [c for c in FEATURE_COLUMNS + ["label"] if c not in df.columns]
        if missing_cols:
            logger.error("Cannot retrain -- missing columns: %s", missing_cols)
            return None

        X = df[FEATURE_COLUMNS]
        y = df["label"]

        try:
            model = train_pairwise(X, y)
        except Exception as exc:
            logger.error("Model training failed: %s", exc)
            return None

        return model

    # --------------------------------------------------------------
    # 5. SAVE MODEL VERSION
    # --------------------------------------------------------------
    def save_model(self, model: Any, df: pd.DataFrame) -> Optional[str]:
        """
        Saves both a timestamped, versioned artifact and overwrites a
        fixed "latest" path -- mirroring exactly how train_model.train()
        persists the success model (predictor_<ts>.joblib +
        predictor.joblib). train_pairwise() itself intentionally doesn't
        do this (see its docstring), so this method owns it.
        """
        timestamp = int(time.time())
        versioned_path = os.path.join(
            self.config.model_dir, f"{PAIRWISE_MODEL_BASENAME}_{timestamp}.joblib"
        )
        latest_path = self.config.latest_model_path

        try:
            joblib.dump(model, versioned_path)
            joblib.dump(model, latest_path)
        except Exception as exc:
            logger.error("Failed to save model to %s: %s", versioned_path, exc)
            return None

        try:
            append_row(self.config.sheet_model_versions, [timestamp, len(df), versioned_path])
        except Exception as exc:
            logger.warning("Model saved locally but failed to log version to Sheets: %s", exc)

        logger.info("New model saved: %s (latest -> %s)", versioned_path, latest_path)
        return versioned_path

    # --------------------------------------------------------------
    # 6. SLACK NOTIFICATION
    # --------------------------------------------------------------
    def notify_slack(self, path: str, dfsize: int) -> None:
        try:
            self.slack.send_message(
                f"New ML model retrained.\nSize: {dfsize} samples\nSaved: {path}"
            )
        except Exception as exc:
            logger.warning("Slack notification failed: %s", exc)

    # --------------------------------------------------------------
    # 7. FULL RETRAINING CYCLE
    # --------------------------------------------------------------
    def run_full_cycle(self) -> Optional[str]:
        logger.info("Starting retraining cycle...")

        df_raw = self.load_training_data()
        if df_raw.empty:
            logger.info("No raw training data -- cycle ending early.")
            return None

        df = self.preprocess_data(df_raw)

        if not self.check_threshold(df):
            return None

        model = self.retrain(df)
        if model is None:
            return None

        path = self.save_model(model, df)
        if path is None:
            return None

        self.notify_slack(path, len(df))

        logger.info("Retraining cycle complete.")
        return path

    # --------------------------------------------------------------
    # Simple run() wrapper for the pipeline
    # --------------------------------------------------------------
    def run(self) -> None:
        """Required by run_pipeline.py -- triggers one full retraining cycle."""
        logger.info("AutoRetrainer.run() invoked.")
        try:
            self.run_full_cycle()
        except Exception as exc:
            logger.error("AutoRetrainer.run() failed: %s", exc)
            raise

    # --------------------------------------------------------------
    # 8. SCHEDULE AUTOMATIC RETRAINING
    # --------------------------------------------------------------
    def schedule_retraining(self, interval_hours: int = 24) -> None:
        """Runs retraining every X hours. Default: daily."""
        if not self.scheduler.running:
            self.scheduler.start()
        self.scheduler.add_job(self.run_full_cycle, "interval", hours=interval_hours)
        logger.info("Auto-retraining scheduled every %d hours.", interval_hours)

    def shutdown(self) -> None:
        """Cleanly stop the scheduler -- call this on app shutdown."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler shut down.")


# --------------------------------------------------------------
# Developer / manual smoke test
# --------------------------------------------------------------
if __name__ == "__main__":
    retrainer = AutoRetrainer()
    retrainer.run_full_cycle()  # manual test
    retrainer.schedule_retraining(1)  # every hour for dev
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.base import JobLookupError

# Optional external integrations (best-effort imports)
try:
    import tweepy
except Exception:
    tweepy = None

from app.integrations.sheets_connector import append_row, read_rows
from app.integrations.social_ingestor import SocialIngestor

# Slack notifier is optional; if missing, posting still works without notifications
try:
    from app.integrations.slack_notifier import SlackNotifier
except Exception:
    SlackNotifier = None


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)


class SocialPoster:
    """
    SocialPoster handles posting and A/B scheduling for social platforms.

    - Currently implements Twitter/X posting using Tweepy (v2 client)
    - Uses APScheduler BackgroundScheduler for scheduling jobs
    - Persists posted content metadata to Google Sheets via sheets_connector
    - Evaluates A/B tests by fetching metrics through SocialIngestor

    Notes:
    - Provide TWITTER_* env vars for posting to work.
    - Provide GOOGLE_SHEET_ID env var to enable Sheets writes.
    - Provide REDDIT or other env keys for SocialIngestor if used elsewhere.
    """

    def __init__(self):
        # Scheduler
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()

        # Social ingestor for reading live metrics
        self.ingestor = SocialIngestor()

        # Sheets enabled flag
        self.sheets_enabled = bool(os.getenv("GOOGLE_SHEET_ID"))

        # Slack notifier (optional)
        self.slack = SlackNotifier() if SlackNotifier is not None else None

        # Twitter client setup
        self.twitter_client = None
        if tweepy is not None:
            api_key = os.getenv("TWITTER_API_KEY")
            api_secret = os.getenv("TWITTER_API_SECRET")
            access_token = os.getenv("TWITTER_ACCESS_TOKEN")
            access_secret = os.getenv("TWITTER_ACCESS_SECRET")
            bearer_token = os.getenv("TWITTER_BEARER_TOKEN")

            if any([api_key, api_secret, access_token, access_secret, bearer_token]):
                try:
                    # Prefer using bearer token + OAuth2 app if available
                    if bearer_token and tweepy.Client:
                        self.twitter_client = tweepy.Client(
                            bearer_token=bearer_token,
                            consumer_key=api_key,
                            consumer_secret=api_secret,
                            access_token=access_token,
                            access_token_secret=access_secret,
                            wait_on_rate_limit=True,
                        )
                    else:
                        # Fallback to older auth (may require tweepy API usage)
                        auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_secret)
                        self.twitter_client = tweepy.API(auth)

                    logger.info("Twitter client initialized.")
                except Exception as e:
                    logger.warning(f"Twitter client init failed: {e}")
                    self.twitter_client = None
            else:
                logger.info("Twitter credentials not found in environment; posting disabled.")
        else:
            logger.info("Tweepy not installed; Twitter posting disabled.")

    # -------------------------
    # Posting helpers
    # -------------------------
    def post_to_twitter(self, text: str) -> Optional[str]:
        """Post content to Twitter/X. Returns post_id (str) or None on failure."""
        if not self.twitter_client:
            logger.error("Twitter client unavailable — cannot post.")
            return None

        try:
            # If using tweepy.Client (v2)
            if hasattr(self.twitter_client, "create_tweet"):
                resp = self.twitter_client.create_tweet(text=text)
                post_id = None
                if resp and hasattr(resp, "data") and isinstance(resp.data, dict):
                    post_id = resp.data.get("id")
                elif resp and hasattr(resp, "data") and hasattr(resp.data, "get"):
                    post_id = resp.data.get("id")

                logger.info(f"Posted tweet (v2) id={post_id}")
            else:
                # Fallback to API.update_status for older tweepy
                status = self.twitter_client.update_status(status=text)
                post_id = getattr(status, "id_str", getattr(status, "id", None))
                logger.info(f"Posted tweet (v1) id={post_id}")

            return str(post_id) if post_id is not None else None

        except Exception as e:
            logger.error(f"Error posting to Twitter: {e}")
            return None

    # -------------------------
    # Scheduling helpers
    # -------------------------
    def schedule_post(self, text: str, run_date: datetime, meta: Optional[Dict[str, Any]] = None) -> str:
        """Schedule a one-off post. Returns job id."""
        meta = meta or {}
        job = self.scheduler.add_job(self._execute_and_persist_post, 'date', run_date=run_date, args=[text, meta])
        logger.info(f"Scheduled post job id={job.id} at {run_date}")
        return job.id

    def _execute_and_persist_post(self, text: str, meta: Dict[str, Any]):
        """Internal: post now and persist metadata to Sheets."""
        post_id = self.post_to_twitter(text)
        ts = datetime.utcnow().isoformat()

        row = [ts, meta.get("campaign_id"), meta.get("variant", ""), post_id, "twitter", text]
        try:
            if self.sheets_enabled:
                append_row("posted_content", row)
                logger.info("Posted content saved to Google Sheets.")
            else:
                logger.info("Sheets disabled — posted content not saved to Sheets.")
        except Exception as e:
            logger.warning(f"Failed to save posted content to sheets: {e}")

        # notify slack if configured
        if self.slack:
            try:
                self.slack.send_message(f"Posted content (campaign={meta.get('campaign_id')}) id={post_id}")
            except Exception:
                pass

    # -------------------------
    # A/B scheduling & evaluation
    # -------------------------
    def schedule_ab_test(self, campaign_id: str, textA: str, textB: str, run_date_A: datetime, run_date_B: datetime, eval_delay_hours: int = 6) -> Dict[str, Any]:
        """
        Schedule an A/B test:
          - post A at run_date_A
          - post B at run_date_B
          - schedule evaluation at run_date_B + eval_delay_hours

        Returns dict with job ids and an internal ab_id.
        """
        ab_id = f"ab_{int(datetime.utcnow().timestamp())}"

        metaA = {"campaign_id": campaign_id, "variant": "A", "ab_id": ab_id}
        metaB = {"campaign_id": campaign_id, "variant": "B", "ab_id": ab_id}

        jobA = self.scheduler.add_job(self._ab_post_A, 'date', run_date=run_date_A, args=[textA, metaA])
        jobB = self.scheduler.add_job(self._ab_post_B, 'date', run_date=run_date_B, args=[textB, metaB])

        eval_time = run_date_B + timedelta(hours=eval_delay_hours)
        jobEval = self.scheduler.add_job(self._evaluate_ab_test, 'date', run_date=eval_time, args=[ab_id])

        logger.info(f"Scheduled A/B test {ab_id}: A_job={jobA.id}, B_job={jobB.id}, Eval_job={jobEval.id}")

        # persist scheduling info to sheets
        if self.sheets_enabled:
            try:
                append_row("ab_schedule", [datetime.utcnow().isoformat(), ab_id, campaign_id, jobA.id, str(run_date_A), jobB.id, str(run_date_B), str(eval_time)])
            except Exception as e:
                logger.warning(f"Could not persist AB schedule: {e}")

        return {"ab_id": ab_id, "jobA": jobA.id, "jobB": jobB.id, "jobEval": jobEval.id}

    def _ab_post_A(self, text: str, meta: Dict[str, Any]):
        post_id = self.post_to_twitter(text)
        ts = datetime.utcnow().isoformat()
        row = [ts, meta.get("ab_id"), meta.get("campaign_id"), "A", post_id]
        try:
            if self.sheets_enabled:
                append_row("ab_posts", row)
        except Exception as e:
            logger.warning(f"Failed to persist AB post A: {e}")

    def _ab_post_B(self, text: str, meta: Dict[str, Any]):
        post_id = self.post_to_twitter(text)
        ts = datetime.utcnow().isoformat()
        row = [ts, meta.get("ab_id"), meta.get("campaign_id"), "B", post_id]
        try:
            if self.sheets_enabled:
                append_row("ab_posts", row)
        except Exception as e:
            logger.warning(f"Failed to persist AB post B: {e}")

    def _evaluate_ab_test(self, ab_id: str):
        """
        Evaluate A/B by reading ab_posts sheet to get the post IDs, fetch metrics via SocialIngestor,
        declare a winner and persist results to sheets and Slack.
        """
        logger.info(f"Evaluating AB test {ab_id}...")

        try:
            rows = read_rows("ab_posts") if self.sheets_enabled else []
            # rows shape: [ts, ab_id, campaign_id, variant, post_id]
            postA = None
            postB = None
            for r in rows:
                try:
                    if len(r) >= 5 and r[1] == ab_id:
                        variant = r[3]
                        post_id = r[4]
                        if variant == "A":
                            postA = post_id
                        elif variant == "B":
                            postB = post_id
                except Exception:
                    continue

            if not postA or not postB:
                logger.warning(f"Could not find both A and B posts for ab_id={ab_id}")
                return

            metricsA = self.ingestor.fetch_post_metrics(postA)
            metricsB = self.ingestor.fetch_post_metrics(postB)

            scoreA = (metricsA.get("likes", 0) + metricsA.get("shares", 0) + metricsA.get("replies", 0))
            scoreB = (metricsB.get("likes", 0) + metricsB.get("shares", 0) + metricsB.get("replies", 0))

            winner = "A" if scoreA > scoreB else ("B" if scoreB > scoreA else "tie")

            result_row = [datetime.utcnow().isoformat(), ab_id, postA, scoreA, postB, scoreB, winner]
            if self.sheets_enabled:
                append_row("ab_test_results", result_row)

            if self.slack:
                try:
                    self.slack.send_message(f"A/B Test {ab_id} completed — winner: {winner} (A={scoreA}, B={scoreB})")
                except Exception:
                    pass

            logger.info(f"A/B Test {ab_id} evaluated. Winner: {winner}")

        except Exception as e:
            logger.error(f"Error evaluating AB test {ab_id}: {e}")

    # -------------------------
    # Utility methods
    # -------------------------
    def list_jobs(self):
        return [j.id for j in self.scheduler.get_jobs()]

    def cancel_job(self, job_id: str) -> bool:
        try:
            self.scheduler.remove_job(job_id)
            return True
        except JobLookupError:
            return False

    def shutdown(self):
        try:
            self.scheduler.shutdown(wait=False)
        except Exception:
            pass


# -------------------------
# Example usage (dev)
# -------------------------
if __name__ == "__main__":
    poster = SocialPoster()

    # Example: schedule two posts 1 and 2 minutes from now and evaluate 10 minutes after B
    now = datetime.utcnow()
    runA = now + timedelta(minutes=1)
    runB = now + timedelta(minutes=2)

    details = poster.schedule_ab_test(
        campaign_id="camp_demo",
        textA="Hello world - variant A",
        textB="Hello world - variant B",
        run_date_A=runA,
        run_date_B=runB,
        eval_delay_hours=0.2  # evaluate after 12 minutes (0.2 hours)
    )

    print("Scheduled:", details)
    print("Current jobs:", poster.list_jobs())

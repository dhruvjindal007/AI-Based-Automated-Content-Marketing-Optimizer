import logging
import pandas as pd

# ---------------------------
# IMPORT UPDATED MODULES
# ---------------------------

from app.content_engine.content_generator import generate_final_variations
from app.sentiment_engine.sentiment_analyzer import analyze_sentiment
from app.metrics_engine.tracker import push_raw_feedback, push_aggregates, log_campaign_event
from app.metrics_engine.metrics_tracker import push_daily_metrics
from app.metrics_engine.metrics_hub import record_campaign_metrics

from app.ab_testing.ab_coach import ABCoach
from app.ml_engine.auto_retrainer import AutoRetrainer

from app.integrations.slack_notifier import SlackNotifier

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger("RUN-PIPELINE")
logging.basicConfig(level=logging.INFO)


# -----------------------------------------------------------
# PIPELINE RUNNER
# -----------------------------------------------------------
def run_pipeline():
    logger.info("\n==============================")
    logger.info("🚀 Starting AI Marketing Workflow")
    logger.info("==============================")

    # --------------------------------------
    # STEP 1: CONTENT GENERATION (includes trend optimization internally)
    # --------------------------------------
    # generate_final_variations() already runs TrendBasedOptimizer.run() on
    # each raw variant as part of its own pipeline (see content_generator.py
    # Step 2 -- hashtag cleanup, trend-line injection, and a trend_scores
    # sheet log all happen in there). v["text"] below is that already-
    # optimized, final copy.
    #
    # FIX: this file used to call optimizer.run(v["text"]) a *second* time
    # on that already-optimized text. That re-analysis would (a) sometimes
    # append a second "Trending now: ..." / "People are searching for: ..."
    # block on top of the first, so sentiment analysis ran on contaminated
    # text instead of real marketing copy, and (b) write a second, spurious
    # row to the trend_scores sheet for the same content. Removed entirely
    # -- there is nothing left to optimize here.
    logger.info("\n[1] Generating content variations (trend-optimized)...")

    variations = generate_final_variations(
        topic="AI in Marketing",
        platform="Twitter",
        keywords=["#AI", "#Marketing"],
        audience="Marketers",
        tone="positive",
        n=2
    )

    for i, v in enumerate(variations, 1):
        logger.info(f"\n--- Variant {i} ---\n{v['text']}\n")

    # --------------------------------------
    # STEP 2: SENTIMENT ANALYSIS
    # --------------------------------------
    logger.info("\n[2] Running Sentiment Analysis...")
    sent_items = [v["text"] for v in variations]
    sentiment_out = analyze_sentiment(sent_items)

    # Raw feedback structure for tracker
    raw_logs = []
    for i, s in enumerate(sentiment_out):
        raw_logs.append({
            "id": f"v{i+1}",
            "text": sent_items[i],
            "source": "generated",
        })

    push_raw_feedback(raw_logs)

    # Sentiment aggregates
    avg_sentiment = sum([s["sentiment_score"] for s in sentiment_out]) / len(sentiment_out)
    push_aggregates({
        "total": len(sentiment_out),
        "avg_score": avg_sentiment,
        "pos_count": sum([1 for s in sentiment_out if s["sentiment_label"] == "POSITIVE"]),
        "neg_count": sum([1 for s in sentiment_out if s["sentiment_label"] == "NEGATIVE"]),
        "neu_count": sum([1 for s in sentiment_out if s["sentiment_label"] == "NEUTRAL"]),
        "pct_positive": 0,
        "pct_negative": 0,
        "avg_toxicity": 0,
        "dominant_emotion": "joy",
    })

    # --------------------------------------
    # STEP 3: A/B TEST (SIMPLE VERSION)
    # --------------------------------------
    logger.info("\n[3] Running A/B Comparison (Simple)...")
    coach = ABCoach()

    A = variations[0]["text"]
    B = variations[1]["text"]

    result = coach.simulate_ab(A, B)

    logger.info(f"\nA/B Result: {result}")
    log_campaign_event("A/B Comparison Completed", result)

    # simulate_ab() always returns both key styles now (scoreA/scoreB/winner
    # AND probA/probB/recommended), so no .get() fallback juggling is needed
    # here anymore -- kept as a defensive read in case that ever changes.
    recommended = result.get("recommended") or result.get("winner")
    scoreA = result.get("probA", result.get("scoreA", 0))
    scoreB = result.get("probB", result.get("scoreB", 0))

    if not recommended:
        recommended = "A" if scoreA >= scoreB else "B"

    # Record campaign metrics for ML
    record_campaign_metrics(
        campaign_id="demo_campaign",
        variant=recommended,
        impressions=1000,
        clicks=80,
        conversions=8,
        sentiment_score=scoreA if recommended == "A" else scoreB,
        trend_score=50.0
    )

    # --------------------------------------
    # STEP 4: PUSH DAILY METRICS TO SHEETS
    # --------------------------------------
    logger.info("\n[4] Pushing Metrics to Google Sheets...")
    df = pd.DataFrame({
        "impressions": [1000],
        "clicks": [80],
        "likes": [50],
        "comments": [10],
        "shares": [15],
        "conversions": [8],
        "sentiment_label": ["POSITIVE"],
        "trend_score": [50],
        "toxicity": [0.1],
        "emotions": [{"joy": 0.8}]
    })

    push_daily_metrics(df)

    # --------------------------------------
    # STEP 5: AUTO RETRAIN MODEL
    # --------------------------------------
    logger.info("\n[5] Training ML Model (Auto Retrainer)...")
    try:
        retrainer = AutoRetrainer()
        retrainer.run()
    except Exception as e:
        logger.error(f"Auto Retrainer failed: {e}")

    # --------------------------------------
    # STEP 6: SLACK SUMMARY
    # --------------------------------------
    logger.info("\n[6] Sending Slack Summary...")
    try:
        slack = SlackNotifier()
        slack.send_message(f"A/B Winner: {recommended}\nScore: {result}")
    except Exception as e:
        logger.warning(f"Slack notification failed: {e}")

    logger.info("\n🎉 FULL WORKFLOW COMPLETED SUCCESSFULLY 🎉")


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()
"""
UPDATED run.py
--------------
This script runs the FULL end-to-end AI Content Marketing Optimizer:

Pipeline Steps:
---------------
1. Generate content variations  (content_generator3)
2. Trend optimization           (trend_based_optimizer3)
3. Sentiment analysis           (sentiment_analyzer2)
4. A/B Comparison (simple) OR Full posting  (ABCoach)
5. Log raw feedback + aggregates (tracker3)
6. Push metrics to Google Sheets (metrics_tracker2)
7. Auto-train ML model           (auto_retrainer)
8. Slack report summary          (slack_notifier2)

Run:
----
python run.py
"""

import logging
import pandas as pd

# ---------------------------
# IMPORT UPDATED MODULES
# ---------------------------

from app.content_engine.content_generator import generate_final_variations
from app.sentiment_engine.sentiment_analyzer import analyze_sentiment
from app.content_engine.trend_based_optimizer import TrendBasedOptimizer
from app.metrics_engine.tracker3 import push_raw_feedback, push_aggregates, log_campaign_event
from app.metrics_engine.metrics_tracker2 import push_daily_metrics
from app.metrics_engine.metrics_hub2 import record_campaign_metrics

from app.ab_testing.ab_coach import ABCoach
from app.ml_engine.auto_retrainer import AutoRetrainer

from app.integrations.slack_notifier3 import SlackNotifier

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Trend optimizer instance
optimizer = TrendBasedOptimizer()

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
    # STEP 1: CONTENT GENERATION
    # --------------------------------------
    logger.info("\n[1] Generating content variations...")

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
    # STEP 2: TREND OPTIMIZATION
    # --------------------------------------
    logger.info("\n[2] Applying Trend Optimization...")
    optimized = []
    for v in variations:
        out = optimizer.run(v["text"])
        optimized.append(out)

    # --------------------------------------
    # STEP 3: SENTIMENT ANALYSIS
    # --------------------------------------
    logger.info("\n[3] Running Sentiment Analysis...")
    sent_items = [v["optimized"] for v in optimized]
    sentiment_out = analyze_sentiment(sent_items)

    # Raw feedback structure for tracker3
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
    # STEP 4: A/B TEST (SIMPLE VERSION)
    # --------------------------------------
    logger.info("\n[4] Running A/B Comparison (Simple)...")
    coach = ABCoach()

    A = optimized[0]["optimized"]
    B = optimized[1]["optimized"]

    result = coach.simulate_ab(A, B)

    logger.info(f"\nA/B Result: {result}")
    log_campaign_event("A/B Comparison Completed", result)

    winner_text = A if result["recommended"] == "A" else B

    # Record campaign metrics for ML
    record_campaign_metrics(
        campaign_id="demo_campaign",
        variant=result["recommended"],
        impressions=1000,
        clicks=80,
        conversions=8,
        sentiment_score = result["probA"] if result["recommended"] == "A" else result["probB"],
        trend_score=50.0
    )

    # --------------------------------------
    # STEP 5: PUSH DAILY METRICS TO SHEETS
    # --------------------------------------
    logger.info("\n[5] Pushing Metrics to Google Sheets...")
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
    # STEP 6: AUTO RETRAIN MODEL
    # --------------------------------------
    logger.info("\n[6] Training ML Model (Auto Retrainer)...")
    try:
        retrainer = AutoRetrainer()
        retrainer.run()
    except Exception as e:
        logger.error(f"Auto Retrainer failed: {e}")

    # --------------------------------------
    # STEP 7: SLACK SUMMARY
    # --------------------------------------
    logger.info("\n[7] Sending Slack Summary...")
    try:
        slack = SlackNotifier()
        slack.send_message(f"A/B Winner: {result['recommended']}\nScore: {result}")
    except Exception as e:
        logger.warning(f"Slack notification failed: {e}")

    logger.info("\n🎉 FULL WORKFLOW COMPLETED SUCCESSFULLY 🎉")


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()

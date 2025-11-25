"""
Streamlit App (app4.py)

Run:
    streamlit run app4.py
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime


from app.content_engine.content_generator import generate_final_variations
from app.content_engine.trend_based_optimizer import TrendBasedOptimizer
from app.sentiment_engine.sentiment_analyzer import analyze_sentiment
from app.ab_testing.ab_coach import ABCoach
from app.metrics_engine.metrics_tracker import push_daily_metrics
from app.metrics_engine.metrics_hub import record_campaign_metrics, fetch_recent_metrics
from app.ml_engine.train_model import train
from app.ml_engine.auto_retrainer import AutoRetrainer
from app.integrations.slack_notifier3 import SlackNotifier


# Safe defaults UI
if generate_final_variations is None:
    st.warning("Warning: content generator not found. UI will show placeholders.")
if analyze_sentiment is None:
    st.warning("Warning: sentiment analyzer not found. Sentiment preview disabled.")
if ABCoach is None:
    st.warning("Warning: A/B coach not found. A/B features disabled.")

if "sentiment_output" not in st.session_state:
    st.session_state["sentiment_output"] = None

# ----------------------------
# Streamlit layout
# ----------------------------
st.set_page_config(page_title="AI Content Marketing Optimizer", layout="wide")
st.markdown("""
<style>

/* REAL TAB TEXT */
.stTabs [data-baseweb="tab"] p {
    font-size: 1.10rem !important;
    font-weight: 500 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Active tab text */
.stTabs [aria-selected="true"] p {
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    color: white !important;
}
            

/* Base tab box styling */
.stTabs [data-baseweb="tab"] {
    padding: 14px 26px !important;
    background: #eef1f6;
    border-radius: 12px !important;
    border: 1px solid #d1d9e6;
    transition: all 0.3s ease-in-out;
}

/* Gradient Hover Animation */
.stTabs [data-baseweb="tab"]:hover {
    background: linear-gradient(135deg, #4b9fff, #6f42c1) !important;
    transform: translateY(-4px);
    box-shadow: 0px 6px 15px rgba(0,0,0,0.25);
}
.stTabs [data-baseweb="tab"]:hover p {
    color: white !important;
}

/* Active tab box styling */
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4b9fff, #1f78ff) !important;
    transform: translateY(-2px);
    box-shadow: 0px 6px 18px rgba(75,159,255,0.5);
}

/* Tab spacing */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px !important;
}


/* ---------------------- BUTTON STYLES ---------------------- */

/* Generate button (first button in your app) */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #4b9fff, #6f42c1);
    color: white;
    padding: 12px 25px;
    border-radius: 12px;
    font-size: 1.1rem;
    font-weight: 700;
    border: none;
    transition: all 0.3s ease-in-out;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.20);
}

/* Hover */
div.stButton > button:first-child:hover {
    transform: translateY(-4px);
    background: linear-gradient(135deg, #6f42c1, #4b9fff);
    box-shadow: 0px 8px 20px rgba(79, 134, 255, 0.6);
    cursor: pointer;
}

/* Click */
div.stButton > button:first-child:active {
    transform: scale(0.97);
    transition: 0.1s;
}

</style>
""", unsafe_allow_html=True)



st.title("AI Content Marketing Optimizer — Dashboard")

tabs = st.tabs(["🚀 Content Generator (Trend-Based)", "💬 Sentiment Engine", "🆚 Performance Comparison (A/B Test)", "⚒️ Model Training & Metrics Hub", "🔔 Slack Notifications"])

# ----------------------------
# Generate Tab
# ----------------------------
with tabs[0]:
    st.header("Generate Content Variations")

    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input("Topic / Headline", value="AI in Marketing")
        platform = st.selectbox("Platform", ["Twitter", "LinkedIn", "Facebook", "Instagram"])
        keywords = st.text_input("Keywords (comma-separated)", value="#AI,#Marketing")
        audience = st.text_input("Audience description", value="digital marketers")
        tone = st.selectbox("Tone", ["positive", "neutral", "funny", "professional"])
        word_count = st.slider("Target word count", 20, 200, 50)
        n_variants = st.number_input("Number of variants", min_value=1, max_value=6, value=2)

    with col2:
        st.write("Quick actions")
        if st.button("Generate Variations"):
            if generate_final_variations is None:
                st.error("Generator module not found.")
            else:
                with st.spinner("Generating..."):
                    kws = [k.strip() for k in keywords.split(",") if k.strip()]
                    variants = generate_final_variations(
                        topic=topic,
                        platform=platform,
                        keywords=kws,
                        audience=audience,
                        tone=tone,
                        n=n_variants,
                        word_count=word_count
                    )
                    st.session_state["generated"] = variants
                    st.success(f"Generated {len(variants)} variants")

    # show generated
    gen = st.session_state.get("generated", None)
    if gen:
        for i, g in enumerate(gen, 1):
            st.subheader(f"Variant {i}")
            st.write(g["text"])
            st.json(g["meta"])

# ----------------------------
# Sentiment Engine Tab
# ----------------------------
with tabs[1]:
    st.header("Sentiment Engine")

    generated_variants = st.session_state.get("generated", [])

    # Initialize session storage for sentiment results
    if "sentiment_results" not in st.session_state:
        st.session_state["sentiment_results"] = []

    # If generated content exists
    if generated_variants:
        st.write("Generated Variants Found:")

        if st.button("Analyze All Variants"):
            st.session_state["sentiment_results"] = []  # reset previous results

            for i, v in enumerate(generated_variants):
                variant_text = v["text"]

                # Run sentiment
                sentiment = analyze_sentiment(variant_text)

                if sentiment and isinstance(sentiment, list):
                    result = sentiment[0]

                    # Inject trend score from generated content
                    trend_score = v["meta"].get("trend_score", 0)
                    result["trend_score"] = trend_score

                    # store in session_state
                    st.session_state["sentiment_results"].append({
                        "variant_index": i + 1,
                        "text": variant_text,
                        "sentiment": result
                    })
                else:
                    st.session_state["sentiment_results"].append({
                        "variant_index": i + 1,
                        "text": variant_text,
                        "sentiment": {"error": "No sentiment returned"}
                    })

    # Always display stored results (persistent like Generate tab)
    if st.session_state["sentiment_results"]:
        st.subheader("Sentiment Analysis Results (Persistent)")
        for item in st.session_state["sentiment_results"]:
            st.write(f"### Variant {item['variant_index']}")
            st.write(item["text"])
            st.json(item["sentiment"])

    # Fallback manual analysis
    if not generated_variants:
        sample_text = st.text_area("Paste text to analyze", value="")
        if st.button("Analyze Now"):
            sentiment = analyze_sentiment(sample_text)
            if sentiment and isinstance(sentiment, list):
                st.session_state["sentiment_results"] = [{
                    "variant_index": "-",
                    "text": sample_text,
                    "sentiment": sentiment[0]
                }]



# ----------------------------
# A/B Test Tab (Simple)
# ----------------------------
with tabs[2]:
    st.header("A/B Comparison")

    # Session state setup for storing A/B results
    if "ab_results" not in st.session_state:
        st.session_state["ab_results"] = None

    generated_variants = st.session_state.get("generated", [])

    st.subheader("Select Variants for A/B Testing")

    # If enough generated variants exist
    if len(generated_variants) >= 2:
        variant_labels = [
            f"Variant {i+1}: {v['text'][:60]}..." 
            for i, v in enumerate(generated_variants)
        ]

        colA, colB = st.columns(2)

        with colA:
            choiceA = st.selectbox("Choose Variant A", variant_labels, key="ab_select_A")
        with colB:
            choiceB = st.selectbox("Choose Variant B", variant_labels, key="ab_select_B")

        indexA = variant_labels.index(choiceA)
        indexB = variant_labels.index(choiceB)

        varA = generated_variants[indexA]["text"]
        varB = generated_variants[indexB]["text"]

        st.write("### Variant A")
        st.write(varA)
        st.write("### Variant B")
        st.write(varB)

    else:
        st.info("Generate at least 2 content variants to auto-fill A/B testing.")
        varA = st.text_area("Variant A text", value="AI tools are transforming marketing.")
        varB = st.text_area("Variant B text", value="Marketing teams must adopt AI to grow.")

    # Compare button
    if st.button("Compare Selected Variants"):
        if ABCoach is None:
            st.error("Simple A/B coach is not available.")
        else:
            coach = ABCoach()
            out = coach.simulate_ab(varA, varB)

            # --- Auto Slack Alert for A/B Winner ---
            try:
                notifier = SlackNotifier()
                ab_message = (
                    f"📊 *A/B Test Completed!*\n"
                    f"• *Variant A Score:* {out['scoreA']:.2f}\n"
                    f"• *Variant B Score:* {out['scoreB']:.2f}\n"
                    f"🏆 *Winner:* Variant {out['winner']}\n"
                    f"💬 *Reason:* {out['explanation']}"
                )
                notifier.send_message(ab_message)
                st.success("A/B Test winner shared with your team on Slack! 🎯")

            except Exception as e:
                st.error(f"Slack alert failed: {e}")


            # Save in session_state
            st.session_state["ab_results"] = {
                "scoreA": out["scoreA"],
                "scoreB": out["scoreB"],
                "winner": out["winner"],
                "explanation": out["explanation"],
                "varA": varA,
                "varB": varB
            }

    # Display A/B test results persistently
    if st.session_state["ab_results"]:
        res = st.session_state["ab_results"]
        st.metric("Score A", res["scoreA"])
        st.metric("Score B", res["scoreB"])
        st.write("Winner:", res["winner"])
        st.write("Explanation:", res["explanation"])

        # Recording button
        if st.button("Record this demo campaign (local + Sheets)"):
            try:
                if record_campaign_metrics is None:
                    st.warning("record_campaign_metrics function not available.")
                else:
                    campaign_id = f"demo_{int(datetime.utcnow().timestamp())}"
                    winner = res["winner"]
                    score = res["scoreA"] if winner == "A" else res["scoreB"]

                    record_campaign_metrics(
                        campaign_id=campaign_id,
                        variant=winner,
                        impressions=1000,
                        clicks=int(100 * score),
                        conversions=int(10 * score),
                        sentiment_score=score,
                        trend_score=50.0,
                        platform="auto-platform",
                        post_id=""
                    )
                    st.success("Demo campaign recorded successfully!")
            except Exception as e:
                st.error(f"Error: {e}")



# ----------------------------
# Metrics & Model Tab
# ----------------------------
with tabs[3]:
    st.header("Metrics & Model")

    st.subheader("Push daily metrics")
    st.write("You can push a sample metrics row to Sheets (or local CSV).")
    if push_daily_metrics is None:
        st.info("Metrics push function not available.")
    else:
        sample_metrics = {
            "impressions": int(st.number_input("Impressions", value=1000)),
            "clicks": int(st.number_input("Clicks", value=80)),
            "likes": int(st.number_input("Likes", value=50)),
            "comments": int(st.number_input("Comments", value=10)),
            "shares": int(st.number_input("Shares", value=15)),
            "conversions": int(st.number_input("Conversions", value=8)),
            "trend_score": float(st.slider("Average trend score", 0.0, 100.0, 50.0))
        }

        if st.button("Push Sample Metrics"):
            df = pd.DataFrame([sample_metrics])
            try:
                res = push_daily_metrics(df)
                st.success("Metrics pushed (or computed).")
                st.json(res)
            except Exception as e:
                st.error(f"Pushing metrics failed: {e}")

    st.write("---")
    st.subheader("Manual Model Training")
    if train is None:
        st.info("Train function not found.")
    else:
        if st.button("Train model now (run train_model3.py)"):
            with st.spinner("Training (this may take a while)..."):
                try:
                    stats = train()
                    st.success("Training completed.")
                    st.json(stats)
                except Exception as e:
                    st.error(f"Training failed: {e}")

    st.write("---")
    st.subheader("Auto Retrainer status")
    if AutoRetrainer is None:
        st.info("AutoRetrainer not available.")
    else:
        if st.button("Run Auto Retrainer Now"):
            with st.spinner("Running AutoRetrainer..."):
                try:
                    retr = AutoRetrainer()
                    retr.run_full_cycle()
                    # --- Auto Slack Alert for Auto Retrainer ---
                    try:
                        notifier = SlackNotifier()
                        notifier.send_message(
                            "🤖 *Auto Retrainer Completed Successfully!*\n"
                            "The ML model has been retrained using the latest campaign data. 🚀"
                        )
                        st.success("Model retrained successfully — notification sent to Slack! 📡")

                    except:
                        pass
                    st.success("Auto retrainer executed.")
                except Exception as e:
                    st.error(f"Auto retrainer failed: {e}")

    st.write("---")
    st.subheader("Recent Campaigns (local CSV)")
    try:
        if fetch_recent_metrics is not None:
            df_recent = fetch_recent_metrics(limit=20)
            if not df_recent.empty:
                st.dataframe(df_recent)
            else:
                st.info("No recent campaign data found.")
        else:
            st.info("fetch_recent_metrics not available.")
    except Exception as e:
        st.error(f"Error fetching recent metrics: {e}")

# ----------------------------
# Slack Tab
# ----------------------------
with tabs[4]:
    st.header("Slack Notifications")
    st.write("Send a test Slack message using your configured notifier (webhook).")
    test_msg = st.text_area("Message", value="Hello from AI Marketing Optimizer Team")

    if st.button("Send Slack Test Message"):
        if SlackNotifier is None:
            st.error("SlackNotifier not available.")
        else:
            notifier = SlackNotifier()
            ok = notifier.send_message(test_msg)
            if ok:
                st.success("Slack message sent")
            else:
                st.error("Slack message failed. Check SLACK_WEBHOOK_URL in .env")

    st.write("---")

# ----------------------------
# Footer
# ----------------------------
st.write("---")
st.write("© 2025 AI Content Marketing Optimizer — All modules active and up to date.")

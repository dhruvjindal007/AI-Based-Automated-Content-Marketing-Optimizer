# 🚀 AI-Based Automated Content Marketing Optimizer

> A complete, intelligent, end-to-end automated platform for generating, optimizing, evaluating, and improving digital marketing content — powered by LLMs, sentiment engines, trend analytics, ML predictions, A/B testing, and automated model retraining.

---

## 📌 Overview

The **AI-Based Automated Content Marketing Optimizer** is an advanced AI system designed to **automate and enhance the full digital content marketing lifecycle**, including content creation, trend optimization, sentiment scoring, performance tracking, A/B testing, and continuous ML-based learning.

The system integrates:

* **LLaMA LLMs/Gemini api** for content generation
* **Advanced Sentiment Analysis** & **Emotion Detection**
* **Trend-based Optimization Engine**
* **A/B Testing Coach**
* **Real-time Slack Alerts**
* **Google Sheets for metrics & logging**
* **Auto-Retrainer** for continuous ML model improvement
* **Interactive Streamlit Dashboard**



---

## 🎯 Key Objectives

### ✔️ Fully Automated Content Pipeline

Automatically generate, optimize, and score content using LLMs + sentiment + trend analysis.


### ✔️ Predictive Content Intelligence

AI predicts which content will perform best based on historical campaigns and A/B tests.


### ✔️ Continuous ML Model Retraining

The system self-improves by learning from engagement metrics, sentiment, and A/B test results.


### ✔️ Centralized Dashboard & Analytics

A beautiful Streamlit dashboard for content generation, sentiment analysis, A/B testing, metrics tracking, and model training.


---

## 🧠 System Architecture (High-Level)

### 1. **Content Engine**

**Files:**

* `content_generator.py`
* `dynamic_prompt.py`
* `trend_based_optimizer.py`

**Functions:**

* Generates content variations
* Builds dynamic prompts
* Applies trend-based scoring & optimization

---

### 2. **Sentiment & Emotion Engine**

**File:** `sentiment_analyzer.py`


Key features:

* HuggingFace sentiment + emotion models
* Trend-aware sentiment scoring
* Toxicity estimation
* Multi-language detection
* Google Sheets logging

---

### 3. **A/B Testing Coach**

**File:** `ab_coach.py`


Capabilities:

* A/B variant scoring
* Probability-based performance prediction
* Automated winner selection
* Optional Slack notifications

---

### 4. **Metrics Hub & Tracker**

**Files:**

* `metrics_hub.py`
* `metrics_tracker.py`
* `tracker.py`

Functions:

* Store daily metrics
* Record campaign performance
* Aggregate sentiment & trend scores
* Feed ML training dataset

---

### 5. **ML Engine (Model Training + Auto Retrainer)**

**Files:**

* `train_model.py`
* `auto_retrainer.py`

Features:

* RandomForestClassifier + GridSearchCV
* SMOTE balancing
* Automatic model versioning
* Continuous retraining pipeline
* Slack notifications on completion

---

### 6. **Integrations Layer**

**Files:**

* `sheets_connector.py`
* `slack_notifier.py`
* `trend_fetcher.py`

Used for:

* Google Sheets APIs
* Slack alerts
* Trend scoring system

---

### 7. **Interactive Streamlit Dashboard**

**File:** `streamlit_app.py`


Tabs include:

* Content generation
* Sentiment engine
* A/B comparison
* Metrics hub
* Manual / Auto model training
* Slack notifications

---

### 8. **Pipeline Orchestration**

**File:** `run.py`
Runs the complete workflow:

1. Generate content
2. Optimize using trends
3. Sentiment analysis
4. A/B test
5. Metrics push
6. Auto retrain
7. Slack summary


---

## 🗂️ Project Folder Structure

```
AI-Content-Marketing-Optimizer/
│
├── app/
│   ├── content_engine/
│   │   ├── content_generator.py
│   │   ├── dynamic_prompt.py
│   │   ├── trend_based_optimizer.py
│   │
│   ├── sentiment_engine/
│   │   └── sentiment_analyzer.py
│   │
│   ├── integrations/
│   │   ├── sheets_connector.py
│   │   ├── slack_notifier.py
│   │   ├── trend_fetcher.py
│   │
│   ├── metrics_engine/
│   │   ├── metrics_hub.py
│   │   ├── metrics_tracker.py
│   │   ├── tracker.py
│   │
│   ├── ab_testing/
│   │   └── ab_coach.py
│   │
│   ├── ml_engine/
│   │   ├── train_model.py
│   │   └── auto_retrainer.py
│
├── streamlit_app.py
├── run.py
├── .gitignore
└── README.md
```

---

## 🌟 Features in Detail

### 🚀 AI Content Generator

* Produces multiple content variations
* Platform-specific formatting
* Keyword-injected prompting
* Audience + tone personalization
* Optional trend-based rewriting

📄 *Source:*
`content_generator.py`


---

### 📊 Trend-Aware Optimization

The optimizer enhances generated content by evaluating trending topics & keywords.

---

### 💬 Deep Sentiment + Emotion Analysis

* Sentiment (Positive/Negative/Neutral)
* Polarity values
* Emotion classification (Joy, Anger, Fear, etc.)
* Trend score fusion
* Toxicity estimation


---

### 🆚 A/B Variant Coach

Uses scoring logic to compute probability of success for content A and B.


---

### 📈 Metrics Engine & Sheets Integration

Stores:

* Impressions
* Clicks
* CTR
* Trend score
* Sentiment
* Conversions

Feeds into ML model.

---

### 🤖 ML Model Training & Auto-Retraining

* RandomForestClassifier
* Hyperparameter tuning via GridSearch
* Balanced via SMOTE
* Auto model versioning
* Slack notification after retrain

---

### 📡 Slack Notification System

* A/B test winner summaries
* Auto-retrainer updates
* Custom user messages

---

### 🖥️ Streamlit Dashboard

Complete UI for interacting with all modules:

* Generate
* Analyze
* Compare
* Train
* Log metrics
* Send alerts


---

## ⚙️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/dhruvjindal007/AI-Based-Automated-Content-Marketing-Optimizer.git
cd AI-Content-Marketing-Optimizer
```

### **2. Create & Activate Virtual Environment**

```
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Place your credentials inside:

```
/credentials
.env
```

### **5. Environment Variables**

Create `.env` file:

```
# Groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.7

# Google Gemini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-1.5-flash

# Twitter / X API Keys
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_secret_api_key
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_SECRET=your_access_secret_token
TWITTER_BEARER_TOKEN=your_bearer_token

# Your Google Sheet ID from the URL:
# https://docs.google.com/spreadsheets/d/<THIS_PART>/edit
GOOGLE_SHEET_ID=your_sheet_id

# Path to credentials JSON 
GOOGLE_APPLICATION_CREDENTIALS=credentials/service_account.json

#slack
SLACK_WEBHOOK_URL = your_slack_webhook_url

MODEL_DIR=models
METRICS_RETRY_LIMIT=3

STREAMLIT_DEBUG=false

ENABLE_TRENDING=true
ENABLE_SENTIMENT=true
ENABLE_SHEETS_LOGGING=true
ENABLE_SLACK_NOTIFICATIONS=true

```

(*Ignored automatically via .gitignore*)


### 6. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

### 7. Or run the full pipeline

```bash
python run.py
```

---

## 📊 How the Full Pipeline Works

*(Managed by `run.py`)*


1️⃣ Generate content variations <br>
2️⃣ Optimize using trend analysis <br>
3️⃣ Sentiment & emotion scoring <br>
4️⃣ A/B evaluation <br>
5️⃣ Push metrics to Google Sheets <br>
6️⃣ Auto-retrain ML model <br>
7️⃣ Slack summary notification

---

## 📉 ML Model Output

The training pipeline outputs:

* `models/predictor.joblib` → latest active model
* `models/predictor_TIMESTAMP.joblib` → historical versions


The **AutoRetrainer** also saves:

* `models/model_TIMESTAMP.pkl`

---
## 👥 Contributors

### 👨‍💼 Project Lead  
- **Charan Teja Mangali** — Lead Developer, System Architect & Mentor

### 🎓 Student Contributors  
- **Dhruv Jindal**

## 🤝 Contributing

Pull requests are welcome.
Open an issue for suggestions or improvements.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## ⭐ Support

If you find this project useful, please give it a **⭐ on GitHub** — it motivates further development!

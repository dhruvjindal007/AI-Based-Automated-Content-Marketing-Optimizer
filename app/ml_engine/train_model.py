"""
train_model3.py — UPDATED & INTEGRATED VERSION
----------------------------------------------

Purpose:
--------
Trains the marketing performance prediction model used by:
- A/B Coach (probability scoring)
- AutoRetrainer (continuous learning)
- Campaign Optimization

Integrated With:
----------------
✓ metrics_hub2.get_ml_training_data()   → ML dataset builder  
✓ sentiment_analyzer2 / trend_fetcher   → Already enriched in dataset  
✓ Google Sheets tracking (indirectly)  
✓ AutoRetrainer (uses train_model function)

Model:
------
RandomForestClassifier + GridSearchCV + SMOTE
Outputs:
--------
• models/predictor_TIMESTAMP.joblib
• models/predictor.joblib (latest)
"""

import os
import joblib
import pandas as pd
import numpy as np
import datetime
import logging

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE

# Updated import — now using metrics_hub2 instead of metrics_hub3
from app.metrics_engine.metrics_hub import get_ml_training_data


# ================================================================
# Logging setup
# ================================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    logger.addHandler(h)


# ================================================================
# MODEL OUTPUT PATHS
# ================================================================

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

LATEST_MODEL_PATH = os.path.join(MODEL_DIR, "predictor.joblib")


# ================================================================
# SUCCESS LABELING
# ================================================================

def compute_success_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    success = 1 when conversion_rate > threshold
    threshold = 2%

    Conversion rate is derived as:
    conversions / max(impressions, 1)

    If impressions not provided, fallback to existing dataframe logic.
    """
    df = df.copy()

    if "impressions" in df.columns:
        df["conversion_rate"] = df["conversions"] / df["impressions"].replace(0, np.nan)
    else:
        # fallback: old method
        df["conversion_rate"] = df["conversions"] / df["ctr"].replace(0, np.nan)

    df["conversion_rate"] = df["conversion_rate"].fillna(0)
    df["success"] = (df["conversion_rate"] > 0.02).astype(int)

    return df


# ================================================================
# FEATURE ENGINEERING
# ================================================================

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create robust normalized features for the RF classifier.
    
    Expected columns from metrics_hub2:
        ctr, sentiment, polarity, conversions, trend_score
    """
    df = df.copy()

    df["ctr_norm"] = df["ctr"].clip(0, 1)
    df["sentiment_norm"] = df["sentiment"].clip(0, 1)
    df["polarity_norm"] = df["polarity"].clip(-1, 1)
    df["trend_norm"] = (df["trend_score"] / 100.0).clip(0, 1)

    features = df[[
        "ctr_norm",
        "sentiment_norm",
        "polarity_norm",
        "trend_norm",
        "conversions"
    ]]

    return features


# ================================================================
# TRAIN FUNCTION
# ================================================================

def train():
    logger.info("Loading ML training dataset...")
    df = get_ml_training_data()

    if df.empty:
        raise ValueError(
            "Training dataset is empty. "
            "Run campaigns and collect A/B metrics first."
        )

    # Step 1 — Labeling
    df = compute_success_label(df)

    # Step 2 — Feature Engineering
    X = feature_engineer(df)
    y = df["success"]

    logger.info(f"Dataset size: {len(df)} rows")
    logger.info("Label counts:\n" + str(y.value_counts()))

    # Step 3 — Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Step 4 — SMOTE Balancing
    logger.info("Balancing classes with SMOTE...")
    sm = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)

    logger.info("Balanced label counts:\n" + str(y_train_balanced.value_counts()))

    # Step 5 — Base Model
    base_model = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        max_depth=None
    )

    # Step 6 — GridSearch Hyperparameter Tuning
    param_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth": [None, 8, 14],
        "min_samples_split": [2, 5],
    }

    grid = GridSearchCV(
        base_model,
        param_grid,
        scoring="roc_auc",
        n_jobs=-1,
        cv=3,
        verbose=1
    )

    logger.info("Starting GridSearchCV...")
    grid.fit(X_train_balanced, y_train_balanced)

    best_model = grid.best_estimator_
    logger.info(f"Best hyperparameters: {grid.best_params_}")

    # Step 7 — Evaluation
    preds = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    pred_labels = best_model.predict(X_test)

    f1 = f1_score(y_test, pred_labels)
    precision = precision_score(y_test, pred_labels)
    recall = recall_score(y_test, pred_labels)
    cm = confusion_matrix(y_test, pred_labels)

    logger.info(f"AUC: {auc:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    # Step 8 — Save Model
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    versioned_path = os.path.join(MODEL_DIR, f"predictor_{timestamp}.joblib")

    joblib.dump(best_model, versioned_path)
    joblib.dump(best_model, LATEST_MODEL_PATH)

    logger.info(f"Model saved: {versioned_path}")
    logger.info(f"Latest model updated: {LATEST_MODEL_PATH}")

    return {
        "auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm.tolist(),
        "model_path": versioned_path,
        "best_params": grid.best_params_
    }


# ================================================================
# RUN MANUALLY
# ================================================================

if __name__ == "__main__":
    results = train()
    print("\nTraining Results:")
    print(results)

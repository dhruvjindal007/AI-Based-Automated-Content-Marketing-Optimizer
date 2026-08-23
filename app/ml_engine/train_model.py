"""
train_model.py
----------------

Trains the marketing performance ("success") prediction model used by:
    - A/B Coach (probability scoring)
    - Campaign Optimization

Pipeline:
    metrics_hub.get_ml_training_data() -> label -> feature engineer
    -> stratified split -> SMOTE (guarded) -> GridSearchCV(RandomForest)
    -> evaluate -> save versioned + latest joblib artifacts

Outputs:
    models/predictor_<timestamp>.joblib
    models/predictor.joblib   (latest)
    models/pairwise_predictor_<timestamp>.joblib
    models/pairwise_predictor.joblib   (latest)

TWO SEPARATE MODELS IN THIS MODULE:
    train(df=None, config=None)
        Single-post conversion "success" predictor, from
        (ctr, sentiment, polarity, trend, conversions). Loads its own
        dataset via get_ml_training_data() unless a df is passed in.
        Used by A/B Coach / Campaign Optimization.

    train_pairwise(X, y, config=None)
        A/B "winner" predictor, from the 6-column pairwise feature set
        AutoRetrainer builds (sentA, sentB, trendA, trendB, engA, engB).
        This is what AutoRetrainer should import and call -- it must
        NOT call `train`, since that function's feature schema and
        label are unrelated to the pairwise A/B-winner problem.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

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

from app.metrics_engine.metrics_hub import get_ml_training_data


# ================================================================
# Logging setup
# ================================================================

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# ================================================================
# CONFIG
# ================================================================

RAW_FEATURE_COLUMNS = ["ctr", "sentiment", "polarity", "trend_score", "conversions"]
ENGINEERED_FEATURE_COLUMNS = [
    "ctr_norm",
    "sentiment_norm",
    "polarity_norm",
    "trend_norm",
    "conversions",
]


@dataclass
class TrainConfig:
    model_dir: str = "models"
    success_threshold: float = 0.02  # conversion_rate above this counts as "success"
    test_size: float = 0.25
    random_state: int = 42
    param_grid: dict = field(
        default_factory=lambda: {
            "n_estimators": [100, 150, 200],
            "max_depth": [None, 8, 14],
            "min_samples_split": [2, 5],
        }
    )
    cv_folds: int = 3

    @property
    def latest_model_path(self) -> str:
        return os.path.join(self.model_dir, "predictor.joblib")


@dataclass
class PairwiseTrainConfig:
    """Config for the A/B-winner model trained on AutoRetrainer's
    pairwise feature set. Kept deliberately separate from TrainConfig
    since the two models have nothing in common except sharing this
    module -- different features, different label, different use."""

    model_dir: str = "models"
    test_size: float = 0.25
    random_state: int = 42
    n_estimators: int = 150
    max_depth: Optional[int] = None

    @property
    def latest_model_path(self) -> str:
        return os.path.join(self.model_dir, "pairwise_predictor.joblib")


# ================================================================
# SUCCESS LABELING
# ================================================================

def compute_success_label(df: pd.DataFrame, threshold: float = 0.02) -> pd.DataFrame:
    """
    success = 1 when conversion_rate > threshold.

    Preferred: conversion_rate = conversions / impressions.
    Fallback (when "impressions" isn't in the dataset): conversions / ctr.
    That fallback is dimensionally unusual -- ctr is a rate, not a count --
    so it's logged loudly rather than applied silently. If you're hitting
    this path often, the upstream dataset is probably missing a column it
    should have.
    """
    df = df.copy()

    if "impressions" in df.columns:
        df["conversion_rate"] = df["conversions"] / df["impressions"].replace(0, np.nan)
    else:
        logger.warning(
            "'impressions' column missing -- falling back to conversions/ctr "
            "for conversion_rate, which may not be meaningful."
        )
        df["conversion_rate"] = df["conversions"] / df["ctr"].replace(0, np.nan)

    df["conversion_rate"] = df["conversion_rate"].fillna(0)
    df["success"] = (df["conversion_rate"] > threshold).astype(int)

    return df


# ================================================================
# FEATURE ENGINEERING
# ================================================================

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build normalized features for the RF classifier.

    Expected raw columns: ctr, sentiment, polarity, trend_score, conversions.
    Raises ValueError (with the missing column names) instead of a raw
    KeyError if the upstream dataset shape has drifted.
    """
    missing = [c for c in RAW_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")

    df = df.copy()

    df["ctr_norm"] = df["ctr"].clip(0, 1)
    df["sentiment_norm"] = df["sentiment"].clip(0, 1)
    df["polarity_norm"] = df["polarity"].clip(-1, 1)
    df["trend_norm"] = (df["trend_score"] / 100.0).clip(0, 1)

    features = df[ENGINEERED_FEATURE_COLUMNS].copy()

    # RandomForestClassifier can't handle NaNs -- impute rather than let
    # a stray missing value blow up training deep inside sklearn.
    if features.isna().any().any():
        na_counts = features.isna().sum()
        logger.warning("Found NaNs in features, filling with column medians: %s",
                        na_counts[na_counts > 0].to_dict())
        features = features.fillna(features.median(numeric_only=True))

    return features


# ================================================================
# CLASS-BALANCING HELPER
# ================================================================

def _safe_smote_resample(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE, but degrade gracefully instead of crashing when the
    minority class is too small. SMOTE's default k_neighbors=5 needs at
    least 6 minority samples; with fewer, it raises. We shrink k or skip
    balancing entirely rather than let the whole training run die.
    """
    class_counts = y_train.value_counts()
    if len(class_counts) < 2:
        logger.warning(
            "Only one class present in training labels (%s) -- skipping SMOTE.",
            class_counts.to_dict(),
        )
        return X_train, y_train

    minority_count = int(class_counts.min())
    if minority_count < 2:
        logger.warning(
            "Minority class has only %d sample(s) -- skipping SMOTE.", minority_count
        )
        return X_train, y_train

    k_neighbors = min(5, minority_count - 1)
    if k_neighbors < 1:
        k_neighbors = 1

    logger.info("Balancing classes with SMOTE (k_neighbors=%d)...", k_neighbors)
    sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_bal, y_bal = sm.fit_resample(X_train, y_train)
    logger.info("Balanced label counts:\n%s", y_bal.value_counts())
    return X_bal, y_bal


# ================================================================
# PAIRWISE A/B-WINNER TRAIN FUNCTION (for AutoRetrainer)
# ================================================================

def train_pairwise(
    X: pd.DataFrame, y: pd.Series, config: Optional[PairwiseTrainConfig] = None
) -> Any:
    """
    Train the A/B "which post won" classifier.

    This is the function AutoRetrainer should import and call:

        from app.ml_engine.train_model import train_pairwise
        model = train_pairwise(X, y)

    X is expected to have exactly the 6 columns AutoRetrainer builds:
    sentA, sentB, trendA, trendB, engA, engB. y is the 0/1 winner label.

    Returns the fitted model object directly (not a dict), matching
    what AutoRetrainer's save_model() expects to joblib.dump().
    """
    cfg = config or PairwiseTrainConfig()
    os.makedirs(cfg.model_dir, exist_ok=True)

    if X.empty or y.empty:
        raise ValueError("train_pairwise received an empty dataset.")

    if X.isna().any().any():
        na_counts = X.isna().sum()
        logger.warning(
            "Found NaNs in pairwise features, filling with column medians: %s",
            na_counts[na_counts > 0].to_dict(),
        )
        X = X.fillna(X.median(numeric_only=True))

    if y.nunique() < 2:
        raise ValueError(
            "Pairwise training labels contain only one class -- "
            "need both A-wins and B-wins examples to train a classifier."
        )

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=stratify
    )

    X_train_balanced, y_train_balanced = _safe_smote_resample(
        X_train, y_train, cfg.random_state
    )

    model = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
    )
    model.fit(X_train_balanced, y_train_balanced)

    # Lightweight evaluation for logging only -- AutoRetrainer's own
    # save_model/versioning handles persistence, so we don't duplicate
    # the joblib.dump calls train() does for the success model.
    if len(X_test) > 0:
        pred_labels = model.predict(X_test)
        try:
            f1 = f1_score(y_test, pred_labels)
            logger.info("Pairwise model F1 on holdout: %.4f", f1)
        except ValueError as exc:
            logger.warning("Could not score pairwise holdout set: %s", exc)

    return model


# ================================================================
# TRAIN FUNCTION
# ================================================================

def train(
    df: Optional[pd.DataFrame] = None, config: Optional[TrainConfig] = None
) -> dict[str, Any]:
    """
    Train the conversion-success predictor.

    Parameters
    ----------
    df:
        Optional pre-loaded dataset (mainly for tests). If omitted, loads
        via get_ml_training_data() as before.
    config:
        Optional TrainConfig to override thresholds/paths/grid without
        editing this module.

    NOTE: this does NOT accept (X, y) pairwise feature arrays -- see the
    module docstring for the mismatch with AutoRetrainer's expected
    `train(X, y)` signature.
    """
    cfg = config or TrainConfig()
    os.makedirs(cfg.model_dir, exist_ok=True)

    if df is None:
        logger.info("Loading ML training dataset...")
        df = get_ml_training_data()

    if df.empty:
        raise ValueError(
            "Training dataset is empty. Run campaigns and collect A/B metrics first."
        )

    # Step 1 — Labeling
    df = compute_success_label(df, threshold=cfg.success_threshold)

    # Step 2 — Feature Engineering
    X = feature_engineer(df)
    y = df["success"]

    logger.info("Dataset size: %d rows", len(df))
    logger.info("Label counts:\n%s", y.value_counts())

    if y.nunique() < 2:
        raise ValueError(
            "Training data contains only one class for 'success' -- "
            "can't train a classifier. Collect more varied outcomes first."
        )

    # Step 3 — Stratified split (keeps both classes present in the test
    # set even when the label is imbalanced; the original random split
    # could leave the test set with a single class, breaking roc_auc_score)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    # Step 4 — SMOTE Balancing (guarded)
    X_train_balanced, y_train_balanced = _safe_smote_resample(
        X_train, y_train, cfg.random_state
    )

    # Step 5 — Base Model
    base_model = RandomForestClassifier(
        n_estimators=120, random_state=cfg.random_state, max_depth=None
    )

    # Step 6 — GridSearch Hyperparameter Tuning
    grid = GridSearchCV(
        base_model,
        cfg.param_grid,
        scoring="roc_auc",
        n_jobs=-1,
        cv=cfg.cv_folds,
        verbose=1,
    )

    logger.info("Starting GridSearchCV...")
    try:
        grid.fit(X_train_balanced, y_train_balanced)
    except Exception as exc:
        logger.error("GridSearchCV failed: %s", exc)
        raise

    best_model = grid.best_estimator_
    logger.info("Best hyperparameters: %s", grid.best_params_)

    # Step 7 — Evaluation
    preds = best_model.predict_proba(X_test)[:, 1]
    pred_labels = best_model.predict(X_test)

    try:
        auc = roc_auc_score(y_test, preds)
    except ValueError as exc:
        # Can still happen if the test split is degenerate despite stratify
        # (e.g. an extremely small dataset). Don't let evaluation crash
        # the whole training run -- surface it as NaN and keep going.
        logger.warning("Could not compute AUC: %s", exc)
        auc = float("nan")

    f1 = f1_score(y_test, pred_labels)
    precision = precision_score(y_test, pred_labels, zero_division=0)
    recall = recall_score(y_test, pred_labels, zero_division=0)
    cm = confusion_matrix(y_test, pred_labels)

    logger.info("AUC: %.4f", auc)
    logger.info("F1 Score: %.4f", f1)
    logger.info("Precision: %.4f", precision)
    logger.info("Recall: %.4f", recall)
    logger.info("Confusion Matrix:\n%s", cm)

    # Step 8 — Save Model
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    versioned_path = os.path.join(cfg.model_dir, f"predictor_{timestamp}.joblib")

    joblib.dump(best_model, versioned_path)
    joblib.dump(best_model, cfg.latest_model_path)

    logger.info("Model saved: %s", versioned_path)
    logger.info("Latest model updated: %s", cfg.latest_model_path)

    return {
        "auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm.tolist(),
        "model_path": versioned_path,
        "best_params": grid.best_params_,
        "n_samples": len(df),
    }


# ================================================================
# RUN MANUALLY
# ================================================================

if __name__ == "__main__":
    results = train()
    print("\nTraining Results:")
    print(results)
"""
Utility functions for evaluating predictive maintenance models.
"""

from __future__ import annotations

import logging

import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def predict_labels(
    model,
    X: pd.DataFrame,
    threshold: float,
):
    """
    Generate prediction probabilities and binary predictions.

    Parameters
    ----------
    model
        Trained classification model.

    X : pd.DataFrame
        Input features.

    threshold : float
        Classification threshold.

    Returns
    -------
    tuple
        (probabilities, predictions)
    """

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return probabilities, predictions


from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    roc_auc_score,
)

def evaluate_predictions(
    y_true,
    y_pred,
    y_prob,
):
    """
    Compute classification metrics.

    Returns
    -------
    tuple
        report_dict,
        report_text,
        roc_auc,
        balanced_accuracy
    """

    # Dictionary for programmatic use
    report_dict = classification_report(
        y_true,
        y_pred,
        output_dict=True,
    )

    # Text version for logging/reporting
    report_text = classification_report(
        y_true,
        y_pred,
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob,
    )

    balanced_acc = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    return (
        report_dict,
        report_text,
        roc_auc,
        balanced_acc,
    )


def log_model_information(
    classifier: xgb.XGBClassifier,
) -> None:
    """
    Log important model information.
    """

    logger.info("=" * 60)
    logger.info("Model Information")
    logger.info("=" * 60)

    logger.info(
        "Estimator : %s",
        classifier,
    )

    logger.info(
        "Classes : %s",
        classifier.classes_,
    )

    logger.info(
        "Objective : %s",
        classifier.get_xgb_params().get("objective"),
    )

    logger.info(
        "Booster : %s",
        classifier.get_xgb_params().get("booster"),
    )

    logger.info(
        "Tree Method : %s",
        classifier.get_xgb_params().get("tree_method"),
    )

    logger.info(
        "Learning Rate : %s",
        classifier.learning_rate,
    )

    logger.info(
        "Max Depth : %s",
        classifier.max_depth,
    )

    logger.info(
        "n_estimators : %s",
        classifier.n_estimators,
    )

    logger.info("=" * 60)

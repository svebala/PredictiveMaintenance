"""
Utility functions for generating and saving model artifacts.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import ConfusionMatrixDisplay

logger = logging.getLogger(__name__)


def save_artifacts(
    classifier,
    search,
    train_report,
    test_report,
    test_auc,
    balanced_acc,
    ytest,
    y_pred_test,
    feature_names,
    artifact_root,
    top_features,
    figsize,
):
    """
    Save all model artifacts locally.

    Returns
    -------
    Path
        Artifact directory.
    """

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving artifacts to %s", artifact_dir)

    # --------------------------------------------------
    # Classification Report
    # --------------------------------------------------

    report_path = artifact_dir / "classification_report.csv"

    pd.DataFrame(test_report).transpose().to_csv(
        report_path
    )

    # --------------------------------------------------
    # Best Parameters
    # --------------------------------------------------

    params_path = artifact_dir / "best_params.json"

    with open(params_path, "w") as fp:

        json.dump(
            search.best_params_,
            fp,
            indent=4,
        )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    metrics = {

        "train_accuracy":
            train_report["accuracy"],

        "test_accuracy":
            test_report["accuracy"],

        "test_precision":
            test_report["1"]["precision"],

        "test_recall":
            test_report["1"]["recall"],

        "test_f1":
            test_report["1"]["f1-score"],

        "roc_auc":
            test_auc,

        "balanced_accuracy":
            balanced_acc,

    }

    metrics_path = artifact_dir / "metrics.json"

    with open(metrics_path, "w") as fp:

        json.dump(
            metrics,
            fp,
            indent=4,
        )

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------

    cm_path = artifact_dir / "confusion_matrix.png"

    fig, ax = plt.subplots(figsize=figsize)

    ConfusionMatrixDisplay.from_predictions(
        ytest,
        y_pred_test,
        ax=ax,
    )

    ax.set_title("Confusion Matrix")

    fig.savefig(cm_path)

    plt.close(fig)

    # --------------------------------------------------
    # Feature Importance
    # --------------------------------------------------

    fi_path = artifact_dir / "feature_importance.png"

    plt.figure(figsize=figsize)

    xgb.plot_importance(
        classifier,
        max_num_features=top_features,
    )

    plt.title("Top Feature Importances")

    plt.tight_layout()

    plt.savefig(fi_path)

    plt.close()

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------

    metadata = {

        "timestamp":
            datetime.now().isoformat(),

        "model":
            "XGBoost",

        "features":
            list(feature_names),

        "best_parameters":
            search.best_params_,

        "metrics":
            metrics,

    }

    metadata_path = artifact_dir / "model_metadata.json"

    with open(metadata_path, "w") as fp:

        json.dump(
            metadata,
            fp,
            indent=4,
        )

    logger.info("Artifacts saved successfully.")

    return artifact_dir

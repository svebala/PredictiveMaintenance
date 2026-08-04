"""
Trains the predictive maintenance model,
tracks experiments with MLflow,
and uploads the trained model to Hugging Face.
"""

import logging
import joblib
import mlflow
import xgboost as xgb
from pathlib import Path

from huggingface_hub import HfApi
from sklearn.compose import make_column_transformer

from sklearn.model_selection import (
    RandomizedSearchCV,
)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from predictive_maintenance.utils.data_utils import load_dataset

from predictive_maintenance.utils.evaluation_utils import (
    predict_labels,
    evaluate_predictions,
    log_model_information,
)

from predictive_maintenance.utils.hf_utils import (
    ensure_repo_exists,
    upload_file,
    upload_directory,
)

from predictive_maintenance.utils.artifact_utils import (
    save_artifacts,
)

# import constants from config file
from predictive_maintenance.config import (
    HF_DATASET_REPO,
    HF_DATASET_TYPE,
    HF_MODEL_REPO,
    HF_MODEL_TYPE,
    HF_TOKEN,
    HF_MAX_RETRIES,
    HF_DATASET_PATH,
    XTRAIN_FILE,
    XTEST_FILE,
    YTRAIN_FILE,
    YTEST_FILE,
    MODEL_FILENAME,
    RANDOM_STATE,
    CLASSIFICATION_THRESHOLD,
    CV_FOLDS,
    N_JOBS,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_ARTIFACT_PATH,
    NUMERIC_FEATURES,
    SCORING_METRIC,
    ARTIFACT_ROOT,
    N_ITER,
    TOP_FEATURES,
    FIGSIZE,
    SAVE_PLOTS,
    UPLOAD_ARTIFACTS,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# Get token information
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set.")

# Authenticate with Hugging Face
api = HfApi(token=HF_TOKEN)

# Load processed splits
XTRAIN_PATH = f"{HF_DATASET_PATH}/{XTRAIN_FILE}"
XTEST_PATH = f"{HF_DATASET_PATH}/{XTEST_FILE}"
YTRAIN_PATH = f"{HF_DATASET_PATH}/{YTRAIN_FILE}"
YTEST_PATH = f"{HF_DATASET_PATH}/{YTEST_FILE}"

Xtrain = load_dataset(XTRAIN_PATH)
Xtest = load_dataset(XTEST_PATH)

ytrain = load_dataset(YTRAIN_PATH).squeeze()
ytest = load_dataset(YTEST_PATH).squeeze()

if Xtrain.empty:
    raise ValueError("Training dataset is empty.")

if Xtest.empty:
    raise ValueError("Test dataset is empty.")

if ytrain.empty or ytest.empty:
    raise ValueError("Target labels are empty.")

# Ensure lists only contain columns that actually exist after dropping unwanted ones
missing_features = [
    col for col in NUMERIC_FEATURES
    if col not in Xtrain.columns
]

if missing_features:
    logger.warning(
        f"Missing numeric features: {missing_features}"
    )

numeric_features = [
    col for col in NUMERIC_FEATURES
    if col in Xtrain.columns
]

# Set the class weight to handle class imbalance
class_counts = ytrain.value_counts()
class_weight = class_counts.loc[0] / class_counts.loc[1]

# Define the preprocessing steps with passthrough for additional columns
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    remainder="passthrough"
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    eval_metric="logloss",
    verbose=2,
)

# Define hyperparameter grid
param_distributions = {
    "xgbclassifier__n_estimators": [100, 150, 200],
    "xgbclassifier__max_depth": [3, 4, 5],                  # Shallowed trees limit memorization
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__subsample": [0.7, 0.8, 0.9],            # Introduces dataset row sampling variance
    "xgbclassifier__colsample_bytree": [0.7, 0.8, 0.9],     # Introduces column feature sampling variance
    "xgbclassifier__min_child_weight": [3, 5, 7],           # Higher minimum limits partition sizes
    "xgbclassifier__gamma": [0.1, 0.2, 0.3],                # Enforces minimum loss reduction split barriers
    "xgbclassifier__reg_alpha": [0.1, 0.5, 1.0],            # L1 Regularization penalizes complex nodes
    "xgbclassifier__reg_lambda": [1.5, 2.0, 3.0]            # L2 Regularization penalizes large leaf weights
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Ensure any active MLflow run is ended before starting a new one
if mlflow.active_run():
    mlflow.end_run()

# Start Model Training & MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    search = RandomizedSearchCV(
        estimator=model_pipeline,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        cv=CV_FOLDS,
        scoring=SCORING_METRIC,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )

    logger.info("Running RandomizedSearchCV...")
    search.fit(Xtrain, ytrain)
    logger.info("Hyperparameter search completed.")

    # Log all parameter combinations and their mean test scores
    results = search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(run_name="XGBoost_Training", nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best hyperparameters
    mlflow.log_params(search.best_params_)

    mlflow.log_metric("best_cv_f1", search.best_score_)
    mlflow.log_param("scale_pos_weight", class_weight)

    # Log model configuration
    mlflow.log_params({
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "scoring_metric": SCORING_METRIC,
    })

    # Store and evaluate the best model
    best_model = search.best_estimator_

    classifier = best_model.named_steps["xgbclassifier"]

    log_model_information(classifier)

    y_pred_train_proba, y_pred_train = predict_labels(
        best_model,
        Xtrain,
        CLASSIFICATION_THRESHOLD,
    )

    y_pred_test_proba, y_pred_test = predict_labels(
        best_model,
        Xtest,
        CLASSIFICATION_THRESHOLD,
    )

    train_report, train_report_text, _, _ = evaluate_predictions(
        ytrain,
        y_pred_train,
        y_pred_train_proba,
    )

    test_report, test_report_text, test_auc, balanced_acc = evaluate_predictions(
        ytest,
        y_pred_test,
        y_pred_test_proba,
    )

    mlflow.log_metric("test_auc", test_auc)
    mlflow.log_metric("balanced_accuracy", balanced_acc)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    logger.info(f"Best Parameters : {search.best_params_}")
    logger.info(f"Best CV Score   : {search.best_score_:.4f}")

    logger.info(
        "Training Classification Report\n%s",
        train_report_text,
    )

    logger.info(
        "Test Classification Report\n%s",
        test_report_text,
    )

    # Save the model locally
    joblib.dump(best_model, MODEL_FILENAME)

    # Log the model artifact
    mlflow.log_artifact(MODEL_FILENAME, artifact_path=MLFLOW_ARTIFACT_PATH)
    logger.info(f"Model saved as artifact at: {MODEL_FILENAME}")

    artifact_dir = None

    if SAVE_PLOTS:
      artifact_dir = save_artifacts(
          classifier=classifier,
          search=search,
          train_report=train_report,
          test_report=test_report,
          test_auc=test_auc,
          balanced_acc=balanced_acc,
          ytest=ytest,
          y_pred_test=y_pred_test,
          feature_names=Xtrain.columns,
          artifact_root=ARTIFACT_ROOT,
          top_features=TOP_FEATURES,
          figsize=FIGSIZE,
      )

    if UPLOAD_ARTIFACTS and artifact_dir is not None:
      upload_directory(
          api=api,
          artifact_dir=artifact_dir,
          repo_id=HF_DATASET_REPO,
          repo_type=HF_DATASET_TYPE,
          run_id=artifact_dir.name,
      )

    # Check if the model repository exists
    ensure_repo_exists(
        api=api,
        repo_id=HF_MODEL_REPO,
        repo_type=HF_MODEL_TYPE,
        token=HF_TOKEN,
        max_retries=HF_MAX_RETRIES,
    )

    # Upload Model
    upload_file(
        api=api,
        file_path=Path(MODEL_FILENAME),
        repo_id=HF_MODEL_REPO,
        repo_type=HF_MODEL_TYPE,
    )

"""
Trains the predictive maintenance model,
tracks experiments with MLflow,
and uploads the trained model to Hugging Face.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# for model serialization
import joblib

import time

# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    HfHubHTTPError,
)
import mlflow

# import constants from config file
from predictive_maintenance.config import (
    HF_DATASET_REPO,
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
)

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

Xtrain = pd.read_csv(XTRAIN_PATH)
Xtest = pd.read_csv(XTEST_PATH)
ytrain = pd.read_csv(YTRAIN_PATH)
ytest = pd.read_csv(YTEST_PATH)

# Ensure lists only contain columns that actually exist after dropping unwanted ones
numeric_features = [col for col in NUMERIC_FEATURES if col in Xtrain.columns]

# Set the class weight to handle class imbalance
class_counts = ytrain.squeeze().value_counts()
class_weight = class_counts.loc[0] / class_counts.loc[1]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=RANDOM_STATE)

# Define hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [100, 200, 300],       # number of tree to build
    "xgbclassifier__max_depth": [3, 5, 7],                # maximum depth of each tree
    "xgbclassifier__colsample_bytree": [0.4, 0.5, 0.6],   # percentage of attributes to be considered (randomly) for each tree
    "xgbclassifier__colsample_bylevel": [0.4, 0.5, 0.6],  # percentage of attributes to be considered (randomly) for each level of a tree
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],    # learning rate
    "xgbclassifier__reg_lambda": [0.4, 0.5, 0.6],         # L2 regularization factor
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=CV_FOLDS,
        scoring=SCORING_METRIC,
        n_jobs=N_JOBS,
    )
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    cv_results = grid_search.cv_results_
    for i in range(len(cv_results['params'])):
        param_set = cv_results['params'][i]
        mean_score = cv_results['mean_test_score'][i]
        std_score = cv_results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_recall", grid_search.best_score_)
    mlflow.log_param("scale_pos_weight", class_weight,)

    # Log model configuration
    mlflow.log_params({
      "classification_threshold": CLASSIFICATION_THRESHOLD,
      "cv_folds": CV_FOLDS,
      "random_state": RANDOM_STATE,
      "scoring_metric": SCORING_METRIC,
    })

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= CLASSIFICATION_THRESHOLD).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= CLASSIFICATION_THRESHOLD).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

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

    # Save the model locally
    joblib.dump(best_model, MODEL_FILENAME)

    # Log the model artifact
    mlflow.log_artifact(MODEL_FILENAME, artifact_path=MLFLOW_ARTIFACT_PATH)
    print(f"Model saved as artifact at: {MODEL_FILENAME}")

    # Check if the model repository exists
    for attempt in range(HF_MAX_RETRIES):
        try:
            api.repo_info(
                repo_id=HF_MODEL_REPO,
                repo_type=HF_MODEL_TYPE,
            )
            print(f"✅ Model '{HF_MODEL_REPO}' already exists.")
            break

        except RepositoryNotFoundError:
            print(f"Model '{HF_MODEL_REPO}' not found. Creating...")
            create_repo(
                repo_id=HF_MODEL_REPO,
                repo_type=HF_MODEL_TYPE,
                private=False,
                token=HF_TOKEN,
                exist_ok=True,
            )
            print("✅ Model created.")
            break

        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                wait = 2 ** attempt
                print(f"Rate limited. Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                raise

    # Upload Model
    api.upload_file(
        path_or_fileobj=MODEL_FILENAME,
        path_in_repo=MODEL_FILENAME,
        repo_id=HF_MODEL_REPO,
        repo_type=HF_MODEL_TYPE,
    )
    print("✅ Model uploaded successfully.")

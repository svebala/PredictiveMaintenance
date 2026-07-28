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

# for creating a folder
import os

# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

# Parameters
## Hugging Face
HF_DATASET_REPO = "BalaSVenkat/predictive-maintenance-dataset"
HF_MODEL_REPO = "BalaSVenkat/predictive-maintenance-model"
MODEL_FILENAME = "engine_predictive_maintenance_model.joblib"
MODEL_REPO_TYPE = "model"
## MLflow
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "engine-predictive-maintenance"
MLFLOW_ARTIFACT_PATH = "model"
## Model
RANDOM_STATE = 42
CLASSIFICATION_THRESHOLD = 0.35 # Lower decision threshold to prioritize recall. In predictive maintenance, missing a faulty engine (false negative) is generally more costly than raising an unnecessary maintenance alert (false positive).
CV_FOLDS = 5
N_JOBS = -1

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

api = HfApi(token=os.getenv("HF_TOKEN"))

# Load processed splits
XTRAIN_PATH = f"hf://datasets/{HF_DATASET_REPO}/Xtrain.csv"
XTEST_PATH = f"hf://datasets/{HF_DATASET_REPO}/Xtest.csv"
YTRAIN_PATH = f"hf://datasets/{HF_DATASET_REPO}/ytrain.csv"
YTEST_PATH = f"hf://datasets/{HF_DATASET_REPO}/ytest.csv"

Xtrain = pd.read_csv(XTRAIN_PATH)
Xtest = pd.read_csv(XTEST_PATH)
ytrain = pd.read_csv(YTRAIN_PATH)
ytest = pd.read_csv(YTEST_PATH)

# List of numerical features in the dataset
numeric_features = [
    'engine_rpm',       # The number of revolutions per minute (RPM) of the engine, indicating engine speed.
    'lub_oil_pressure', # The pressure of the lubricating oil in the engine, essential for reducing friction and wear.
    'fuel_pressure',    # The pressure at which fuel is supplied to the engine, critical for proper combustion.
    'coolant_pressure', # The pressure of the engine coolant, affecting engine temperature regulation.
    'lub_oil_temp',     # The temperature of the lubricating oil, which impacts viscosity and engine performance.
    'coolant_temp'      # The temperature of the engine coolant, crucial for preventing overheating.
]

# Ensure lists only contain columns that actually exist after dropping unwanted ones
numeric_features = [col for col in numeric_features if col in Xtrain.columns]

# Set the class weight to handle class imbalance
counts = ytrain.squeeze().value_counts()
class_weight = counts.loc[0] / counts.loc[1]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=RANDOM_STATE)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [100, 200, 300],       # number of tree to build
    'xgbclassifier__max_depth': [3, 5, 7],                # maximum depth of each tree
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],   # percentage of attributes to be considered (randomly) for each tree
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],  # percentage of attributes to be considered (randomly) for each level of a tree
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],    # learning rate
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],         # L2 regularization factor
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=CV_FOLDS, scoring="recall", n_jobs=N_JOBS)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Log model configuration
    mlflow.log_params({
      "classification_threshold": CLASSIFICATION_THRESHOLD,
      "cv_folds": CV_FOLDS,
      "random_state": RANDOM_STATE,
      "scoring_metric": "recall",
    })
    mlflow.log_params({
    "classification_threshold": CLASSIFICATION_THRESHOLD,
    "cv_folds": CV_FOLDS,
    "random_state": RANDOM_STATE,
    "scoring_metric": "recall"})

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

    # Upload to Hugging Face
    repo_id = HF_MODEL_REPO
    repo_type = MODEL_REPO_TYPE

    # Check if the model exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model '{repo_id}' not found. Creating new model...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Model '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj=MODEL_FILENAME,
        path_in_repo=MODEL_FILENAME,
        repo_id=repo_id,
        repo_type=repo_type,
    )

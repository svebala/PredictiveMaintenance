import os
from zoneinfo import ZoneInfo

# Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")

HF_DATASET_REPO = "BalaSVenkat/predictive-maintenance-dataset"
HF_DATASET_TYPE = "dataset"

HF_MODEL_REPO = "BalaSVenkat/predictive-maintenance-model"
HF_MODEL_TYPE = "model"

HF_SPACE_REPO = "BalaSVenkat/predictive-maintenance-space"
HF_SPACE_TYPE = "space"

HF_MAX_RETRIES = 5

# Filenames
MODEL_FILENAME = "engine_predictive_maintenance_model.joblib"

DATASET_FILENAME = "engine_data.csv"

CLEANED_DATA = "cleaned_engine_data.csv"

XTRAIN_FILE = "Xtrain.csv"
XTEST_FILE = "Xtest.csv"
YTRAIN_FILE = "ytrain.csv"
YTEST_FILE = "ytest.csv"

# Dataset Paths
HF_DATASET_PATH = (
    f"hf://datasets/{HF_DATASET_REPO}"
)

TARGET_COLUMN = "engine_condition"

NUMERIC_FEATURES = [
    "engine_rpm",
    "lub_oil_pressure",
    "fuel_pressure",
    "coolant_pressure",
    "lub_oil_temp",
    "coolant_temp",
]

# Model
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Decision threshold used during deployment.
# Lower than the default (0.50) to improve recall,
# reducing the chance of missing faulty engines.
CLASSIFICATION_THRESHOLD = 0.35

LOW_RISK_THRESHOLD = 0.20
CV_FOLDS = 5
N_JOBS = -1

SCORING_METRIC = "recall"

# MLflow
MLFLOW_TRACKING_URI = "mlruns"
MLFLOW_EXPERIMENT_NAME = "engine-predictive-maintenance"
MLFLOW_ARTIFACT_PATH = "model"

# Timezone used for timestamps displayed in the application
TIMEZONE = ZoneInfo("Asia/Kolkata")

SENSOR_LABELS = {
    "engine_rpm": "Engine RPM (rpm)",
    "lub_oil_pressure": "Lubricating Oil Pressure (bar)",
    "fuel_pressure": "Fuel Pressure (bar)",
    "coolant_pressure": "Coolant Pressure (bar)",
    "lub_oil_temp": "Lubricating Oil Temperature (°C)",
    "coolant_temp": "Coolant Temperature (°C)"
}


# for data manipulation
import pandas as pd

# for creating a folder
import os

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# for hugging face space authentication to upload files
from huggingface_hub import HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/BalaSVenkat/predictive-maintanence-dataset/engine_state.csv"
maintanence_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully from Hugging Face.")
print(f"Initial Dataset Shape: {maintanence_dataset.shape}")

# Define the target variable for the classification task
target = 'engine_condition'

# List of numerical features in the dataset
numeric_features = [
    'engine_rpm',       # The number of revolutions per minute (RPM) of the engine, indicating engine speed.
    'lub_oil_pressure', # The pressure of the lubricating oil in the engine, essential for reducing friction and wear.
    'fuel_pressure',    # The pressure at which fuel is supplied to the engine, critical for proper combustion.
    'coolant_pressure', # The pressure of the engine coolant, affecting engine temperature regulation.
    'lub_oil_temp',     # The temperature of the lubricating oil, which impacts viscosity and engine performance.
    'coolant_temp'      # The temperature of the engine coolant, crucial for preventing overheating.
]

# Define predictor matrix (X) using selected numeric and categorical features
X = maintanence_dataset[numeric_features]

# Define target variable
y = maintanence_dataset[target]

# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.20,    # 20% of the data is reserved for testing
    random_state=42,   # Ensures reproducibility by setting a fixed random seed
    stratify=y         # preserves the same class distribution in both train and test datasets.
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="BalaSVenkat/predictive-maintanence-dataset",
        repo_type="dataset",
    )

"""
Prepares the predictive maintenance dataset by
cleaning, preprocessing, splitting into train/test sets,
and uploading the processed files to Hugging Face.
"""

# for data manipulation
import pandas as pd

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# for hugging face space authentication to upload files
from huggingface_hub import HfApi

# import constants from config file
from predictive_maintenance.config import (
    HF_DATASET_PATH,
    TARGET_COLUMN,
    NUMERIC_FEATURES,
    TEST_SIZE,
    RANDOM_STATE,
    HF_DATASET_REPO,
    HF_DATASET_TYPE,
    HF_TOKEN,
    CLEANED_DATA,
    XTRAIN_FILE,
    XTEST_FILE,
    YTRAIN_FILE,
    YTEST_FILE,
)

# Get token information
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set.")

# Authenticate with Hugging Face
api = HfApi(token=HF_TOKEN)

# Load dataset from Hugging Face
maintenance_dataset = pd.read_csv(HF_DATASET_PATH/{DATASET_FILENAME})
print("Dataset loaded successfully from Hugging Face.")
print(f"Initial Dataset Shape: {maintenance_dataset.shape}")

# Convert all attribute names to small letter and replace spaces with underscores
maintenance_dataset.columns = maintenance_dataset.columns.str.lower().str.replace(' ', '_')

# Handle Missing Values
numeric_cols = maintenance_dataset.select_dtypes(include="number").columns

# Remove duplicate records (if exist)
maintenance_dataset = maintenance_dataset.drop_duplicates().reset_index(drop=True)
print(f"Dataset shape after duplicate removals: {maintenance_dataset.shape}")

# Numerical columns -> Median Imputation
for col in numeric_cols:
    maintenance_dataset[col] = maintenance_dataset[col].fillna(maintenance_dataset[col].median())

# Save Cleaned Dataset
maintenance_dataset.to_csv(CLEANED_DATA, index=False)

# Select predictor features
X = maintenance_dataset[NUMERIC_FEATURES]

# Define target variable
y = maintenance_dataset[TARGET_COLUMN]

# Split dataset into train and test
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,                       
    test_size=TEST_SIZE,        
    random_state=RANDOM_STATE,  
    stratify=y                  
)

Xtrain.to_csv(XTRAIN_FILE, index=False)
Xtest.to_csv(XTEST_FILE, index=False)
ytrain.to_csv(YTRAIN_FILE, index=False)
ytest.to_csv(YTEST_FILE, index=False)

files = (
    CLEANED_DATA,
    XTRAIN_FILE,
    XTEST_FILE,
    YTRAIN_FILE,
    YTEST_FILE,
)

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=HF_DATASET_REPO,
        repo_type=HF_DATASET_TYPE
    )
print("✅ All processed datasets uploaded successfully.")

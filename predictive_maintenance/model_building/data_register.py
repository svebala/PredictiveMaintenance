"""
Registers the dataset repository on Hugging Face Hub
and uploads the local dataset.
"""

import time

from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo

# import constants from config file
from predictive_maintenance.config import (
    HF_DATASET_REPO,
    HF_DATASET_TYPE,
    HF_TOKEN,
    HF_MAX_RETRIES,
)

# Get token information
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set.")

# Authenticate with Hugging Face
api = HfApi(token=HF_TOKEN)

# Step 1: Check if the dataset exists
for attempt in range(HF_MAX_RETRIES):
    try:
        api.repo_info(
            repo_id=HF_DATASET_REPO,
            repo_type=HF_DATASET_TYPE,
        )
        print(f"✅ Dataset '{HF_DATASET_REPO}' already exists.")
        break

    except RepositoryNotFoundError:
        print(f"Dataset '{HF_DATASET_REPO}' not found. Creating...")
        create_repo(
            repo_id=HF_DATASET_REPO,
            repo_type=HF_DATASET_TYPE,
            private=False,
            token=HF_TOKEN,
            exist_ok=True,
        )
        print("✅ Dataset created.")
        break

    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            wait = 2 ** attempt
            print(f"Rate limited. Retrying in {wait} seconds...")
            time.sleep(wait)
        else:
            raise

# Upload dataset
api.upload_folder(
    folder_path="predictive_maintenance/data",
    repo_id=HF_DATASET_REPO,
    repo_type=HF_DATASET_TYPE
)
print("✅ Dataset uploaded successfully.")


import os
import time

from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.getenv("HF_TOKEN")

repo_id = "BalaSVenkat/predictive-maintenance-dataset"
repo_type = "dataset"

# Initialize API client
api = HfApi()

# Step 1: Check if the dataset exists
for attempt in range(5):
    try:
        api.repo_info(repo_id=repo_id,repo_type=repo_type,token=HF_TOKEN)
        print(f"✅ Dataset '{repo_id}' already exists.")
        break

    except RepositoryNotFoundError:
        print(f"Dataset '{repo_id}' not found. Creating...")
        create_repo(repo_id=repo_id,repo_type=repo_type,private=False,token=HF_TOKEN,exist_ok=True)
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
    repo_id=repo_id,
    repo_type=repo_type,
    token=HF_TOKEN
)

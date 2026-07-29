"""
Creates the Hugging Face Space (if required)
and uploads the Streamlit deployment files.
"""

import time
from huggingface_hub import HfApi, create_repo
from huggingface_hub.errors import (
    RepositoryNotFoundError,
    HfHubHTTPError,
)

from predictive_maintenance.config import (
    HF_SPACE_REPO,
    HF_SPACE_TYPE,
    HF_TOKEN,
    HF_MAX_RETRIES,
)

# Get token information
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set.")

# Authenticate with Hugging Face
api = HfApi(token=HF_TOKEN)

# Check if the Space repository exists
for attempt in range(HF_MAX_RETRIES):
    try:
        api.repo_info(
            repo_id=HF_SPACE_REPO,
            repo_type=HF_SPACE_TYPE,
        )
        print(f"✅ Space '{HF_SPACE_REPO}' already exists.")
        break

    except RepositoryNotFoundError:
        print(f"Space '{HF_SPACE_REPO}' not found. Creating...")
        create_repo(
            repo_id=HF_SPACE_REPO,
            repo_type=HF_SPACE_TYPE,
            space_sdk="streamlit",
            private=False,
            token=HF_TOKEN,
            exist_ok=True,
        )
        print("✅ Space created.")
        break

    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            wait = 2 ** attempt
            print(f"Rate limited. Retrying in {wait} seconds...")
            time.sleep(wait)
        else:
            raise

# Upload Space
api.upload_folder(
    folder_path="predictive_maintenance/deployment",
    repo_id=HF_SPACE_REPO,
    repo_type=HF_SPACE_TYPE,
    path_in_repo=""
)
print("✅ Space uploaded successfully.")

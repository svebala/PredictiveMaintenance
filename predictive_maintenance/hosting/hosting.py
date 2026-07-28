
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.errors import RepositoryNotFoundError

# Configuration
HF_SPACE_REPO = "BalaSVenkat/predictive-maintenance-space"
HF_REPO_TYPE = "space"
DEPLOYMENT_FOLDER = "predictive_maintenance/deployment"

# Get token information
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set.")

# Authenticate with Hugging Face
api = HfApi(token=HF_TOKEN)

# Ensure the Hugging Face Space exists
try:
    api.repo_info(
        repo_id=HF_SPACE_REPO,
        repo_type=HF_REPO_TYPE
    )
    print(f"Space '{HF_SPACE_REPO}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{HF_SPACE_REPO}' not found. Creating new space...")

    create_repo(
        repo_id=HF_SPACE_REPO,
        repo_type=HF_REPO_TYPE,
        private=False
    )

    print(f"Space '{HF_SPACE_REPO}' created successfully.")

# Upload deployment files to the Space
api.upload_folder(
    folder_path=DEPLOYMENT_FOLDER,
    repo_id=HF_SPACE_REPO,
    repo_type=HF_REPO_TYPE,
    path_in_repo=""
)

print(f"Deployment uploaded successfully to '{HF_SPACE_REPO}'.")

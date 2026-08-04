"""
Utility functions for Hugging Face repository management
and artifact uploads.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    HfHubHTTPError,
)

logger = logging.getLogger(__name__)


def ensure_repo_exists(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    token: str,
    max_retries: int,
) -> None:
    """
    Create a Hugging Face repository if it does not exist.

    Parameters
    ----------
    api : HfApi
        Hugging Face API client.

    repo_id : str
        Repository ID.

    repo_type : str
        Repository type ("dataset" or "model").

    token : str
        Hugging Face access token.

    max_retries : int
        Maximum retry attempts for rate limiting.
    """

    for attempt in range(max_retries):

        try:

            api.repo_info(
                repo_id=repo_id,
                repo_type=repo_type,
            )

            logger.info(
                "Repository '%s' already exists.",
                repo_id,
            )

            return

        except RepositoryNotFoundError:

            logger.info(
                "Creating repository '%s'...",
                repo_id,
            )

            create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                private=False,
                token=token,
                exist_ok=True,
            )

            logger.info(
                "Repository created successfully."
            )

            return

        except HfHubHTTPError as error:

            if (
                error.response is not None
                and error.response.status_code == 429
            ):

                wait = 2 ** attempt

                logger.warning(
                    "Rate limited. Retrying in %s seconds...",
                    wait,
                )

                time.sleep(wait)

            else:
                raise

def upload_file(
    api: HfApi,
    file_path: Path,
    repo_id: str,
    repo_type: str,
) -> None:
    """
    Upload a single file to Hugging Face.
    """

    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.name,
        repo_id=repo_id,
        repo_type=repo_type,
    )

    logger.info(
        "Uploaded %s",
        file_path.name,
    )

def upload_directory(
    api: HfApi,
    artifact_dir: Path,
    repo_id: str,
    repo_type: str,
    run_id: str,
) -> None:
    """
    Upload all files in an artifact directory.
    """

    for file in artifact_dir.iterdir():

        if not file.is_file():
            continue

        try:

            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=f"{run_id}/{file.name}",
                repo_id=repo_id,
                repo_type=repo_type,
            )

            logger.info(
                "Uploaded %s",
                file.name,
            )

        except Exception:

            logger.exception(
                "Failed uploading %s",
                file.name,
            )

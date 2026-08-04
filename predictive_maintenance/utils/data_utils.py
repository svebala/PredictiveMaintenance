"""
Utility functions for loading datasets and computing
class weights for predictive maintenance models.
"""

from pathlib import Path

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV dataset after validating that it exists.

    Parameters
    ----------
    file_path : str | Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset does not exist.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    logger.info("Loading dataset: %s", file_path)

    return pd.read_csv(file_path)

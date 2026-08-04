"""
Utility functions for loading datasets and computing
class weights for predictive maintenance models.
"""

from pathlib import Path

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a CSV dataset from a Hugging Face Dataset repository.

    Parameters
    ----------
    file_path : str
        Hugging Face dataset path
        (e.g., hf://datasets/<username>/<repo>/Xtrain.csv)

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file cannot be found.

    RuntimeError
        If the dataset cannot be loaded for any other reason.
    """

    logger.info("Loading dataset from: %s", file_path)

    try:
        df = pd.read_csv(file_path)

        logger.info(
            "Successfully loaded %d rows and %d columns.",
            df.shape[0],
            df.shape[1],
        )

        return df

    except FileNotFoundError as excptn:
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        ) from excptn

    except Exception as excptn:
        raise RuntimeError(
            f"Failed to load dataset from '{file_path}'. "
            f"Error: {excptn}"
        ) from excptn

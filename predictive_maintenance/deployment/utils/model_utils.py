"""
Utility functions for loading the predictive maintenance model.
"""

import joblib
import streamlit as st

from huggingface_hub import hf_hub_download

from config import (
    HF_MODEL_REPO,
    MODEL_FILENAME,
)


@st.cache_resource
def load_model():
    """
    Download and load the trained model from Hugging Face Hub.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Trained prediction pipeline.
    """

    model_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=MODEL_FILENAME,
    )

    model = joblib.load(model_path)

    return model


def display_model_information(model):
    """
    Display model information in the Streamlit sidebar.
    """

    st.sidebar.success("✅ Model Loaded")
    st.sidebar.write(type(model).__name__)

    with st.sidebar.expander("Model Information"):

        st.write("**Model Type:**", type(model).__name__)

        if hasattr(model, "named_steps"):

            st.write("**Pipeline Steps:**")

            for name, step in model.named_steps.items():

                st.write(
                    f"• {name}: {type(step).__name__}"
                )

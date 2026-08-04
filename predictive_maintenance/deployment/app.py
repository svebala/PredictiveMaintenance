import streamlit as st
from utils.ui_utils import (
    load_css,
    get_sensor_inputs,
    display_recommendation,
)

from utils.model_utils import (
    load_model,
    display_model_information,
)

from config import CLASSIFICATION_THRESHOLD
from utils.visualization_utils import risk_progress_bar

# Set page configuration
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🚗",
    layout="centered"
)

load_css()

# Streamlit UI
st.title("🚗 Engine Predictive Maintenance")

st.info("""
Enter the engine sensor readings below.

The trained machine learning model analyzes these readings to estimate the probability that the engine requires maintenance and provides an appropriate maintenance recommendation.
""")

# Download the model from Hugging Face Hub and Load the trained model
try:

    model = load_model()
    display_model_information(model)

except Exception as e:
    st.error("Unable to load prediction model.")
    st.exception(e)
    st.stop()

# Sensor Details
input_data = get_sensor_inputs()

# Prediction
if st.button("Run Diagnostics", use_container_width=True):

    prediction_proba = float(model.predict_proba(input_data)[0, 1])
    healthy_prob = (1 - prediction_proba) * 100

    # Display prediction probability
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.markdown("<div style='margin-top:35px'></div>", unsafe_allow_html=True)
        st.metric(
            label="Failure Probability",
            value=f"{prediction_proba:.2%}",
            delta=f"{prediction_proba - CLASSIFICATION_THRESHOLD:+.2%}",
            delta_color="inverse"
        )

    with col2:
        st.markdown(
            """
            <div style="
                text-align:center;
                margin-top:20px;
                color:#1F4E79;
                font-size:28px;
                font-weight:600;
            ">
                Diagnostic<br>Results
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown("<div style='margin-top:35px'></div>", unsafe_allow_html=True)
        st.metric(
            label="Healthy Probability",
            value=f"{healthy_prob:.2f}%"
        )

    risk_progress_bar(prediction_proba)

    # Risk classification
    display_recommendation(prediction_proba)

st.divider()

st.caption(
    "Predictions are intended to support maintenance decisions and "
    "should not replace physical inspection."
)

import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from zoneinfo import ZoneInfo
from huggingface_hub import hf_hub_download

# import constants from config file
from config import (
    LOW_RISK_THRESHOLD,
    CLASSIFICATION_THRESHOLD,
    TIMEZONE,
    SENSOR_LABELS,
    HF_MODEL_REPO,
    MODEL_FILENAME,
)

# Set page configuration
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🚗",
    layout="centered"
)

st.markdown("""
<style>

/* Run Diagnostics button */
div.stButton > button:first-child {
    background-color: #F57C00;
    color: white;
    border-radius: 8px;
    border: none;
    font-size: 18px;
    font-weight: 600;
    height: 3em;
}

/* Button hover */
div.stButton > button:first-child:hover {
    background-color: #EF6C00;
    color: white;
}

/* Page title */
h1 {
    color: #1F4E79;
}

/* Sub headings */
h2, h3 {
    color: #1F4E79;
}

/* Success box */
div[data-testid="stSuccess"] {
    background-color: #E8F5E9;
    border-left: 6px solid #2E7D32;
}

/* Warning box */
div[data-testid="stWarning"] {
    background-color: #FFF8E1;
    border-left: 6px solid #F9A825;
}

/* Error box */
div[data-testid="stError"] {
    background-color: #FFEBEE;
    border-left: 6px solid #C62828;
}

</style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("🚗 Engine Predictive Maintenance")

st.info("""
Enter the engine sensor readings below.

The trained machine learning model analyzes these readings to estimate the probability that the engine requires maintenance and provides an appropriate maintenance recommendation.
""")

# Download the model from Hugging Face Hub and Load the trained model
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=MODEL_FILENAME
    )
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error("Unable to load prediction model.")
    st.exception(e)
    st.stop()

# Sensor Details
left, right = st.columns(2)

with left:
    engine_rpm = st.number_input(SENSOR_LABELS["engine_rpm"], min_value=61, max_value=2239, value=791, step=1, help="Rotational speed of the engine in revolutions per minute.")
    lub_oil_pressure = st.number_input(SENSOR_LABELS["lub_oil_pressure"], min_value=0.003, max_value=7.266, value=3.304, step=0.01, format="%.3f", help="Pressure of the engine's lubricating oil system.")
    fuel_pressure = st.number_input(SENSOR_LABELS["fuel_pressure"], min_value=0.003, max_value=21.138, value=6.656, step=0.01, format="%.3f", help="Fuel pressure supplied to the engine.")

with right:
    coolant_pressure = st.number_input(SENSOR_LABELS["coolant_pressure"], min_value=0.002, max_value=7.479, value=2.335, step=0.01, format="%.3f", help="Pressure within the engine cooling system.")
    lub_oil_temp = st.number_input(SENSOR_LABELS["lub_oil_temp"], min_value=71.322, max_value=89.581, value=77.643, step=0.1, format="%.3f", help="Temperature of the lubricating oil.")
    coolant_temp = st.number_input(SENSOR_LABELS["coolant_temp"], min_value=61.673, max_value=195.528, value=78.427, step=0.1, format="%.3f", help="Temperature of the engine coolant.")

# Prepare input data
input_data = pd.DataFrame([{
    "engine_rpm": engine_rpm,
    "lub_oil_pressure": lub_oil_pressure,
    "fuel_pressure": fuel_pressure,
    "coolant_pressure": coolant_pressure,
    "lub_oil_temp": lub_oil_temp,
    "coolant_temp": coolant_temp
}])

# Prediction
if st.button("Run Diagnostics", use_container_width=True):

    prediction_proba = model.predict_proba(input_data)[0, 1]
    healthy_prob = (1 - prediction_proba) * 100

    st.markdown(
        """
        <h3 style="
            text-align:center;
            color:#1F4E79;
            margin-top:10px;
            margin-bottom:15px;">
            Diagnostic Results
        </h3>
        """,
        unsafe_allow_html=True
    )

    # Display prediction probability
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Maintenance Probability",
            value=f"{prediction_proba:.2%}",
            delta=f"{prediction_proba - CLASSIFICATION_THRESHOLD:+.2%} vs threshold",
            delta_color="inverse",
        )

    with col2:
        st.metric(
            "Normal Operation Probability",
            f"{healthy_prob:.2f}%"
        )

    # Risk classification
    if prediction_proba < LOW_RISK_THRESHOLD:

        st.success("🟢 Healthy Engine")

        st.markdown("""
    **Risk Level:** Low

    The engine is operating within normal conditions.

    **Recommended Action**
    - Continue normal operation
    - Follow the regular maintenance schedule
    - Monitor sensor readings periodically
    """)

    elif prediction_proba < CLASSIFICATION_THRESHOLD:

        st.warning("🟡 Moderate Risk")

        st.markdown(f"""
    The engine is showing **early signs of abnormal behaviour**.

    Current fault probability is **{prediction_proba:.2%}**, which is below the decision threshold of **{CLASSIFICATION_THRESHOLD:.0%}**.

    **Recommended Action**
    - Monitor engine performance closely
    - Inspect lubrication and cooling systems
    - Repeat diagnostics if sensor values change
    """)

    else:

        st.error("🔴 High Risk – Maintenance Required")

        st.markdown(f"""
    The estimated probability of engine failure is **{prediction_proba:.2%}**, which exceeds the decision threshold of **{CLASSIFICATION_THRESHOLD:.0%}**.

    **Recommended Action**
    - Schedule maintenance immediately
    - Inspect lubrication system
    - Inspect coolant system
    - Check fuel pressure
    - Perform a complete engine inspection
    """)

    st.divider()

    with st.expander("View Submitted Sensor Readings"):

        sensor_df = (
            input_data.T
            .rename(columns={0: "Reading"})
            .rename_axis("Sensor")
        )
        sensor_df.rename(index=SENSOR_LABELS, inplace=True)
        st.dataframe(sensor_df, use_container_width=True)

    st.caption(
        f"Diagnostics generated on: "
        f"{datetime.now(TIMEZONE).strftime('%d %b %Y, %I:%M %p')} IST"
    )

st.divider()

st.caption(
    "Predictions are intended to support maintenance decisions and "
    "should not replace physical inspection."
)

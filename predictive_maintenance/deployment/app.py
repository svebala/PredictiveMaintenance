import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

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

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

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

# Download the model from Hugging Face Hub and Load the trained model
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="BalaSVenkat/predictive-maintenance-model",
        filename="mlops_predictive_maintenance_model.joblib"
    )
    return joblib.load(model_path)

try:
    model = load_model()
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.exception(e)
    st.stop()

# Streamlit UI
st.title("🚗 Predictive Maintenance for Engine Health")

st.info(
    """
Enter the engine sensor readings below.

The trained machine learning model will analyze the sensor readings
and predict whether the engine is operating normally or requires maintenance.
"""
)

# Sensor Details
engine_rpm = st.number_input( "Engine RPM", min_value=61, max_value=2239, value=791, step=1)
lub_oil_pressure = st.number_input("Lubricating Oil Pressure (bar)", min_value=0.003, max_value=7.266, value=3.304, step=0.01, format="%.3f")
fuel_pressure = st.number_input("Fuel Pressure (bar)", min_value=0.003, max_value=21.138, value=6.656, step=0.01, format="%.3f")
coolant_pressure = st.number_input("Coolant Pressure (bar)", min_value=0.002, max_value=7.479, value=2.335, step=0.01, format="%.3f")
lub_oil_temp = st.number_input("Lubricating Oil Temperature (°C)", min_value=71.322, max_value=89.581, value=77.643, step=0.1, format="%.3f")
coolant_temp = st.number_input("Coolant Temperature (°C)", min_value=61.673, max_value=195.528, value=78.427, step=0.1, format="%.3f")

# Prepare input data
input_data = pd.DataFrame([{
    "engine_rpm": engine_rpm,
    "lub_oil_pressure": lub_oil_pressure,
    "fuel_pressure": fuel_pressure,
    "coolant_pressure": coolant_pressure,
    "lub_oil_temp": lub_oil_temp,
    "coolant_temp": coolant_temp
}])

# Classification threshold
classification_threshold = 0.35

# Prediction
if st.button("Run Diagnostics", use_container_width=True):

    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = int(prediction_proba >= classification_threshold)

    # Display prediction probability
    st.metric(
        label="Probability of Engine Requiring Maintenance",
        value=f"{prediction_proba:.2%}"
    )

    st.progress(min(prediction_proba, 1.0))

    # Risk classification
    if prediction_proba < 0.20:

        st.success("🟢 Healthy Engine")

        st.markdown("""
    **Risk Level:** Low

    The engine is operating within normal conditions.

    **Recommended Action**
    - Continue normal operation
    - Follow the regular maintenance schedule
    - Monitor sensor readings periodically
    """)

    elif prediction_proba < classification_threshold:

        st.warning("🟡 Moderate Risk")

        st.markdown(f"""
    The engine is showing **early signs of abnormal behaviour**.

    Current fault probability is **{prediction_proba:.2%}**, which is below the decision threshold of **{classification_threshold:.0%}**.

    **Recommended Action**
    - Monitor engine performance closely
    - Inspect lubrication and cooling systems
    - Repeat diagnostics if sensor values change
    """)

    else:

        st.error("🔴 High Risk – Maintenance Required")

        st.markdown(f"""
    The estimated probability of engine failure is **{prediction_proba:.2%}**, which exceeds the decision threshold of **{classification_threshold:.0%}**.

    **Recommended Action**
    - Schedule maintenance immediately
    - Inspect lubrication system
    - Inspect coolant system
    - Check fuel pressure
    - Perform a complete engine inspection
    """)

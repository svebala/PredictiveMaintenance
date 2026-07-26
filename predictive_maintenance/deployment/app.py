import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

st.write("Checkpoint 1")
# Set page configuration
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🚗",
    layout="centered"
)

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
    st.write("Checkpoint 2")
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
engine_rpm = st.number_input( "Engine RPM", min_value=61.0, max_value=2239.0, value=791.24, step=1.0)
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

st.write("Checkpoint 3")

# Classification threshold
classification_threshold = 0.5

# Prediction
if st.button("Predict", use_container_width=True):

    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = int(prediction_proba >= classification_threshold)

    # Display prediction probability
    st.metric(
        label="Probability of Normal Engine Condition",
        value=f"{prediction_proba:.2%}"
    )

    if prediction == 1:
        st.success("✅ Engine is operating normally. No immediate maintenance is required.")
    else:
        st.error("⚠️ Engine requires maintenance. Please inspect the engine at the earliest opportunity.")

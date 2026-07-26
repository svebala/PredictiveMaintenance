import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="BalaSVenkat/predictive-maintanence-model",
    filename="mlops_predective_maintanence_model.joblib"
)

# Load the trained model
model = joblib.load(model_path)

# Streamlit UI
st.title("Machine Failure Prediction App")
st.write(
    "This internal application predicts a machine failure "
    "based on historical and real-time engine sensor data."
)
st.write("Please enter the sensor details below.")

# Customer Details
engine_rpm = st.number_input("Number of Persons Visiting",min_value=1, max_value=10, value=2)
lub_oil_pressure = st.number_input("Number of Persons Visiting",min_value=1, max_value=10, value=2)
fuel_pressure = st.number_input("Number of Persons Visiting",min_value=1, max_value=10, value=2)
coolant_pressure = st.number_input("Number of Persons Visiting",min_value=1, max_value=10, value=2)
lub_oil_temp = st.number_input("Number of Persons Visiting",min_value=1, max_value=10, value=2)
coolant_temp = st.number_input("Number of Persons Visiting",min_value=1, max_value=10, value=2)

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
classification_threshold = 0.5

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)

    if prediction == 1:
        st.success("✅ The machine will not fail.")
    else:
        st.error("❌ The machine is likely to fail.")

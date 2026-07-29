"""
Configuration used by the Streamlit deployment.
"""

from zoneinfo import ZoneInfo

# Hugging Face Model
HF_MODEL_REPO = "BalaSVenkat/predictive-maintenance-model"
MODEL_FILENAME = "engine_predictive_maintenance_model.joblib"

# Decision Thresholds
LOW_RISK_THRESHOLD = 0.20
CLASSIFICATION_THRESHOLD = 0.35

# Timezone
TIMEZONE = ZoneInfo("Asia/Kolkata")

# Sensor Labels
SENSOR_LABELS = {
    "engine_rpm": "Engine RPM (rpm)",
    "lub_oil_pressure": "Lubricating Oil Pressure (bar)",
    "fuel_pressure": "Fuel Pressure (bar)",
    "coolant_pressure": "Coolant Pressure (bar)",
    "lub_oil_temp": "Lubricating Oil Temperature (°C)",
    "coolant_temp": "Coolant Temperature (°C)",
}

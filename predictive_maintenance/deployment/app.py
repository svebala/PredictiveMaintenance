import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from zoneinfo import ZoneInfo
from huggingface_hub import hf_hub_download
import streamlit.components.v1 as components

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
    model = joblib.load(model_path)

    st.sidebar.success("✅ Model Loaded")
    st.sidebar.write(type(model).__name__)

    return model

try:
    model = load_model()
    with st.sidebar.expander("Model Information"):
        st.write("Model Type:", type(model).__name__)
        if hasattr(model, "named_steps"):
            st.write("Pipeline Steps")

            for name, step in model.named_steps.items():
                st.write(f"• {name}: {type(step).__name__}")
except Exception as e:
    st.error("Unable to load prediction model.")
    st.exception(e)
    st.stop()

# Sensor Details
left, right = st.columns(2)

with left:
    engine_rpm = st.number_input(SENSOR_LABELS["engine_rpm"], min_value=60, max_value=2500, value=750, step=1, help="Rotational speed of the engine in revolutions per minute.")
    lub_oil_pressure = st.number_input(SENSOR_LABELS["lub_oil_pressure"], min_value=0.10, max_value=20.00, value=3.30, step=0.01, format="%.2f", help="Pressure of the engine's lubricating oil system.")
    fuel_pressure = st.number_input(SENSOR_LABELS["fuel_pressure"], min_value=0.10, max_value=20.00, value=6.65, step=0.01, format="%.2f", help="Fuel pressure supplied to the engine.")

with right:
    coolant_pressure = st.number_input(SENSOR_LABELS["coolant_pressure"], min_value=0.10, max_value=20.00, value=2.33, step=0.01, format="%.2f", help="Pressure within the engine cooling system.")
    lub_oil_temp = st.number_input(SENSOR_LABELS["lub_oil_temp"], min_value=60.00, max_value=120.00, value=77.64, step=0.1, format="%.2f", help="Temperature of the lubricating oil.")
    coolant_temp = st.number_input(SENSOR_LABELS["coolant_temp"], min_value=60.00, max_value=120.00, value=78.47, step=0.1, format="%.2f", help="Temperature of the engine coolant.")

# Prepare input data
input_data = pd.DataFrame([{
    "engine_rpm": engine_rpm,
    "lub_oil_pressure": lub_oil_pressure,
    "fuel_pressure": fuel_pressure,
    "coolant_pressure": coolant_pressure,
    "lub_oil_temp": lub_oil_temp,
    "coolant_temp": coolant_temp
}])

def risk_progress_bar(probability):

    percent = round(float(probability) * 100, 2)

    # Badge colors
    if percent <= 20:
        badge_fill = "#E8F5E9"
        badge_border = "#2E7D32"
        badge_text = "#1B5E20"
    elif percent <= 35:
        badge_fill = "#FFF8E1"
        badge_border = "#F9A825"
        badge_text = "#EF6C00"
    else:
        badge_fill = "#FFEBEE"
        badge_border = "#C62828"
        badge_text = "#B71C1C"

    width = 760
    height = 175

    bar_x = 70
    bar_y = 70
    bar_width = 620
    bar_height = 30

    marker_x = bar_x + (percent / 100) * bar_width
    fill_width = (percent / 100) * bar_width

    svg = f"""
<svg width="100%" viewBox="0 0 {width} {height}"
     xmlns="http://www.w3.org/2000/svg">

<defs>

    <!-- Shadow -->
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="2"
                      stdDeviation="2"
                      flood-color="#888888"
                      flood-opacity="0.35"/>
    </filter>

    <!-- Risk Gradient -->
    <linearGradient id="riskGradient" x1="0%" y1="0%" x2="100%" y2="0%">

        <stop offset="0%" stop-color="#2E7D32"/>
        <stop offset="20%" stop-color="#2E7D32"/>

        <stop offset="20%" stop-color="#F9A825"/>
        <stop offset="35%" stop-color="#F9A825"/>

        <stop offset="35%" stop-color="#C62828"/>
        <stop offset="100%" stop-color="#C62828"/>

    </linearGradient>

    <!-- Clip only the completed portion -->
    <clipPath id="clipFill">
        <rect
            x="{bar_x}"
            y="{bar_y}"
            width="{fill_width}"
            height="{bar_height}"
            rx="12"
            ry="12"/>
    </clipPath>

</defs>

<!-- Percentage Badge -->

<rect
    x="{marker_x-42}"
    y="10"
    width="84"
    height="30"
    rx="8"
    fill="{badge_fill}"
    stroke="{badge_border}"
    stroke-width="1.5"
    filter="url(#shadow)"
/>

<text
    x="{marker_x}"
    y="30"
    text-anchor="middle"
    font-size="15"
    font-weight="bold"
    fill="{badge_text}">
    {percent:.2f}%
</text>

<!-- Pointer -->

<polygon
    points="{marker_x-7},46 {marker_x+7},46 {marker_x},58"
    fill="#1F4E79"/>

<!-- Background -->

<rect
    x="{bar_x}"
    y="{bar_y}"
    width="{bar_width}"
    height="{bar_height}"
    rx="12"
    ry="12"
    fill="#ECECEC"/>

<!-- Gradient Fill -->

<rect
    x="{bar_x}"
    y="{bar_y}"
    width="{bar_width}"
    height="{bar_height}"
    rx="12"
    ry="12"
    fill="url(#riskGradient)"
    clip-path="url(#clipFill)"/>

<!-- Threshold Markers -->

<circle cx="{bar_x}" cy="106" r="3" fill="#444"/>

<circle
    cx="{bar_x+0.20*bar_width}"
    cy="106"
    r="5"
    fill="white"
    stroke="#2E7D32"
    stroke-width="2"/>

<circle
    cx="{bar_x+0.35*bar_width}"
    cy="106"
    r="5"
    fill="white"
    stroke="#F9A825"
    stroke-width="2"/>

<circle
    cx="{bar_x+bar_width}"
    cy="106"
    r="5"
    fill="white"
    stroke="#C62828"
    stroke-width="2"/>

<!-- Percentage Labels -->

<text x="{bar_x}" y="126"
      text-anchor="middle"
      font-size="13"
      font-weight="bold">0%</text>

<text x="{bar_x+0.20*bar_width}" y="126"
      text-anchor="middle"
      font-size="13"
      font-weight="bold">20%</text>

<text x="{bar_x+0.35*bar_width}" y="126"
      text-anchor="middle"
      font-size="13"
      font-weight="bold">35%</text>

<text x="{bar_x+bar_width}" y="126"
      text-anchor="middle"
      font-size="13"
      font-weight="bold">100%</text>

<!-- Risk Labels -->

<text
    x="{bar_x+0.10*bar_width}"
    y="155"
    text-anchor="middle"
    font-size="14"
    font-weight="bold"
    fill="#2E7D32">
    Low Risk
</text>

<text
    x="{bar_x+0.275*bar_width}"
    y="155"
    text-anchor="middle"
    font-size="14"
    font-weight="bold"
    fill="#F9A825">
    Moderate
</text>

<text
    x="{bar_x+0.675*bar_width}"
    y="155"
    text-anchor="middle"
    font-size="14"
    font-weight="bold"
    fill="#C62828">
    High Risk
</text>

</svg>
"""
    components.html(svg, height=175)

# Prediction
if st.button("Run Diagnostics", use_container_width=True):

    prediction_proba = model.predict_proba(input_data)[0, 1]
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

    st.caption(
        f"Generated on: "
        f"{datetime.now(TIMEZONE).strftime('%d %b %Y, %I:%M %p')} IST"
    )

st.divider()

st.caption(
    "Predictions are intended to support maintenance decisions and "
    "should not replace physical inspection."
)

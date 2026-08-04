"""
Utility functions for the Streamlit user interface.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from datetime import datetime

from config import (
    SENSOR_LABELS,
    LOW_RISK_THRESHOLD,
    CLASSIFICATION_THRESHOLD,
    TIMEZONE,
)


def load_css() -> None:
    """
    Load custom CSS for the Streamlit application.
    """

    css_path = (
        Path(__file__).parent.parent
        / "assets"
        / "styles.css"
    )

    with open(css_path, encoding="utf-8") as css_file:
        st.markdown(
            f"<style>{css_file.read()}</style>",
            unsafe_allow_html=True,
        )


def get_sensor_inputs() -> pd.DataFrame:
    """
    Display sensor input controls and return the values
    as a DataFrame.
    """

    left, right = st.columns(2)

    with left:

        engine_rpm = st.number_input(
            SENSOR_LABELS["engine_rpm"],
            min_value=60,
            max_value=2500,
            value=750,
            step=1,
            help="Rotational speed of the engine in revolutions per minute.",
        )

        lub_oil_pressure = st.number_input(
            SENSOR_LABELS["lub_oil_pressure"],
            min_value=0.10,
            max_value=20.00,
            value=3.30,
            step=0.01,
            format="%.2f",
            help="Pressure of the engine's lubricating oil system.",
        )

        fuel_pressure = st.number_input(
            SENSOR_LABELS["fuel_pressure"],
            min_value=0.10,
            max_value=20.00,
            value=6.65,
            step=0.01,
            format="%.2f",
            help="Fuel pressure supplied to the engine.",
        )

    with right:

        coolant_pressure = st.number_input(
            SENSOR_LABELS["coolant_pressure"],
            min_value=0.10,
            max_value=20.00,
            value=2.33,
            step=0.01,
            format="%.2f",
            help="Pressure within the engine cooling system.",
        )

        lub_oil_temp = st.number_input(
            SENSOR_LABELS["lub_oil_temp"],
            min_value=60.00,
            max_value=120.00,
            value=77.64,
            step=0.10,
            format="%.2f",
            help="Temperature of the lubricating oil.",
        )

        coolant_temp = st.number_input(
            SENSOR_LABELS["coolant_temp"],
            min_value=60.00,
            max_value=120.00,
            value=78.47,
            step=0.10,
            format="%.2f",
            help="Temperature of the engine coolant.",
        )

    return pd.DataFrame([{
        "engine_rpm": engine_rpm,
        "lub_oil_pressure": lub_oil_pressure,
        "fuel_pressure": fuel_pressure,
        "coolant_pressure": coolant_pressure,
        "lub_oil_temp": lub_oil_temp,
        "coolant_temp": coolant_temp,
    }])


def display_recommendation(prediction_proba: float) -> None:
    """
    Display the maintenance recommendation based on
    the predicted failure probability.
    """

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

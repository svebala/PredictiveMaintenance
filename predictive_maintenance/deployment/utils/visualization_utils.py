"""
Visualization utilities for the Streamlit application.
"""

import streamlit.components.v1 as components

from config import (
    LOW_RISK_THRESHOLD,
    CLASSIFICATION_THRESHOLD,
)


def risk_progress_bar(probability: float) -> None:
    """
    Display a dynamic risk progress bar using SVG.

    Parameters
    ----------
    probability : float
        Predicted failure probability.
    """

    percent = round(float(probability) * 100, 2)

    low_threshold = LOW_RISK_THRESHOLD
    high_threshold = CLASSIFICATION_THRESHOLD

    low_threshold_perc = low_threshold * 100
    high_threshold_perc = high_threshold * 100

    # Badge colors
    if percent <= low_threshold_perc:
        badge_fill = "#E8F5E9"
        badge_border = "#2E7D32"
        badge_text = "#1B5E20"

    elif percent <= high_threshold_perc:
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

    low_pos = low_threshold * bar_width
    high_pos = high_threshold * bar_width

    mid_pos = (
        (low_threshold + high_threshold) / 2
    ) * bar_width

    high_label_pos = (
        (high_threshold + 1.0) / 2
    ) * bar_width

    marker_x = (
        bar_x + (percent / 100) * bar_width
    )

    fill_width = (
        (percent / 100) * bar_width
    )

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

            <stop offset="{low_threshold_perc}%" stop-color="#F9A825"/>
            <stop offset="{high_threshold_perc}%" stop-color="#F9A825"/>

            <stop offset="{high_threshold_perc}%" stop-color="#C62828"/>
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
        cx="{bar_x + low_pos}"
        cy="106"
        r="5"
        fill="white"
        stroke="#2E7D32"
        stroke-width="2"/>

    <circle
        cx="{bar_x + high_pos}"
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

    <text x="{bar_x + low_pos}" y="126"
          text-anchor="middle"
          font-size="13"
          font-weight="bold">{low_threshold_perc:.0f}%</text>

    <text x="{bar_x + high_pos}" y="126"
          text-anchor="middle"
          font-size="13"
          font-weight="bold">{high_threshold_perc:.0f}%</text>

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
        x="{bar_x + mid_pos}"
        y="155"
        text-anchor="middle"
        font-size="14"
        font-weight="bold"
        fill="#F9A825">
        Moderate
    </text>

    <text
        x="{bar_x + high_label_pos}"
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

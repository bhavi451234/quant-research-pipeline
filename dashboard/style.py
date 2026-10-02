"""
Shared styling helpers for every dashboard page. Import and call
apply_shared_style() once near the top of each page script.
"""

import streamlit as st

ACCENT = "#2DD4BF"
POSITIVE = "#34D399"
NEGATIVE = "#F87171"
MUTED = "#9CA3AF"


def apply_shared_style():
    st.markdown(
        """
        <style>
        /* Rounded, shadowed metric cards */
        div[data-testid="stMetric"] {
            background-color: #1A1F29;
            border: 1px solid #2A313D;
            border-radius: 12px;
            padding: 16px 18px 12px 18px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
        }
        div[data-testid="stMetricLabel"] {
            color: #9CA3AF;
            font-size: 0.85rem;
        }
        /* Tabs: a bit more breathing room */
        button[data-baseweb="tab"] {
            font-size: 0.95rem;
            padding-top: 8px;
            padding-bottom: 8px;
        }
        /* Sidebar section spacing */
        section[data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_color(value) -> str:
    """Returns a hex color for a numeric value: green if positive, red if
    negative, muted gray if NaN/None. Used to color Plotly bars by sign."""
    import math
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return MUTED
    return POSITIVE if value >= 0 else NEGATIVE
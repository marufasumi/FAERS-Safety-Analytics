"""
Shared styling utilities for the FDA FAERS Streamlit application.

The application uses a professional healthcare analytics theme based on:
- deep navy;
- teal accents;
- light neutral backgrounds;
- white analytical cards;
- restrained borders and shadows.
"""

from __future__ import annotations

import html

import streamlit as st


# ==========================================================
# Application Color Palette
# ==========================================================

NAVY = "#12304A"
DARK_NAVY = "#0B2239"
TEAL = "#0E7490"
DARK_TEAL = "#0F5F73"
LIGHT_TEAL = "#E6F4F7"

PAGE_BACKGROUND = "#F5F7FA"
CARD_BACKGROUND = "#FFFFFF"
SIDEBAR_BACKGROUND = "#EDF2F7"

PRIMARY_TEXT = "#1F2937"
SECONDARY_TEXT = "#64748B"
BORDER_COLOR = "#D9E2EC"

SUCCESS = "#167D5A"
SUCCESS_BACKGROUND = "#ECF8F3"

INFO = "#2563EB"
INFO_BACKGROUND = "#EFF6FF"

WARNING = "#B45309"
WARNING_BACKGROUND = "#FFF7ED"

DANGER = "#B42318"
DANGER_BACKGROUND = "#FEF3F2"


# ==========================================================
# Global Styling
# ==========================================================

def apply_global_styles() -> None:
    """
    Apply the shared professional dashboard styling.
    """
    st.markdown(
        f"""
        <style>
        /* ==================================================
           Main application
           ================================================== */

        .stApp {{
            background-color: {PAGE_BACKGROUND};
            color: {PRIMARY_TEXT};
        }}

        .block-container {{
            max-width: 1280px;
            padding-top: 2rem;
            padding-bottom: 4rem;
            padding-left: 2.5rem;
            padding-right: 2.5rem;
        }}

        html,
        body,
        [class*="css"] {{
            font-family:
                Inter,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                Roboto,
                Helvetica,
                Arial,
                sans-serif;
        }}


        /* ==================================================
           Main headings
           ================================================== */

        h1 {{
            color: {NAVY};
            font-size: 2.25rem;
            font-weight: 750;
            letter-spacing: -0.035em;
            line-height: 1.2;
            margin-bottom: 0.75rem;
        }}

        h2 {{
            color: {NAVY};
            font-size: 1.55rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-top: 2rem;
            margin-bottom: 0.8rem;
        }}

        h3 {{
            color: {DARK_NAVY};
            font-size: 1.15rem;
            font-weight: 680;
            margin-top: 1.25rem;
            margin-bottom: 0.6rem;
        }}

        p,
        li {{
            color: {PRIMARY_TEXT};
            line-height: 1.65;
        }}

        small,
        .stCaption,
        [data-testid="stCaptionContainer"] {{
            color: {SECONDARY_TEXT};
        }}

        hr {{
            border: none;
            border-top: 1px solid {BORDER_COLOR};
            margin-top: 2rem;
            margin-bottom: 2rem;
        }}


        /* ==================================================
           Sidebar
           ================================================== */

        [data-testid="stSidebar"] {{
            background:
                linear-gradient(
                    180deg,
                    {DARK_NAVY} 0%,
                    {NAVY} 100%
                );
            border-right: none;
        }}

        [data-testid="stSidebar"] > div:first-child {{
            padding-top: 1.25rem;
        }}

        [data-testid="stSidebar"] * {{
            color: #F8FAFC;
        }}

        [data-testid="stSidebar"] a {{
            border-radius: 8px;
            margin-top: 0.15rem;
            margin-bottom: 0.15rem;
            transition:
                background-color 0.2s ease,
                transform 0.2s ease;
        }}

        [data-testid="stSidebar"] a:hover {{
            background-color: rgba(255, 255, 255, 0.12);
            transform: translateX(2px);
        }}

        [data-testid="stSidebar"] a[aria-current="page"] {{
            background-color: {TEAL};
            color: #FFFFFF;
            font-weight: 650;
        }}

        [data-testid="stSidebarNav"] span {{
            color: #F8FAFC;
        }}

        [data-testid="stSidebarNav"] svg {{
            color: #E2E8F0;
        }}

        [data-testid="stSidebarCollapsedControl"] {{
            color: {NAVY};
        }}


        /* ==================================================
           Metric cards
           ================================================== */

        [data-testid="stMetric"] {{
            background-color: {CARD_BACKGROUND};
            border: 1px solid {BORDER_COLOR};
            border-radius: 14px;
            padding: 1rem 1.1rem;
            min-height: 118px;
            box-shadow:
                0 2px 5px rgba(15, 39, 68, 0.04),
                0 10px 25px rgba(15, 39, 68, 0.05);
            transition:
                transform 0.2s ease,
                box-shadow 0.2s ease;
        }}

        [data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            box-shadow:
                0 5px 12px rgba(15, 39, 68, 0.06),
                0 16px 30px rgba(15, 39, 68, 0.08);
        }}

        [data-testid="stMetricLabel"] {{
            color: {SECONDARY_TEXT};
            font-size: 0.82rem;
            font-weight: 650;
            letter-spacing: 0.02em;
        }}

        [data-testid="stMetricValue"] {{
            color: {NAVY};
            font-size: 1.8rem;
            font-weight: 750;
            letter-spacing: -0.03em;
        }}

        [data-testid="stMetricDelta"] {{
            font-size: 0.82rem;
        }}


        /* ==================================================
           Buttons
           ================================================== */

        .stButton > button,
        .stFormSubmitButton > button {{
            background:
                linear-gradient(
                    135deg,
                    {TEAL},
                    {DARK_TEAL}
                );
            color: #FFFFFF;
            border: none;
            border-radius: 9px;
            min-height: 2.8rem;
            padding: 0.65rem 1.25rem;
            font-weight: 650;
            box-shadow:
                0 4px 10px rgba(14, 116, 144, 0.18);
            transition:
                transform 0.15s ease,
                box-shadow 0.15s ease,
                opacity 0.15s ease;
        }}

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {{
            color: #FFFFFF;
            border: none;
            transform: translateY(-1px);
            box-shadow:
                0 7px 16px rgba(14, 116, 144, 0.25);
        }}

        .stButton > button:active,
        .stFormSubmitButton > button:active {{
            transform: translateY(0);
            opacity: 0.95;
        }}

        .stButton > button:focus,
        .stFormSubmitButton > button:focus {{
            color: #FFFFFF;
            border-color: {TEAL};
            box-shadow:
                0 0 0 0.2rem rgba(14, 116, 144, 0.18);
        }}


        /* ==================================================
           Inputs
           ================================================== */

        [data-baseweb="input"] > div,
        [data-baseweb="select"] > div,
        [data-baseweb="textarea"] > div {{
            background-color: #FFFFFF;
            border-color: {BORDER_COLOR};
            border-radius: 8px;
        }}

        [data-baseweb="input"] > div:focus-within,
        [data-baseweb="select"] > div:focus-within,
        [data-baseweb="textarea"] > div:focus-within {{
            border-color: {TEAL};
            box-shadow:
                0 0 0 2px rgba(14, 116, 144, 0.12);
        }}

        [data-testid="stNumberInput"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stTextInput"] label,
        [data-testid="stRadio"] label,
        [data-testid="stMultiSelect"] label {{
            color: {DARK_NAVY};
            font-weight: 600;
        }}

        div[role="radiogroup"] {{
            gap: 0.5rem;
        }}


        /* ==================================================
           Tabs
           ================================================== */

        [data-baseweb="tab-list"] {{
            gap: 0.35rem;
            background-color: #EAF0F5;
            border-radius: 10px;
            padding: 0.3rem;
        }}

        [data-baseweb="tab"] {{
            height: 2.75rem;
            border-radius: 7px;
            padding-left: 1rem;
            padding-right: 1rem;
            color: {SECONDARY_TEXT};
            font-weight: 600;
        }}

        [aria-selected="true"][data-baseweb="tab"] {{
            background-color: #FFFFFF;
            color: {TEAL};
            box-shadow:
                0 2px 7px rgba(15, 39, 68, 0.08);
        }}

        [data-baseweb="tab-highlight"] {{
            display: none;
        }}

        [data-baseweb="tab-border"] {{
            display: none;
        }}


        /* ==================================================
           Expanders
           ================================================== */

        [data-testid="stExpander"] {{
            background-color: {CARD_BACKGROUND};
            border: 1px solid {BORDER_COLOR};
            border-radius: 10px;
            box-shadow:
                0 2px 6px rgba(15, 39, 68, 0.035);
            overflow: hidden;
        }}

        [data-testid="stExpander"] summary {{
            color: {DARK_NAVY};
            font-weight: 620;
            padding-top: 0.15rem;
            padding-bottom: 0.15rem;
        }}

        [data-testid="stExpander"] summary:hover {{
            color: {TEAL};
        }}


        /* ==================================================
           Forms
           ================================================== */

        [data-testid="stForm"] {{
            background-color: rgba(255, 255, 255, 0.72);
            border: 1px solid {BORDER_COLOR};
            border-radius: 14px;
            padding: 1.25rem;
            box-shadow:
                0 4px 18px rgba(15, 39, 68, 0.04);
        }}


        /* ==================================================
           Dataframes and tables
           ================================================== */

        [data-testid="stDataFrame"] {{
            background-color: {CARD_BACKGROUND};
            border: 1px solid {BORDER_COLOR};
            border-radius: 10px;
            overflow: hidden;
            box-shadow:
                0 2px 7px rgba(15, 39, 68, 0.04);
        }}

        [data-testid="stTable"] {{
            background-color: {CARD_BACKGROUND};
            border-radius: 10px;
            overflow: hidden;
        }}


        /* ==================================================
           Plotly charts and images
           ================================================== */

        [data-testid="stPlotlyChart"] {{
            background-color: {CARD_BACKGROUND};
            border: 1px solid {BORDER_COLOR};
            border-radius: 14px;
            padding: 0.75rem;
            box-shadow:
                0 5px 18px rgba(15, 39, 68, 0.05);
        }}

        [data-testid="stImage"] img {{
            border-radius: 12px;
            border: 1px solid {BORDER_COLOR};
            box-shadow:
                0 5px 18px rgba(15, 39, 68, 0.06);
        }}


        /* ==================================================
           Streamlit alerts
           ================================================== */

        [data-testid="stAlert"] {{
            border-radius: 10px;
            border-width: 1px;
        }}


        /* ==================================================
           Download and toolbar controls
           ================================================== */

        [data-testid="stDownloadButton"] > button {{
            border-radius: 8px;
            border: 1px solid {TEAL};
            color: {TEAL};
            background-color: #FFFFFF;
        }}

        [data-testid="stDownloadButton"] > button:hover {{
            background-color: {LIGHT_TEAL};
            color: {DARK_TEAL};
            border-color: {DARK_TEAL};
        }}


        /* ==================================================
           Hide default Streamlit chrome
           ================================================== */

        #MainMenu {{
            visibility: hidden;
        }}

        footer {{
            visibility: hidden;
        }}

        [data-testid="stStatusWidget"] {{
            visibility: hidden;
        }}


        /* ==================================================
           Responsive layout
           ================================================== */

        @media (max-width: 900px) {{
            .block-container {{
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: 1.25rem;
            }}

            h1 {{
                font-size: 1.8rem;
            }}

            h2 {{
                font-size: 1.35rem;
            }}

            [data-testid="stMetric"] {{
                min-height: 100px;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# Reusable Information Box
# ==========================================================

def render_info_box(message: str) -> None:
    """
    Render a professional informational callout.
    """
    safe_message = message.strip()

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(
                135deg,
                {INFO_BACKGROUND},
                #FFFFFF
            );
            border: 1px solid #BFDBFE;
            border-left: 5px solid {INFO};
            border-radius: 10px;
            padding: 1rem 1.15rem;
            margin-top: 0.75rem;
            margin-bottom: 1.25rem;
            color: {PRIMARY_TEXT};
            line-height: 1.6;
            box-shadow: 0 3px 10px rgba(37, 99, 235, 0.05);
        ">
            <div style="
                color: {PRIMARY_TEXT};
                font-size: 0.96rem;
            ">
                {safe_message}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# Reusable Result Box
# ==========================================================

def render_result_box(message: str) -> None:
    """
    Render a professional positive result callout.
    """
    safe_message = message.strip()

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(
                135deg,
                {SUCCESS_BACKGROUND},
                #FFFFFF
            );
            border: 1px solid #B7E4D3;
            border-left: 5px solid {SUCCESS};
            border-radius: 10px;
            padding: 1rem 1.15rem;
            margin-top: 0.75rem;
            margin-bottom: 1.25rem;
            color: {PRIMARY_TEXT};
            line-height: 1.6;
            box-shadow: 0 3px 10px rgba(22, 125, 90, 0.06);
        ">
            <div style="
                color: {PRIMARY_TEXT};
                font-size: 0.97rem;
            ">
                {safe_message}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# Reusable Disclaimer
# ==========================================================

def render_disclaimer(disclaimer: str) -> None:
    """
    Render the application disclaimer.
    """
    safe_disclaimer = html.escape(
        disclaimer.strip()
    )

    st.markdown(
        f"""
        <div style="
            margin-top: 2.5rem;
            padding: 1rem 1.15rem;
            background-color: {WARNING_BACKGROUND};
            border: 1px solid #FED7AA;
            border-left: 5px solid {WARNING};
            border-radius: 10px;
            color: #7C2D12;
            font-size: 0.88rem;
            line-height: 1.55;
        ">
            <div style="
                font-weight: 700;
                margin-bottom: 0.3rem;
                color: {WARNING};
            ">
                Important Disclaimer
            </div>

            {safe_disclaimer}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# Image Caption
# ==========================================================

def render_image_caption(caption: str) -> None:
    """
    Render a consistent analytical figure caption.
    """
    safe_caption = html.escape(
        caption.strip()
    )

    st.markdown(
        f"""
        <div style="
            color: {SECONDARY_TEXT};
            font-size: 0.84rem;
            line-height: 1.5;
            text-align: center;
            margin-top: 0.4rem;
            margin-bottom: 1.25rem;
            padding-left: 1rem;
            padding-right: 1rem;
        ">
            <strong style="color: {NAVY};">
                Figure.
            </strong>
            {safe_caption}
        </div>
        """,
        unsafe_allow_html=True,
    )
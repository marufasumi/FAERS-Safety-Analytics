"""
Home page for the FDA FAERS Serious Adverse Event Prediction application.

This page introduces the project, summarizes the main analytical results,
and directs users to the detailed Streamlit pages.
"""

import streamlit as st

from utils.constants import (
    DISCLAIMER,
    KMEANS_K,
    RANDOM_FOREST_ACCURACY,
    RANDOM_FOREST_F1,
    RANDOM_FOREST_ROC_AUC,
    TOTAL_MODELS,
    TOTAL_PREDICTORS,
    TOTAL_REPORTS,
)
from utils.styling import (
    apply_global_styles,
    render_disclaimer,
    render_info_box,
)


# ==========================================================
# Streamlit Page Configuration
# ==========================================================

st.set_page_config(
    page_title="FAERS Serious Event Prediction",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_styles()


# ==========================================================
# Sidebar
# ==========================================================

with st.sidebar:
    st.title("FAERS Analytics")

    st.markdown(
        """
        Explore the project using the pages in the navigation menu.
        """
    )

    st.divider()

    st.markdown("### Project Scope")

    st.markdown(
        f"""
        - **Reports:** {TOTAL_REPORTS:,}
        - **Predictors:** {TOTAL_PREDICTORS}
        - **Models evaluated:** {TOTAL_MODELS}
        - **K-Means clusters:** {KMEANS_K}
        """
    )

    st.divider()

    st.markdown(
        """
        **Data source**

        FDA Adverse Event Reporting System  
        2024 Quarter 1 ASCII files
        """
    )


# ==========================================================
# Header
# ==========================================================

st.title(
    "Predicting Serious Adverse Drug Event Reports Using FDA FAERS Data"
)

st.markdown(
    """
    A statistical learning and machine learning application for analyzing
    reported adverse drug events and estimating whether a FAERS report is
    classified as serious.
    """
)

render_info_box(
    """
    This portfolio application demonstrates an end-to-end machine learning
    pipeline for analyzing FDA Adverse Event Reporting System (FAERS) data.
    It showcases data engineering, exploratory analysis, statistical
    learning, machine learning, clustering, and real-time prediction of
    serious adverse drug event reports.
    """
)


# ==========================================================
# Main Project Metrics
# ==========================================================

st.subheader("Project at a Glance")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric(
        label="FAERS Reports",
        value=f"{TOTAL_REPORTS:,}",
    )

with metric_col2:
    st.metric(
        label="Model Predictors",
        value=str(TOTAL_PREDICTORS),
    )

with metric_col3:
    st.metric(
        label="Models Evaluated",
        value=str(TOTAL_MODELS),
    )

with metric_col4:
    st.metric(
        label="Best Test Accuracy",
        value=f"{RANDOM_FOREST_ACCURACY:.1%}",
    )


# ==========================================================
# Best Model Summary
# ==========================================================

st.subheader("Best Predictive Model")

model_col1, model_col2, model_col3 = st.columns(3)

with model_col1:
    st.metric(
        label="Random Forest Accuracy",
        value=f"{RANDOM_FOREST_ACCURACY:.3f}",
    )

with model_col2:
    st.metric(
        label="Random Forest ROC AUC",
        value=f"{RANDOM_FOREST_ROC_AUC:.3f}",
    )

with model_col3:
    st.metric(
        label="Random Forest F1 Score",
        value=f"{RANDOM_FOREST_F1:.3f}",
    )

st.markdown(
    """
    The Random Forest model achieved the strongest overall predictive
    performance among the nine evaluated methods. The final deployed model
    uses a saved scikit-learn pipeline containing the original preprocessing
    steps and fitted classifier.
    """
)


# ==========================================================
# Project Components
# ==========================================================

st.subheader("Application Sections")

section_col1, section_col2 = st.columns(2)

with section_col1:
    st.markdown(
        """
        ### Project Overview

        Review the research question, data source, analytical dataset,
        target variable, and engineered feature groups.

        ### Exploratory Data Analysis

        Examine target balance, missing values, demographics, medication
        burden, reaction burden, indication burden, therapy characteristics,
        and reporting patterns.

        ### Statistical Analysis

        Review Welch t-tests, Mann–Whitney U tests, and chi-square tests used
        to compare serious and non-serious reports.
        """
    )

with section_col2:
    st.markdown(
        """
        ### Machine Learning

        Compare the performance of nine supervised learning models using
        accuracy, precision, recall, specificity, F1 score, ROC AUC, and
        cross-validation results.

        ### Clustering

        Explore the seven K-Means patient-safety profiles identified from
        medication burden, reactions, indications, therapy, and reporting
        characteristics.

        ### Predict Serious Event

        Enter report-level characteristics and obtain an educational
        classification from the saved Random Forest pipeline.
        """
    )


# ==========================================================
# Analytical Workflow
# ==========================================================

st.subheader("Analytical Workflow")

st.markdown(
    """
    **Raw FDA FAERS tables**
    → **report-level data integration**
    → **feature engineering**
    → **exploratory and statistical analysis**
    → **supervised model comparison**
    → **K-Means clustering**
    → **interactive prediction**
    """
)


# ==========================================================
# Key Findings
# ==========================================================

st.subheader("Key Findings")

st.markdown(
    """
    - Serious and non-serious reports differed significantly across multiple
      demographic, medication, reaction, indication, therapy, and reporting
      characteristics.
    - Medication burden and clinical complexity were strongly associated with
      serious-report classification.
    - Random Forest produced the strongest overall predictive performance.
    - K-Means clustering identified distinct report profiles, including small
      high-complexity clusters with very high serious-report proportions.
    """
)


# ==========================================================
# Disclaimer
# ==========================================================

render_disclaimer(DISCLAIMER)

st.caption(
    "This is an educational application and must not be used for clinical diagnosis or treatment decisions."
)

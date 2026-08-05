"""
Project overview page for the FDA FAERS serious-event application.

This page describes the research objective, source data, analytical dataset,
target variable, engineered features, and end-to-end analytical workflow.
"""

import streamlit as st

from utils.constants import (
    DISCLAIMER,
    KMEANS_K,
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
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Project Overview | FAERS Analytics",
    page_icon="📘",
    layout="wide",
)

apply_global_styles()


# ==========================================================
# Header
# ==========================================================

st.title("Project Overview")

st.markdown(
    """
    This project uses FDA Adverse Event Reporting System data and statistical
    learning methods to study factors associated with serious adverse drug
    event reports.
    """
)

render_info_box(
    """
    The final analytical dataset contains one row per FAERS report. Information
    from seven raw FDA tables was aggregated and integrated using the report
    identifier before statistical analysis, machine learning, and clustering.
    """
)


# ==========================================================
# Research Objective
# ==========================================================

st.header("Research Objective")

st.markdown(
    """
    The primary objective was to develop and compare statistical learning
    models that classify whether an FDA FAERS report is serious or
    non-serious.

    The analysis also examined:

    - how serious and non-serious reports differ across demographic,
      medication, reaction, indication, therapy, and reporting characteristics;
    - which supervised learning model provides the strongest predictive
      performance;
    - whether unsupervised learning can identify distinct adverse-event
      report profiles.
    """
)


# ==========================================================
# Project Scale
# ==========================================================

st.header("Project Scale")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric(
        label="Analytical Reports",
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
        label="K-Means Clusters",
        value=str(KMEANS_K),
    )


# ==========================================================
# Data Source
# ==========================================================

st.header("Data Source")

st.markdown(
    """
    The project used the **FDA Adverse Event Reporting System 2024 Quarter 1
    ASCII data files**.

    FAERS is a spontaneous reporting system containing adverse-event reports
    submitted to the FDA by healthcare professionals, consumers,
    manufacturers, and other reporting sources.
    """
)

data_col1, data_col2 = st.columns(2)

with data_col1:
    st.subheader("Raw Tables")

    st.markdown(
        """
        - **DEMO** — demographic and administrative report information
        - **DRUG** — reported medications and drug roles
        - **REAC** — reported adverse reactions
        - **OUTC** — reported patient outcomes
        """
    )

with data_col2:
    st.subheader("Additional Tables")

    st.markdown(
        """
        - **THER** — therapy start and end dates
        - **INDI** — reported drug indications
        - **RPSR** — report-source information
        """
    )

st.warning(
    """
    FAERS is a spontaneous reporting database. It cannot establish causal
    relationships, incidence rates, prevalence, or comparative drug safety
    without additional epidemiological evidence.
    """
)


# ==========================================================
# Analytical Dataset
# ==========================================================

st.header("Analytical Dataset")

st.markdown(
    """
    The raw FAERS tables contain multiple rows for many reports because one
    report may include several medications, reactions, indications, outcomes,
    therapy records, or reporter sources.

    To support report-level modeling, each table was aggregated using
    `primaryid`. The aggregated tables were then joined to the DEMO table,
    producing one analytical row per report.
    """
)

st.code(
    """
Raw FDA tables
      ↓
Table-specific cleaning
      ↓
Aggregation by primaryid
      ↓
Report-level table integration
      ↓
Feature engineering
      ↓
406,184 rows × 49 model predictors
    """,
    language="text",
)


# ==========================================================
# Target Variable
# ==========================================================

st.header("Target Variable")

target_col1, target_col2 = st.columns(2)

with target_col1:
    st.subheader("Serious Report")

    st.markdown(
        """
        A report was classified as serious when at least one serious outcome
        was recorded in the FAERS outcome table.

        Examples of serious outcomes include:

        - death;
        - life-threatening event;
        - hospitalization;
        - disability;
        - congenital anomaly;
        - another medically important condition.
        """
    )

with target_col2:
    st.subheader("Non-Serious Report")

    st.markdown(
        """
        A report was classified as non-serious when no serious outcome record
        was identified for the report.

        Model encoding:

        - `1` = serious report
        - `0` = non-serious report
        """
    )

st.markdown(
    """
    The target distribution was moderately balanced:

    - **Serious reports:** 222,364 — 54.74%
    - **Non-serious reports:** 183,820 — 45.26%
    """
)


# ==========================================================
# Feature Engineering
# ==========================================================

st.header("Engineered Feature Groups")

feature_col1, feature_col2 = st.columns(2)

with feature_col1:
    st.subheader("Demographic Features")

    st.markdown(
        """
        - age in years;
        - body weight in kilograms;
        - sex;
        - reporting and occurrence countries;
        - reporter and report types.
        """
    )

    st.subheader("Medication Burden")

    st.markdown(
        """
        - total and unique drugs;
        - primary and secondary suspect drugs;
        - concomitant and interacting drugs;
        - medication-class indicators.
        """
    )

    st.subheader("Reaction Burden")

    st.markdown(
        """
        - total and unique reactions;
        - acute kidney injury;
        - diabetic ketoacidosis;
        - hypoglycemia;
        - lactic acidosis;
        - amputation;
        - genital infection.
        """
    )

with feature_col2:
    st.subheader("Indication Burden")

    st.markdown(
        """
        - total and unique indications;
        - diabetes indicators;
        - hypertension;
        - chronic kidney disease;
        - heart failure;
        - obesity;
        - unknown indication.
        """
    )

    st.subheader("Therapy Characteristics")

    st.markdown(
        """
        - number of therapy records;
        - mean, median, minimum, and maximum therapy duration;
        - short-term and long-term therapy indicators;
        - therapy-duration outlier indicator.
        """
    )

    st.subheader("Reporting Characteristics")

    st.markdown(
        """
        - number of reporter sources;
        - healthcare-professional report;
        - consumer report;
        - foreign report.
        """
    )


# ==========================================================
# Analytical Methods
# ==========================================================

st.header("Analytical Methods")

method_col1, method_col2, method_col3 = st.columns(3)

with method_col1:
    st.subheader("Exploratory Analysis")

    st.markdown(
        """
        Distribution analysis, missing-value assessment, summary statistics,
        visual comparisons, and correlation analysis.
        """
    )

with method_col2:
    st.subheader("Statistical Testing")

    st.markdown(
        """
        Welch t-tests, Mann–Whitney U tests, and chi-square tests were used to
        compare serious and non-serious reports.
        """
    )

with method_col3:
    st.subheader("Statistical Learning")

    st.markdown(
        """
        Nine supervised models were compared, and K-Means clustering was used
        to identify distinct report profiles.
        """
    )


# ==========================================================
# Models Evaluated
# ==========================================================

st.header("Supervised Models Evaluated")

st.markdown(
    """
    The project compared the following classification methods:

    1. Logistic Regression
    2. Linear Discriminant Analysis
    3. Quadratic Discriminant Analysis
    4. Gaussian Naive Bayes
    5. K-Nearest Neighbors
    6. Decision Tree
    7. Random Forest
    8. Support Vector Machine
    9. Neural Network
    """
)


# ==========================================================
# End-to-End Workflow
# ==========================================================

st.header("End-to-End Workflow")

st.markdown(
    """
    **1. Data acquisition**  
    Download and inspect the seven FDA FAERS ASCII tables.

    **2. Data cleaning**  
    Standardize identifiers, demographics, dates, medication fields,
    reactions, indications, outcomes, and report-source variables.

    **3. Data integration**  
    Aggregate each table to report level and merge using `primaryid`.

    **4. Feature engineering**  
    Create burden measures, clinical-category flags, therapy-duration
    measures, and reporting indicators.

    **5. Exploratory and statistical analysis**  
    Compare serious and non-serious reports and quantify significant
    differences.

    **6. Supervised modeling**  
    Train, tune, and evaluate nine classification methods.

    **7. Unsupervised modeling**  
    Apply K-Means clustering to identify distinct report profiles.

    **8. Deployment**  
    Load the saved Random Forest pipeline into this Streamlit application for
    educational prediction demonstrations.
    """
)


# ==========================================================
# Disclaimer
# ==========================================================

render_disclaimer(DISCLAIMER)
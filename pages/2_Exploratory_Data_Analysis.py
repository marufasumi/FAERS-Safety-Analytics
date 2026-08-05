"""
Exploratory Data Analysis page for the FDA FAERS application.

This page presents previously exported EDA figures and summary tables.
It does not reload or recompute the original 2024 Q1 raw FAERS data.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.constants import (
    ASSETS_DIR,
    DATA_DIR,
    DISCLAIMER,
    TOTAL_REPORTS,
)
from utils.loaders import load_csv, load_image
from utils.styling import (
    apply_global_styles,
    render_disclaimer,
    render_image_caption,
    render_info_box,
)


# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Exploratory Data Analysis | FAERS Analytics",
    page_icon="📊",
    layout="wide",
)

apply_global_styles()


# ==========================================================
# File Paths
# ==========================================================

TARGET_DISTRIBUTION_IMAGE = ASSETS_DIR / "target_distribution.png"

TARGET_DISTRIBUTION_CSV = DATA_DIR / "target_distribution.csv"
MISSING_VALUES_CSV = DATA_DIR / "missing_values.csv"
NUMERIC_SUMMARY_CSV = DATA_DIR / "numeric_summary.csv"
DEMOGRAPHIC_SUMMARY_CSV = DATA_DIR / "demographic_summary.csv"
DRUG_BURDEN_SUMMARY_CSV = DATA_DIR / "drug_burden_summary.csv"
REACTION_BURDEN_SUMMARY_CSV = DATA_DIR / "reaction_burden_summary.csv"
INDICATION_BURDEN_SUMMARY_CSV = DATA_DIR / "indication_burden_summary.csv"
THERAPY_SUMMARY_CSV = DATA_DIR / "therapy_summary.csv"
REPORTER_SUMMARY_CSV = DATA_DIR / "reporter_summary.csv"


# ==========================================================
# Helper Functions
# ==========================================================

def show_table(
    dataframe: pd.DataFrame | None,
    title: str,
    description: str,
) -> None:
    """
    Display an exported EDA table with a heading and description.
    """
    st.subheader(title)
    st.markdown(description)

    if dataframe is None or dataframe.empty:
        st.info("The exported table is unavailable.")
        return

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
    )


def find_first_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """
    Find the first matching column name using case-insensitive comparison.
    """
    normalized_columns = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        normalized_candidate = candidate.strip().lower()

        if normalized_candidate in normalized_columns:
            return normalized_columns[normalized_candidate]

    return None


def render_missing_values_chart(
    dataframe: pd.DataFrame | None,
) -> None:
    """
    Create a missing-value bar chart when suitable exported columns exist.
    """
    if dataframe is None or dataframe.empty:
        return

    variable_column = find_first_column(
        dataframe,
        [
            "Variable",
            "variable",
            "Feature",
            "feature",
            "Column",
            "column",
        ],
    )

    percentage_column = find_first_column(
        dataframe,
        [
            "Missing_Percentage",
            "missing_percentage",
            "Missing Percentage",
            "Missing %",
            "Percent_Missing",
            "percent_missing",
        ],
    )

    count_column = find_first_column(
        dataframe,
        [
            "Missing_Count",
            "missing_count",
            "Missing Count",
            "Count_Missing",
            "count_missing",
        ],
    )

    value_column = percentage_column or count_column

    if variable_column is None or value_column is None:
        return

    chart_df = dataframe[
        [variable_column, value_column]
    ].copy()

    chart_df[value_column] = pd.to_numeric(
        chart_df[value_column],
        errors="coerce",
    )

    chart_df = (
        chart_df
        .dropna(subset=[value_column])
        .sort_values(value_column, ascending=False)
        .head(15)
        .set_index(variable_column)
    )

    if chart_df.empty:
        return

    st.bar_chart(chart_df)


# ==========================================================
# Header
# ==========================================================

st.title("Exploratory Data Analysis")

st.markdown(
    """
    This page summarizes the structure, distributions, missingness, and
    report-level characteristics of the final FDA FAERS analytical dataset.
    """
)

render_info_box(
    """
    The visualizations and tables shown here were exported from the completed
    analytical notebooks. The Streamlit application presents the saved
    results without rerunning the full raw-data preparation pipeline.
    """
)


# ==========================================================
# Dataset Overview
# ==========================================================

st.header("Dataset Overview")

overview_col1, overview_col2, overview_col3 = st.columns(3)

with overview_col1:
    st.metric(
        label="Total Reports",
        value=f"{TOTAL_REPORTS:,}",
    )

with overview_col2:
    st.metric(
        label="Serious Reports",
        value="222,364",
        delta="54.74% of reports",
    )

with overview_col3:
    st.metric(
        label="Non-Serious Reports",
        value="183,820",
        delta="45.26% of reports",
    )

st.markdown(
    """
    Each analytical row represents one FAERS report. Variables were engineered
    from demographic, drug, reaction, indication, therapy, outcome, and
    report-source tables.
    """
)


# ==========================================================
# Load Exported Results
# ==========================================================

target_distribution_df = load_csv(TARGET_DISTRIBUTION_CSV)
missing_values_df = load_csv(MISSING_VALUES_CSV)
numeric_summary_df = load_csv(NUMERIC_SUMMARY_CSV)
demographic_summary_df = load_csv(DEMOGRAPHIC_SUMMARY_CSV)
drug_burden_summary_df = load_csv(DRUG_BURDEN_SUMMARY_CSV)
reaction_burden_summary_df = load_csv(REACTION_BURDEN_SUMMARY_CSV)
indication_burden_summary_df = load_csv(INDICATION_BURDEN_SUMMARY_CSV)
therapy_summary_df = load_csv(THERAPY_SUMMARY_CSV)
reporter_summary_df = load_csv(REPORTER_SUMMARY_CSV)


# ==========================================================
# Navigation Tabs
# ==========================================================

(
    target_tab,
    missing_tab,
    demographic_tab,
    clinical_tab,
    therapy_tab,
) = st.tabs(
    [
        "Target Distribution",
        "Missing Values",
        "Demographics",
        "Clinical Burden",
        "Therapy and Reporting",
    ]
)


# ==========================================================
# Target Distribution Tab
# ==========================================================

with target_tab:
    st.header("Target Distribution")

    target_image = load_image(TARGET_DISTRIBUTION_IMAGE)

    image_col, summary_col = st.columns([1.4, 1])

    with image_col:
        if target_image is not None:
            st.image(
                target_image,
                use_container_width=True,
            )

            render_image_caption(
                "Distribution of serious and non-serious FAERS reports."
            )
        else:
            st.info(
                "The target-distribution image is unavailable."
            )

    with summary_col:
        st.subheader("Interpretation")

        st.markdown(
            """
            The target variable was moderately balanced:

            - **Serious:** 222,364 reports
            - **Non-serious:** 183,820 reports
            - **Serious proportion:** 54.74%
            - **Non-serious proportion:** 45.26%

            Because both classes were well represented, model evaluation was
            not dominated by an extremely rare outcome class.
            """
        )

        st.info(
            """
            Accuracy was considered together with precision, recall,
            specificity, F1 score, ROC AUC, and cross-validation performance.
            """
        )

    show_table(
        dataframe=target_distribution_df,
        title="Exported Target Summary",
        description=(
            "This table contains the report counts and proportions used in "
            "the target-distribution analysis."
        ),
    )


# ==========================================================
# Missing Values Tab
# ==========================================================

with missing_tab:
    st.header("Missing-Value Analysis")

    missing_col1, missing_col2 = st.columns(2)

    with missing_col1:
        st.metric(
            label="Age Missing",
            value="38.84%",
        )

    with missing_col2:
        st.metric(
            label="Weight Missing",
            value="83.10%",
        )

    st.markdown(
        """
        Missingness was especially high for body weight, while age was
        available for a larger—but still incomplete—subset of reports.

        Missing values were retained as meaningful data-quality characteristics
        and were handled through the preprocessing steps embedded in the saved
        machine learning pipeline.
        """
    )

    render_missing_values_chart(missing_values_df)

    show_table(
        dataframe=missing_values_df,
        title="Missing-Value Summary",
        description=(
            "Variables are summarized using exported missing counts and "
            "percentages from the analytical dataset."
        ),
    )


# ==========================================================
# Demographic Tab
# ==========================================================

with demographic_tab:
    st.header("Demographic Characteristics")

    age_col, weight_col = st.columns(2)

    with age_col:
        st.metric(
            label="Mean Reported Age",
            value="55.21 years",
        )

        st.caption(
            "Calculated among reports with non-missing age."
        )

    with weight_col:
        st.metric(
            label="Mean Reported Weight",
            value="73.42 kg",
        )

        st.caption(
            "Calculated among reports with non-missing weight."
        )

    st.markdown(
        """
        Demographic features provide important context, but their
        interpretation requires caution because spontaneous reports may
        contain incomplete, inconsistent, or selectively reported patient
        information.
        """
    )

    show_table(
        dataframe=demographic_summary_df,
        title="Demographic Summary",
        description=(
            "Summary statistics for demographic variables overall and, where "
            "available, by serious-report classification."
        ),
    )

    show_table(
        dataframe=numeric_summary_df,
        title="Numeric Feature Summary",
        description=(
            "Descriptive statistics for the numeric variables included in "
            "the report-level analytical dataset."
        ),
    )


# ==========================================================
# Clinical Burden Tab
# ==========================================================

with clinical_tab:
    st.header("Clinical and Report Complexity")

    st.markdown(
        """
        Report complexity was represented through counts and indicators
        describing medications, adverse reactions, and treatment indications.
        """
    )

    burden_section = st.radio(
        "Select a feature group",
        options=[
            "Medication Burden",
            "Reaction Burden",
            "Indication Burden",
        ],
        horizontal=True,
    )

    if burden_section == "Medication Burden":
        show_table(
            dataframe=drug_burden_summary_df,
            title="Medication Burden Summary",
            description=(
                "Summary statistics for total drugs, unique drugs, suspect "
                "drug roles, concomitant drugs, and interacting drugs."
            ),
        )

        st.markdown(
            """
            Medication burden captures the number and role of medications
            included in each report. Higher counts may reflect polypharmacy,
            clinical complexity, or more detailed case documentation.
            """
        )

    elif burden_section == "Reaction Burden":
        show_table(
            dataframe=reaction_burden_summary_df,
            title="Reaction Burden Summary",
            description=(
                "Summary statistics for the total and unique adverse "
                "reactions recorded per report."
            ),
        )

        st.markdown(
            """
            Reaction burden measures how many adverse-event terms were
            associated with a report. It does not measure clinical incidence
            or prove that a medication caused the reported reactions.
            """
        )

    else:
        show_table(
            dataframe=indication_burden_summary_df,
            title="Indication Burden Summary",
            description=(
                "Summary statistics for total indications, unique "
                "indications, and selected clinical-condition indicators."
            ),
        )

        st.markdown(
            """
            Indication features describe the reported reasons for medication
            use. Approximately one-third of reports contained an unknown or
            unavailable indication in the completed analysis.
            """
        )


# ==========================================================
# Therapy and Reporting Tab
# ==========================================================

with therapy_tab:
    st.header("Therapy and Reporting Characteristics")

    st.markdown(
        """
        Therapy features summarize reported treatment-duration information.
        Reporting features describe the number and type of report sources
        associated with each analytical record.
        """
    )

    therapy_section, reporting_section = st.columns(2)

    with therapy_section:
        show_table(
            dataframe=therapy_summary_df,
            title="Therapy Summary",
            description=(
                "Duration and count summaries derived from reported therapy "
                "start and end dates."
            ),
        )

    with reporting_section:
        show_table(
            dataframe=reporter_summary_df,
            title="Reporter Summary",
            description=(
                "Summary of healthcare-professional, consumer, foreign, and "
                "multiple-source reporting indicators."
            ),
        )

    st.warning(
        """
        Therapy dates and reporter characteristics may be incomplete or
        inconsistently reported. These features should therefore be interpreted
        as characteristics of submitted FAERS reports rather than complete
        patient histories.
        """
    )


# ==========================================================
# Main EDA Conclusions
# ==========================================================

st.header("Main Exploratory Findings")

st.markdown(
    """
    - The target variable was moderately balanced between serious and
      non-serious reports.
    - Age and especially body weight contained substantial missingness.
    - Report-level complexity varied considerably across medication,
      reaction, indication, therapy, and source characteristics.
    - Several count variables were strongly right-skewed and contained
      extreme values.
    - These patterns supported the use of nonparametric statistical tests
      for many group comparisons and flexible nonlinear models for
      classification.
    """
)


# ==========================================================
# Disclaimer
# ==========================================================

render_disclaimer(DISCLAIMER)
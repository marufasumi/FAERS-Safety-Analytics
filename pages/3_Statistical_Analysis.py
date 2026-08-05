"""
Statistical analysis page for the FDA FAERS application.

This page presents the exported inferential test results comparing serious
and non-serious adverse-event reports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.constants import DATA_DIR, DISCLAIMER
from utils.loaders import load_csv
from utils.styling import (
    apply_global_styles,
    render_disclaimer,
    render_info_box,
)


# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Statistical Analysis | FAERS Analytics",
    page_icon="📐",
    layout="wide",
)

apply_global_styles()


# ==========================================================
# File Paths
# ==========================================================

DEMOGRAPHIC_TESTS_PATH = DATA_DIR / "demographic_tests.csv"
DRUG_TESTS_PATH = DATA_DIR / "drug_tests.csv"
REACTION_TESTS_PATH = DATA_DIR / "reaction_tests.csv"
INDICATION_TESTS_PATH = DATA_DIR / "indication_tests.csv"
THERAPY_TESTS_PATH = DATA_DIR / "therapy_tests.csv"
REPORTER_TESTS_PATH = DATA_DIR / "reporter_tests.csv"

REACTION_CATEGORY_TESTS_PATH = (
    DATA_DIR / "reaction_category_tests.csv"
)

INDICATION_CATEGORY_TESTS_PATH = (
    DATA_DIR / "indication_category_tests.csv"
)


# ==========================================================
# Helper Functions
# ==========================================================

def format_p_value(value: object) -> str:
    """
    Format p-values for readable scientific display.
    """
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if np.isnan(numeric_value):
        return "NA"

    if numeric_value == 0:
        return "< 1e-300"

    if numeric_value < 0.001:
        return f"{numeric_value:.2e}"

    return f"{numeric_value:.4f}"


def add_significance_columns(
    dataframe: pd.DataFrame | None,
    alpha: float,
) -> pd.DataFrame | None:
    """
    Add readable p-value and significance columns.
    """
    if dataframe is None or dataframe.empty:
        return dataframe

    result = dataframe.copy()

    result["p-value"] = pd.to_numeric(
        result["p-value"],
        errors="coerce",
    )

    result["Formatted p-value"] = result["p-value"].apply(
        format_p_value
    )

    result["Significant"] = np.where(
        result["p-value"] < alpha,
        "Yes",
        "No",
    )

    return result


def display_test_table(
    dataframe: pd.DataFrame | None,
    title: str,
    description: str,
    alpha: float,
) -> None:
    """
    Display an inferential test table with readable formatting.
    """
    st.subheader(title)
    st.markdown(description)

    formatted_df = add_significance_columns(
        dataframe=dataframe,
        alpha=alpha,
    )

    if formatted_df is None or formatted_df.empty:
        st.info(
            "The exported statistical test table is unavailable."
        )
        return

    display_columns = [
        column
        for column in [
            "Variable",
            "Test",
            "Statistic",
            "Degrees of Freedom",
            "Formatted p-value",
            "Significant",
        ]
        if column in formatted_df.columns
    ]

    st.dataframe(
        formatted_df[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Statistic": st.column_config.NumberColumn(
                "Statistic",
                format="%.4f",
            ),
            "Degrees of Freedom": (
                st.column_config.NumberColumn(
                    "Degrees of Freedom",
                    format="%d",
                )
            ),
        },
    )


def count_significant_tests(
    dataframes: list[pd.DataFrame | None],
    alpha: float,
) -> tuple[int, int]:
    """
    Count significant and total tests across exported tables.
    """
    significant_count = 0
    total_count = 0

    for dataframe in dataframes:
        if dataframe is None or dataframe.empty:
            continue

        p_values = pd.to_numeric(
            dataframe["p-value"],
            errors="coerce",
        ).dropna()

        total_count += len(p_values)

        significant_count += int(
            (p_values < alpha).sum()
        )

    return significant_count, total_count


def prettify_variable_name(variable_name: str) -> str:
    """
    Convert exported variable names into readable labels.
    """
    replacements = {
        "flag_AKI": "Acute kidney injury",
        "flag_DKA": "Diabetic ketoacidosis",
        "flag_hypoglycemia": "Hypoglycemia",
        "flag_lactic_acidosis": "Lactic acidosis",
        "flag_amputation": "Amputation",
        "flag_genital_infection": "Genital infection",
        "flag_type2_diabetes": "Type 2 diabetes",
        "flag_type1_diabetes": "Type 1 diabetes",
        "flag_any_diabetes": "Any diabetes",
        "flag_hypertension": "Hypertension",
        "flag_chronic_kidney_disease": (
            "Chronic kidney disease"
        ),
        "flag_heart_failure": "Heart failure",
        "flag_obesity": "Obesity",
        "flag_unknown_indication": "Unknown indication",
    }

    if variable_name in replacements:
        return replacements[variable_name]

    return (
        variable_name
        .replace("_", " ")
        .strip()
        .title()
    )


def build_significance_chart(
    dataframe: pd.DataFrame | None,
    alpha: float,
):
    """
    Build an interactive Plotly chart of -log10(p-value).

    Larger values indicate stronger statistical evidence against
    the null hypothesis.
    """
    if dataframe is None or dataframe.empty:
        return None

    required_columns = {"Variable", "p-value"}

    if not required_columns.issubset(dataframe.columns):
        return None

    chart_df = dataframe[
        ["Variable", "p-value"]
    ].copy()

    chart_df["p-value"] = pd.to_numeric(
        chart_df["p-value"],
        errors="coerce",
    )

    chart_df = chart_df.dropna(
        subset=["p-value"]
    )

    if chart_df.empty:
        return None

    # Exact zero cannot be log-transformed.
    # Replace it with a very small positive value.
    chart_df["Adjusted p-value"] = (
        chart_df["p-value"].replace(
            0,
            1e-300,
        )
    )

    chart_df["-log10(p-value)"] = -np.log10(
        chart_df["Adjusted p-value"]
    )

    chart_df["Variable"] = (
        chart_df["Variable"]
        .astype(str)
        .apply(prettify_variable_name)
    )

    chart_df["Formatted p-value"] = (
        chart_df["p-value"].apply(
            format_p_value
        )
    )

    chart_df = chart_df.sort_values(
        "-log10(p-value)",
        ascending=True,
    )

    significance_threshold = -np.log10(alpha)

    figure = px.bar(
        chart_df,
        x="-log10(p-value)",
        y="Variable",
        orientation="h",
        title=(
            "Statistical Evidence by Clinical Category"
        ),
        labels={
            "-log10(p-value)": "−log10(p-value)",
            "Variable": "Clinical category",
        },
        custom_data=[
            "Formatted p-value",
        ],
    )

    figure.add_vline(
        x=significance_threshold,
        line_dash="dash",
        annotation_text=f"α = {alpha}",
        annotation_position="top",
    )

    figure.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "−log10(p-value): %{x:.2f}<br>"
            "p-value: %{customdata[0]}"
            "<extra></extra>"
        )
    )

    figure.update_layout(
        height=max(
            420,
            len(chart_df) * 55,
        ),
        margin={
            "l": 20,
            "r": 20,
            "t": 70,
            "b": 20,
        },
        xaxis_title="−log10(p-value)",
        yaxis_title=None,
        hoverlabel={
            "namelength": -1,
        },
    )

    return figure


# ==========================================================
# Header
# ==========================================================

st.title("Statistical Analysis")

st.markdown(
    """
    Inferential tests were used to compare serious and non-serious FDA FAERS
    reports across demographic, medication, reaction, indication, therapy,
    reporter, and clinical-category characteristics.
    """
)

render_info_box(
    """
    Statistical significance indicates evidence of an association or
    distributional difference in this dataset. It does not establish
    causation, clinical importance, or medication-specific risk.
    """
)


# ==========================================================
# Load Exported Results
# ==========================================================

demographic_tests_df = load_csv(
    DEMOGRAPHIC_TESTS_PATH
)

drug_tests_df = load_csv(
    DRUG_TESTS_PATH
)

reaction_tests_df = load_csv(
    REACTION_TESTS_PATH
)

indication_tests_df = load_csv(
    INDICATION_TESTS_PATH
)

therapy_tests_df = load_csv(
    THERAPY_TESTS_PATH
)

reporter_tests_df = load_csv(
    REPORTER_TESTS_PATH
)

reaction_category_tests_df = load_csv(
    REACTION_CATEGORY_TESTS_PATH
)

indication_category_tests_df = load_csv(
    INDICATION_CATEGORY_TESTS_PATH
)

all_test_dataframes = [
    demographic_tests_df,
    drug_tests_df,
    reaction_tests_df,
    indication_tests_df,
    therapy_tests_df,
    reporter_tests_df,
    reaction_category_tests_df,
    indication_category_tests_df,
]


# ==========================================================
# Significance Level
# ==========================================================

st.header("Analysis Settings")

alpha = st.selectbox(
    "Statistical significance level",
    options=[
        0.05,
        0.01,
        0.001,
    ],
    index=0,
    format_func=lambda value: f"α = {value}",
)

significant_count, total_count = (
    count_significant_tests(
        dataframes=all_test_dataframes,
        alpha=alpha,
    )
)

metric_col1, metric_col2, metric_col3 = (
    st.columns(3)
)

with metric_col1:
    st.metric(
        label="Exported Tests",
        value=str(total_count),
    )

with metric_col2:
    st.metric(
        label="Statistically Significant",
        value=str(significant_count),
    )

with metric_col3:
    percentage = (
        significant_count / total_count
        if total_count
        else 0
    )

    st.metric(
        label="Significant Proportion",
        value=f"{percentage:.1%}",
    )

st.caption(
    """
    The selected alpha level changes only the displayed significance labels.
    It does not rerun the original statistical tests.
    """
)


# ==========================================================
# Test Selection Rationale
# ==========================================================

st.header("Why These Tests Were Used")

test_col1, test_col2, test_col3 = (
    st.columns(3)
)

with test_col1:
    st.subheader("Welch t-test")

    st.markdown(
        """
        Used for continuous demographic variables such as age and weight.

        Welch's version does not require equal variances between the serious
        and non-serious groups.
        """
    )

with test_col2:
    st.subheader("Mann–Whitney U")

    st.markdown(
        """
        Used for skewed count and duration variables.

        It is a nonparametric test that compares the distributions of two
        independent groups.
        """
    )

with test_col3:
    st.subheader("Chi-square")

    st.markdown(
        """
        Used for categorical variables and binary clinical indicators.

        It evaluates whether category frequencies differ by serious-report
        classification.
        """
    )


# ==========================================================
# Result Tabs
# ==========================================================

(
    demographic_tab,
    burden_tab,
    category_tab,
    reporter_tab,
    interpretation_tab,
) = st.tabs(
    [
        "Demographics",
        "Burden and Therapy",
        "Clinical Categories",
        "Reporter Variables",
        "Interpretation",
    ]
)


# ==========================================================
# Demographic Tests
# ==========================================================

with demographic_tab:
    st.header("Demographic Comparisons")

    display_test_table(
        dataframe=demographic_tests_df,
        title="Age and Weight",
        description=(
            "Welch t-tests compared the mean reported age and body weight "
            "between serious and non-serious reports."
        ),
        alpha=alpha,
    )

    st.markdown(
        """
        ### Main result

        Both age and weight showed statistically significant group
        differences:

        - age: p ≈ 4.74 × 10⁻²⁹²;
        - weight: p ≈ 1.34 × 10⁻⁸⁷.

        Because the analytical dataset is very large, even modest differences
        can produce extremely small p-values. Statistical significance should
        therefore be interpreted together with descriptive statistics and
        practical relevance.
        """
    )


# ==========================================================
# Burden and Therapy Tests
# ==========================================================

with burden_tab:
    st.header(
        "Burden and Therapy Comparisons"
    )

    selected_group = st.radio(
        "Select a statistical test group",
        options=[
            "Medication Burden",
            "Reaction Burden",
            "Indication Burden",
            "Therapy Characteristics",
        ],
        horizontal=True,
    )

    if selected_group == "Medication Burden":
        display_test_table(
            dataframe=drug_tests_df,
            title="Medication Burden Tests",
            description=(
                "Mann–Whitney U tests compared drug-count and drug-role "
                "variables between serious and non-serious reports."
            ),
            alpha=alpha,
        )

        st.markdown(
            """
            All medication-burden variables were statistically significant.
            The primary-suspect count had the largest p-value in this group,
            but it still met the conventional 0.05 significance threshold.
            """
        )

    elif selected_group == "Reaction Burden":
        display_test_table(
            dataframe=reaction_tests_df,
            title="Reaction Burden Tests",
            description=(
                "Mann–Whitney U tests compared total and unique reaction "
                "counts between serious and non-serious reports."
            ),
            alpha=alpha,
        )

        st.markdown(
            """
            Both total reactions and unique reactions differed significantly
            between the two report classes.
            """
        )

    elif selected_group == "Indication Burden":
        display_test_table(
            dataframe=indication_tests_df,
            title="Indication Burden Tests",
            description=(
                "Mann–Whitney U tests compared total and unique indication "
                "counts between serious and non-serious reports."
            ),
            alpha=alpha,
        )

        st.markdown(
            """
            Total and unique indication counts showed statistically
            significant distributional differences.
            """
        )

    else:
        display_test_table(
            dataframe=therapy_tests_df,
            title="Therapy Tests",
            description=(
                "Mann–Whitney U tests compared therapy-count and duration "
                "variables between serious and non-serious reports."
            ),
            alpha=alpha,
        )

        st.markdown(
            """
            Therapy-count and duration measures were statistically significant,
            suggesting that treatment-history characteristics differed across
            report classes.
            """
        )


# ==========================================================
# Clinical Category Tests
# ==========================================================

with category_tab:
    st.header(
        "Clinical Category Associations"
    )

    category_choice = st.radio(
        "Select a category group",
        options=[
            "Reaction Categories",
            "Indication Categories",
        ],
        horizontal=True,
    )

    if category_choice == "Reaction Categories":
        selected_df = reaction_category_tests_df
        table_title = (
            "Reaction Category Chi-square Tests"
        )
        table_description = (
            "These tests evaluated associations between selected reaction "
            "indicators and serious-report classification."
        )

    else:
        selected_df = indication_category_tests_df
        table_title = (
            "Indication Category Chi-square Tests"
        )
        table_description = (
            "These tests evaluated associations between selected indication "
            "indicators and serious-report classification."
        )

    display_test_table(
        dataframe=selected_df,
        title=table_title,
        description=table_description,
        alpha=alpha,
    )

    significance_figure = (
        build_significance_chart(
            dataframe=selected_df,
            alpha=alpha,
        )
    )

    if significance_figure is not None:
        st.subheader(
            "Relative Statistical Evidence"
        )

        st.markdown(
            """
            The chart displays `-log10(p-value)`. Larger values indicate
            stronger statistical evidence against the null hypothesis.
            The dashed vertical line represents the selected significance
            threshold.
            """
        )

        st.plotly_chart(
            significance_figure,
            use_container_width=True,
            config={
                "displaylogo": False,
                "responsive": True,
            },
        )

    else:
        st.info(
            "The significance chart could not be generated."
        )

    st.warning(
        """
        The height of a significance bar does not measure clinical effect
        size, severity, causality, or public-health importance.
        """
    )


# ==========================================================
# Reporter Tests
# ==========================================================

with reporter_tab:
    st.header(
        "Reporter and Report-Type Associations"
    )

    display_test_table(
        dataframe=reporter_tests_df,
        title=(
            "Reporter Variable Chi-square Tests"
        ),
        description=(
            "Chi-square tests evaluated whether reporter type and report type "
            "were associated with serious-report classification."
        ),
        alpha=alpha,
    )

    st.markdown(
        """
        Reporter type and report type both showed strong statistical
        associations with the target. These variables may capture differences
        in reporting behavior, case documentation, submission channels, or
        regulatory reporting requirements.
        """
    )


# ==========================================================
# Interpretation
# ==========================================================

with interpretation_tab:
    st.header(
        "How to Interpret the Results"
    )

    st.markdown(
        """
        ### Statistical significance

        A p-value below the selected alpha level indicates that the observed
        result would be unlikely under the corresponding null hypothesis.

        ### Large-sample sensitivity

        With more than 400,000 reports, the tests have substantial statistical
        power. Very small differences may therefore become statistically
        significant.

        ### Distributional differences

        A significant Mann–Whitney U result indicates a difference in
        distributions or rank patterns. It does not automatically mean that
        group means differ by a large amount.

        ### Association is not causation

        A significant chi-square result indicates an association between
        categorical variables. It does not establish that a drug, condition,
        reaction, reporter type, or therapy pattern caused a serious outcome.

        ### Practical interpretation

        These statistical results were used together with descriptive
        analysis and predictive modeling. No single p-value was treated as
        sufficient evidence for a clinical conclusion.
        """
    )


# ==========================================================
# Main Statistical Findings
# ==========================================================

st.header("Main Statistical Findings")

st.markdown(
    """
    - Reported age and weight differed significantly between serious and
      non-serious reports.
    - Drug, reaction, indication, and therapy burden measures showed
      statistically significant distributional differences.
    - Selected reaction categories—including acute kidney injury, diabetic
      ketoacidosis, hypoglycemia, lactic acidosis, amputation, and genital
      infection—were significantly associated with report seriousness.
    - Several indication categories, including diabetes, hypertension,
      chronic kidney disease, and heart failure, were significantly associated
      with report classification.
    - Reporter type and report type showed strong associations with the target.
    """
)


# ==========================================================
# Disclaimer
# ==========================================================

render_disclaimer(DISCLAIMER)
"""
Interactive prediction page for the FDA FAERS application.

This page:
- loads the saved Random Forest pipeline;
- loads the exact 49-feature schema;
- collects one report-level input;
- validates the input schema;
- generates a serious/non-serious classification;
- displays class probabilities using Plotly.

The prediction is an educational model demonstration and is not intended
for clinical decision-making.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.constants import (
    DISCLAIMER,
    FEATURE_COLUMNS_PATH,
    MODEL_PATH,
    TOTAL_PREDICTORS,
)
from utils.loaders import (
    load_feature_columns,
    load_model,
)
from utils.prediction import predict_serious_report
from utils.styling import (
    apply_global_styles,
    render_disclaimer,
    render_info_box,
    render_result_box,
)


# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Predict Serious Event | FAERS Analytics",
    page_icon="🩺",
    layout="wide",
)

apply_global_styles()


# ==========================================================
# Feature Metadata
# ==========================================================

CATEGORICAL_FEATURES = {
    "sex",
    "reporter_country",
    "occur_country",
    "reporter_type",
    "report_type",
}

BINARY_FEATURES = {
    # Medication indicators
    "contains_insulin",
    "contains_sglt2",
    "contains_sulfonylurea",

    # Reaction indicators
    "flag_AKI",
    "flag_DKA",
    "flag_hypoglycemia",
    "flag_lactic_acidosis",
    "flag_amputation",
    "flag_genital_infection",

    # Indication indicators
    "flag_type2_diabetes",
    "flag_type1_diabetes",
    "flag_any_diabetes",
    "flag_hypertension",
    "flag_chronic_kidney_disease",
    "flag_heart_failure",
    "flag_obesity",
    "flag_unknown_indication",

    # Reporter indicators
    "has_health_professional_report",
    "has_consumer_report",
    "has_foreign_report",

    # Therapy indicators
    "short_term_therapy",
    "long_term_therapy",
    "therapy_duration_outlier_present",
}

INTEGER_FEATURES = {
    # Drug burden
    "total_drugs",
    "num_unique_drugs",
    "num_primary_suspect",
    "num_secondary_suspect",
    "num_concomitant",
    "num_interacting",

    # Reaction burden
    "total_reactions",
    "num_unique_reactions",

    # Indication burden
    "total_indications",
    "num_unique_indications",

    # Reporter burden
    "num_reporter_sources",

    # Therapy burden
    "total_therapies",
    "num_therapies_with_duration",
}

FLOAT_FEATURES = {
    "age_years",
    "weight_kg",
    "mean_therapy_duration_days",
    "median_therapy_duration_days",
    "min_therapy_duration_days",
    "max_therapy_duration_days",
}


# ==========================================================
# Readable Feature Labels
# ==========================================================

FEATURE_LABELS = {
    "age_years": "Age (years)",
    "weight_kg": "Weight (kg)",
    "sex": "Sex",
    "reporter_country": "Reporter country",
    "occur_country": "Occurrence country",
    "reporter_type": "Reporter type",
    "report_type": "Report type",

    "total_drugs": "Total drugs",
    "num_unique_drugs": "Unique drugs",
    "num_primary_suspect": "Primary suspect drugs",
    "num_secondary_suspect": "Secondary suspect drugs",
    "num_concomitant": "Concomitant drugs",
    "num_interacting": "Interacting drugs",
    "contains_insulin": "Contains insulin",
    "contains_sglt2": "Contains an SGLT2 inhibitor",
    "contains_sulfonylurea": "Contains a sulfonylurea",

    "total_reactions": "Total reactions",
    "num_unique_reactions": "Unique reactions",
    "flag_AKI": "Acute kidney injury",
    "flag_DKA": "Diabetic ketoacidosis",
    "flag_hypoglycemia": "Hypoglycemia",
    "flag_lactic_acidosis": "Lactic acidosis",
    "flag_amputation": "Amputation",
    "flag_genital_infection": "Genital infection",

    "total_indications": "Total indications",
    "num_unique_indications": "Unique indications",
    "flag_type2_diabetes": "Type 2 diabetes",
    "flag_type1_diabetes": "Type 1 diabetes",
    "flag_any_diabetes": "Any diabetes",
    "flag_hypertension": "Hypertension",
    "flag_chronic_kidney_disease": "Chronic kidney disease",
    "flag_heart_failure": "Heart failure",
    "flag_obesity": "Obesity",
    "flag_unknown_indication": "Unknown indication",

    "num_reporter_sources": "Number of reporter sources",
    "has_health_professional_report": (
        "Healthcare-professional report"
    ),
    "has_consumer_report": "Consumer report",
    "has_foreign_report": "Foreign report",

    "total_therapies": "Total therapy records",
    "num_therapies_with_duration": (
        "Therapies with recorded duration"
    ),
    "mean_therapy_duration_days": (
        "Mean therapy duration (days)"
    ),
    "median_therapy_duration_days": (
        "Median therapy duration (days)"
    ),
    "min_therapy_duration_days": (
        "Minimum therapy duration (days)"
    ),
    "max_therapy_duration_days": (
        "Maximum therapy duration (days)"
    ),
    "short_term_therapy": "Short-term therapy",
    "long_term_therapy": "Long-term therapy",
    "therapy_duration_outlier_present": (
        "Therapy-duration outlier present"
    ),
}


# ==========================================================
# Feature Groups
# ==========================================================

FEATURE_GROUPS = {
    "Demographics and Report Information": {
        "age_years",
        "weight_kg",
        "sex",
        "reporter_country",
        "occur_country",
        "reporter_type",
        "report_type",
    },
    "Medication Burden": {
        "total_drugs",
        "num_unique_drugs",
        "num_primary_suspect",
        "num_secondary_suspect",
        "num_concomitant",
        "num_interacting",
        "contains_insulin",
        "contains_sglt2",
        "contains_sulfonylurea",
    },
    "Reaction Burden": {
        "total_reactions",
        "num_unique_reactions",
        "flag_AKI",
        "flag_DKA",
        "flag_hypoglycemia",
        "flag_lactic_acidosis",
        "flag_amputation",
        "flag_genital_infection",
    },
    "Indication Burden": {
        "total_indications",
        "num_unique_indications",
        "flag_type2_diabetes",
        "flag_type1_diabetes",
        "flag_any_diabetes",
        "flag_hypertension",
        "flag_chronic_kidney_disease",
        "flag_heart_failure",
        "flag_obesity",
        "flag_unknown_indication",
    },
    "Therapy Characteristics": {
        "total_therapies",
        "num_therapies_with_duration",
        "mean_therapy_duration_days",
        "median_therapy_duration_days",
        "min_therapy_duration_days",
        "max_therapy_duration_days",
        "short_term_therapy",
        "long_term_therapy",
        "therapy_duration_outlier_present",
    },
    "Reporting Characteristics": {
        "num_reporter_sources",
        "has_health_professional_report",
        "has_consumer_report",
        "has_foreign_report",
    },
}


# ==========================================================
# Categorical Input Options
# ==========================================================

SEX_OPTIONS = {
    "Female": "F",
    "Male": "M",
    "Unknown / not reported": "UNK",
}

REPORTER_TYPE_OPTIONS = {
    "Physician": "MD",
    "Pharmacist": "PH",
    "Other health professional": "OT",
    "Lawyer": "LW",
    "Consumer": "CN",
    "Unknown / not reported": "UNK",
}

REPORT_TYPE_OPTIONS = {
    "Expedited report": "EXP",
    "Periodic report": "PER",
    "Direct report": "DIR",
    "Unknown / not reported": "UNK",
}

COUNTRY_OPTIONS = {
    "United States": "US",
    "Canada": "CA",
    "United Kingdom": "GB",
    "Germany": "DE",
    "France": "FR",
    "Japan": "JP",
    "India": "IN",
    "China": "CN",
    "Other / unknown": "UNK",
}


# ==========================================================
# Helper Functions
# ==========================================================

def get_feature_label(feature_name: str) -> str:
    """
    Return a readable label for a model feature.
    """
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]

    return (
        feature_name
        .replace("_", " ")
        .strip()
        .title()
    )


def binary_input(
    feature_name: str,
    key_prefix: str,
) -> int:
    """
    Render a binary Yes/No selector and return 1 or 0.
    """
    answer = st.selectbox(
        label=get_feature_label(feature_name),
        options=["No", "Yes"],
        index=0,
        key=f"{key_prefix}_{feature_name}",
    )

    return 1 if answer == "Yes" else 0


def integer_input(
    feature_name: str,
    key_prefix: str,
) -> int:
    """
    Render a non-negative integer input.
    """
    default_values = {
        "total_drugs": 1,
        "num_unique_drugs": 1,
        "num_primary_suspect": 1,
        "total_reactions": 1,
        "num_unique_reactions": 1,
        "total_indications": 1,
        "num_unique_indications": 1,
        "num_reporter_sources": 1,
        "total_therapies": 1,
    }

    default_value = default_values.get(
        feature_name,
        0,
    )

    return int(
        st.number_input(
            label=get_feature_label(feature_name),
            min_value=0,
            value=default_value,
            step=1,
            key=f"{key_prefix}_{feature_name}",
        )
    )


def float_input(
    feature_name: str,
    key_prefix: str,
) -> float:
    """
    Render a non-negative numeric input.
    """
    default_values = {
        "age_years": 55.0,
        "weight_kg": 73.0,
        "mean_therapy_duration_days": 0.0,
        "median_therapy_duration_days": 0.0,
        "min_therapy_duration_days": 0.0,
        "max_therapy_duration_days": 0.0,
    }

    default_value = default_values.get(
        feature_name,
        0.0,
    )

    return float(
        st.number_input(
            label=get_feature_label(feature_name),
            min_value=0.0,
            value=default_value,
            step=1.0,
            key=f"{key_prefix}_{feature_name}",
        )
    )


def categorical_input(
    feature_name: str,
    key_prefix: str,
) -> str:
    """
    Render a categorical selector for known categorical features.
    """
    if feature_name == "sex":
        option_map = SEX_OPTIONS

    elif feature_name == "reporter_type":
        option_map = REPORTER_TYPE_OPTIONS

    elif feature_name == "report_type":
        option_map = REPORT_TYPE_OPTIONS

    elif feature_name in {
        "reporter_country",
        "occur_country",
    }:
        option_map = COUNTRY_OPTIONS

    else:
        return st.text_input(
            label=get_feature_label(feature_name),
            value="UNK",
            key=f"{key_prefix}_{feature_name}",
        )

    display_value = st.selectbox(
        label=get_feature_label(feature_name),
        options=list(option_map.keys()),
        key=f"{key_prefix}_{feature_name}",
    )

    return option_map[display_value]


def generic_input(
    feature_name: str,
    key_prefix: str,
) -> Any:
    """
    Render an input for an unexpected feature in the saved schema.

    This prevents the page from silently omitting a model feature.
    """
    return st.number_input(
        label=get_feature_label(feature_name),
        value=0.0,
        step=1.0,
        key=f"{key_prefix}_{feature_name}",
        help=(
            "This feature was found in the saved model schema but "
            "does not have a custom input definition."
        ),
    )


def render_feature_input(
    feature_name: str,
    key_prefix: str,
) -> Any:
    """
    Render the correct Streamlit input widget for one feature.
    """
    if feature_name in CATEGORICAL_FEATURES:
        return categorical_input(
            feature_name,
            key_prefix,
        )

    if feature_name in BINARY_FEATURES:
        return binary_input(
            feature_name,
            key_prefix,
        )

    if feature_name in INTEGER_FEATURES:
        return integer_input(
            feature_name,
            key_prefix,
        )

    if feature_name in FLOAT_FEATURES:
        return float_input(
            feature_name,
            key_prefix,
        )

    return generic_input(
        feature_name,
        key_prefix,
    )


def get_group_features(
    feature_columns: list[str],
    group_name: str,
) -> list[str]:
    """
    Return group features in the model's exact saved order.
    """
    group_features = FEATURE_GROUPS[
        group_name
    ]

    return [
        feature
        for feature in feature_columns
        if feature in group_features
    ]


def build_probability_chart(
    serious_probability: float,
    non_serious_probability: float,
):
    """
    Build an interactive Plotly probability chart.
    """
    probability_df = pd.DataFrame(
        {
            "Classification": [
                "Serious",
                "Non-serious",
            ],
            "Probability": [
                serious_probability * 100,
                non_serious_probability * 100,
            ],
        }
    )

    figure = px.bar(
        probability_df,
        x="Probability",
        y="Classification",
        orientation="h",
        title="Predicted Class Probabilities",
        labels={
            "Probability": "Probability (%)",
            "Classification": (
                "Report classification"
            ),
        },
        text="Probability",
    )

    figure.update_traces(
        texttemplate="%{x:.2f}%",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Probability: %{x:.2f}%"
            "<extra></extra>"
        ),
    )

    figure.update_layout(
        height=370,
        margin={
            "l": 20,
            "r": 70,
            "t": 70,
            "b": 20,
        },
        xaxis_title="Probability (%)",
        yaxis_title=None,
    )

    figure.update_xaxes(
        range=[0, 105]
    )

    return figure


def check_logical_consistency(
    input_values: dict[str, Any],
) -> list[str]:
    """
    Identify simple logical inconsistencies in submitted counts.

    These checks do not change the prediction. They only help the user
    review the entered values.
    """
    warnings: list[str] = []

    count_relationships = [
        (
            "num_unique_drugs",
            "total_drugs",
            "Unique drugs cannot exceed total drugs.",
        ),
        (
            "num_unique_reactions",
            "total_reactions",
            "Unique reactions cannot exceed total reactions.",
        ),
        (
            "num_unique_indications",
            "total_indications",
            "Unique indications cannot exceed total indications.",
        ),
        (
            "num_therapies_with_duration",
            "total_therapies",
            (
                "Therapies with recorded duration cannot exceed "
                "total therapy records."
            ),
        ),
    ]

    for smaller_feature, larger_feature, message in (
        count_relationships
    ):
        if (
            smaller_feature in input_values
            and larger_feature in input_values
            and input_values[smaller_feature]
            > input_values[larger_feature]
        ):
            warnings.append(message)

    role_features = [
        "num_primary_suspect",
        "num_secondary_suspect",
        "num_concomitant",
        "num_interacting",
    ]

    if (
        "total_drugs" in input_values
        and all(
            feature in input_values
            for feature in role_features
        )
    ):
        role_total = sum(
            input_values[feature]
            for feature in role_features
        )

        if role_total > input_values["total_drugs"]:
            warnings.append(
                """
                The combined drug-role counts exceed the entered total
                number of drugs.
                """
            )

    duration_features = [
        "min_therapy_duration_days",
        "median_therapy_duration_days",
        "mean_therapy_duration_days",
        "max_therapy_duration_days",
    ]

    if all(
        feature in input_values
        for feature in duration_features
    ):
        minimum = input_values[
            "min_therapy_duration_days"
        ]
        median = input_values[
            "median_therapy_duration_days"
        ]
        maximum = input_values[
            "max_therapy_duration_days"
        ]

        if minimum > maximum:
            warnings.append(
                """
                Minimum therapy duration cannot exceed maximum therapy
                duration.
                """
            )

        if median < minimum or median > maximum:
            warnings.append(
                """
                Median therapy duration should fall between the entered
                minimum and maximum durations.
                """
            )

    return warnings


# ==========================================================
# Header
# ==========================================================

st.title("Predict Serious Event Classification")

st.markdown(
    """
    Enter one set of report-level characteristics to obtain an educational
    classification from the saved Random Forest pipeline.
    """
)

render_info_box(
    """
    The model predicts whether a submitted report resembles reports classified
    as serious or non-serious in the 2024 Q1 FDA FAERS analytical dataset.
    It does not predict whether a patient will experience an adverse event.
    """
)


# ==========================================================
# Load Model and Feature Schema
# ==========================================================

try:
    model = load_model(
        MODEL_PATH
    )

    feature_columns = load_feature_columns(
        FEATURE_COLUMNS_PATH
    )

except Exception as error:
    st.error(
        "The saved prediction resources could not be loaded."
    )

    st.exception(error)
    st.stop()


# ==========================================================
# Model Status
# ==========================================================

st.header("Model Status")

status_col1, status_col2, status_col3 = (
    st.columns(3)
)

with status_col1:
    st.metric(
        label="Model",
        value="Random Forest",
    )

with status_col2:
    st.metric(
        label="Expected Predictors",
        value=str(
            len(feature_columns)
        ),
    )

with status_col3:
    schema_status = (
        "Verified"
        if len(feature_columns)
        == TOTAL_PREDICTORS
        else "Review Required"
    )

    st.metric(
        label="Feature Schema",
        value=schema_status,
    )

if len(feature_columns) != TOTAL_PREDICTORS:
    st.warning(
        f"""
        The saved schema contains {len(feature_columns)} features, while the
        application constant expects {TOTAL_PREDICTORS}. The prediction form
        will continue to trust the saved JSON schema.
        """
    )


# ==========================================================
# Prediction Form
# ==========================================================

st.header("Report Characteristics")

st.markdown(
    """
    Complete the sections below. Default values are included only to make the
    interface easier to test and do not represent a real patient or report.
    """
)

with st.form(
    "faers_prediction_form",
    clear_on_submit=False,
):
    input_values: dict[str, Any] = {}

    for group_name in FEATURE_GROUPS:
        group_features = get_group_features(
            feature_columns=feature_columns,
            group_name=group_name,
        )

        if not group_features:
            continue

        with st.expander(
            group_name,
            expanded=(
                group_name
                == "Demographics and Report Information"
            ),
        ):
            input_columns = st.columns(2)

            for index, feature_name in enumerate(
                group_features
            ):
                selected_column = input_columns[
                    index % 2
                ]

                with selected_column:
                    input_values[feature_name] = (
                        render_feature_input(
                            feature_name=feature_name,
                            key_prefix="prediction",
                        )
                    )

    known_group_features = set().union(
        *FEATURE_GROUPS.values()
    )

    additional_features = [
        feature
        for feature in feature_columns
        if feature not in known_group_features
    ]

    if additional_features:
        with st.expander(
            "Additional Model Features",
            expanded=False,
        ):
            st.caption(
                """
                These fields were found in the saved feature schema but were
                not included in the predefined application groups.
                """
            )

            extra_columns = st.columns(2)

            for index, feature_name in enumerate(
                additional_features
            ):
                selected_column = extra_columns[
                    index % 2
                ]

                with selected_column:
                    input_values[feature_name] = (
                        render_feature_input(
                            feature_name=feature_name,
                            key_prefix="additional",
                        )
                    )

    submitted = st.form_submit_button(
        "Generate Prediction",
        use_container_width=True,
        type="primary",
    )


# ==========================================================
# Prediction Results
# ==========================================================

if submitted:
    consistency_warnings = (
        check_logical_consistency(
            input_values
        )
    )

    if consistency_warnings:
        st.warning(
            """
            Please review the following input consistency issues before
            interpreting the prediction:
            """
        )

        for warning_message in consistency_warnings:
            st.markdown(
                f"- {warning_message.strip()}"
            )

    input_df = pd.DataFrame(
        [input_values]
    )

    try:
        result = predict_serious_report(
            model=model,
            input_df=input_df,
            feature_columns=feature_columns,
        )

    except Exception as error:
        st.error(
            "The prediction could not be generated."
        )

        st.exception(error)

    else:
        st.header("Prediction Result")

        predicted_class = result[
            "predicted_class"
        ]

        if predicted_class == 1:
            render_result_box(
                """
                <strong>Predicted classification: Serious report</strong><br>
                The submitted characteristics more closely resemble reports
                classified as serious in the training dataset.
                """
            )

        else:
            render_result_box(
                """
                <strong>Predicted classification: Non-serious report</strong>
                <br>
                The submitted characteristics more closely resemble reports
                classified as non-serious in the training dataset.
                """
            )

        result_col1, result_col2, result_col3 = (
            st.columns(3)
        )

        with result_col1:
            st.metric(
                label="Predicted Class",
                value=(
                    "Serious"
                    if predicted_class == 1
                    else "Non-serious"
                ),
            )

        with result_col2:
            st.metric(
                label="Serious Probability",
                value=(
                    f"{result['serious_probability']:.2%}"
                ),
            )

        with result_col3:
            st.metric(
                label="Model Confidence",
                value=(
                    f"{result['confidence']:.2%}"
                ),
            )

        probability_figure = (
            build_probability_chart(
                serious_probability=result[
                    "serious_probability"
                ],
                non_serious_probability=result[
                    "non_serious_probability"
                ],
            )
        )

        st.plotly_chart(
            probability_figure,
            use_container_width=True,
            config={
                "displaylogo": False,
                "responsive": True,
            },
        )

        st.markdown(
            """
            A probability near 50% indicates greater model uncertainty.
            A higher probability indicates stronger model preference for one
            class, but it does not guarantee that the classification is
            clinically correct.
            """
        )

        with st.expander(
            "View submitted model inputs"
        ):
            submitted_input_df = input_df.T.reset_index()

            submitted_input_df.columns = [
                "Feature",
                "Submitted Value",
            ]

            submitted_input_df[
                "Display Name"
            ] = submitted_input_df[
                "Feature"
            ].apply(
                get_feature_label
            )

            submitted_input_df = (
                submitted_input_df[
                    [
                        "Display Name",
                        "Feature",
                        "Submitted Value",
                    ]
                ]
            )

            st.dataframe(
                submitted_input_df,
                use_container_width=True,
                hide_index=True,
            )

        with st.expander(
            "How this prediction was generated"
        ):
            st.markdown(
                """
                1. The submitted values were converted into a one-row pandas
                   DataFrame.
                2. The input was validated against the exact 49-feature schema
                   saved with the project.
                3. Columns were reordered to match the training schema.
                4. The saved scikit-learn pipeline applied the original
                   preprocessing.
                5. The fitted Random Forest generated the class prediction and
                   class probabilities.
                """
            )


# ==========================================================
# Interpretation Guidance
# ==========================================================

st.header("Interpretation Guidance")

guidance_col1, guidance_col2 = st.columns(2)

with guidance_col1:
    st.subheader("What the result means")

    st.markdown(
        """
        The output indicates which report class the entered characteristics
        most closely resemble according to the fitted Random Forest model.
        """
    )

with guidance_col2:
    st.subheader("What the result does not mean")

    st.markdown(
        """
        The output does not determine causality, diagnose a condition, estimate
        adverse-event incidence, or recommend treatment.
        """
    )


# ==========================================================
# Disclaimer
# ==========================================================

render_disclaimer(DISCLAIMER)
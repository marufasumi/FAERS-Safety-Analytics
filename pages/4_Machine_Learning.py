"""
Machine learning results page for the FDA FAERS application.

This page presents:
- performance comparisons across nine supervised models;
- interactive Plotly charts;
- cross-validation results;
- weighted model rankings;
- saved diagnostic figures;
- Random Forest feature importance;
- final model selection.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.constants import (
    ASSETS_DIR,
    DATA_DIR,
    DISCLAIMER,
    RANDOM_FOREST_ACCURACY,
    RANDOM_FOREST_F1,
    RANDOM_FOREST_ROC_AUC,
    TOTAL_MODELS,
)
from utils.loaders import load_csv, load_image
from utils.styling import (
    apply_global_styles,
    render_disclaimer,
    render_image_caption,
    render_info_box,
    render_result_box,
)


# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Machine Learning | FAERS Analytics",
    page_icon="🤖",
    layout="wide",
)

apply_global_styles()


# ==========================================================
# Data Paths
# ==========================================================

MODEL_PERFORMANCE_PATH = (
    DATA_DIR / "model_performance_summary.csv"
)

WEIGHTED_RANKINGS_PATH = (
    DATA_DIR / "weighted_model_rankings.csv"
)


# ==========================================================
# Saved Figure Paths
# ==========================================================

COMPARISON_IMAGE_PATHS = {
    "Accuracy": ASSETS_DIR / "accuracy_comparison.png",
    "Precision": ASSETS_DIR / "precision_comparison.png",
    "Recall": ASSETS_DIR / "recall_comparison.png",
    "Specificity": ASSETS_DIR / "specificity_comparison.png",
    "F1 Score": ASSETS_DIR / "f1_comparison.png",
    "ROC AUC": ASSETS_DIR / "roc_auc_comparison.png",
    "Cross-Validation Accuracy": (
        ASSETS_DIR / "cv_accuracy_comparison.png"
    ),
}

MODEL_FILE_NAMES = {
    "Logistic Regression": "logistic_regression",
    "Linear Discriminant Analysis": (
        "linear_discriminant_analysis"
    ),
    "Quadratic Discriminant Analysis": (
        "quadratic_discriminant_analysis"
    ),
    "Gaussian Naive Bayes": "gaussian_naive_bayes",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "Support Vector Machine": (
        "support_vector_machine"
    ),
    "Neural Network": "neural_network",
}

DIAGNOSTIC_SUFFIXES = {
    "Confusion Matrix": "confusion_matrix.png",
    "ROC Curve": "roc_curve.png",
    "Precision–Recall Curve": (
        "precision_recall_curve.png"
    ),
}

RANDOM_FOREST_FEATURE_IMPORTANCE_PATH = (
    ASSETS_DIR / "random_forest_feature_importance.png"
)

RANDOM_FOREST_CONFUSION_MATRIX_PATH = (
    ASSETS_DIR / "random_forest_confusion_matrix.png"
)

RANDOM_FOREST_ROC_CURVE_PATH = (
    ASSETS_DIR / "random_forest_roc_curve.png"
)

RANDOM_FOREST_PRECISION_RECALL_PATH = (
    ASSETS_DIR
    / "random_forest_precision_recall_curve.png"
)


# ==========================================================
# Helper Functions
# ==========================================================

def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """
    Find a column using case-insensitive aliases.
    """
    normalized_columns = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        normalized_candidate = (
            candidate.strip().lower()
        )

        if normalized_candidate in normalized_columns:
            return normalized_columns[
                normalized_candidate
            ]

    return None


def standardize_performance_table(
    dataframe: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """
    Standardize exported model-performance column names.

    This allows the page to tolerate minor naming differences
    in the exported CSV file.
    """
    if dataframe is None or dataframe.empty:
        return dataframe

    model_column = find_column(
        dataframe,
        [
            "Model",
            "model",
            "Model Name",
            "model_name",
        ],
    )

    metric_aliases = {
        "Accuracy": [
            "Accuracy",
            "accuracy",
            "Test Accuracy",
            "test_accuracy",
        ],
        "Precision": [
            "Precision",
            "precision",
        ],
        "Recall": [
            "Recall",
            "recall",
            "Sensitivity",
            "sensitivity",
        ],
        "Specificity": [
            "Specificity",
            "specificity",
        ],
        "F1 Score": [
            "F1",
            "F1 Score",
            "F1_Score",
            "f1",
            "f1_score",
        ],
        "ROC AUC": [
            "ROC AUC",
            "ROC_AUC",
            "ROC-AUC",
            "roc_auc",
        ],
        "CV Accuracy": [
            "CV Accuracy",
            "CV_Accuracy",
            "Cross-Validation Accuracy",
            "cross_validation_accuracy",
        ],
    }

    if model_column is None:
        return None

    standardized_df = pd.DataFrame(
        {
            "Model": dataframe[
                model_column
            ].astype(str)
        }
    )

    for standard_name, candidates in (
        metric_aliases.items()
    ):
        source_column = find_column(
            dataframe,
            candidates,
        )

        if source_column is not None:
            standardized_df[standard_name] = (
                pd.to_numeric(
                    dataframe[source_column],
                    errors="coerce",
                )
            )

    return standardized_df


def standardize_ranking_table(
    dataframe: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """
    Prepare the weighted model-ranking table.
    """
    if dataframe is None or dataframe.empty:
        return dataframe

    result = dataframe.copy()

    numeric_candidates = [
        "Overall Rank",
        "Accuracy",
        "ROC AUC",
        "F1",
        "F1 Score",
        "CV Accuracy",
        "Weighted Score",
    ]

    for column in numeric_candidates:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    rank_column = find_column(
        result,
        [
            "Overall Rank",
            "Rank",
            "overall_rank",
        ],
    )

    if rank_column is not None:
        result = result.sort_values(
            rank_column,
            ascending=True,
        )

    return result


def build_metric_chart(
    performance_df: pd.DataFrame,
    metric: str,
):
    """
    Build an interactive horizontal Plotly bar chart.
    """
    if metric not in performance_df.columns:
        return None

    chart_df = performance_df[
        ["Model", metric]
    ].dropna()

    if chart_df.empty:
        return None

    chart_df = chart_df.sort_values(
        metric,
        ascending=True,
    )

    figure = px.bar(
        chart_df,
        x=metric,
        y="Model",
        orientation="h",
        title=f"{metric} by Model",
        labels={
            metric: metric,
            "Model": "Model",
        },
        text=metric,
        custom_data=[
            "Model",
            metric,
        ],
    )

    figure.update_traces(
        texttemplate="%{x:.3f}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            f"{metric}: "
            "%{customdata[1]:.4f}"
            "<extra></extra>"
        ),
    )

    figure.update_layout(
        height=max(
            470,
            len(chart_df) * 52,
        ),
        margin={
            "l": 20,
            "r": 60,
            "t": 70,
            "b": 20,
        },
        xaxis_title=metric,
        yaxis_title=None,
    )

    minimum_value = chart_df[metric].min()

    if minimum_value >= 0.5:
        figure.update_xaxes(
            range=[
                max(
                    0,
                    minimum_value - 0.08,
                ),
                1.02,
            ]
        )
    else:
        figure.update_xaxes(
            range=[0, 1.02]
        )

    return figure


def build_multi_metric_chart(
    performance_df: pd.DataFrame,
    selected_metrics: list[str],
):
    """
    Build a grouped Plotly comparison chart.
    """
    available_metrics = [
        metric
        for metric in selected_metrics
        if metric in performance_df.columns
    ]

    if not available_metrics:
        return None

    chart_df = performance_df[
        ["Model"] + available_metrics
    ].copy()

    long_df = chart_df.melt(
        id_vars="Model",
        value_vars=available_metrics,
        var_name="Metric",
        value_name="Score",
    )

    long_df = long_df.dropna(
        subset=["Score"]
    )

    if long_df.empty:
        return None

    figure = px.bar(
        long_df,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Selected Performance Metrics by Model",
        labels={
            "Model": "Model",
            "Score": "Score",
            "Metric": "Metric",
        },
    )

    figure.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "%{fullData.name}: %{y:.4f}"
            "<extra></extra>"
        )
    )

    figure.update_layout(
        height=560,
        margin={
            "l": 20,
            "r": 20,
            "t": 70,
            "b": 120,
        },
        xaxis_tickangle=-35,
        yaxis_range=[0, 1.02],
        legend_title_text="Metric",
    )

    return figure


def build_accuracy_auc_scatter(
    performance_df: pd.DataFrame,
):
    """
    Build a Plotly scatter chart comparing accuracy and ROC AUC.
    """
    required_columns = {
        "Model",
        "Accuracy",
        "ROC AUC",
    }

    if not required_columns.issubset(
        performance_df.columns
    ):
        return None

    chart_df = performance_df[
        [
            "Model",
            "Accuracy",
            "ROC AUC",
        ]
    ].dropna()

    if chart_df.empty:
        return None

    figure = px.scatter(
        chart_df,
        x="Accuracy",
        y="ROC AUC",
        text="Model",
        title="Accuracy versus ROC AUC",
        labels={
            "Accuracy": "Test accuracy",
            "ROC AUC": "ROC AUC",
        },
        custom_data=[
            "Model",
        ],
    )

    figure.update_traces(
        textposition="top center",
        marker={
            "size": 13,
        },
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Accuracy: %{x:.4f}<br>"
            "ROC AUC: %{y:.4f}"
            "<extra></extra>"
        ),
    )

    figure.update_layout(
        height=560,
        margin={
            "l": 20,
            "r": 20,
            "t": 70,
            "b": 20,
        },
    )

    return figure


# def show_saved_image(
#     image_path: Path,
#     caption: str,
# ) -> None:
#     """
#     Load and display one saved analytical figure.
#     """
#     image = load_image(image_path)

#     if image is None:
#         st.info(
#             f"Figure unavailable: {image_path.name}"
#         )
#         return

#     st.image(
#         image,
#         use_container_width=True,
#     )

def show_saved_image(
    image_path: Path,
    caption: str,
    width: int = 700,
) -> None:
    """
    Load and display one saved analytical figure.

    Parameters
    ----------
    image_path : Path
        Location of the saved image.

    caption : str
        Figure caption displayed below the image.

    width : int, default=700
        Display width of the image in pixels.
    """
    image = load_image(image_path)

    if image is None:
        st.info(
            f"Figure unavailable: {image_path.name}"
        )
        return

    # Center the image on the page
    left, center, right = st.columns([1, 2.5, 1])

    with center:
        st.image(
            image,
            width=width,
        )

    render_image_caption(caption)



def get_diagnostic_image_path(
    model_name: str,
    diagnostic_type: str,
) -> Path | None:
    """
    Construct a saved diagnostic-image path.
    """
    model_file_name = MODEL_FILE_NAMES.get(
        model_name
    )

    diagnostic_suffix = DIAGNOSTIC_SUFFIXES.get(
        diagnostic_type
    )

    if (
        model_file_name is None
        or diagnostic_suffix is None
    ):
        return None

    return ASSETS_DIR / (
        f"{model_file_name}_{diagnostic_suffix}"
    )


# ==========================================================
# Header
# ==========================================================

st.title("Machine Learning Results")

st.markdown(
    """
    Nine supervised learning methods were trained and evaluated to classify
    FDA FAERS reports as serious or non-serious.
    """
)

render_info_box(
    """
    All model results shown on this page were exported from the completed
    modeling notebooks. This Streamlit application does not retrain the
    models when the page loads.
    """
)


# ==========================================================
# Load Exported Results
# ==========================================================

raw_performance_df = load_csv(
    MODEL_PERFORMANCE_PATH
)

raw_ranking_df = load_csv(
    WEIGHTED_RANKINGS_PATH
)

performance_df = standardize_performance_table(
    raw_performance_df
)

ranking_df = standardize_ranking_table(
    raw_ranking_df
)


# ==========================================================
# Final Model Summary
# ==========================================================

st.header("Final Model")

render_result_box(
    """
    <strong>Random Forest was selected as the final model.</strong>
    It achieved the strongest overall combination of test accuracy,
    ROC AUC, F1 score, and cross-validation performance.
    """
)

metric_col1, metric_col2, metric_col3, metric_col4 = (
    st.columns(4)
)

with metric_col1:
    st.metric(
        label="Models Compared",
        value=str(TOTAL_MODELS),
    )

with metric_col2:
    st.metric(
        label="Random Forest Accuracy",
        value=f"{RANDOM_FOREST_ACCURACY:.4f}",
    )

with metric_col3:
    st.metric(
        label="Random Forest ROC AUC",
        value=f"{RANDOM_FOREST_ROC_AUC:.4f}",
    )

with metric_col4:
    st.metric(
        label="Random Forest F1",
        value=f"{RANDOM_FOREST_F1:.4f}",
    )


# ==========================================================
# Analysis Tabs
# ==========================================================

(
    comparison_tab,
    combined_tab,
    rankings_tab,
    diagnostics_tab,
    final_model_tab,
) = st.tabs(
    [
        "Metric Comparison",
        "Combined View",
        "Model Rankings",
        "Model Diagnostics",
        "Random Forest",
    ]
)


# ==========================================================
# Metric Comparison
# ==========================================================

with comparison_tab:
    st.header("Interactive Metric Comparison")

    if performance_df is None or performance_df.empty:
        st.info(
            "The model-performance table is unavailable."
        )

    else:
        available_metrics = [
            column
            for column in [
                "Accuracy",
                "Precision",
                "Recall",
                "Specificity",
                "F1 Score",
                "ROC AUC",
                "CV Accuracy",
            ]
            if column in performance_df.columns
        ]

        selected_metric = st.selectbox(
            "Select a performance metric",
            options=available_metrics,
            index=0,
        )

        metric_figure = build_metric_chart(
            performance_df=performance_df,
            metric=selected_metric,
        )

        if metric_figure is not None:
            st.plotly_chart(
                metric_figure,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "responsive": True,
                },
            )

        st.markdown(
            """
            Higher values indicate stronger performance. No single metric was
            used alone for final model selection.
            """
        )

        with st.expander(
            "View exported performance table"
        ):
            st.dataframe(
                performance_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    metric: (
                        st.column_config.NumberColumn(
                            metric,
                            format="%.4f",
                        )
                    )
                    for metric in available_metrics
                },
            )

        saved_image_metric = st.selectbox(
            "Optional: view the original saved comparison figure",
            options=list(
                COMPARISON_IMAGE_PATHS.keys()
            ),
        )

        saved_metric_path = (
            COMPARISON_IMAGE_PATHS[
                saved_image_metric
            ]
        )

        with st.expander(
            "Show original notebook figure"
        ):
            show_saved_image(
                saved_metric_path,
                (
                    f"Original exported "
                    f"{saved_image_metric.lower()} "
                    "comparison figure."
                ),
            )


# ==========================================================
# Combined Performance View
# ==========================================================

with combined_tab:
    st.header("Combined Model Performance")

    if performance_df is None or performance_df.empty:
        st.info(
            "The model-performance table is unavailable."
        )

    else:
        available_metrics = [
            column
            for column in [
                "Accuracy",
                "Precision",
                "Recall",
                "Specificity",
                "F1 Score",
                "ROC AUC",
                "CV Accuracy",
            ]
            if column in performance_df.columns
        ]

        default_metrics = [
            metric
            for metric in [
                "Accuracy",
                "F1 Score",
                "ROC AUC",
            ]
            if metric in available_metrics
        ]

        selected_metrics = st.multiselect(
            "Choose metrics to compare",
            options=available_metrics,
            default=default_metrics,
        )

        if selected_metrics:
            combined_figure = (
                build_multi_metric_chart(
                    performance_df,
                    selected_metrics,
                )
            )

            if combined_figure is not None:
                st.plotly_chart(
                    combined_figure,
                    use_container_width=True,
                    config={
                        "displaylogo": False,
                        "responsive": True,
                    },
                )
        else:
            st.info(
                "Select at least one metric."
            )

        scatter_figure = (
            build_accuracy_auc_scatter(
                performance_df
            )
        )

        if scatter_figure is not None:
            st.subheader(
                "Accuracy and Discrimination"
            )

            st.markdown(
                """
                This chart compares overall classification accuracy with
                ROC AUC, which measures discrimination across thresholds.
                """
            )

            st.plotly_chart(
                scatter_figure,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "responsive": True,
                },
            )


# ==========================================================
# Weighted Rankings
# ==========================================================

with rankings_tab:
    st.header("Weighted Model Ranking")

    st.markdown(
        """
        The weighted ranking combined multiple evaluation measures so that
        final model selection was not determined by one metric alone.
        """
    )

    if ranking_df is None or ranking_df.empty:
        st.info(
            "The weighted-ranking table is unavailable."
        )

    else:
        st.dataframe(
            ranking_df,
            use_container_width=True,
            hide_index=True,
        )

        rank_column = find_column(
            ranking_df,
            [
                "Overall Rank",
                "Rank",
                "overall_rank",
            ],
        )

        model_column = find_column(
            ranking_df,
            [
                "Model",
                "model",
            ],
        )

        if (
            rank_column is not None
            and model_column is not None
        ):
            ranking_chart_df = ranking_df[
                [
                    model_column,
                    rank_column,
                ]
            ].copy()

            ranking_chart_df[
                rank_column
            ] = pd.to_numeric(
                ranking_chart_df[
                    rank_column
                ],
                errors="coerce",
            )

            ranking_chart_df = (
                ranking_chart_df
                .dropna(subset=[rank_column])
                .sort_values(
                    rank_column,
                    ascending=False,
                )
            )

            ranking_figure = px.bar(
                ranking_chart_df,
                x=rank_column,
                y=model_column,
                orientation="h",
                title="Overall Model Ranking",
                labels={
                    rank_column: "Overall rank",
                    model_column: "Model",
                },
                text=rank_column,
            )

            ranking_figure.update_traces(
                texttemplate="%{x:.1f}",
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Overall rank: %{x:.1f}"
                    "<extra></extra>"
                ),
            )

            ranking_figure.update_layout(
                height=max(
                    470,
                    len(ranking_chart_df) * 52,
                ),
                margin={
                    "l": 20,
                    "r": 50,
                    "t": 70,
                    "b": 20,
                },
                yaxis_title=None,
            )

            ranking_figure.update_xaxes(
                autorange="reversed"
            )

            st.plotly_chart(
                ranking_figure,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "responsive": True,
                },
            )

        st.caption(
            """
            A lower overall rank indicates stronger combined model performance.
            """
        )


# ==========================================================
# Model Diagnostics
# ==========================================================

with diagnostics_tab:
    st.header("Saved Model Diagnostics")

    st.markdown(
        """
        Select a model and diagnostic figure. These images were exported from
        the original model-evaluation notebook.
        """
    )

    diagnostic_col1, diagnostic_col2 = (
        st.columns(2)
    )

    with diagnostic_col1:
        selected_model = st.selectbox(
            "Select a model",
            options=list(
                MODEL_FILE_NAMES.keys()
            ),
        )

    with diagnostic_col2:
        selected_diagnostic = st.selectbox(
            "Select a diagnostic",
            options=list(
                DIAGNOSTIC_SUFFIXES.keys()
            ),
        )

    diagnostic_path = (
        get_diagnostic_image_path(
            model_name=selected_model,
            diagnostic_type=selected_diagnostic,
        )
    )

    if diagnostic_path is not None:
        show_saved_image(
            diagnostic_path,
            (
                f"{selected_diagnostic} for "
                f"{selected_model}."
            ),
        )

    st.info(
        """
        K-Nearest Neighbors is included in the tabular comparison but does not
        have a corresponding exported diagnostic image in the current assets
        folder.
        """
    )


# ==========================================================
# Random Forest Detail
# ==========================================================

with final_model_tab:
    st.header("Random Forest Detail")

    st.markdown(
        """
        The final Random Forest used:

        - `n_estimators = 200`
        - `max_depth = 20`
        - `min_samples_leaf = 1`

        The deployed artifact is a fitted scikit-learn pipeline containing
        both preprocessing and classification steps.
        """
    )

    rf_figure_choice = st.radio(
        "Select a Random Forest result",
        options=[
            "Feature Importance",
            "Confusion Matrix",
            "ROC Curve",
            "Precision–Recall Curve",
        ],
        horizontal=True,
    )

    if rf_figure_choice == "Feature Importance":
        selected_rf_path = (
            RANDOM_FOREST_FEATURE_IMPORTANCE_PATH
        )
        selected_rf_caption = (
            "Random Forest feature-importance "
            "results exported from the modeling notebook."
        )

    elif rf_figure_choice == "Confusion Matrix":
        selected_rf_path = (
            RANDOM_FOREST_CONFUSION_MATRIX_PATH
        )
        selected_rf_caption = (
            "Random Forest confusion matrix on "
            "the held-out test dataset."
        )

    elif rf_figure_choice == "ROC Curve":
        selected_rf_path = (
            RANDOM_FOREST_ROC_CURVE_PATH
        )
        selected_rf_caption = (
            "Random Forest receiver operating "
            "characteristic curve."
        )

    else:
        selected_rf_path = (
            RANDOM_FOREST_PRECISION_RECALL_PATH
        )
        selected_rf_caption = (
            "Random Forest precision–recall curve."
        )

    show_saved_image(
        selected_rf_path,
        selected_rf_caption,
    )

    st.warning(
        """
        Feature importance indicates how strongly a variable contributed to
        the fitted model's splits. It does not measure causality or determine
        whether a feature is clinically responsible for serious outcomes.
        """
    )


# ==========================================================
# Main Modeling Findings
# ==========================================================

st.header("Main Modeling Findings")

st.markdown(
    """
    - Random Forest achieved the strongest overall performance.
    - Decision Tree produced nearly identical test accuracy but offered less
      ensemble stability than Random Forest.
    - Neural Network, Support Vector Machine, and Logistic Regression also
      achieved strong classification results.
    - ROC AUC values were high across several models, indicating that the
      engineered features contained substantial discriminatory information.
    - Quadratic Discriminant Analysis and Gaussian Naive Bayes produced weaker
      overall classification performance.
    - Final model selection considered test performance, cross-validation,
      model stability, and deployment suitability.
    """
)


# ==========================================================
# Disclaimer
# ==========================================================

render_disclaimer(DISCLAIMER)
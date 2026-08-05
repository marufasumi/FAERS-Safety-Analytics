"""
Clustering results page for the FDA FAERS application.

This page presents the exported K-Means clustering results, including:
- cluster sizes;
- serious-report proportions;
- cluster-level summary statistics;
- elbow and silhouette diagnostics;
- PCA visualization;
- centroid heatmap;
- interpretation of identified report profiles.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.constants import (
    ASSETS_DIR,
    DATA_DIR,
    DISCLAIMER,
    KMEANS_K,
    TOTAL_REPORTS,
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
    page_title="Clustering | FAERS Analytics",
    page_icon="🧩",
    layout="wide",
)

apply_global_styles()


# ==========================================================
# Data and Figure Paths
# ==========================================================

CLUSTER_SUMMARY_PATH = (
    DATA_DIR / "cluster_summary.csv"
)

CLUSTER_SIZES_IMAGE_PATH = (
    ASSETS_DIR / "cluster_sizes.png"
)

CLUSTER_SERIOUS_DISTRIBUTION_IMAGE_PATH = (
    ASSETS_DIR / "cluster_serious_distribution.png"
)

ELBOW_METHOD_IMAGE_PATH = (
    ASSETS_DIR / "elbow_method.png"
)

SILHOUETTE_SCORES_IMAGE_PATH = (
    ASSETS_DIR / "silhouette_scores.png"
)

PCA_CLUSTERS_IMAGE_PATH = (
    ASSETS_DIR / "pca_clusters.png"
)

CLUSTER_CENTROID_HEATMAP_PATH = (
    ASSETS_DIR / "cluster_centroid_heatmap.png"
)


# ==========================================================
# Helper Functions
# ==========================================================

def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """
    Find a dataframe column using case-insensitive aliases.
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


def prepare_cluster_summary(
    dataframe: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """
    Standardize key cluster-summary columns.

    The function preserves all original exported columns while adding
    standardized fields when suitable source columns are available.
    """
    if dataframe is None or dataframe.empty:
        return dataframe

    result = dataframe.copy()

    cluster_column = find_column(
        result,
        [
            "Cluster",
            "cluster",
            "Cluster_ID",
            "cluster_id",
        ],
    )

    size_column = find_column(
        result,
        [
            "Count",
            "count",
            "Cluster Size",
            "cluster_size",
            "Size",
            "size",
            "Number of Reports",
            "number_of_reports",
        ],
    )

    serious_count_column = find_column(
        result,
        [
            "Serious",
            "serious",
            "Serious Count",
            "serious_count",
            "Serious Reports",
            "serious_reports",
        ],
    )

    non_serious_count_column = find_column(
        result,
        [
            "Non-Serious",
            "non_serious",
            "Non-Serious Count",
            "non_serious_count",
            "Non-Serious Reports",
            "non_serious_reports",
        ],
    )

    serious_percentage_column = find_column(
        result,
        [
            "Serious Percentage",
            "serious_percentage",
            "Serious Proportion",
            "serious_proportion",
            "Serious Rate",
            "serious_rate",
            "Serious %",
            "Percent Serious",
            "percent_serious",
        ],
    )

    if cluster_column is not None:
        result["Cluster Label"] = (
            "Cluster "
            + result[cluster_column].astype(str)
        )

    if size_column is not None:
        result["Cluster Size"] = pd.to_numeric(
            result[size_column],
            errors="coerce",
        )

    if serious_count_column is not None:
        result["Serious Count"] = pd.to_numeric(
            result[serious_count_column],
            errors="coerce",
        )

    if non_serious_count_column is not None:
        result["Non-Serious Count"] = pd.to_numeric(
            result[non_serious_count_column],
            errors="coerce",
        )

    if serious_percentage_column is not None:
        result["Serious Percentage"] = (
            pd.to_numeric(
                result[
                    serious_percentage_column
                ],
                errors="coerce",
            )
        )

        # Convert proportions such as 0.98 into percentages.
        non_missing = result[
            "Serious Percentage"
        ].dropna()

        if (
            not non_missing.empty
            and non_missing.max() <= 1
        ):
            result["Serious Percentage"] = (
                result["Serious Percentage"] * 100
            )

    elif (
        "Serious Count" in result.columns
        and "Cluster Size" in result.columns
    ):
        result["Serious Percentage"] = (
            result["Serious Count"]
            / result["Cluster Size"]
            * 100
        )

    return result


def build_cluster_size_chart(
    cluster_df: pd.DataFrame | None,
):
    """
    Build an interactive Plotly cluster-size chart.
    """
    required_columns = {
        "Cluster Label",
        "Cluster Size",
    }

    if (
        cluster_df is None
        or cluster_df.empty
        or not required_columns.issubset(
            cluster_df.columns
        )
    ):
        return None

    chart_df = cluster_df[
        [
            "Cluster Label",
            "Cluster Size",
        ]
    ].dropna()

    if chart_df.empty:
        return None

    chart_df = chart_df.sort_values(
        "Cluster Size",
        ascending=True,
    )

    figure = px.bar(
        chart_df,
        x="Cluster Size",
        y="Cluster Label",
        orientation="h",
        title="Number of Reports by Cluster",
        labels={
            "Cluster Size": "Number of reports",
            "Cluster Label": "Cluster",
        },
        text="Cluster Size",
    )

    figure.update_traces(
        texttemplate="%{x:,.0f}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Reports: %{x:,.0f}"
            "<extra></extra>"
        ),
    )

    figure.update_layout(
        height=max(
            460,
            len(chart_df) * 60,
        ),
        margin={
            "l": 20,
            "r": 70,
            "t": 70,
            "b": 20,
        },
        yaxis_title=None,
    )

    return figure


def build_serious_percentage_chart(
    cluster_df: pd.DataFrame | None,
):
    """
    Build an interactive Plotly chart of serious-report percentage.
    """
    required_columns = {
        "Cluster Label",
        "Serious Percentage",
    }

    if (
        cluster_df is None
        or cluster_df.empty
        or not required_columns.issubset(
            cluster_df.columns
        )
    ):
        return None

    chart_df = cluster_df[
        [
            "Cluster Label",
            "Serious Percentage",
        ]
    ].dropna()

    if chart_df.empty:
        return None

    chart_df = chart_df.sort_values(
        "Serious Percentage",
        ascending=True,
    )

    figure = px.bar(
        chart_df,
        x="Serious Percentage",
        y="Cluster Label",
        orientation="h",
        title="Serious-Report Percentage by Cluster",
        labels={
            "Serious Percentage": (
                "Serious reports (%)"
            ),
            "Cluster Label": "Cluster",
        },
        text="Serious Percentage",
    )

    figure.update_traces(
        texttemplate="%{x:.2f}%",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Serious reports: %{x:.2f}%"
            "<extra></extra>"
        ),
    )

    figure.update_layout(
        height=max(
            460,
            len(chart_df) * 60,
        ),
        margin={
            "l": 20,
            "r": 70,
            "t": 70,
            "b": 20,
        },
        xaxis_title="Serious reports (%)",
        yaxis_title=None,
    )

    figure.update_xaxes(
        range=[0, 105]
    )

    return figure


def build_cluster_composition_chart(
    cluster_df: pd.DataFrame | None,
):
    """
    Build a stacked Plotly chart of serious and non-serious counts.
    """
    required_columns = {
        "Cluster Label",
        "Serious Count",
        "Non-Serious Count",
    }

    if (
        cluster_df is None
        or cluster_df.empty
        or not required_columns.issubset(
            cluster_df.columns
        )
    ):
        return None

    chart_df = cluster_df[
        [
            "Cluster Label",
            "Serious Count",
            "Non-Serious Count",
        ]
    ].copy()

    chart_df = chart_df.dropna(
        subset=[
            "Serious Count",
            "Non-Serious Count",
        ]
    )

    if chart_df.empty:
        return None

    long_df = chart_df.melt(
        id_vars="Cluster Label",
        value_vars=[
            "Serious Count",
            "Non-Serious Count",
        ],
        var_name="Report Classification",
        value_name="Count",
    )

    figure = px.bar(
        long_df,
        x="Cluster Label",
        y="Count",
        color="Report Classification",
        barmode="stack",
        title=(
            "Serious and Non-Serious Reports "
            "within Each Cluster"
        ),
        labels={
            "Cluster Label": "Cluster",
            "Count": "Number of reports",
            "Report Classification": (
                "Classification"
            ),
        },
    )

    figure.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "%{fullData.name}: %{y:,.0f}"
            "<extra></extra>"
        )
    )

    figure.update_layout(
        height=540,
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
#     Display one exported clustering figure.
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

#     render_image_caption(caption)

def show_saved_image(
    image_path: Path,
    caption: str,
    width: int = 550,
) -> None:
    """
    Display one exported clustering figure.

    Parameters
    ----------
    image_path : Path
        Path to the saved image.

    caption : str
        Caption displayed below the image.

    width : int, default=550
        Display width of the image in pixels.
    """
    image = load_image(image_path)

    if image is None:
        st.info(
            f"Figure unavailable: {image_path.name}"
        )
        return

    # Center the image on the page
    left, center, right = st.columns([1.5, 3, 1.5])

    with center:
        st.image(
            image,
            width=width,
        )

    render_image_caption(caption)
# ==========================================================
# Header
# ==========================================================

st.title("K-Means Clustering")

st.markdown(
    """
    Unsupervised learning was used to identify distinct adverse-event report
    profiles based on medication burden, reaction burden, indication burden,
    therapy characteristics, reporting characteristics, and demographics.
    """
)

render_info_box(
    """
    K-Means clustering was exploratory. The cluster labels describe groups of
    similar FAERS reports and should not be interpreted as clinical diagnoses
    or validated patient phenotypes.
    """
)


# ==========================================================
# Load Results
# ==========================================================

raw_cluster_df = load_csv(
    CLUSTER_SUMMARY_PATH
)

cluster_df = prepare_cluster_summary(
    raw_cluster_df
)


# ==========================================================
# Clustering Summary
# ==========================================================

st.header("Clustering Summary")

metric_col1, metric_col2, metric_col3 = (
    st.columns(3)
)

with metric_col1:
    st.metric(
        label="Selected Clusters",
        value=str(KMEANS_K),
    )

with metric_col2:
    st.metric(
        label="Reports Clustered",
        value=f"{TOTAL_REPORTS:,}",
    )

with metric_col3:
    st.metric(
        label="Algorithm",
        value="K-Means",
    )

render_result_box(
    """
    Seven clusters were selected after reviewing elbow and silhouette
    diagnostics. The resulting groups differed substantially in size,
    report complexity, and serious-report proportion.
    """
)


# ==========================================================
# Main Tabs
# ==========================================================

(
    overview_tab,
    distribution_tab,
    diagnostics_tab,
    profiles_tab,
    interpretation_tab,
) = st.tabs(
    [
        "Cluster Overview",
        "Serious Distribution",
        "Model Diagnostics",
        "Cluster Profiles",
        "Interpretation",
    ]
)


# ==========================================================
# Cluster Overview
# ==========================================================

with overview_tab:
    st.header("Cluster Sizes")

    cluster_size_figure = (
        build_cluster_size_chart(
            cluster_df
        )
    )

    if cluster_size_figure is not None:
        st.plotly_chart(
            cluster_size_figure,
            use_container_width=True,
            config={
                "displaylogo": False,
                "responsive": True,
            },
        )

    else:
        st.info(
            """
            The interactive cluster-size chart could not be generated from
            the exported table. The original notebook figure is shown below.
            """
        )

        show_saved_image(
            CLUSTER_SIZES_IMAGE_PATH,
            (
                "Original exported cluster-size "
                "figure."
            ),
        )

    st.markdown(
        """
        Cluster sizes were highly unequal. Most reports belonged to a large
        general cluster, while several smaller clusters represented unusual
        combinations of report burden and clinical complexity.
        """
    )

    if cluster_df is not None and not cluster_df.empty:
        st.subheader("Exported Cluster Summary")

        st.dataframe(
            cluster_df,
            use_container_width=True,
            hide_index=True,
        )


# ==========================================================
# Serious Distribution
# ==========================================================

with distribution_tab:
    st.header(
        "Serious-Report Distribution by Cluster"
    )

    serious_percentage_figure = (
        build_serious_percentage_chart(
            cluster_df
        )
    )

    if serious_percentage_figure is not None:
        st.plotly_chart(
            serious_percentage_figure,
            use_container_width=True,
            config={
                "displaylogo": False,
                "responsive": True,
            },
        )

    else:
        show_saved_image(
            CLUSTER_SERIOUS_DISTRIBUTION_IMAGE_PATH,
            (
                "Original exported serious-report "
                "distribution by cluster."
            ),
        )

    composition_figure = (
        build_cluster_composition_chart(
            cluster_df
        )
    )

    if composition_figure is not None:
        st.subheader(
            "Cluster Composition"
        )

        st.plotly_chart(
            composition_figure,
            use_container_width=True,
            config={
                "displaylogo": False,
                "responsive": True,
            },
        )

    st.markdown(
        """
        The clustering analysis found substantial variation in serious-report
        prevalence across clusters. Some small, high-complexity clusters
        contained almost exclusively serious reports.
        """
    )

    st.warning(
        """
        Serious-report concentration within a cluster is descriptive. It does
        not prove that the cluster characteristics caused serious outcomes.
        """
    )


# ==========================================================
# Model Diagnostics
# ==========================================================

with diagnostics_tab:
    st.header(
        "Choosing the Number of Clusters"
    )

    diagnostic_choice = st.radio(
        "Select a clustering diagnostic",
        options=[
            "Elbow Method",
            "Silhouette Scores",
        ],
        horizontal=True,
    )

    if diagnostic_choice == "Elbow Method":
        show_saved_image(
            ELBOW_METHOD_IMAGE_PATH,
            (
                "Elbow-method diagnostic used to "
                "assess candidate values of K."
            ),
        )

        st.markdown(
            """
            The elbow method examines how within-cluster variation decreases
            as the number of clusters increases. A point of diminishing
            improvement helps identify a reasonable candidate value for K.
            """
        )

    else:
        show_saved_image(
            SILHOUETTE_SCORES_IMAGE_PATH,
            (
                "Silhouette scores across candidate "
                "numbers of clusters."
            ),
        )

        st.markdown(
            """
            The silhouette score measures how well each observation fits
            within its assigned cluster relative to neighboring clusters.
            Higher values generally indicate clearer separation.
            """
        )

    st.info(
        """
        The final choice of K = 7 considered both diagnostic evidence and the
        interpretability of the resulting report profiles.
        """
    )


# ==========================================================
# Cluster Profiles
# ==========================================================

with profiles_tab:
    st.header("Cluster Profiles")

    profile_choice = st.radio(
        "Select a cluster-profile visualization",
        options=[
            "Centroid Heatmap",
            "PCA Visualization",
        ],
        horizontal=True,
    )

    if profile_choice == "Centroid Heatmap":
        show_saved_image(
            CLUSTER_CENTROID_HEATMAP_PATH,
            (
                "Standardized cluster centroids "
                "across selected features."
            ),
        )

        st.markdown(
            """
            The centroid heatmap shows the relative feature pattern associated
            with each cluster. Larger positive values indicate that a cluster
            was above the overall average for a feature after standardization.
            """
        )

    else:
        show_saved_image(
            PCA_CLUSTERS_IMAGE_PATH,
            (
                "Two-dimensional PCA projection "
                "of the K-Means clusters."
            ),
        )

        st.markdown(
            """
            Principal component analysis was used only for two-dimensional
            visualization. The K-Means model itself was fitted using the
            selected standardized feature space rather than only the plotted
            principal components.
            """
        )


# ==========================================================
# Interpretation
# ==========================================================

with interpretation_tab:
    st.header(
        "Interpreting the Seven Clusters"
    )

    st.markdown(
        """
        ### High-complexity clusters

        Several small clusters had extremely high medication burden and very
        high serious-report prevalence. For example:

        - one extreme-drug-count cluster contained approximately 1,309 reports
          and was about 98.78% serious;
        - another high-drug-count cluster contained approximately 739 reports
          and was about 99.86% serious.

        ### Moderate-high complexity cluster

        A larger cluster with moderate-to-high report complexity contained
        approximately 54,746 reports and had a serious-report proportion of
        about 76.90%.

        ### Largest general cluster

        The largest cluster contained approximately 324,549 reports and had
        a serious-report proportion of about 50.77%.

        ### Practical meaning

        These patterns suggest that medication burden and overall report
        complexity are useful for distinguishing report profiles. However,
        the clusters remain exploratory and require external validation before
        being interpreted as clinically meaningful subgroups.
        """
    )


# ==========================================================
# Main Findings
# ==========================================================

st.header("Main Clustering Findings")

st.markdown(
    """
    - K-Means identified seven distinct FAERS report profiles.
    - Cluster sizes were strongly imbalanced.
    - Small clusters with extreme medication burden had very high
      serious-report proportions.
    - The largest cluster represented a more common, lower-complexity report
      profile with a serious-report proportion close to the overall dataset.
    - Clustering complemented supervised learning by revealing heterogeneity
      that a single binary prediction does not describe.
    """
)


# ==========================================================
# Disclaimer
# ==========================================================

render_disclaimer(DISCLAIMER)
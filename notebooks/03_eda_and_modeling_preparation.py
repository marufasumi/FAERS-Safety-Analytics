# Extracted from: 03_EDA_and_Modeling_Preparation.ipynb.ipynb
# Complete executable code cells in notebook order.


# ============================================================================
# Code Cell 1
# ============================================================================
# ==========================================================
# Section 1. Load the Modeling Dataset
# ==========================================================

import pandas as pd
from pathlib import Path

# ----------------------------------------------------------
# Define project paths
# ----------------------------------------------------------

PROJECT_ROOT = Path.cwd().parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"

MODELING_DATASET = DATA_PATH / "faers_modeling_dataset.parquet"

# ----------------------------------------------------------
# Load dataset
# ----------------------------------------------------------

df = pd.read_parquet(MODELING_DATASET)

# ----------------------------------------------------------
# Basic validation
# ----------------------------------------------------------

print("=" * 70)
print("Modeling Dataset Successfully Loaded")
print("=" * 70)

print(f"Dataset Path : {MODELING_DATASET}")
print(f"Rows         : {df.shape[0]:,}")
print(f"Columns      : {df.shape[1]}")
print()

print("First Five Rows")
display(df.head())

# ============================================================================
# Code Cell 2
# ============================================================================
# ==========================================================
# Validation
# ==========================================================

EXPECTED_ROWS = 406_184
EXPECTED_COLUMNS = 51

print("=" * 70)
print("Dataset Validation")
print("=" * 70)

print(f"Expected Rows    : {EXPECTED_ROWS:,}")
print(f"Actual Rows      : {df.shape[0]:,}")
print()

print(f"Expected Columns : {EXPECTED_COLUMNS}")
print(f"Actual Columns   : {df.shape[1]}")
print()

print(f"Unique PRIMARYIDs: {df['primaryid'].nunique():,}")

assert df.shape[0] == EXPECTED_ROWS, "Unexpected number of rows."
assert df.shape[1] == EXPECTED_COLUMNS, "Unexpected number of columns."
assert df["primaryid"].is_unique, "PRIMARYID is not unique."

print("\n✅ Dataset validation passed successfully.")

# ============================================================================
# Code Cell 3
# ============================================================================
# ==========================================================
# Section 2. Inspect Dataset Structure
# ==========================================================

print("=" * 70)
print("Dataset Overview")
print("=" * 70)

print(f"Rows      : {df.shape[0]:,}")
print(f"Columns   : {df.shape[1]}")
print()

print("=" * 70)
print("Column Names")
print("=" * 70)

for i, col in enumerate(df.columns, start=1):
    print(f"{i:2d}. {col}")

print("\n")

print("=" * 70)
print("Dataset Information")
print("=" * 70)

df.info()

print("\n")

print("=" * 70)
print("Memory Usage")
print("=" * 70)

memory_mb = df.memory_usage(deep=True).sum() / 1024**2

print(f"Total Memory Usage : {memory_mb:.2f} MB")

print("\n")

print("=" * 70)
print("First Five Records")
print("=" * 70)

display(df.head())

# ============================================================================
# Code Cell 4
# ============================================================================
# ==========================================================
# Validation
# ==========================================================

print("=" * 70)
print("Data Type Summary")
print("=" * 70)

dtype_summary = (
    df.dtypes
      .astype(str)
      .value_counts()
      .rename_axis("Data Type")
      .reset_index(name="Count")
)

display(dtype_summary)

print("\n")

print("=" * 70)
print("Duplicate Column Names")
print("=" * 70)

print(df.columns.duplicated().sum())

assert df.columns.duplicated().sum() == 0

print("\n✅ Dataset structure validation completed successfully.")

# ============================================================================
# Code Cell 5
# ============================================================================
# ==========================================================
# Section 3. Missing Value Analysis
# ==========================================================

import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Calculate missing values
# ----------------------------------------------------------

missing_summary = (
    pd.DataFrame({
        "Missing Count": df.isna().sum(),
        "Missing Percent": df.isna().mean() * 100
    })
    .sort_values("Missing Count", ascending=False)
)

print("=" * 70)
print("Missing Value Summary")
print("=" * 70)

display(missing_summary)

# ----------------------------------------------------------
# Plot variables with missing values only
# ----------------------------------------------------------

missing_plot = missing_summary[missing_summary["Missing Count"] > 0]

plt.figure(figsize=(10,5))

plt.bar(
    missing_plot.index,
    missing_plot["Missing Percent"]
)

plt.ylabel("Missing (%)")
plt.xlabel("Variable")
plt.title("Percentage of Missing Values by Variable")

plt.xticks(rotation=45, ha="right")

for i, value in enumerate(missing_plot["Missing Percent"]):
    label = f"{value:.1f}%" if value >= 1 else f"{value:.3f}%"

    plt.text(
    i,
    value + 1,
    label,
    ha="center",
    fontsize=9
       )

plt.tight_layout()
plt.show()

# ============================================================================
# Code Cell 6
# ============================================================================
# ==========================================================
# Section 4. Summary Statistics
# ==========================================================

continuous_features = [
    "age_years",
    "weight_kg",
    "total_drugs",
    "num_unique_drugs",
    "num_primary_suspect",
    "num_secondary_suspect",
    "num_concomitant",
    "num_interacting",
    "total_reactions",
    "num_unique_reactions",
    "total_indications",
    "num_unique_indications",
    "num_reporter_sources",
    "total_therapies",
    "num_therapies_with_duration",
    "mean_therapy_duration_days",
    "median_therapy_duration_days",
    "min_therapy_duration_days",
    "max_therapy_duration_days",
]

summary_stats = (
    df[continuous_features]
    .describe()
    .T
)

summary_stats["Missing"] = df[continuous_features].isna().sum()
summary_stats["Missing (%)"] = (
    df[continuous_features].isna().mean() * 100
)

summary_stats = summary_stats[
    [
        "count",
        "Missing",
        "Missing (%)",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    ]
]

summary_stats = summary_stats.round(2)

print("=" * 70)
print("Summary Statistics")
print("=" * 70)

display(summary_stats)

# ============================================================================
# Code Cell 7
# ============================================================================
# ==========================================================
# Validation
# ==========================================================

print("=" * 70)
print("Validation")
print("=" * 70)

print(f"Continuous variables analyzed : {len(continuous_features)}")

assert len(summary_stats) == len(continuous_features)

print("Summary statistics generated successfully.")

# ============================================================================
# Code Cell 8
# ============================================================================
# ==========================================================
# Initialize Plotting Style
# ==========================================================

import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3
})

print("✅ Global plotting style initialized.")

# ============================================================================
# Code Cell 9
# ============================================================================
# ==========================================================
# Section 5. Target Variable Distribution
# ==========================================================

target_summary = (
    df["is_serious"]
    .value_counts()
    .sort_index()                      # <-- keeps 0 then 1
    .rename_axis("Class")
    .reset_index(name="Count")
)

target_summary["Percent"] = (
    target_summary["Count"] / len(df) * 100
).round(2)

target_summary["Class"] = target_summary["Class"].map({
    0: "Non-Serious",
    1: "Serious"
})

display(target_summary)

# ----------------------------------------------------------
# Bar Chart
# ----------------------------------------------------------

plt.figure(figsize=(7, 5))

bars = plt.bar(
    target_summary["Class"],
    target_summary["Count"],
    edgecolor="black"
)

plt.title("Distribution of Serious Adverse Event Reports")
plt.ylabel("Number of Reports")
plt.xlabel("Outcome Class")

for bar, pct in zip(bars, target_summary["Percent"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 3000,
        f"{pct:.2f}%",
        ha="center",
        fontsize=11
    )

plt.tight_layout()
plt.show()

# ============================================================================
# Code Cell 10
# ============================================================================
# ==========================================================
# Validation
# ==========================================================

print("=" * 70)
print("Validation")
print("=" * 70)

print(f"Total Reports : {target_summary['Count'].sum():,}")

assert target_summary["Count"].sum() == len(df)
assert set(df["is_serious"].unique()) == {0, 1}

print("\nTarget Classes")
display(target_summary)

print("\n✅ Target variable validation passed.")

# ============================================================================
# Code Cell 11
# ============================================================================
# ==========================================================
# Section 6. Distribution of Demographic Variables
# ==========================================================

demographic_features = [
    "age_years",
    "weight_kg"
]

for feature in demographic_features:

    data = df[feature].dropna()

    mean_value = data.mean()
    median_value = data.median()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14,5)
    )

    # ------------------------------------------------------
    # Histogram
    # ------------------------------------------------------

    axes[0].hist(
        data,
        bins=40,
        edgecolor="black"
    )

    axes[0].axvline(
        mean_value,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_value:.1f}"
    )

    axes[0].axvline(
        median_value,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median = {median_value:.1f}"
    )

    axes[0].set_title(f"{feature} Distribution")
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # ------------------------------------------------------
    # Boxplot
    # ------------------------------------------------------

    axes[1].boxplot(
        data,
        vert=True,
        patch_artist=True
    )

    axes[1].set_title(f"{feature} Boxplot")
    axes[1].set_ylabel(feature)

    plt.tight_layout()
    plt.show()

# ============================================================================
# Code Cell 12
# ============================================================================
# ==========================================================
# Validation
# ==========================================================

print("=" * 70)
print("Demographic Variable Summary")
print("=" * 70)

validation_table = []

for feature in demographic_features:

    data = df[feature]

    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = ((data < lower) | (data > upper)).sum()

    validation_table.append({
        "Variable": feature,
        "Non-Missing": data.notna().sum(),
        "Missing": data.isna().sum(),
        "Missing (%)": round(data.isna().mean() * 100, 2),
        "Mean": round(data.mean(), 2),
        "Median": round(data.median(), 2),
        "Outliers": outliers,
        "Outlier (%)": round(outliers / data.notna().sum() * 100, 2)
    })

validation_table = pd.DataFrame(validation_table)

display(validation_table)

print("\n✅ Demographic variable validation completed.")

# ============================================================================
# Code Cell 13
# ============================================================================
# ==========================================================
# Section 7. Distribution of Drug-Related Features
# ==========================================================

drug_features = [
    "total_drugs",
    "num_unique_drugs",
    "num_primary_suspect",
    "num_secondary_suspect",
    "num_concomitant",
    "num_interacting"
]

drug_feature_labels = {
    "total_drugs": "Total Drugs",
    "num_unique_drugs": "Unique Drugs",
    "num_primary_suspect": "Primary Suspect Drugs",
    "num_secondary_suspect": "Secondary Suspect Drugs",
    "num_concomitant": "Concomitant Drugs",
    "num_interacting": "Interacting Drugs"
}

for feature in drug_features:

    data = df[feature]

    mean_value = data.mean()
    median_value = data.median()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14,5)
    )

    # ------------------------------------------------------
    # Histogram
    # ------------------------------------------------------

    axes[0].hist(
        data,
        bins=40,
        edgecolor="black"
    )

    axes[0].axvline(
        mean_value,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_value:.2f}"
    )

    axes[0].axvline(
        median_value,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median = {median_value:.2f}"
    )

    axes[0].set_title(f"{drug_feature_labels[feature]} Distribution")
    axes[0].set_xlabel(drug_feature_labels[feature])
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # ------------------------------------------------------
    # Boxplot
    # ------------------------------------------------------

    axes[1].boxplot(
        data,
        vert=True,
        patch_artist=True
    )

    axes[1].set_title(f"{drug_feature_labels[feature]} Boxplot")
    axes[1].set_ylabel(drug_feature_labels[feature])

    plt.tight_layout()
    plt.show()

# ============================================================================
# Code Cell 14
# ============================================================================
# ==========================================================
# Create Figures Directory
# ==========================================================

from pathlib import Path

PROJECT_ROOT = Path.cwd().parent

FIGURES_DIR = PROJECT_ROOT / "figures"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 70)
print("Figures Directory")
print("=" * 70)
print(FIGURES_DIR)

# ============================================================================
# Code Cell 15
# ============================================================================
# ==========================================================
# Section 7. Distribution of Drug-Related Features
# ==========================================================

drug_features = [
    "total_drugs",
    "num_unique_drugs",
    "num_primary_suspect",
    "num_secondary_suspect",
    "num_concomitant",
    "num_interacting"
]

drug_feature_labels = {
    "total_drugs": "Total Drugs",
    "num_unique_drugs": "Unique Drugs",
    "num_primary_suspect": "Primary Suspect Drugs",
    "num_secondary_suspect": "Secondary Suspect Drugs",
    "num_concomitant": "Concomitant Drugs",
    "num_interacting": "Interacting Drugs"
}

for feature in drug_features:

    data = df[feature]

    mean_value = data.mean()
    median_value = data.median()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14,5)
    )

    # ------------------------------------------------------
    # Histogram
    # ------------------------------------------------------

    axes[0].hist(
        data,
        bins=40,
        edgecolor="black"
    )

    axes[0].axvline(
        mean_value,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_value:.2f}"
    )

    axes[0].axvline(
        median_value,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median = {median_value:.2f}"
    )

    axes[0].set_title(f"{drug_feature_labels[feature]} Distribution")
    axes[0].set_xlabel(drug_feature_labels[feature])
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # ------------------------------------------------------
    # Boxplot
    # ------------------------------------------------------

    axes[1].boxplot(
        data,
        vert=True,
        patch_artist=True
    )

    axes[1].set_title(f"{drug_feature_labels[feature]} Boxplot")
    axes[1].set_ylabel(drug_feature_labels[feature])

    plt.tight_layout()

    # ------------------------------------------------------
    # Save Figure
    # ------------------------------------------------------

    figure_path = FIGURES_DIR / f"section07_{feature}.png"

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"✅ Saved: {figure_path.name}")

# ============================================================================
# Code Cell 16
# ============================================================================
# ==========================================================
# Section 8. Distribution of Reaction-Related Features
# ==========================================================

reaction_features = [
    "total_reactions",
    "num_unique_reactions"
]

reaction_labels = {
    "total_reactions": "Total Reactions",
    "num_unique_reactions": "Unique Reactions"
}

for feature in reaction_features:

    data = df[feature]

    mean_value = data.mean()
    median_value = data.median()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14,5)
    )

    # ------------------------------------------------------
    # Histogram
    # ------------------------------------------------------

    axes[0].hist(
        data,
        bins=40,
        edgecolor="black"
    )

    axes[0].axvline(
        mean_value,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_value:.2f}"
    )

    axes[0].axvline(
        median_value,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median = {median_value:.2f}"
    )

    axes[0].set_title(f"{reaction_labels[feature]} Distribution")
    axes[0].set_xlabel(reaction_labels[feature])
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # ------------------------------------------------------
    # Boxplot
    # ------------------------------------------------------

    axes[1].boxplot(
        data,
        vert=True,
        patch_artist=True
    )

    axes[1].set_title(f"{reaction_labels[feature]} Boxplot")
    axes[1].set_ylabel(reaction_labels[feature])

    plt.tight_layout()

    # ------------------------------------------------------
    # Save Figure
    # ------------------------------------------------------

    figure_path = FIGURES_DIR / f"section08_{feature}.png"

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"✅ Saved: {figure_path.name}")

# ============================================================================
# Code Cell 17
# ============================================================================
# ==========================================================
# Validation
# ==========================================================

print("=" * 90)
print("Reaction Feature Summary")
print("=" * 90)

validation_table = []

for feature in reaction_features:

    data = df[feature]

    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = ((data < lower) | (data > upper)).sum()

    validation_table.append({
        "Variable": reaction_labels[feature],
        "Mean": round(data.mean(),2),
        "Median": round(data.median(),2),
        "Minimum": int(data.min()),
        "Maximum": int(data.max()),
        "Outliers": outliers,
        "Outlier (%)": round(outliers / len(data) * 100,2)
    })

validation_table = pd.DataFrame(validation_table)

display(validation_table)

print("\n✅ Reaction-related feature validation completed.")

# ============================================================================
# Code Cell 18
# ============================================================================
# ==========================================================
# Section 9. Distribution of Binary Clinical Reaction Flags
# ==========================================================

reaction_flags = [
    "flag_AKI",
    "flag_DKA",
    "flag_hypoglycemia",
    "flag_lactic_acidosis",
    "flag_amputation",
    "flag_genital_infection"
]

reaction_labels = {
    "flag_AKI": "AKI",
    "flag_DKA": "DKA",
    "flag_hypoglycemia": "Hypoglycemia",
    "flag_lactic_acidosis": "Lactic Acidosis",
    "flag_amputation": "Amputation",
    "flag_genital_infection": "Genital Infection"
}

summary = []

for feature in reaction_flags:

    count = df[feature].sum()

    percent = count / len(df) * 100

    summary.append({
        "Reaction": reaction_labels[feature],
        "Count": int(count),
        "Percent": percent
    })

reaction_summary = (
    pd.DataFrame(summary)
    .sort_values("Percent", ascending=False)
    .reset_index(drop=True)
)

display(reaction_summary)

# ============================================================================
# Code Cell 19
# ============================================================================
# ==========================================================
# Reaction Flag Prevalence
# ==========================================================

plt.figure(figsize=(10,6))

bars = plt.bar(
    reaction_summary["Reaction"],
    reaction_summary["Percent"],
    edgecolor="black"
)

plt.title("Prevalence of Clinically Important Reaction Categories")
plt.ylabel("Reports (%)")

plt.ylim(
    0,
    reaction_summary["Percent"].max() * 1.15
)
plt.xlabel("Reaction Category")

plt.xticks(rotation=20)

for bar, pct in zip(bars, reaction_summary["Percent"]):

    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.2,
        f"{pct:.2f}%",
        ha="center",
        fontsize=10
    )

plt.tight_layout()

# ----------------------------------------------------------
# Save Figure
# ----------------------------------------------------------

figure_path = FIGURES_DIR / "section09_reaction_flag_prevalence.png"

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"✅ Saved: {figure_path.name}")

# ============================================================================
# Code Cell 20
# ============================================================================
# ==========================================================
# Validation
# ==========================================================

print("=" * 90)
print("Reaction Flag Summary")
print("=" * 90)

display(reaction_summary)

print()

print("Total Reports :", f"{len(df):,}")

print("\n✅ Binary reaction flag validation completed.")

# ============================================================================
# Code Cell 21
# ============================================================================
# ==========================================================
# Section 10A. Distribution of Indication Count Variables
# ==========================================================

indication_features = [
    "total_indications",
    "num_unique_indications"
]

indication_labels = {
    "total_indications": "Total Indications",
    "num_unique_indications": "Unique Indications"
}

for feature in indication_features:

    data = df[feature]

    mean_value = data.mean()
    median_value = data.median()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14,5)
    )

    # ------------------------------------------------------
    # Histogram
    # ------------------------------------------------------

    axes[0].hist(
        data,
        bins=40,
        edgecolor="black"
    )

    axes[0].axvline(
        mean_value,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_value:.2f}"
    )

    axes[0].axvline(
        median_value,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median = {median_value:.2f}"
    )

    axes[0].set_title(f"{indication_labels[feature]} Distribution")
    axes[0].set_xlabel(indication_labels[feature])
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # ------------------------------------------------------
    # Boxplot
    # ------------------------------------------------------

    axes[1].boxplot(
        data,
        vert=True,
        patch_artist=True
    )

    axes[1].set_title(f"{indication_labels[feature]} Boxplot")
    axes[1].set_ylabel(indication_labels[feature])

    plt.tight_layout()

    # ------------------------------------------------------
    # Save Figure
    # ------------------------------------------------------

    figure_path = FIGURES_DIR / f"section10A_{feature}.png"

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"✅ Saved: {figure_path.name}")

# ============================================================================
# Code Cell 22
# ============================================================================
# ==========================================================
# Validation
# ==========================================================

print("=" * 90)
print("Indication Count Feature Summary")
print("=" * 90)

validation_table = []

for feature in indication_features:

    data = df[feature]

    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = ((data < lower) | (data > upper)).sum()

    validation_table.append({
        "Variable": indication_labels[feature],
        "Mean": round(data.mean(), 2),
        "Median": round(data.median(), 2),
        "Minimum": int(data.min()),
        "Maximum": int(data.max()),
        "Outliers": outliers,
        "Outlier (%)": round(outliers / len(data) * 100, 2)
    })

validation_table = pd.DataFrame(validation_table)

display(validation_table)

print("\n✅ Indication count variable validation completed.")

# ============================================================================
# Code Cell 23
# ============================================================================
# ==========================================================
# Section 10B
# Clinical Indication Flags
# ==========================================================

indication_flags = {
    "Type 2 Diabetes": "flag_type2_diabetes",
    "Type 1 Diabetes": "flag_type1_diabetes",
    "Any Diabetes": "flag_any_diabetes",
    "Hypertension": "flag_hypertension",
    "Chronic Kidney Disease": "flag_chronic_kidney_disease",
    "Heart Failure": "flag_heart_failure",
    "Obesity": "flag_obesity",
    "Unknown Indication": "flag_unknown_indication"
}

summary = pd.DataFrame({
    "Indication": indication_flags.keys(),
    "Count": [df[col].sum() for col in indication_flags.values()]
})

summary["Percent"] = summary["Count"] / len(df) * 100

display(summary)

# ============================================================================
# Code Cell 24
# ============================================================================
# ==========================================================
# Clinical Indication Flags
# ==========================================================

plt.figure(figsize=(12, 6))

bars = plt.bar(
    summary["Indication"],
    summary["Percent"],
    edgecolor="black"
)

plt.title(
    "Prevalence of Clinically Important Indication Categories",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Indication Category", fontsize=12)
plt.ylabel("Reports (%)", fontsize=12)
plt.xticks(rotation=20)

# Add percentage labels
for bar, pct in zip(bars, summary["Percent"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.15,
        f"{pct:.2f}%",
        ha="center",
        va="bottom",
        fontsize=11
    )

plt.tight_layout()

# ==========================================================
# Save Figure
# ==========================================================

plt.savefig(
    FIGURES_DIR / "section10B_indication_flags.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================================
# Code Cell 25
# ============================================================================
# ==========================================================
# Section 11
# Reporter Characteristics
# ==========================================================

categorical_vars = [
    "reporter_type",
    "report_type"
]

for col in categorical_vars:

    print("="*90)
    print(col.upper())
    print("="*90)

    summary = (
        df[col]
        .value_counts(dropna=False)
        .reset_index()
    )

    summary.columns = [col, "Count"]
    summary["Percent"] = summary["Count"] / len(df) * 100

    display(summary)

# ============================================================================
# Code Cell 26
# ============================================================================
# ==========================================================
# Reporter Type
# ==========================================================

summary = (
    df["reporter_type"]
    .value_counts()
    .reset_index()
)

summary.columns = ["Reporter Type", "Count"]
summary["Percent"] = summary["Count"] / len(df) * 100

plt.figure(figsize=(10,6))

bars = plt.bar(
    summary["Reporter Type"],
    summary["Percent"],
    edgecolor="black"
)

plt.title(
    "Distribution of Reporter Types",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Reporter Type")
plt.ylabel("Reports (%)")

plt.xticks(rotation=20)

for bar, pct in zip(bars, summary["Percent"]):
    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.3,
        f"{pct:.2f}%",
        ha="center",
        fontsize=11
    )

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "section11_reporter_type.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================================
# Code Cell 27
# ============================================================================
# ==========================================================
# Report Type
# ==========================================================

summary = (
    df["report_type"]
    .value_counts()
    .reset_index()
)

summary.columns = ["Report Type", "Count"]
summary["Percent"] = summary["Count"] / len(df) * 100

plt.figure(figsize=(8,5))

bars = plt.bar(
    summary["Report Type"],
    summary["Percent"],
    edgecolor="black"
)

plt.title(
    "Distribution of Report Types",
    fontsize=15,
    fontweight="bold"
)

plt.ylabel("Reports (%)")

for bar, pct in zip(bars, summary["Percent"]):
    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.3,
        f"{pct:.2f}%",
        ha="center",
        fontsize=11
    )

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "section11_report_type.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================================
# Code Cell 28
# ============================================================================
# ==========================================================
# Reporter Indicator Variables
# ==========================================================

reporter_flags = {
    "Health Professional":"has_health_professional_report",
    "Consumer":"has_consumer_report",
    "Foreign":"has_foreign_report"
}

summary = pd.DataFrame({
    "Reporter Source": reporter_flags.keys(),
    "Count":[df[c].sum() for c in reporter_flags.values()]
})

summary["Percent"] = summary["Count"]/len(df)*100

display(summary)

plt.figure(figsize=(8,5))

bars = plt.bar(
    summary["Reporter Source"],
    summary["Percent"],
    edgecolor="black"
)

plt.title(
    "Reporter Source Indicators",
    fontsize=15,
    fontweight="bold"
)

plt.ylabel("Reports (%)")

for bar,pct in zip(bars,summary["Percent"]):
    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.3,
        f"{pct:.2f}%",
        ha="center",
        fontsize=11
    )

plt.tight_layout()

plt.savefig(
    FIGURES_DIR/"section11_reporter_flags.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================================
# Code Cell 29
# ============================================================================
# ==========================================================
# Number of Reporter Sources
# ==========================================================

summary = (
    df["num_reporter_sources"]
    .value_counts()
    .sort_index()
    .reset_index()
)

summary.columns = ["Number of Sources","Count"]
summary["Percent"] = summary["Count"]/len(df)*100

display(summary)

plt.figure(figsize=(7,5))

bars = plt.bar(
    summary["Number of Sources"].astype(str),
    summary["Percent"],
    edgecolor="black"
)

plt.title(
    "Number of Reporter Sources per Case",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Number of Reporter Sources")
plt.ylabel("Reports (%)")

for bar,pct in zip(bars,summary["Percent"]):
    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.3,
        f"{pct:.2f}%",
        ha="center",
        fontsize=11
    )

plt.tight_layout()

plt.savefig(
    FIGURES_DIR/"section11_num_reporter_sources.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================================
# Code Cell 30
# ============================================================================
print("="*90)
print("Reporter Characteristics Summary")
print("="*90)

print(f"Total Reports : {len(df):,}")

print()

display(df[
    [
        "reporter_type",
        "report_type",
        "has_health_professional_report",
        "has_consumer_report",
        "has_foreign_report",
        "num_reporter_sources"
    ]
].describe(include="all"))

print("\n✅ Reporter characteristic validation completed.")

# ============================================================================
# Code Cell 31
# ============================================================================
# ==========================================================
# Section 12
# Therapy Duration Features
# ==========================================================
import numpy as np

therapy_vars = [
    "total_therapies",
    "num_therapies_with_duration",
    "therapy_duration_available",
    "has_missing_duration",
    "short_term_therapy",
    "long_term_therapy",
    "therapy_duration_outlier_present"
]

summary = []

for col in therapy_vars:

    summary.append({

        "Variable": col,

        "Count": df[col].sum(),

        "Percent": df[col].mean()*100
        if set(df[col].dropna().unique()).issubset({0,1})
        else np.nan
    })

summary = pd.DataFrame(summary)

display(summary)

# ============================================================================
# Code Cell 32
# ============================================================================
#Distribution of Therapy Counts

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)

plt.hist(
    df["total_therapies"],
    bins=40,
    edgecolor="black"
)

plt.axvline(
    df["total_therapies"].mean(),
    color="red",
    linestyle="--",
    label=f"Mean = {df['total_therapies'].mean():.2f}"
)

plt.axvline(
    df["total_therapies"].median(),
    color="green",
    linestyle="--",
    label=f"Median = {df['total_therapies'].median():.2f}"
)

plt.title("Total Therapies Distribution")
plt.xlabel("Total Therapies")
plt.ylabel("Frequency")
plt.legend()

plt.subplot(1,2,2)

plt.boxplot(df["total_therapies"])

plt.title("Total Therapies Boxplot")
plt.ylabel("Therapies")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "section12_total_therapies.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================================
# Code Cell 33
# ============================================================================
# Therapy Duration Availability
summary = pd.DataFrame({

    "Category":[
        "Duration Available",
        "Missing Duration",
        "Long-term Therapy",
        "Short-term Therapy",
        "Duration Outlier"
    ],

    "Percent":[

        df["therapy_duration_available"].mean()*100,
        df["has_missing_duration"].mean()*100,
        df["long_term_therapy"].mean()*100,
        df["short_term_therapy"].mean()*100,
        df["therapy_duration_outlier_present"].mean()*100

    ]
})

display(summary)

plt.figure(figsize=(10,5))

bars = plt.bar(
    summary["Category"],
    summary["Percent"],
    edgecolor="black"
)

plt.title(
    "Therapy Duration Characteristics",
    fontsize=16,
    fontweight="bold"
)

plt.ylabel("Reports (%)")

plt.xticks(rotation=20)

for bar,pct in zip(bars,summary["Percent"]):

    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.2,
        f"{pct:.2f}%",
        ha="center",
        fontsize=11
    )

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "section12_duration_flags.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================================
# Code Cell 34
# ============================================================================
# validation
print("="*90)
print("Therapy Feature Summary")
print("="*90)

display(summary)

display(
    df[
        [
            "total_therapies",
            "num_therapies_with_duration"
        ]
    ].describe()
)

print()

print("✅ Therapy duration feature validation completed.")

# ============================================================================
# Code Cell 35
# ============================================================================
# ==========================================================
# Target Variable Distribution
# ==========================================================

target_summary = (
    df["is_serious"]
    .value_counts()
    .rename(index={0:"Non-Serious",1:"Serious"})
    .reset_index()
)

target_summary.columns=["Class","Count"]

target_summary["Percent"] = (
    target_summary["Count"] /
    len(df) *100
)

display(target_summary)

plt.figure(figsize=(7,5))

bars = plt.bar(
    target_summary["Class"],
    target_summary["Percent"],
    edgecolor="black"
)

plt.title(
    "Distribution of Serious Adverse Event Reports",
    fontsize=18,
    fontweight="bold"
)

plt.ylabel("Reports (%)")

for bar,pct in zip(bars,target_summary["Percent"]):

    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.4,
        f"{pct:.2f}%",
        ha="center",
        fontsize=12
    )

plt.tight_layout()

plt.savefig(
    FIGURES_DIR/"section13_target_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================================
# Code Cell 36
# ============================================================================
# ==========================================================
# Section 13A
# Demographic Variables vs Seriousness
# ==========================================================

demo_summary = (

    df.groupby("is_serious")[

        [
            "age_years",
            "weight_kg"
        ]

    ]

    .agg(
        ["count","mean","median","std"]
    )

)

display(demo_summary.round(2))

# ============================================================================
# Code Cell 37
# ============================================================================
# age comaprison
plt.figure(figsize=(8,6))

df.boxplot(
    column="age_years",
    by="is_serious",
    grid=False
)

plt.title(
    "Patient Age by Seriousness",
    fontsize=16,
    fontweight="bold"
)

plt.suptitle("")

plt.xlabel("Seriousness")
plt.ylabel("Age (Years)")

plt.xticks(
    [1,2],
    ["Non-Serious","Serious"]
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "section13A_age_by_target.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================================
# Code Cell 38
# ============================================================================
# weight comparison
plt.figure(figsize=(8,6))

df.boxplot(
    column="weight_kg",
    by="is_serious",
    grid=False
)

plt.title(
    "Body Weight by Seriousness",
    fontsize=16,
    fontweight="bold"
)

plt.suptitle("")

plt.xlabel("Seriousness")
plt.ylabel("Weight (kg)")

plt.xticks(
    [1,2],
    ["Non-Serious","Serious"]
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "section13A_weight_by_target.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================================
# Code Cell 39
# ============================================================================
# Statistical Test
from scipy.stats import ttest_ind

age_test = ttest_ind(

    df.loc[df.is_serious==0,"age_years"].dropna(),

    df.loc[df.is_serious==1,"age_years"].dropna(),

    equal_var=False

)

weight_test = ttest_ind(

    df.loc[df.is_serious==0,"weight_kg"].dropna(),

    df.loc[df.is_serious==1,"weight_kg"].dropna(),

    equal_var=False

)

test_summary = pd.DataFrame({

    "Variable":[
        "Age",
        "Weight"
    ],

    "p-value":[
        age_test.pvalue,
        weight_test.pvalue
    ]

})

display(test_summary)

# ============================================================================
# Code Cell 40
# ============================================================================
# validation
print("="*90)
print("Demographic Variables vs Seriousness")
print("="*90)

print()

print("Target Counts")

display(
    df["is_serious"]
    .value_counts()
)

print()

display(
    demo_summary.round(2)
)

print()

display(
    test_summary
)

print()

print("✅ Demographic comparison completed.")

# ============================================================================
# Code Cell 41
# ============================================================================
# ==========================================================
# Section 13B. Drug Burden vs Seriousness
# ==========================================================

from scipy.stats import mannwhitneyu

drug_burden_features = [
    "total_drugs",
    "num_unique_drugs",
    "num_primary_suspect",
    "num_secondary_suspect",
    "num_concomitant",
    "num_interacting"
]

drug_burden_labels = {
    "total_drugs": "Total Drugs",
    "num_unique_drugs": "Unique Drugs",
    "num_primary_suspect": "Primary Suspect Drugs",
    "num_secondary_suspect": "Secondary Suspect Drugs",
    "num_concomitant": "Concomitant Drugs",
    "num_interacting": "Interacting Drugs"
}

# ----------------------------------------------------------
# Summary statistics by target class
# ----------------------------------------------------------

drug_burden_summary = (
    df.groupby("is_serious")[drug_burden_features]
      .agg(["count", "mean", "median", "std"])
      .round(2)
)

display(drug_burden_summary)

# ============================================================================
# Code Cell 42
# ============================================================================
# statistical test
# ==========================================================
# Mann-Whitney U Tests
# ==========================================================

test_results = []

for feature in drug_burden_features:

    non_serious = df.loc[df["is_serious"] == 0, feature]
    serious = df.loc[df["is_serious"] == 1, feature]

    statistic, p_value = mannwhitneyu(
        non_serious,
        serious,
        alternative="two-sided"
    )

    test_results.append({
        "Variable": drug_burden_labels[feature],
        "Mann-Whitney U Statistic": statistic,
        "p-value": p_value
    })

drug_burden_tests = pd.DataFrame(test_results)

drug_burden_tests["p-value"] = drug_burden_tests["p-value"].apply(
    lambda x: f"{x:.2e}"
)

display(drug_burden_tests)

# ============================================================================
# Code Cell 43
# ============================================================================
# ==========================================================
# Drug Burden Boxplots by Seriousness
# ==========================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(14, 14)
)

axes = axes.flatten()

for ax, feature in zip(axes, drug_burden_features):

    df.boxplot(
        column=feature,
        by="is_serious",
        grid=False,
        showfliers=False,
        ax=ax
    )

    ax.set_title(drug_burden_labels[feature])
    ax.set_xlabel("Seriousness")
    ax.set_ylabel(drug_burden_labels[feature])
    ax.set_xticklabels(["Non-Serious", "Serious"])

plt.suptitle("")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "section13B_drug_burden_by_target.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================================
# Code Cell 44
# ============================================================================
# ==========================================================
# Validation
# ==========================================================

print("=" * 90)
print("Drug Burden vs Seriousness")
print("=" * 90)

print("\nTarget Counts")
display(df["is_serious"].value_counts().sort_index())

print("\nDrug Burden Summary")
display(drug_burden_summary)

print("\nStatistical Tests")
display(drug_burden_tests)

print("\n✅ Drug burden comparison completed.")

# ============================================================================
# Code Cell 45
# ============================================================================
# ==========================================================
# Section 13C. Reaction Burden vs Seriousness
# ==========================================================

from scipy.stats import mannwhitneyu

reaction_features = [
    "total_reactions",
    "num_unique_reactions"
]

reaction_labels = {
    "total_reactions": "Total Reactions",
    "num_unique_reactions": "Unique Reactions"
}

reaction_summary = (
    df.groupby("is_serious")[reaction_features]
      .agg(["count", "mean", "median", "std"])
      .round(2)
)

display(reaction_summary)

# ============================================================================
# Code Cell 46
# ============================================================================
# ==========================================================
# Mann-Whitney U Tests
# ==========================================================

reaction_tests = []

for feature in reaction_features:

    non_serious = df.loc[df.is_serious == 0, feature]
    serious = df.loc[df.is_serious == 1, feature]

    statistic, p_value = mannwhitneyu(
        non_serious,
        serious,
        alternative="two-sided"
    )

    reaction_tests.append({
        "Variable": reaction_labels[feature],
        "Mann-Whitney U Statistic": statistic,
        "p-value": p_value
    })

reaction_tests = pd.DataFrame(reaction_tests)

reaction_tests["p-value"] = reaction_tests["p-value"].apply(
    lambda x: f"{x:.2e}"
)

display(reaction_tests)

# ============================================================================
# Code Cell 47
# ============================================================================
# ==========================================================
# Reaction Burden Boxplots
# ==========================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12,5)
)

for ax, feature in zip(axes, reaction_features):

    df.boxplot(
        column=feature,
        by="is_serious",
        showfliers=False,
        grid=False,
        ax=ax
    )

    ax.set_title(reaction_labels[feature])
    ax.set_xlabel("Seriousness")
    ax.set_ylabel(reaction_labels[feature])
    ax.set_xticklabels(["Non-Serious", "Serious"])

plt.suptitle("")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "section13C_reaction_burden_by_target.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================================
# Code Cell 48
# ============================================================================
# ==========================================================
# Validation
# ==========================================================

print("=" * 90)
print("Reaction Burden vs Seriousness")
print("=" * 90)

print("\nTarget Counts")
display(df["is_serious"].value_counts().sort_index())

print("\nReaction Burden Summary")
display(reaction_summary)

print("\nStatistical Tests")
display(reaction_tests)

print("\n✅ Reaction burden comparison completed.")

# ============================================================================
# Code Cell 49
# ============================================================================
# ==========================================================
# Section 13D. Indication Burden vs Seriousness
# ==========================================================

from scipy.stats import mannwhitneyu

variables = [
    "total_indications",
    "num_unique_indications"
]

summary_rows = []
stats_rows = []

for var in variables:

    non_serious = df.loc[
        df["is_serious"] == 0,
        var
    ]

    serious = df.loc[
        df["is_serious"] == 1,
        var
    ]

    summary_rows.append([
        var,
        non_serious.count(),
        non_serious.mean(),
        non_serious.median(),
        non_serious.std(),
        serious.count(),
        serious.mean(),
        serious.median(),
        serious.std()
    ])

    u_stat, p_value = mannwhitneyu(
        non_serious,
        serious,
        alternative="two-sided"
    )

    stats_rows.append([
        var,
        u_stat,
        p_value
    ])

summary_df = pd.DataFrame(
    summary_rows,
    columns=[
        "Variable",
        "NS Count",
        "NS Mean",
        "NS Median",
        "NS Std",
        "S Count",
        "S Mean",
        "S Median",
        "S Std"
    ]
)

stats_df = pd.DataFrame(
    stats_rows,
    columns=[
        "Variable",
        "Mann-Whitney U Statistic",
        "p-value"
    ]
)

stats_df["p-value"] = stats_df["p-value"].apply(lambda x: f"{x:.2e}")

display(summary_df.round(2))
display(stats_df)

# ============================================================================
# Code Cell 50
# ============================================================================
# ==========================================================
# Boxplots
# ==========================================================

fig, axes = plt.subplots(1, 2, figsize=(14,6))

plot_vars = [
    ("total_indications", "Total Indications"),
    ("num_unique_indications", "Unique Indications")
]

for ax, (var, title) in zip(axes, plot_vars):

    df.boxplot(
        column=var,
        by="is_serious",
        ax=ax,
        grid=False
    )

    ax.set_title(title)
    ax.set_xlabel("Seriousness")
    ax.set_ylabel(title)

    ax.set_xticklabels([
        "Non-Serious",
        "Serious"
    ])

plt.suptitle("")
plt.tight_layout()

# Save figure
plt.savefig(
    FIGURES_DIR / "section13D_indication_burden_by_target.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================================
# Code Cell 51
# ============================================================================
# ==========================================================
# Validation
# ==========================================================

print("="*90)
print("Indication Burden vs Seriousness")
print("="*90)

print("\nTarget Counts")
print(df["is_serious"].value_counts())

print("\nIndication Burden Summary")
display(summary_df.round(2))

print("\nStatistical Tests")
display(stats_df)

print("\n✅ Indication burden comparison completed.")

# ============================================================================
# Code Cell 52
# ============================================================================
# ==========================================================
# Section 13E. Clinically Important Reaction Categories vs Seriousness
# ==========================================================

from scipy.stats import chi2_contingency

reaction_flags = [
    "flag_AKI",
    "flag_lactic_acidosis",
    "flag_hypoglycemia",
    "flag_DKA",
    "flag_genital_infection",
    "flag_amputation"
]

reaction_labels = {
    "flag_AKI": "AKI",
    "flag_lactic_acidosis": "Lactic Acidosis",
    "flag_hypoglycemia": "Hypoglycemia",
    "flag_DKA": "DKA",
    "flag_genital_infection": "Genital Infection",
    "flag_amputation": "Amputation"
}

summary_rows = []
stats_rows = []

for var in reaction_flags:

    table = pd.crosstab(df["is_serious"], df[var])

    chi2, p_value, dof, expected = chi2_contingency(table)

    ns_pct = df.loc[df["is_serious"] == 0, var].mean() * 100
    s_pct = df.loc[df["is_serious"] == 1, var].mean() * 100

    summary_rows.append({
        "Reaction": reaction_labels[var],
        "Non-Serious (%)": ns_pct,
        "Serious (%)": s_pct
    })

    stats_rows.append({
        "Reaction": reaction_labels[var],
        "Chi-square": chi2,
        "p-value": p_value
    })

summary_df = pd.DataFrame(summary_rows)
stats_df = pd.DataFrame(stats_rows)

stats_df["p-value"] = stats_df["p-value"].apply(lambda x: f"{x:.2e}")

display(summary_df.round(3))
display(stats_df)

# ============================================================================
# Code Cell 53
# ============================================================================
# ==========================================================
# Grouped Bar Chart
# ==========================================================

plot_df = summary_df.set_index("Reaction")

ax = plot_df.plot(
    kind="bar",
    figsize=(12,6)
)

plt.title(
    "Clinically Important Reaction Categories by Seriousness",
    fontsize=16,
    weight="bold"
)

plt.ylabel("Reports (%)")
plt.xlabel("Reaction Category")
plt.xticks(rotation=20)
plt.legend(["Non-Serious","Serious"])

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f%%",
        fontsize=9
    )

plt.tight_layout()

# Save figure
plt.savefig(
    FIGURES_DIR / "section13E_reaction_flags_by_target.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================================
# Code Cell 54
# ============================================================================
# ==========================================================
# Validation
# ==========================================================

print("="*90)
print("Reaction Categories vs Seriousness")
print("="*90)

print("\nTarget Counts")
print(df["is_serious"].value_counts())

print("\nReaction Summary")
display(summary_df.round(3))

print("\nChi-square Tests")
display(stats_df)

print("\n✅ Reaction category comparison completed.")

# ============================================================================
# Code Cell 55
# ============================================================================
print(df.columns.tolist())

# ============================================================================
# Code Cell 56
# ============================================================================
# ==========================================================
# Section 13F - Clinical Indication Categories vs Seriousness
# ==========================================================

from scipy.stats import chi2_contingency

# ----------------------------------------------------------
# Indication flag columns
# ----------------------------------------------------------

indication_flags = [
    "flag_type2_diabetes",
    "flag_type1_diabetes",
    "flag_any_diabetes",
    "flag_hypertension",
    "flag_chronic_kidney_disease",
    "flag_heart_failure",
    "flag_obesity",
    "flag_unknown_indication"
]

indication_labels = {
    "flag_type2_diabetes": "Type 2 Diabetes",
    "flag_type1_diabetes": "Type 1 Diabetes",
    "flag_any_diabetes": "Any Diabetes",
    "flag_hypertension": "Hypertension",
    "flag_chronic_kidney_disease": "Chronic Kidney Disease",
    "flag_heart_failure": "Heart Failure",
    "flag_obesity": "Obesity",
    "flag_unknown_indication": "Unknown Indication"
}

# ----------------------------------------------------------
# Summary tables
# ----------------------------------------------------------

summary_rows = []
stats_rows = []

for var in indication_flags:

    table = pd.crosstab(df["is_serious"], df[var])

    chi2, p_value, dof, expected = chi2_contingency(table)

    ns_pct = (
        df.loc[df["is_serious"] == 0, var].mean() * 100
    )

    s_pct = (
        df.loc[df["is_serious"] == 1, var].mean() * 100
    )

    summary_rows.append([
        indication_labels[var],
        ns_pct,
        s_pct
    ])

    stats_rows.append([
        indication_labels[var],
        chi2,
        p_value
    ])

summary_df = pd.DataFrame(
    summary_rows,
    columns=[
        "Indication",
        "Non-Serious (%)",
        "Serious (%)"
    ]
)

stats_df = pd.DataFrame(
    stats_rows,
    columns=[
        "Indication",
        "Chi-square",
        "p-value"
    ]
)

display(summary_df.round(3))
display(stats_df)

# ============================================================================
# Code Cell 57
# ============================================================================
# ==========================================================
# Visualization
# ==========================================================

x = np.arange(len(summary_df))
width = 0.38

plt.figure(figsize=(12,6))

bars1 = plt.bar(
    x - width/2,
    summary_df["Non-Serious (%)"],
    width,
    label="Non-Serious"
)

bars2 = plt.bar(
    x + width/2,
    summary_df["Serious (%)"],
    width,
    label="Serious"
)

plt.xticks(
    x,
    summary_df["Indication"],
    rotation=20
)

plt.ylabel("Reports (%)")
plt.xlabel("Clinical Indication")
plt.title("Clinical Indication Categories by Seriousness")

plt.legend()

for bars in [bars1, bars2]:
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.2f}%",
            ha="center",
            fontsize=10
        )

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "section13F_indication_categories_vs_seriousness.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================================
# Code Cell 58
# ============================================================================
# ==========================================================
# Console Summary
# ==========================================================

print("="*90)
print("Indication Categories vs Seriousness")
print("="*90)

print("\nTarget Counts")
print(df["is_serious"].value_counts())

print("\nIndication Summary")
display(summary_df.round(3))

print("\nChi-square Tests")
display(stats_df)

print("\n✅ Indication category comparison completed.")

# ============================================================================
# Code Cell 59
# ============================================================================
# ==========================================================
# Section 14A - Correlation Heatmap
# ==========================================================
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

numeric_df = df.select_dtypes(include=np.number).copy()

# Remove target variable
numeric_df = numeric_df.drop(columns=["is_serious"])

corr = numeric_df.corr()

plt.figure(figsize=(18,15))

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.25,
    cbar_kws={"shrink":0.8}
)

plt.title("Correlation Matrix of Numeric Features")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "section14A_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================================
# Code Cell 60
# ============================================================================
# ==========================================================
# Section 14B - Highly Correlated Variable Pairs
# ==========================================================

# Use correlation matrix from previous section
corr_matrix = numeric_df.corr().abs()

# Upper triangle only
upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

# Threshold
threshold = 0.80

high_corr = (
    upper.stack()
         .reset_index()
)

high_corr.columns = [
    "Variable 1",
    "Variable 2",
    "Correlation"
]

high_corr = (
    high_corr[
        high_corr["Correlation"] >= threshold
    ]
    .sort_values(
        "Correlation",
        ascending=False
    )
    .reset_index(drop=True)
)

display(high_corr.round(3))

print("="*90)
print("Highly Correlated Variable Pairs")
print("="*90)
print(f"Correlation Threshold : |r| ≥ {threshold}")
print(f"Number of Pairs       : {len(high_corr)}")

display(high_corr.round(3))

print("\n✅ High correlation analysis completed.")

# ============================================================================
# Code Cell 61
# ============================================================================
# ==========================================================
# Section 14C - Variance Inflation Factor (VIF)
# ==========================================================

from statsmodels.stats.outliers_influence import variance_inflation_factor

# ----------------------------------------------------------
# Numeric predictors only
# ----------------------------------------------------------

vif_df = numeric_df.copy()

# Remove identifier
if "primaryid" in vif_df.columns:
    vif_df = vif_df.drop(columns=["primaryid"])

# Remove target if present
if "is_serious" in vif_df.columns:
    vif_df = vif_df.drop(columns=["is_serious"])

# Fill missing values temporarily
vif_df = vif_df.fillna(0)

# ----------------------------------------------------------
# Calculate VIF
# ----------------------------------------------------------

vif_results = pd.DataFrame({
    "Variable": vif_df.columns,
    "VIF": [
        variance_inflation_factor(
            vif_df.values,
            i
        )
        for i in range(vif_df.shape[1])
    ]
})

vif_results = (
    vif_results
    .sort_values("VIF", ascending=False)
    .reset_index(drop=True)
)

display(vif_results.round(2))

# ----------------------------------------------------------
# High VIF variables
# ----------------------------------------------------------

high_vif = vif_results[vif_results["VIF"] >= 10]

print("="*90)
print("Variance Inflation Factor (VIF)")
print("="*90)

print(f"Variables evaluated : {len(vif_results)}")
print(f"Variables with VIF ≥ 10 : {len(high_vif)}")

display(high_vif.round(2))

print("\n✅ VIF analysis completed.")

# ============================================================================
# Code Cell 62
# ============================================================================
print("="*90)
print("FINAL MODELING DATASET")
print("="*90)

print(f"Rows            : {len(df):,}")
print(f"Columns         : {df.shape[1]}")
print(f"Predictors      : {df.shape[1]-1}")
print(f"Target Variable : is_serious")

display(
    df["is_serious"]
      .value_counts()
      .rename_axis("Class")
      .reset_index(name="Count")
)

# ============================================================================
# Code Cell 63
# ============================================================================
# ==========================================================
# Export Important EDA Tables
# ==========================================================

from pathlib import Path

PROJECT_DIR = Path.cwd().parent
EDA_RESULTS_DIR = PROJECT_DIR / "results" / "eda"
EDA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Export only tables that exist
if "missing_summary" in globals():
    missing_summary.to_csv(EDA_RESULTS_DIR / "missing_values.csv", index=False)

if "target_summary" in globals():
    target_summary.to_csv(EDA_RESULTS_DIR / "target_distribution.csv", index=False)

if "summary_stats" in globals():
    summary_stats.to_csv(EDA_RESULTS_DIR / "summary_statistics.csv", index=False)

if "high_corr" in globals():
    high_corr.to_csv(EDA_RESULTS_DIR / "high_correlations.csv", index=False)

if "vif_results" in globals():
    vif_results.to_csv(EDA_RESULTS_DIR / "vif_results.csv", index=False)

print("=" * 70)
print("Important EDA tables exported successfully.")
print("=" * 70)
print(f"Location: {EDA_RESULTS_DIR}")

# ============================================================================
# Code Cell 64
# ============================================================================
# ==========================================================
# Create EDA Results Directory Structure
# ==========================================================

from pathlib import Path

PROJECT_DIR = Path.cwd().parent

RESULTS_DIR = PROJECT_DIR / "results"
EDA_DIR = RESULTS_DIR / "eda"

TABLE_DIR = EDA_DIR / "tables"
STAT_DIR = EDA_DIR / "statistics"
FIGURE_DIR = EDA_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("EDA Export Directories Created")
print("=" * 70)

print(TABLE_DIR)
print(STAT_DIR)
print(FIGURE_DIR)

# ============================================================================
# Code Cell 65
# ============================================================================
# ==========================================================
# Export Summary Tables
# ==========================================================

tables = {}

if "target_summary" in globals():
    tables["target_distribution"] = target_summary

if "missing_summary" in globals():
    tables["missing_values"] = missing_summary

if "summary_stats" in globals():
    tables["numeric_summary"] = summary_stats

if "demo_summary" in globals():
    tables["demographic_summary"] = demo_summary

if "therapy_summary" in globals():
    tables["therapy_summary"] = therapy_summary

if "drug_burden_summary" in globals():
    tables["drug_burden_summary"] = drug_burden_summary

if "reaction_summary" in globals():
    tables["reaction_burden_summary"] = reaction_summary

if "indication_summary" in globals():
    tables["indication_burden_summary"] = indication_summary

if "reporter_summary" in globals():
    tables["reporter_summary"] = reporter_summary

if "reaction_flag_summary" in globals():
    tables["reaction_category_summary"] = reaction_flag_summary

if "indication_flag_summary" in globals():
    tables["indication_category_summary"] = indication_flag_summary

if "high_corr" in globals():
    tables["high_correlations"] = high_corr

if "vif_results" in globals():
    tables["vif_results"] = vif_results

for name, table in tables.items():
    table.to_csv(TABLE_DIR / f"{name}.csv", index=False)

print(f"Exported {len(tables)} summary tables.")

# ============================================================================
# Code Cell 66
# ============================================================================
# ==========================================================
# Export Statistical Test Results
# ==========================================================

stats = {}

if "demo_stats" in globals():
    stats["demographic_tests"] = demo_stats

if "therapy_stats" in globals():
    stats["therapy_tests"] = therapy_stats

if "drug_stats" in globals():
    stats["drug_tests"] = drug_stats

if "reaction_stats" in globals():
    stats["reaction_tests"] = reaction_stats

if "indication_stats" in globals():
    stats["indication_tests"] = indication_stats

if "reporter_stats" in globals():
    stats["reporter_tests"] = reporter_stats

if "reaction_flag_stats" in globals():
    stats["reaction_category_tests"] = reaction_flag_stats

if "indication_flag_stats" in globals():
    stats["indication_category_tests"] = indication_flag_stats

for name, table in stats.items():
    table.to_csv(STAT_DIR / f"{name}.csv", index=False)

print(f"Exported {len(stats)} statistical result tables.")

# ============================================================================
# Code Cell 67
# ============================================================================
# ==========================================================
# Export Feature Dictionary
# ==========================================================

feature_dictionary = pd.DataFrame({
    "Feature": df.columns,
    "Data Type": df.dtypes.astype(str)
})

feature_dictionary.to_csv(
    TABLE_DIR / "feature_dictionary.csv",
    index=False
)

feature_dictionary.head()

# ============================================================================
# Code Cell 68
# ============================================================================
# ==========================================================
# Export Feature List
# ==========================================================

feature_list = pd.DataFrame({

    "Feature": df.columns,

    "Feature Type": [

        "Target" if col=="is_serious"

        else "Binary"

        if df[col].dropna().isin([0,1]).all()

        else "Numeric"

        for col in df.columns

    ]

})

feature_list.to_csv(
    TABLE_DIR / "feature_list.csv",
    index=False
)

feature_list.head()

# ============================================================================
# Code Cell 69
# ============================================================================
# ==========================================================
# Export Modeling Dataset Summary
# ==========================================================

modeling_summary = pd.DataFrame({

    "Metric":[

        "Rows",
        "Columns",
        "Predictors",
        "Target Variable",
        "Serious Reports",
        "Non-Serious Reports"

    ],

    "Value":[

        len(df),

        len(df.columns),

        len(df.columns)-1,

        "is_serious",

        int(df.is_serious.sum()),

        int((df.is_serious==0).sum())

    ]

})

modeling_summary.to_csv(
    TABLE_DIR/"modeling_dataset_summary.csv",
    index=False
)

modeling_summary

# ============================================================================
# Code Cell 70
# ============================================================================
# ==========================================================
# Export Column Types
# ==========================================================

column_types = pd.DataFrame({

    "Column":df.columns,

    "dtype":df.dtypes.astype(str)

})

column_types.to_csv(
    TABLE_DIR/"column_types.csv",
    index=False
)

column_types.head()

# ============================================================================
# Code Cell 71
# ============================================================================
# ==========================================================
# Export Correlation Matrix
# ==========================================================

numeric_df = df.select_dtypes(include="number")

corr = numeric_df.corr()

corr.to_csv(
    TABLE_DIR/"correlation_matrix.csv"
)

print("Correlation matrix exported.")

# ============================================================================
# Code Cell 72
# ============================================================================
# ==========================================================
# Export Missing Value Report
# ==========================================================

missing_report = (

    df.isna()

      .sum()

      .reset_index()

)

missing_report.columns=["Variable","Missing Count"]

missing_report["Missing Percent"] = (

    missing_report["Missing Count"]/len(df)*100

)

missing_report.to_csv(

    TABLE_DIR/"missing_values.csv",

    index=False

)

missing_report.head()

# ============================================================================
# Code Cell 73
# ============================================================================
# ==========================================================
# Export Summary
# ==========================================================

print("="*80)
print("EDA EXPORT COMPLETED")
print("="*80)

print(f"Tables      : {TABLE_DIR}")
print(f"Statistics  : {STAT_DIR}")
print(f"Figures     : {FIGURE_DIR}")

print("\nNotebook 4 can directly use these exported files.")

# ============================================================================
# Code Cell 74
# ============================================================================
# ==========================================================
# Recreate and Export EDA Statistical Test Results
# ==========================================================

from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency

STAT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------
# 1. Demographic Tests
# ----------------------------------------------------------

demo_test_rows = []

for col in ["age_years", "weight_kg"]:

    non_serious = df.loc[df["is_serious"] == 0, col].dropna()
    serious = df.loc[df["is_serious"] == 1, col].dropna()

    stat, p_value = ttest_ind(
        non_serious,
        serious,
        equal_var=False
    )

    demo_test_rows.append({
        "Variable": col,
        "Test": "Welch t-test",
        "Statistic": stat,
        "p-value": p_value
    })

demo_stats = pd.DataFrame(demo_test_rows)
demo_stats.to_csv(STAT_DIR / "demographic_tests.csv", index=False)


# ----------------------------------------------------------
# 2. Drug Burden Tests
# ----------------------------------------------------------

drug_features = [
    "total_drugs",
    "num_unique_drugs",
    "num_primary_suspect",
    "num_secondary_suspect",
    "num_concomitant",
    "num_interacting"
]

drug_test_rows = []

for col in drug_features:

    non_serious = df.loc[df["is_serious"] == 0, col]
    serious = df.loc[df["is_serious"] == 1, col]

    stat, p_value = mannwhitneyu(
        non_serious,
        serious,
        alternative="two-sided"
    )

    drug_test_rows.append({
        "Variable": col,
        "Test": "Mann-Whitney U",
        "Statistic": stat,
        "p-value": p_value
    })

drug_stats = pd.DataFrame(drug_test_rows)
drug_stats.to_csv(STAT_DIR / "drug_tests.csv", index=False)


# ----------------------------------------------------------
# 3. Reaction Burden Tests
# ----------------------------------------------------------

reaction_features = [
    "total_reactions",
    "num_unique_reactions"
]

reaction_test_rows = []

for col in reaction_features:

    non_serious = df.loc[df["is_serious"] == 0, col]
    serious = df.loc[df["is_serious"] == 1, col]

    stat, p_value = mannwhitneyu(
        non_serious,
        serious,
        alternative="two-sided"
    )

    reaction_test_rows.append({
        "Variable": col,
        "Test": "Mann-Whitney U",
        "Statistic": stat,
        "p-value": p_value
    })

reaction_stats = pd.DataFrame(reaction_test_rows)
reaction_stats.to_csv(STAT_DIR / "reaction_tests.csv", index=False)


# ----------------------------------------------------------
# 4. Indication Burden Tests
# ----------------------------------------------------------

indication_features = [
    "total_indications",
    "num_unique_indications"
]

indication_test_rows = []

for col in indication_features:

    non_serious = df.loc[df["is_serious"] == 0, col]
    serious = df.loc[df["is_serious"] == 1, col]

    stat, p_value = mannwhitneyu(
        non_serious,
        serious,
        alternative="two-sided"
    )

    indication_test_rows.append({
        "Variable": col,
        "Test": "Mann-Whitney U",
        "Statistic": stat,
        "p-value": p_value
    })

indication_stats = pd.DataFrame(indication_test_rows)
indication_stats.to_csv(STAT_DIR / "indication_tests.csv", index=False)


# ----------------------------------------------------------
# 5. Therapy Tests
# ----------------------------------------------------------

therapy_features = [
    "total_therapies",
    "num_therapies_with_duration",
    "mean_therapy_duration_days",
    "median_therapy_duration_days",
    "min_therapy_duration_days",
    "max_therapy_duration_days"
]

therapy_test_rows = []

for col in therapy_features:

    non_serious = df.loc[df["is_serious"] == 0, col]
    serious = df.loc[df["is_serious"] == 1, col]

    stat, p_value = mannwhitneyu(
        non_serious,
        serious,
        alternative="two-sided"
    )

    therapy_test_rows.append({
        "Variable": col,
        "Test": "Mann-Whitney U",
        "Statistic": stat,
        "p-value": p_value
    })

therapy_stats = pd.DataFrame(therapy_test_rows)
therapy_stats.to_csv(STAT_DIR / "therapy_tests.csv", index=False)


# ----------------------------------------------------------
# 6. Reporter Categorical Tests
# ----------------------------------------------------------

reporter_test_rows = []

for col in ["reporter_type", "report_type"]:

    table = pd.crosstab(df[col], df["is_serious"])

    stat, p_value, dof, expected = chi2_contingency(table)

    reporter_test_rows.append({
        "Variable": col,
        "Test": "Chi-square",
        "Statistic": stat,
        "Degrees of Freedom": dof,
        "p-value": p_value
    })

reporter_stats = pd.DataFrame(reporter_test_rows)
reporter_stats.to_csv(STAT_DIR / "reporter_tests.csv", index=False)


# ----------------------------------------------------------
# 7. Reaction Category Flag Tests
# ----------------------------------------------------------

reaction_flags = [
    "flag_AKI",
    "flag_lactic_acidosis",
    "flag_hypoglycemia",
    "flag_DKA",
    "flag_genital_infection",
    "flag_amputation"
]

reaction_flag_test_rows = []

for col in reaction_flags:

    table = pd.crosstab(df[col], df["is_serious"])

    stat, p_value, dof, expected = chi2_contingency(table)

    reaction_flag_test_rows.append({
        "Variable": col,
        "Test": "Chi-square",
        "Statistic": stat,
        "Degrees of Freedom": dof,
        "p-value": p_value
    })

reaction_flag_stats = pd.DataFrame(reaction_flag_test_rows)
reaction_flag_stats.to_csv(STAT_DIR / "reaction_category_tests.csv", index=False)


# ----------------------------------------------------------
# 8. Indication Category Flag Tests
# ----------------------------------------------------------

indication_flags = [
    "flag_type2_diabetes",
    "flag_type1_diabetes",
    "flag_any_diabetes",
    "flag_hypertension",
    "flag_chronic_kidney_disease",
    "flag_heart_failure",
    "flag_obesity",
    "flag_unknown_indication"
]

indication_flag_test_rows = []

for col in indication_flags:

    table = pd.crosstab(df[col], df["is_serious"])

    stat, p_value, dof, expected = chi2_contingency(table)

    indication_flag_test_rows.append({
        "Variable": col,
        "Test": "Chi-square",
        "Statistic": stat,
        "Degrees of Freedom": dof,
        "p-value": p_value
    })

indication_flag_stats = pd.DataFrame(indication_flag_test_rows)
indication_flag_stats.to_csv(STAT_DIR / "indication_category_tests.csv", index=False)


# ----------------------------------------------------------
# Validation
# ----------------------------------------------------------

exported_stats = [
    "demographic_tests.csv",
    "drug_tests.csv",
    "reaction_tests.csv",
    "indication_tests.csv",
    "therapy_tests.csv",
    "reporter_tests.csv",
    "reaction_category_tests.csv",
    "indication_category_tests.csv"
]

print("=" * 80)
print("STATISTICAL TEST EXPORT COMPLETED")
print("=" * 80)

for file in exported_stats:
    print(f"✅ {file}")

print("\nLocation:")
print(STAT_DIR)

# ============================================================================
# Code Cell 75
# ============================================================================
# ==========================================================
# Export Final Modeling Dataset
# ==========================================================

from pathlib import Path

PROJECT_DIR = Path.cwd().parent

MODELING_DIR = PROJECT_DIR / "data" / "modeling"
MODELING_DIR.mkdir(parents=True, exist_ok=True)

output_file = MODELING_DIR / "faers_modeling_dataset.parquet"

df.to_parquet(
    output_file,
    index=False
)

print("=" * 80)
print("FINAL MODELING DATASET SAVED")
print("=" * 80)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")
print(f"Location: {output_file}")

# Extracted from: 02_Constructing the Report-Level Analytical Dataset.ipynb
# Complete executable code cells in notebook order.


# ============================================================================
# Code Cell 1
# ============================================================================
# ==========================================================
# Notebook Setup
# ==========================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 120)

sns.set_theme(style="whitegrid")

# Define project directories
PROJECT_DIR = Path.cwd().parent

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "faers_ascii_2024q1" / "ASCII"
PROCESSED_DIR = DATA_DIR / "processed"

FIGURE_DIR = PROJECT_DIR / "figures" / "notebook02"
DOCS_DIR = PROJECT_DIR / "docs"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("Notebook setup completed successfully.")
print("Project directory:", PROJECT_DIR)
print("Raw data directory:", RAW_DIR)
print("Processed data directory:", PROCESSED_DIR)
print("Notebook 02 figure directory:", FIGURE_DIR)

# ============================================================================
# Code Cell 2
# ============================================================================
# ==========================================================
# Verify Raw FAERS Files
# ==========================================================

expected_files = [
    "DEMO24Q1.txt",
    "DRUG24Q1.txt",
    "REAC24Q1.txt",
    "OUTC24Q1.txt",
    "RPSR24Q1.txt",
    "THER24Q1.txt",
    "INDI24Q1.txt",
]

print("Raw directory:")
print(RAW_DIR)
print()

for file_name in expected_files:
    file_path = RAW_DIR / file_name
    status = "Found" if file_path.exists() else "Missing"
    print(f"{file_name:<15} {status}")

# ============================================================================
# Code Cell 3
# ============================================================================
# ==========================================================
# Load Raw FAERS Tables
# ==========================================================

print("Loading FAERS datasets...\n")

demo = pd.read_csv(
    RAW_DIR / "DEMO24Q1.txt",
    sep="$",
    low_memory=False
)

drug = pd.read_csv(
    RAW_DIR / "DRUG24Q1.txt",
    sep="$",
    low_memory=False
)

reac = pd.read_csv(
    RAW_DIR / "REAC24Q1.txt",
    sep="$",
    low_memory=False
)

outc = pd.read_csv(
    RAW_DIR / "OUTC24Q1.txt",
    sep="$",
    low_memory=False
)

rpsr = pd.read_csv(
    RAW_DIR / "RPSR24Q1.txt",
    sep="$",
    low_memory=False
)

ther = pd.read_csv(
    RAW_DIR / "THER24Q1.txt",
    sep="$",
    low_memory=False
)

indi = pd.read_csv(
    RAW_DIR / "INDI24Q1.txt",
    sep="$",
    low_memory=False
)

print("All FAERS tables loaded successfully.")

# ============================================================================
# Code Cell 4
# ============================================================================
# ==========================================================
# Validate Loaded Table Dimensions
# ==========================================================

table_summary = pd.DataFrame({
    "Table": ["DEMO", "DRUG", "REAC", "OUTC", "RPSR", "THER", "INDI"],
    "Rows": [
        demo.shape[0],
        drug.shape[0],
        reac.shape[0],
        outc.shape[0],
        rpsr.shape[0],
        ther.shape[0],
        indi.shape[0]
    ],
    "Columns": [
        demo.shape[1],
        drug.shape[1],
        reac.shape[1],
        outc.shape[1],
        rpsr.shape[1],
        ther.shape[1],
        indi.shape[1]
    ]
})

print("=" * 55)
print("FAERS Dataset Dimension Summary")
print("=" * 55)

display(table_summary)

# ============================================================================
# Code Cell 5
# ============================================================================
# ==============================================================================
# Construct the binary response variable
# ==============================================================================

# Create one record per PRIMARYID from the OUTC table
is_serious = (
    outc.groupby("primaryid")
        .size()
        .reset_index(name="outcome_count")
)

# Every report appearing in OUTC is considered serious
is_serious["is_serious"] = 1

# Keep only the response variable
is_serious = is_serious[["primaryid", "is_serious"]]

# Merge with DEMO so every report receives a target value
target = demo[["primaryid"]].merge(
    is_serious,
    on="primaryid",
    how="left"
)

# Reports without an OUTC record are considered non-serious
target["is_serious"] = target["is_serious"].fillna(0).astype(int)

print("Target variable created successfully.\n")

display(target.head())

print("\nTarget dataset shape:", target.shape)

# ============================================================================
# Code Cell 6
# ============================================================================
# ==========================================================
# Validate Target Variable Distribution
# ==========================================================

# Count observations in each class
target_summary = (
    target["is_serious"]
    .value_counts()
    .sort_index()
    .rename_axis("is_serious")
    .reset_index(name="Count")
)

# Calculate percentages
target_summary["Percentage"] = (
    target_summary["Count"] /
    target_summary["Count"].sum() * 100
).round(2)

print("=" * 60)
print("Target Variable Distribution")
print("=" * 60)

display(target_summary)

# ----------------------------------------------------------
# Visualization
# ----------------------------------------------------------

plt.figure(figsize=(6, 5))

ax = sns.countplot(
    data=target,
    x="is_serious",
    order=[0, 1]
)

ax.set_title("Distribution of Target Variable (is_serious)", fontsize=14)
ax.set_xlabel("is_serious")
ax.set_ylabel("Number of Reports")

# Add count labels
for container in ax.containers:
    ax.bar_label(container, fmt="%.0f")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "target_variable_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================================
# Code Cell 7
# ============================================================================
# ==========================================================
# DEMO Data Quality Assessment
# ==========================================================

print("=" * 70)
print("DEMO Table Overview")
print("=" * 70)

print(f"Rows              : {demo.shape[0]:,}")
print(f"Columns           : {demo.shape[1]}")

print(f"\nUnique PRIMARYIDs : {demo['primaryid'].nunique():,}")

duplicate_primaryids = demo.duplicated(subset="primaryid").sum()

print(f"Duplicate PRIMARYIDs : {duplicate_primaryids:,}")

print("\n" + "=" * 70)
print("Missing Values")
print("=" * 70)

missing_summary = (
    demo.isna()
        .sum()
        .to_frame("Missing")
)

missing_summary["Percent"] = (
    missing_summary["Missing"] /
    len(demo) * 100
).round(2)

missing_summary = (
    missing_summary
    .sort_values("Missing", ascending=False)
)

display(missing_summary)

# ============================================================================
# Code Cell 8
# ============================================================================
# ==========================================================
# DEMO Variable Selection Plan
# ==========================================================

demo_feature_plan = pd.DataFrame({
    "Original Variable": [
        "primaryid",
        "caseid",
        "age",
        "age_cod",
        "sex",
        "wt",
        "wt_cod",
        "reporter_country",
        "occr_country",
        "occp_cod",
        "rept_cod",
        "event_dt",
        "age_grp",
        "mfr_num",
        "mfr_sndr",
        "to_mfr",
        "lit_ref",
        "auth_num",
        "e_sub",
        "i_f_code",
        "caseversion",
        "init_fda_dt",
        "fda_dt",
        "mfr_dt",
        "rept_dt"
    ],

    "Keep": [
        "Key",
        "No",
        "Yes",
        "Yes",
        "Yes",
        "Yes",
        "Yes",
        "Yes",
        "Yes",
        "Yes",
        "Investigate",
        "No",
        "No",
        "No",
        "No",
        "No",
        "No",
        "No",
        "No",
        "No",
        "No",
        "No",
        "No",
        "No",
        "No"
    ],

    "Reason": [
        "Primary key",
        "Duplicate identifier",
        "Important demographic predictor",
        "Convert age to years",
        "Important demographic predictor",
        "Clinically relevant",
        "Convert weight to kilograms",
        "Geographic predictor",
        "Geographic predictor",
        "Reporter occupation",
        "May contain reporting information",
        "Administrative date",
        "Derived from age",
        "Administrative identifier",
        "Manufacturer identifier",
        "Administrative variable",
        "Literature reference",
        "Authorization number",
        "Submission indicator",
        "Follow-up indicator",
        "Administrative version",
        "Administrative date",
        "Administrative date",
        "Administrative date",
        "Administrative date"
    ],

    "Planned Feature": [
        "Primary Key",
        "-",
        "age_years",
        "Used for conversion",
        "sex",
        "weight_kg",
        "Used for conversion",
        "reporter_country",
        "occur_country",
        "reporter_type",
        "To be evaluated",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-"
    ]
})

display(demo_feature_plan)

# ============================================================================
# Code Cell 9
# ============================================================================
# ==========================================================
# Investigate Reporting Variables
# ==========================================================

reporting_variables = ["rept_cod", "occp_cod"]

for variable in reporting_variables:

    print("\n" + "=" * 70)
    print(f"{variable} Distribution")
    print("=" * 70)

    frequency = (
        demo[variable]
        .value_counts(dropna=False)
        .rename_axis(variable)
        .reset_index(name="Count")
    )

    frequency["Percentage"] = (
        frequency["Count"] / len(demo) * 100
    ).round(2)

    display(frequency)

    print(f"\nNumber of unique categories: {demo[variable].nunique(dropna=True)}")
    print(f"Missing values: {demo[variable].isna().sum():,}")

# ============================================================================
# Code Cell 10
# ============================================================================
# ==========================================================
# DEMO Feature Engineering Specification
# ==========================================================

demo_feature_spec = pd.DataFrame({

    "Source Variable": [
        "primaryid",
        "age",
        "age_cod",
        "sex",
        "wt",
        "wt_cod",
        "reporter_country",
        "occr_country",
        "occp_cod",
        "rept_cod"
    ],

    "Feature Type": [
        "Identifier",
        "Numeric",
        "Support",
        "Categorical",
        "Numeric",
        "Support",
        "Categorical",
        "Categorical",
        "Categorical",
        "Categorical"
    ],

    "Clinical Importance": [
        "Merge key",
        "Patient age",
        "Age unit",
        "Patient sex",
        "Body weight",
        "Weight unit",
        "Reporter location",
        "Event location",
        "Reporter occupation",
        "Report type"
    ],

    "Missing (%)": [
        0.00,
        38.84,
        38.84,
        15.40,
        83.10,
        83.10,
        0.00,
        9.28,
        1.68,
        0.00
    ],

    "Transformation": [
        "Keep",
        "Convert to years",
        "Support conversion",
        "Replace missing with 'Unknown'",
        "Convert to kilograms",
        "Support conversion",
        "Keep",
        "Replace missing with 'Unknown'",
        "Replace missing with 'Unknown'",
        "Keep"
    ],

    "Encoding Strategy": [
        "None",
        "Numeric",
        "-",
        "One-Hot",
        "Numeric",
        "-",
        "One-Hot",
        "One-Hot",
        "One-Hot",
        "One-Hot"
    ],

    "Final Feature": [
        "primaryid",
        "age_years",
        "-",
        "sex",
        "weight_kg",
        "-",
        "reporter_country",
        "occur_country",
        "reporter_type",
        "report_type"
    ]
})

display(demo_feature_spec)

# ============================================================================
# Code Cell 11
# ============================================================================
# ==========================================================
# Create Working Copy of DEMO
# ==========================================================

demo_features = demo.copy()

# ==========================================================
# Standardize Patient Age
# ==========================================================

# Conversion factors to years
age_conversion = {
    "YR": 1,
    "MON": 1 / 12,
    "WK": 1 / 52.1429,
    "DY": 1 / 365.25,
    "HR": 1 / (24 * 365.25),
    "DEC": 10
}

# Create standardized age feature
demo_features["age_years"] = (
    demo_features["age"] *
    demo_features["age_cod"].map(age_conversion)
)

# Display summary statistics
print("=" * 70)
print("Age Standardization Summary")
print("=" * 70)

display(demo_features["age_years"].describe())

print("\nMissing age_years:", demo_features["age_years"].isna().sum())

# ============================================================================
# Code Cell 12
# ============================================================================
# ==========================================================
# Validate Standardized Age
# ==========================================================

print("=" * 70)
print("Age Validation")
print("=" * 70)

print("Minimum age :", demo_features["age_years"].min())
print("Maximum age :", demo_features["age_years"].max())

# Count implausible ages
implausible_age = (
    (demo_features["age_years"] < 0) |
    (demo_features["age_years"] > 120)
)

print("\nReports with age < 0:", (demo_features["age_years"] < 0).sum())
print("Reports with age > 120:", (demo_features["age_years"] > 120).sum())

print("\nTotal implausible ages:", implausible_age.sum())

# Display extreme values
display(
    demo_features.loc[
        implausible_age,
        ["primaryid", "age", "age_cod", "age_years"]
    ].head(20)
)

# ============================================================================
# Code Cell 13
# ============================================================================
# ==========================================================
# Handle Implausible Ages
# ==========================================================

# Replace biologically implausible ages with missing values
demo_features.loc[
    demo_features["age_years"] > 120,
    "age_years"
] = np.nan

print("=" * 70)
print("Age Cleaning Summary")
print("=" * 70)

print("Remaining ages >120:",
      (demo_features["age_years"] > 120).sum())

print("Missing age_years:",
      demo_features["age_years"].isna().sum())

print("\nSummary statistics after cleaning:")

display(demo_features["age_years"].describe())

# ============================================================================
# Code Cell 14
# ============================================================================
# ==========================================================
# Investigate Weight Units
# ==========================================================

print("=" * 70)
print("Weight Unit Distribution")
print("=" * 70)

weight_units = (
    demo_features["wt_cod"]
    .value_counts(dropna=False)
    .rename_axis("wt_cod")
    .reset_index(name="Count")
)

weight_units["Percentage"] = (
    weight_units["Count"] / len(demo_features) * 100
).round(2)

display(weight_units)

print("\nNumber of unique units:",
      demo_features["wt_cod"].nunique(dropna=True))

print("Missing weight units:",
      demo_features["wt_cod"].isna().sum())

# ============================================================================
# Code Cell 15
# ============================================================================
# ==========================================================
# Standardize Patient Weight
# ==========================================================

# Conversion factors to kilograms
weight_conversion = {
    "KG": 1.0,
    "LBS": 0.45359237
}

# Create standardized weight feature
demo_features["weight_kg"] = (
    demo_features["wt"] *
    demo_features["wt_cod"].map(weight_conversion)
)

print("=" * 70)
print("Weight Standardization Summary")
print("=" * 70)

display(demo_features["weight_kg"].describe())

print("\nMissing weight_kg:",
      demo_features["weight_kg"].isna().sum())

# ============================================================================
# Code Cell 16
# ============================================================================
# ==========================================================
# Validate Standardized Weight
# ==========================================================

print("=" * 70)
print("Weight Validation")
print("=" * 70)

print("Minimum weight :", demo_features["weight_kg"].min())
print("Maximum weight :", demo_features["weight_kg"].max())

# Define implausible weights
implausible_weight = (
    (demo_features["weight_kg"] <= 0) |
    (demo_features["weight_kg"] > 500)
)

print("\nReports with weight <= 0:",
      (demo_features["weight_kg"] <= 0).sum())

print("Reports with weight > 500:",
      (demo_features["weight_kg"] > 500).sum())

print("\nTotal implausible weights:",
      implausible_weight.sum())

display(
    demo_features.loc[
        implausible_weight,
        ["primaryid", "wt", "wt_cod", "weight_kg"]
    ].head(20)
)

# ============================================================================
# Code Cell 17
# ============================================================================
# ==========================================================
# Handle Implausible Weights
# ==========================================================

# Replace implausible weights with missing values
demo_features.loc[
    (demo_features["weight_kg"] <= 0) |
    (demo_features["weight_kg"] > 500),
    "weight_kg"
] = np.nan

print("=" * 70)
print("Weight Cleaning Summary")
print("=" * 70)

print("Remaining weights <= 0:",
      (demo_features["weight_kg"] <= 0).sum())

print("Remaining weights > 500:",
      (demo_features["weight_kg"] > 500).sum())

print("Missing weight_kg:",
      demo_features["weight_kg"].isna().sum())

print("\nSummary statistics after cleaning:")

display(demo_features["weight_kg"].describe())

# ============================================================================
# Code Cell 18
# ============================================================================
# ==========================================================
# Engineer Patient Sex
# ==========================================================

print("=" * 70)
print("Patient Sex")
print("=" * 70)

# Distribution before cleaning
print("\nOriginal Distribution")

sex_before = (
    demo_features["sex"]
    .value_counts(dropna=False)
    .rename_axis("sex")
    .reset_index(name="Count")
)

sex_before["Percentage"] = (
    sex_before["Count"] / len(demo_features) * 100
).round(2)

display(sex_before)

# ==========================================================
# Standardize patient sex
# ==========================================================

demo_features["sex"] = demo_features["sex"].replace({
    "UNK": "Unknown"
})

demo_features["sex"] = demo_features["sex"].fillna("Unknown")

print("\nDistribution After Cleaning")

sex_after = (
    demo_features["sex"]
    .value_counts(dropna=False)
    .rename_axis("sex")
    .reset_index(name="Count")
)

sex_after["Percentage"] = (
    sex_after["Count"] / len(demo_features) * 100
).round(2)

display(sex_after)

# ============================================================================
# Code Cell 19
# ============================================================================
# ==========================================================
# Investigate Reporter Country
# ==========================================================

print("=" * 70)
print("Reporter Country")
print("=" * 70)

country_before = (
    demo_features["reporter_country"]
    .value_counts(dropna=False)
    .rename_axis("reporter_country")
    .reset_index(name="Count")
)

country_before["Percentage"] = (
    country_before["Count"] /
    len(demo_features) * 100
).round(2)

display(country_before.head(20))

print("\nNumber of unique countries:",
      demo_features["reporter_country"].nunique(dropna=True))

print("Missing values:",
      demo_features["reporter_country"].isna().sum())

# ============================================================================
# Code Cell 20
# ============================================================================
# ==========================================================
# Investigate Occurrence Country
# ==========================================================

print("=" * 70)
print("Occurrence Country")
print("=" * 70)

occur_before = (
    demo_features["occr_country"]
    .value_counts(dropna=False)
    .rename_axis("occur_country")
    .reset_index(name="Count")
)

occur_before["Percentage"] = (
    occur_before["Count"] / len(demo_features) * 100
).round(2)

display(occur_before.head(20))

print("\nNumber of unique countries:",
      demo_features["occr_country"].nunique(dropna=True))

print("Missing values:",
      demo_features["occr_country"].isna().sum())

# ============================================================================
# Code Cell 21
# ============================================================================
# ==========================================================
# Standardize Occurrence Country
# ==========================================================

# Replace missing values with a standardized category
demo_features["occur_country"] = (
    demo_features["occr_country"]
    .fillna("Unknown")
)

print("=" * 70)
print("Occurrence Country After Cleaning")
print("=" * 70)

occur_after = (
    demo_features["occur_country"]
    .value_counts(dropna=False)
    .rename_axis("occur_country")
    .reset_index(name="Count")
)

occur_after["Percentage"] = (
    occur_after["Count"] /
    len(demo_features) * 100
).round(2)

display(occur_after.head(20))

print("\nNumber of unique categories:",
      demo_features["occur_country"].nunique())

# ============================================================================
# Code Cell 22
# ============================================================================
# ==========================================================
# Engineer Reporter Occupation
# ==========================================================

print("=" * 70)
print("Reporter Occupation")
print("=" * 70)

reporter_before = (
    demo_features["occp_cod"]
    .value_counts(dropna=False)
    .rename_axis("reporter_type")
    .reset_index(name="Count")
)

reporter_before["Percentage"] = (
    reporter_before["Count"] / len(demo_features) * 100
).round(2)

display(reporter_before)

# FDA occupation code mapping
occupation_map = {
    "MD": "Physician",
    "PH": "Pharmacist",
    "HP": "Health Professional",
    "CN": "Consumer",
    "LW": "Lawyer"
}

demo_features["reporter_type"] = (
    demo_features["occp_cod"]
    .map(occupation_map)
    .fillna("Unknown")
)

print("\n" + "=" * 70)
print("Reporter Occupation After Cleaning")
print("=" * 70)

reporter_after = (
    demo_features["reporter_type"]
    .value_counts()
    .rename_axis("reporter_type")
    .reset_index(name="Count")
)

reporter_after["Percentage"] = (
    reporter_after["Count"] / len(demo_features) * 100
).round(2)

display(reporter_after)

# ============================================================================
# Code Cell 23
# ============================================================================
# ==========================================================
# Engineer Report Type
# ==========================================================

print("=" * 70)
print("Report Type")
print("=" * 70)

# Distribution before cleaning
report_type_before = (
    demo_features["rept_cod"]
    .value_counts(dropna=False)
    .rename_axis("report_type")
    .reset_index(name="Count")
)

report_type_before["Percentage"] = (
    report_type_before["Count"] /
    len(demo_features) * 100
).round(2)

display(report_type_before)

# FDA report type mapping
report_type_map = {
    "EXP": "Expedited",
    "PER": "Periodic",
    "DIR": "Direct",
    "30DAY": "30-Day",
    "5DAY": "5-Day"
}

# Create standardized feature
demo_features["report_type"] = (
    demo_features["rept_cod"]
    .map(report_type_map)
)

print("\n" + "=" * 70)
print("Report Type After Cleaning")
print("=" * 70)

report_type_after = (
    demo_features["report_type"]
    .value_counts(dropna=False)
    .rename_axis("report_type")
    .reset_index(name="Count")
)

report_type_after["Percentage"] = (
    report_type_after["Count"] /
    len(demo_features) * 100
).round(2)

display(report_type_after)

print("\nMissing values:",
      demo_features["report_type"].isna().sum())

# ============================================================================
# Code Cell 24
# ============================================================================
# ==========================================================
# Create Final DEMO Feature Table
# ==========================================================

demo_features_final = demo_features[
    [
        "primaryid",
        "age_years",
        "weight_kg",
        "sex",
        "reporter_country",
        "occur_country",
        "reporter_type",
        "report_type"
    ]
].copy()

# ==========================================================
# Validate DEMO Feature Table
# ==========================================================

print("=" * 70)
print("DEMO Feature Table Validation")
print("=" * 70)

print(f"Rows                 : {demo_features_final.shape[0]:,}")
print(f"Columns              : {demo_features_final.shape[1]}")
print(f"Unique PRIMARYIDs    : {demo_features_final['primaryid'].nunique():,}")
print(f"Duplicate PRIMARYIDs : {demo_features_final.duplicated('primaryid').sum():,}")

print("\nMissing Values")
print("-" * 70)

missing_summary = (
    demo_features_final.isna()
    .sum()
    .to_frame("Missing")
)

missing_summary["Percent"] = (
    missing_summary["Missing"] /
    len(demo_features_final) * 100
).round(2)

display(missing_summary)

print("\nData Types")
print("-" * 70)

display(
    demo_features_final.dtypes.to_frame("Data Type")
)

# ============================================================================
# Code Cell 25
# ============================================================================
# ==========================================================
# Save DEMO Feature Table
# ==========================================================

demo_features_final = demo_features.copy()

demo_path = PROCESSED_DIR / "demo_features.csv"

demo_features_final.to_csv(
    demo_path,
    index=False
)

print("=" * 70)
print("DEMO feature table saved")
print("=" * 70)
print(demo_path)

# ============================================================================
# Code Cell 26
# ============================================================================
# ==========================================================
# Save Final DEMO Feature Table (Parquet)
# ==========================================================

demo_features_final = demo_features.copy()

# Output path
demo_parquet_path = PROCESSED_DIR / "demo_features.parquet"

# Save as Parquet
demo_features_final.to_parquet(
    demo_parquet_path,
    index=False
)

print("=" * 70)
print("DEMO Feature Table Saved Successfully (Parquet)")
print("=" * 70)

print(f"Rows    : {demo_features_final.shape[0]:,}")
print(f"Columns : {demo_features_final.shape[1]}")
print(f"File    : {demo_parquet_path}")

# ============================================================================
# Code Cell 27
# ============================================================================
# ==========================================================
# DRUG Data Quality Assessment
# ==========================================================

print("=" * 70)
print("DRUG Table Overview")
print("=" * 70)

print(f"Rows                    : {drug.shape[0]:,}")
print(f"Columns                 : {drug.shape[1]}")
print(f"Unique PRIMARYIDs       : {drug['primaryid'].nunique():,}")

duplicate_drug_rows = drug.duplicated().sum()
print(f"Duplicate full rows     : {duplicate_drug_rows:,}")

print("\nAverage drug records per report:")
print(round(drug.shape[0] / drug["primaryid"].nunique(), 2))

print("\n" + "=" * 70)
print("Missing Values")
print("=" * 70)

drug_missing = drug.isna().sum().to_frame("Missing")
drug_missing["Percent"] = (
    drug_missing["Missing"] / len(drug) * 100
).round(2)

drug_missing = drug_missing.sort_values("Missing", ascending=False)

display(drug_missing)

print("\n" + "=" * 70)
print("Drug Role Code Distribution")
print("=" * 70)

role_distribution = (
    drug["role_cod"]
    .value_counts(dropna=False)
    .rename_axis("role_cod")
    .reset_index(name="Count")
)

role_distribution["Percentage"] = (
    role_distribution["Count"] / len(drug) * 100
).round(2)

display(role_distribution)

# ============================================================================
# Code Cell 28
# ============================================================================
# ==========================================================
# DRUG Variable Selection Plan
# ==========================================================

drug_variable_plan = pd.DataFrame({

    "Original Variable": [

        "primaryid",
        "drug_seq",
        "drugname",
        "role_cod",
        "prod_ai",

        "route",
        "dose_amt",
        "dose_unit",
        "dose_freq",
        "dose_form",
        "dose_vbm",

        "cum_dose_chr",
        "cum_dose_unit",

        "dechal",
        "rechal",

        "nda_num",
        "lot_num",
        "exp_dt",

        "caseid",
        "val_vbm"

    ],

    "Decision": [

        "Keep",
        "Support",
        "Keep",
        "Keep",
        "Review",

        "Drop",
        "Drop",
        "Drop",
        "Drop",
        "Drop",
        "Drop",

        "Drop",
        "Drop",

        "Drop",
        "Drop",

        "Drop",
        "Drop",
        "Drop",

        "Drop",
        "Drop"

    ],

    "Reason": [

        "Merge key",

        "Maintains drug order",

        "Drug identity",

        "Drug role",

        "Potential drug class mapping",

        "High missingness",
        "High missingness",
        "High missingness",
        "High missingness",
        "High missingness",
        "High missingness",

        "High missingness",
        "High missingness",

        "High missingness",
        "High missingness",

        "Administrative",
        "Administrative",
        "Administrative",

        "Duplicate identifier",
        "Administrative"

    ],

    "Planned Use": [

        "Merge",

        "Support aggregation",

        "Drug counts / drug classes",

        "Role-based features",

        "Evaluate",

        "-",
        "-",
        "-",
        "-",
        "-",
        "-",

        "-",
        "-",

        "-",
        "-",

        "-",
        "-",
        "-",

        "-",
        "-"

    ]

})

display(drug_variable_plan)

# ============================================================================
# Code Cell 29
# ============================================================================
# ==========================================================
# DRUG Feature Engineering Specification
# ==========================================================

drug_feature_spec = pd.DataFrame({

    "Engineered Feature": [

        "total_drugs",
        "num_unique_drugs",

        "num_primary_suspect",
        "num_secondary_suspect",
        "num_concomitant",
        "num_interacting",

        "contains_insulin",
        "contains_sglt2",
        "contains_sulfonylurea"

    ],

    "Feature Type": [

        "Numeric",
        "Numeric",

        "Numeric",
        "Numeric",
        "Numeric",
        "Numeric",

        "Binary",
        "Binary",
        "Binary"

    ],

    "Source Variable(s)": [

        "drugname",
        "drugname",

        "role_cod",
        "role_cod",
        "role_cod",
        "role_cod",

        "drugname",
        "drugname",
        "drugname"

    ],

    "Aggregation Method": [

        "Count",
        "Unique Count",

        "Conditional Count",
        "Conditional Count",
        "Conditional Count",
        "Conditional Count",

        "Any Match",
        "Any Match",
        "Any Match"

    ],

    "Clinical Interpretation": [

        "Overall medication burden",

        "Medication diversity",

        "Primary suspected drugs",

        "Secondary suspected drugs",

        "Concomitant medications",

        "Interacting medications",

        "Exposure to insulin",

        "Exposure to SGLT2 inhibitors",

        "Exposure to sulfonylureas"

    ]

})

display(drug_feature_spec)

# ============================================================================
# Code Cell 30
# ============================================================================
# ==========================================================
# Initialize DRUG Feature Table
# ==========================================================

drug_features = pd.DataFrame({
    "primaryid": sorted(drug["primaryid"].unique())
})

# ==========================================================
# Engineer Total Drug Count
# ==========================================================

total_drugs = (
    drug
    .groupby("primaryid")
    .size()
    .rename("total_drugs")
    .reset_index()
)

drug_features = drug_features.merge(
    total_drugs,
    on="primaryid",
    how="left"
)

print("=" * 70)
print("Total Drug Count Summary")
print("=" * 70)

display(drug_features["total_drugs"].describe())

print("\nMissing values:",
      drug_features["total_drugs"].isna().sum())

# ============================================================================
# Code Cell 31
# ============================================================================
# ==========================================================
# Validate Total Drug Count
# ==========================================================

print("=" * 70)
print("Total Drug Count Validation")
print("=" * 70)

print("Minimum drugs per report:",
      drug_features["total_drugs"].min())

print("Maximum drugs per report:",
      drug_features["total_drugs"].max())

print("\nReports with >20 drugs:",
      (drug_features["total_drugs"] > 20).sum())

print("Reports with >50 drugs:",
      (drug_features["total_drugs"] > 50).sum())

print("Reports with >100 drugs:",
      (drug_features["total_drugs"] > 100).sum())

print("\nTop 20 reports with the largest drug counts")

display(
    drug_features
    .sort_values("total_drugs", ascending=False)
    .head(20)
)

# ============================================================================
# Code Cell 32
# ============================================================================
# ==========================================================
# Engineer Unique Drug Count
# ==========================================================

unique_drugs = (
    drug
    .groupby("primaryid")["drugname"]
    .nunique()
    .rename("num_unique_drugs")
    .reset_index()
)

drug_features = drug_features.merge(
    unique_drugs,
    on="primaryid",
    how="left"
)

print("=" * 70)
print("Unique Drug Count Summary")
print("=" * 70)

display(drug_features["num_unique_drugs"].describe())

print("\nMissing values:",
      drug_features["num_unique_drugs"].isna().sum())

# ============================================================================
# Code Cell 33
# ============================================================================
# ==========================================================
# Compare Total vs Unique Drug Counts
# ==========================================================

# Calculate duplicate drug records
drug_features["duplicate_drug_records"] = (
    drug_features["total_drugs"] -
    drug_features["num_unique_drugs"]
)

print("=" * 70)
print("Comparison of Drug Count Features")
print("=" * 70)

print("Reports where total == unique:",
      (drug_features["duplicate_drug_records"] == 0).sum())

print("Reports with repeated drug records:",
      (drug_features["duplicate_drug_records"] > 0).sum())

print("\nSummary of duplicate drug records")

display(
    drug_features["duplicate_drug_records"].describe()
)

print("\nTop 20 reports with largest differences")

display(
    drug_features
    .sort_values("duplicate_drug_records", ascending=False)
    .head(20)
)

# ============================================================================
# Code Cell 34
# ============================================================================
# ==========================================================
# Engineer Drug Role Features
# ==========================================================

# Count drug roles for each report
drug_role_features = (
    drug
    .groupby(["primaryid", "role_cod"])
    .size()
    .unstack(fill_value=0)
)

# Rename columns
drug_role_features = drug_role_features.rename(columns={
    "PS": "num_primary_suspect",
    "SS": "num_secondary_suspect",
    "C": "num_concomitant",
    "I": "num_interacting"
})

# Ensure all expected columns exist
expected_columns = [
    "num_primary_suspect",
    "num_secondary_suspect",
    "num_concomitant",
    "num_interacting"
]

for col in expected_columns:
    if col not in drug_role_features.columns:
        drug_role_features[col] = 0

drug_role_features = (
    drug_role_features[expected_columns]
    .reset_index()
)

# Merge into DRUG feature table
drug_features = drug_features.merge(
    drug_role_features,
    on="primaryid",
    how="left"
)

print("=" * 70)
print("Drug Role Features Created")
print("=" * 70)

display(drug_features[expected_columns].describe())

print("\nMissing values")

display(
    drug_features[expected_columns]
    .isna()
    .sum()
    .to_frame("Missing")
)

# ============================================================================
# Code Cell 35
# ============================================================================
# ==========================================================
# Validate Drug Role Features
# ==========================================================

# Calculate total role counts
drug_features["role_sum"] = (
    drug_features["num_primary_suspect"] +
    drug_features["num_secondary_suspect"] +
    drug_features["num_concomitant"] +
    drug_features["num_interacting"]
)

# Compare with total drug count
drug_features["role_check"] = (
    drug_features["role_sum"] ==
    drug_features["total_drugs"]
)

print("=" * 70)
print("Drug Role Validation")
print("=" * 70)

print("Reports passing validation:",
      drug_features["role_check"].sum())

print("Reports failing validation:",
      (~drug_features["role_check"]).sum())

print("\nValidation rate:")

print(
    f"{drug_features['role_check'].mean() * 100:.2f}%"
)

# Display any mismatches
mismatches = drug_features.loc[
    ~drug_features["role_check"],
    [
        "primaryid",
        "total_drugs",
        "role_sum",
        "num_primary_suspect",
        "num_secondary_suspect",
        "num_concomitant",
        "num_interacting"
    ]
]

if len(mismatches) > 0:
    print("\nReports with mismatched counts:")
    display(mismatches.head(20))
else:
    print("\n✓ All reports passed validation.")

# ============================================================================
# Code Cell 36
# ============================================================================
# ==========================================================
# Remove Temporary Validation Columns
# ==========================================================

drug_features.drop(
    columns=["role_sum", "role_check"],
    inplace=True
)

# ============================================================================
# Code Cell 37
# ============================================================================
# ==========================================================
# Investigate Drug Names
# ==========================================================

print("=" * 70)
print("Drug Name Investigation")
print("=" * 70)

print(f"Total drug records : {len(drug):,}")
print(f"Unique drug names  : {drug['drugname'].nunique():,}")

print("\nMost Common Drug Names")
print("-" * 70)

top_drugs = (
    drug["drugname"]
    .value_counts()
    .head(50)
    .rename_axis("drugname")
    .reset_index(name="Count")
)

display(top_drugs)

print("\nDrug Name Examples")

sample_names = (
    drug["drugname"]
    .drop_duplicates()
    .sort_values()
    .head(50)
)

display(sample_names.reset_index(drop=True))

# ============================================================================
# Code Cell 38
# ============================================================================
# ==========================================================
# Standardize Drug Names
# ==========================================================

import re

def standardize_drug_name(name):
    """
    Standardize FAERS drug names for drug class mapping.
    """
    if pd.isna(name):
        return np.nan

    # Convert to uppercase
    name = str(name).upper()

    # Remove leading/trailing whitespace
    name = name.strip()

    # Remove extra spaces
    name = re.sub(r"\s+", " ", name)

    # Remove common punctuation
    name = re.sub(r"[^\w\s]", " ", name)

    # Collapse spaces again after punctuation removal
    name = re.sub(r"\s+", " ", name).strip()

    return name


drug["drugname_std"] = (
    drug["drugname"]
    .apply(standardize_drug_name)
)

print("=" * 70)
print("Drug Name Standardization")
print("=" * 70)

comparison = pd.DataFrame({
    "Original": drug["drugname"].head(20),
    "Standardized": drug["drugname_std"].head(20)
})

display(comparison)

print("\nUnique standardized drug names:",
      drug["drugname_std"].nunique())

# ============================================================================
# Code Cell 39
# ============================================================================
# ==========================================================
# Drug Class Dictionaries
# ==========================================================

# --------------------------
# SGLT2 Inhibitors
# --------------------------

SGLT2_DRUGS = {

    # Generic names
    "EMPAGLIFLOZIN",
    "DAPAGLIFLOZIN",
    "CANAGLIFLOZIN",
    "ERTUGLIFLOZIN",
    "BEXAGLIFLOZIN",

    # Brand names
    "JARDIANCE",
    "FARXIGA",
    "INVOKANA",
    "STEGLATRO",
    "BRENZAVVY"
}

# --------------------------
# Sulfonylureas
# --------------------------

SULFONYLUREA_DRUGS = {

    # Generic
    "GLIMEPIRIDE",
    "GLIPIZIDE",
    "GLYBURIDE",
    "GLIBENCLAMIDE",
    "GLICLAZIDE",
    "CHLORPROPAMIDE",
    "TOLAZAMIDE",
    "TOLBUTAMIDE",

    # Brand
    "AMARYL",
    "DIABETA",
    "MICRONASE",
    "GLYNASE"
}

# --------------------------
# Insulin Products
# --------------------------

INSULIN_DRUGS = {

    # Generic
    "INSULIN",
    "INSULIN ASPART",
    "INSULIN GLARGINE",
    "INSULIN LISPRO",
    "INSULIN DETEMIR",
    "INSULIN DEGLUDEC",
    "INSULIN GLULISINE",
    "INSULIN HUMAN",
    "INSULIN NPH",
    "INSULIN REGULAR",
    "INSULIN NOS",

    # Brand
    "NOVOLOG",
    "NOVORAPID",
    "LANTUS",
    "LEVEMIR",
    "TRESIBA",
    "HUMALOG",
    "APIDRA",
    "HUMULIN",
    "NOVOLIN",
    "BASAGLAR",
    "FIASP",
    "TOUJEO"
}

print("=" * 70)
print("Drug Class Dictionary Summary")
print("=" * 70)

print(f"SGLT2 drugs        : {len(SGLT2_DRUGS)}")
print(f"Insulin products   : {len(INSULIN_DRUGS)}")
print(f"Sulfonylureas      : {len(SULFONYLUREA_DRUGS)}")

# ============================================================================
# Code Cell 40
# ============================================================================
# ==========================================================
# Engineer Drug Class Indicators - Improved Version
# ==========================================================

def contains_drug_class(drug_names, drug_dictionary):
    """
    Return 1 if any standardized drug name contains
    a keyword from the therapeutic drug class dictionary.
    """
    for name in drug_names:
        for keyword in drug_dictionary:
            if keyword in name:
                return 1
    return 0


# Aggregate standardized drug names
drug_name_lists = (
    drug
    .groupby("primaryid")["drugname_std"]
    .apply(set)
)

# Create report-level indicators
drug_class_features = pd.DataFrame({
    "primaryid": drug_name_lists.index,

    "contains_insulin": drug_name_lists.apply(
        lambda x: contains_drug_class(x, INSULIN_DRUGS)
    ),

    "contains_sglt2": drug_name_lists.apply(
        lambda x: contains_drug_class(x, SGLT2_DRUGS)
    ),

    "contains_sulfonylurea": drug_name_lists.apply(
        lambda x: contains_drug_class(x, SULFONYLUREA_DRUGS)
    )
}).reset_index(drop=True)

# Drop old drug-class indicator columns if they already exist
drug_features = drug_features.drop(
    columns=[
        "contains_insulin",
        "contains_sglt2",
        "contains_sulfonylurea"
    ],
    errors="ignore"
)

# Merge updated indicators into DRUG feature table
drug_features = drug_features.merge(
    drug_class_features,
    on="primaryid",
    how="left"
)

print("=" * 70)
print("Drug Class Indicators - Improved Matching")
print("=" * 70)

summary = (
    drug_features[
        [
            "contains_insulin",
            "contains_sglt2",
            "contains_sulfonylurea"
        ]
    ]
    .sum()
    .to_frame("Reports")
)

summary["Percent"] = (
    summary["Reports"] /
    len(drug_features) * 100
).round(2)

display(summary)

print("\nMissing values")

display(
    drug_features[
        [
            "contains_insulin",
            "contains_sglt2",
            "contains_sulfonylurea"
        ]
    ]
    .isna()
    .sum()
    .to_frame("Missing")
)

# ============================================================================
# Code Cell 41
# ============================================================================
# ==========================================================
# Validate Final DRUG Feature Table
# ==========================================================

print("=" * 70)
print("DRUG Feature Table Validation")
print("=" * 70)

print(f"Rows                 : {drug_features.shape[0]:,}")
print(f"Columns              : {drug_features.shape[1]}")
print(f"Unique PRIMARYIDs    : {drug_features['primaryid'].nunique():,}")
print(f"Duplicate PRIMARYIDs : {drug_features.duplicated('primaryid').sum():,}")

print("\nMissing Values")
print("-" * 70)

missing_summary = (
    drug_features.isna()
    .sum()
    .to_frame("Missing")
)

missing_summary["Percent"] = (
    missing_summary["Missing"] /
    len(drug_features) * 100
).round(2)

display(missing_summary)

print("\nData Types")
print("-" * 70)

display(
    drug_features.dtypes.to_frame("Data Type")
)

print("\nFirst Five Rows")
print("-" * 70)

display(drug_features.head())

# ============================================================================
# Code Cell 42
# ============================================================================
# ==========================================================
# Remove Temporary / Duplicate Columns
# ==========================================================

drug_features = drug_features.drop(
    columns=[
        "contains_insulin_x",
        "contains_sglt2_x",
        "contains_sulfonylurea_x",

        "contains_insulin_y",
        "contains_sglt2_y",
        "contains_sulfonylurea_y",

        # Validation-only feature
        "duplicate_drug_records"
    ],
    errors="ignore"
)

print("=" * 70)
print("Columns After Cleanup")
print("=" * 70)

display(drug_features.columns.to_frame("Feature"))

# ============================================================================
# Code Cell 43
# ============================================================================
# ==========================================================
# Save DRUG Feature Table
# ==========================================================

drug_features_final = drug_features.copy()

drug_path = PROCESSED_DIR / "drug_features.csv"

drug_features_final.to_csv(
    drug_path,
    index=False
)

print("=" * 70)
print("DRUG feature table saved")
print("=" * 70)
print(drug_path)

# ============================================================================
# Code Cell 44
# ============================================================================
# ==========================================================
# Save Final DRUG Feature Table (Parquet)
# ==========================================================

drug_features_final = drug_features.copy()

# Output path
drug_parquet_path = PROCESSED_DIR / "drug_features.parquet"

# Save as Parquet
drug_features_final.to_parquet(
    drug_parquet_path,
    index=False
)

print("=" * 70)
print("DRUG Feature Table Saved Successfully (Parquet)")
print("=" * 70)

print(f"Rows    : {drug_features_final.shape[0]:,}")
print(f"Columns : {drug_features_final.shape[1]}")
print(f"File    : {drug_parquet_path}")

# ============================================================================
# Code Cell 45
# ============================================================================
# ==========================================================
# REAC Data Quality Assessment
# ==========================================================

print("=" * 70)
print("REAC Table Overview")
print("=" * 70)

print(f"Rows                    : {reac.shape[0]:,}")
print(f"Columns                 : {reac.shape[1]}")
print(f"Unique PRIMARYIDs       : {reac['primaryid'].nunique():,}")

duplicate_reac_rows = reac.duplicated().sum()
print(f"Duplicate full rows     : {duplicate_reac_rows:,}")

print("\nAverage reaction records per report:")
print(round(reac.shape[0] / reac["primaryid"].nunique(), 2))

print("\n" + "=" * 70)
print("Missing Values")
print("=" * 70)

reac_missing = reac.isna().sum().to_frame("Missing")
reac_missing["Percent"] = (
    reac_missing["Missing"] / len(reac) * 100
).round(2)

reac_missing = reac_missing.sort_values("Missing", ascending=False)

display(reac_missing)

print("\n" + "=" * 70)
print("Most Common Reaction Terms")
print("=" * 70)

top_reactions = (
    reac["pt"]
    .value_counts(dropna=False)
    .head(30)
    .rename_axis("pt")
    .reset_index(name="Count")
)

top_reactions["Percentage"] = (
    top_reactions["Count"] / len(reac) * 100
).round(2)

display(top_reactions)

# ============================================================================
# Code Cell 46
# ============================================================================
# ==========================================================
# Remove Duplicate REAC Records
# ==========================================================

print("=" * 70)
print("Removing Duplicate REAC Records")
print("=" * 70)

rows_before = len(reac)

reac = reac.drop_duplicates()

rows_after = len(reac)

duplicates_removed = rows_before - rows_after

print(f"Rows before cleaning : {rows_before:,}")
print(f"Rows after cleaning  : {rows_after:,}")
print(f"Duplicates removed   : {duplicates_removed:,}")

print("\nUnique PRIMARYIDs:",
      reac["primaryid"].nunique())

# ============================================================================
# Code Cell 47
# ============================================================================
# ==========================================================
# REAC Variable Selection
# ==========================================================

reac_variable_selection = pd.DataFrame({

    "Original Variable": [
        "primaryid",
        "pt",
        "drug_rec_act",
        "caseid"
    ],

    "Decision": [
        "Keep",
        "Keep",
        "Drop",
        "Drop"
    ],

    "Reason": [
        "Merge key",
        "Clinical reaction term (MedDRA Preferred Term)",
        "99.93% missing values",
        "Administrative identifier"
    ],

    "Planned Use": [
        "Merge",
        "Reaction feature engineering",
        "-",
        "-"
    ]

})

display(reac_variable_selection)

# ============================================================================
# Code Cell 48
# ============================================================================
# ==========================================================
# REAC Feature Specification
# ==========================================================

reac_feature_specification = pd.DataFrame({

    "Engineered Feature": [

        "total_reactions",
        "num_unique_reactions",

        "flag_AKI",
        "flag_DKA",
        "flag_hypoglycemia",
        "flag_lactic_acidosis",
        "flag_amputation",
        "flag_genital_infection"

    ],

    "Feature Type": [

        "Numeric",
        "Numeric",

        "Binary",
        "Binary",
        "Binary",
        "Binary",
        "Binary",
        "Binary"

    ],

    "Source Variable": [

        "pt",
        "pt",

        "pt",
        "pt",
        "pt",
        "pt",
        "pt",
        "pt"

    ],

    "Aggregation Method": [

        "Count",
        "Unique Count",

        "Any Match",
        "Any Match",
        "Any Match",
        "Any Match",
        "Any Match",
        "Any Match"

    ],

    "Clinical Interpretation": [

        "Overall adverse reaction burden",
        "Reaction diversity",

        "Acute kidney injury reported",
        "Diabetic ketoacidosis reported",
        "Hypoglycemia reported",
        "Lactic acidosis reported",
        "Lower limb amputation reported",
        "Genital infection reported"

    ]

})

display(reac_feature_specification)

# ============================================================================
# Code Cell 49
# ============================================================================
# ==========================================================
# Investigate Disease-Specific Reaction Terms
# ==========================================================

keywords = [
    "kidney",
    "renal",
    "hypogly",
    "keto",
    "lactic",
    "amput",
    "genital",
    "infection"
]

for keyword in keywords:

    print("=" * 70)
    print(f"Reaction terms containing '{keyword}'")
    print("=" * 70)

    matches = (
        reac.loc[
            reac["pt"].str.contains(
                keyword,
                case=False,
                na=False
            ),
            "pt"
        ]
        .drop_duplicates()
        .sort_values()
    )

    display(
        pd.DataFrame(
            {"Reaction Term": matches.values}
        )
    )

# ============================================================================
# Code Cell 50
# ============================================================================
# ==========================================================
# Disease-Specific MedDRA Dictionaries
# ==========================================================

AKI_TERMS = {
    "Acute kidney injury",
    "Acute renal failure",
    "Renal failure",
    "Renal impairment"
}

DKA_TERMS = {
    "Diabetic ketoacidosis",
    "Euglycaemic diabetic ketoacidosis",
    "Diabetic ketoacidotic hyperglycaemic coma",
    "Ketoacidosis"
}

HYPOGLYCEMIA_TERMS = {
    "Hypoglycaemia",
    "Hypoglycaemic coma",
    "Hypoglycaemic seizure",
    "Hypoglycaemic unconsciousness",
    "Hypoglycaemic encephalopathy",
    "Hypoglycaemia unawareness"
}

LACTIC_ACIDOSIS_TERMS = {
    "Lactic acidosis"
}

AMPUTATION_TERMS = {
    "Amputation",
    "Toe amputation",
    "Foot amputation",
    "Leg amputation",
    "Limb amputation",
    "Finger amputation",
    "Arm amputation"
}

GENITAL_INFECTION_TERMS = {
    "Genital infection",
    "Genital candidiasis",
    "Vulvovaginal candidiasis",
    "Vaginal infection",
    "Balanitis",
    "Balanoposthitis",
    "Vulvovaginal mycotic infection",
    "Fungal genital infection"
}

print("=" * 70)
print("Disease Reaction Dictionary Summary")
print("=" * 70)

print(f"AKI terms                 : {len(AKI_TERMS)}")
print(f"DKA terms                 : {len(DKA_TERMS)}")
print(f"Hypoglycemia terms        : {len(HYPOGLYCEMIA_TERMS)}")
print(f"Lactic acidosis terms     : {len(LACTIC_ACIDOSIS_TERMS)}")
print(f"Amputation terms          : {len(AMPUTATION_TERMS)}")
print(f"Genital infection terms   : {len(GENITAL_INFECTION_TERMS)}")

# ============================================================================
# Code Cell 51
# ============================================================================
# ==========================================================
# Engineer General Reaction Features
# ==========================================================

# Total reaction count
total_reactions = (
    reac
    .groupby("primaryid")
    .size()
    .rename("total_reactions")
)

# Number of unique reaction terms
unique_reactions = (
    reac
    .groupby("primaryid")["pt"]
    .nunique()
    .rename("num_unique_reactions")
)

# Create report-level feature table
reac_features = pd.concat(
    [
        total_reactions,
        unique_reactions
    ],
    axis=1
).reset_index()

print("=" * 70)
print("General Reaction Features")
print("=" * 70)

display(
    reac_features[
        [
            "total_reactions",
            "num_unique_reactions"
        ]
    ].describe()
)

print("\nMissing values")

display(
    reac_features[
        [
            "total_reactions",
            "num_unique_reactions"
        ]
    ]
    .isna()
    .sum()
    .to_frame("Missing")
)

# ============================================================================
# Code Cell 52
# ============================================================================
# ==========================================================
# Validate Total Reaction Count
# ==========================================================

print("=" * 70)
print("Total Reaction Count Validation")
print("=" * 70)

print(f"Minimum reactions per report : {reac_features['total_reactions'].min()}")
print(f"Maximum reactions per report : {reac_features['total_reactions'].max()}")

print()

print(
    "Reports with >20 reactions:",
    (reac_features["total_reactions"] > 20).sum()
)

print(
    "Reports with >50 reactions:",
    (reac_features["total_reactions"] > 50).sum()
)

print(
    "Reports with >100 reactions:",
    (reac_features["total_reactions"] > 100).sum()
)

print("\nTop 20 reports with largest reaction counts")

display(

    reac_features
    .sort_values(
        "total_reactions",
        ascending=False
    )
    .head(20)

)

# ============================================================================
# Code Cell 53
# ============================================================================
# ==========================================================
# Compare Total vs Unique Reaction Counts
# ==========================================================

reac_features["duplicate_reaction_records"] = (
    reac_features["total_reactions"] -
    reac_features["num_unique_reactions"]
)

print("=" * 70)
print("Comparison of Reaction Count Features")
print("=" * 70)

print(
    "Reports where total == unique:",
    (reac_features["duplicate_reaction_records"] == 0).sum()
)

print(
    "Reports with repeated reaction terms:",
    (reac_features["duplicate_reaction_records"] > 0).sum()
)

print("\nSummary of repeated reaction terms")

display(
    reac_features["duplicate_reaction_records"].describe()
)

print("\nTop 20 reports with largest differences")

display(
    reac_features
    .sort_values(
        "duplicate_reaction_records",
        ascending=False
    )
    .head(20)
)

# ============================================================================
# Code Cell 54
# ============================================================================
# ==========================================================
# Engineer Disease-Specific Reaction Flags
# ==========================================================

# Aggregate reaction terms by report
reaction_lists = (
    reac
    .groupby("primaryid")["pt"]
    .apply(set)
)

# Helper function
def contains_reaction(reactions, dictionary):
    """
    Return 1 if any MedDRA Preferred Term
    belongs to the specified reaction dictionary.
    """
    return int(any(pt in dictionary for pt in reactions))


# Create binary reaction flags
reaction_flags = pd.DataFrame({

    "primaryid": reaction_lists.index,

    "flag_AKI":
        reaction_lists.apply(
            lambda x: contains_reaction(x, AKI_TERMS)
        ),

    "flag_DKA":
        reaction_lists.apply(
            lambda x: contains_reaction(x, DKA_TERMS)
        ),

    "flag_hypoglycemia":
        reaction_lists.apply(
            lambda x: contains_reaction(x, HYPOGLYCEMIA_TERMS)
        ),

    "flag_lactic_acidosis":
        reaction_lists.apply(
            lambda x: contains_reaction(x, LACTIC_ACIDOSIS_TERMS)
        ),

    "flag_amputation":
        reaction_lists.apply(
            lambda x: contains_reaction(x, AMPUTATION_TERMS)
        ),

    "flag_genital_infection":
        reaction_lists.apply(
            lambda x: contains_reaction(x, GENITAL_INFECTION_TERMS)
        )

}).reset_index(drop=True)


# Merge into REAC feature table
reac_features = reac_features.merge(
    reaction_flags,
    on="primaryid",
    how="left"
)

print("=" * 70)
print("Disease-Specific Reaction Flags")
print("=" * 70)

summary = (
    reac_features[
        [
            "flag_AKI",
            "flag_DKA",
            "flag_hypoglycemia",
            "flag_lactic_acidosis",
            "flag_amputation",
            "flag_genital_infection"
        ]
    ]
    .sum()
    .to_frame("Reports")
)

summary["Percent"] = (
    summary["Reports"] /
    len(reac_features) * 100
).round(2)

display(summary)

print("\nMissing Values")

display(
    reac_features[
        [
            "flag_AKI",
            "flag_DKA",
            "flag_hypoglycemia",
            "flag_lactic_acidosis",
            "flag_amputation",
            "flag_genital_infection"
        ]
    ]
    .isna()
    .sum()
    .to_frame("Missing")
)

# ============================================================================
# Code Cell 55
# ============================================================================
# ==========================================================
# Validate Disease-Specific Reaction Dictionaries
# ==========================================================

reaction_dictionaries = {
    "AKI": AKI_TERMS,
    "DKA": DKA_TERMS,
    "Hypoglycemia": HYPOGLYCEMIA_TERMS,
    "Lactic Acidosis": LACTIC_ACIDOSIS_TERMS,
    "Amputation": AMPUTATION_TERMS,
    "Genital Infection": GENITAL_INFECTION_TERMS
}

print("=" * 70)
print("Disease-Specific Reaction Dictionary Validation")
print("=" * 70)

for name, dictionary in reaction_dictionaries.items():

    matched = sorted(
        set(reac.loc[reac["pt"].isin(dictionary), "pt"])
    )

    print(f"\n{name}")
    print("-" * 50)
    print(f"Matched PTs: {len(matched)}")

    display(
        pd.DataFrame({
            "Matched MedDRA PT": matched
        })
    )

# ============================================================================
# Code Cell 56
# ============================================================================
# ==========================================================
# Validate Final REAC Feature Table
# ==========================================================

# Remove validation-only column if it exists
reac_features = reac_features.drop(
    columns=["duplicate_reaction_records"],
    errors="ignore"
)

print("=" * 70)
print("REAC Feature Table Validation")
print("=" * 70)

print(f"Rows                 : {reac_features.shape[0]:,}")
print(f"Columns              : {reac_features.shape[1]}")
print(f"Unique PRIMARYIDs    : {reac_features['primaryid'].nunique():,}")
print(f"Duplicate PRIMARYIDs : {reac_features.duplicated('primaryid').sum():,}")

print("\nMissing Values")
print("-" * 70)

missing = (
    reac_features
    .isna()
    .sum()
    .to_frame("Missing")
)

missing["Percent"] = (
    missing["Missing"] /
    len(reac_features) * 100
).round(2)

display(missing)

print("\nData Types")
print("-" * 70)

display(
    reac_features.dtypes.to_frame("Data Type")
)

print("\nFirst Five Rows")
print("-" * 70)

display(
    reac_features.head()
)

# ============================================================================
# Code Cell 57
# ============================================================================
# ==========================================================
# Save Final REAC Feature Table (CSV + Parquet)
# ==========================================================

reac_features_final = reac_features.copy()

# Output paths
reac_csv_path = PROCESSED_DIR / "reac_features.csv"
reac_parquet_path = PROCESSED_DIR / "reac_features.parquet"

# Save CSV
reac_features_final.to_csv(
    reac_csv_path,
    index=False
)

# Save Parquet
reac_features_final.to_parquet(
    reac_parquet_path,
    index=False
)

print("=" * 70)
print("REAC Feature Table Saved Successfully")
print("=" * 70)

print(f"Rows    : {reac_features_final.shape[0]:,}")
print(f"Columns : {reac_features_final.shape[1]}")
print(f"CSV     : {reac_csv_path}")
print(f"Parquet : {reac_parquet_path}")

# ============================================================================
# Code Cell 58
# ============================================================================
# ==========================================================
# INDI Data Quality Assessment
# ==========================================================

print("=" * 70)
print("INDI Table Overview")
print("=" * 70)

print(f"Rows                    : {indi.shape[0]:,}")
print(f"Columns                 : {indi.shape[1]}")
print(f"Unique PRIMARYIDs       : {indi['primaryid'].nunique():,}")

duplicate_rows = indi.duplicated().sum()

print(f"Duplicate full rows     : {duplicate_rows:,}")

print("\nAverage indication records per report:")

print(
    round(
        indi.shape[0] /
        indi["primaryid"].nunique(),
        2
    )
)

print("\n" + "=" * 70)
print("Missing Values")
print("=" * 70)

missing = (
    indi
    .isna()
    .sum()
    .to_frame("Missing")
)

missing["Percent"] = (
    missing["Missing"] /
    len(indi) * 100
).round(2)

display(
    missing.sort_values(
        "Missing",
        ascending=False
    )
)

print("\n" + "=" * 70)
print("Most Common Indications")
print("=" * 70)

top_indications = (
    indi["indi_pt"]
    .value_counts()
    .head(30)
    .rename_axis("indi_pt")
    .reset_index(name="Count")
)

top_indications["Percentage"] = (
    top_indications["Count"] /
    len(indi) * 100
).round(2)

display(top_indications)

# ============================================================================
# Code Cell 59
# ============================================================================
# ==========================================================
# Remove Duplicate INDI Records
# ==========================================================

print("=" * 70)
print("Removing Duplicate INDI Records")
print("=" * 70)

rows_before = len(indi)

indi = indi.drop_duplicates()

rows_after = len(indi)

duplicates_removed = rows_before - rows_after

print(f"Rows before cleaning : {rows_before:,}")
print(f"Rows after cleaning  : {rows_after:,}")
print(f"Duplicates removed   : {duplicates_removed:,}")

print("\nUnique PRIMARYIDs:",
      indi["primaryid"].nunique())

# ============================================================================
# Code Cell 60
# ============================================================================
# ==========================================================
# Save Cleaned INDI Table (CSV + Parquet)
# ==========================================================

indi_csv_path = PROCESSED_DIR / "indi_clean.csv"
indi_parquet_path = PROCESSED_DIR / "indi_clean.parquet"

indi.to_csv(
    indi_csv_path,
    index=False
)

indi.to_parquet(
    indi_parquet_path,
    index=False
)

print("=" * 70)
print("Cleaned INDI Table Saved Successfully")
print("=" * 70)

print(f"Rows    : {indi.shape[0]:,}")
print(f"Columns : {indi.shape[1]}")
print(f"CSV     : {indi_csv_path}")
print(f"Parquet : {indi_parquet_path}")

# ============================================================================
# Code Cell 61
# ============================================================================
# ==========================================================
# INDI Variable Selection
# ==========================================================

indi_variable_selection = pd.DataFrame({

    "Original Variable": [
        "primaryid",
        "indi_pt",
        "indi_drug_seq",
        "caseid"
    ],

    "Decision": [
        "Keep",
        "Keep",
        "Drop",
        "Drop"
    ],

    "Reason": [
        "Merge key",
        "Clinical indication (MedDRA Preferred Term)",
        "Administrative sequence identifier",
        "Administrative identifier"
    ],

    "Planned Use": [
        "Merge",
        "Indication feature engineering",
        "-",
        "-"
    ]

})

display(indi_variable_selection)

# ============================================================================
# Code Cell 62
# ============================================================================
# ==========================================================
# INDI Feature Specification
# ==========================================================

indi_feature_specification = pd.DataFrame({

    "Engineered Feature":[

        "total_indications",
        "num_unique_indications",

        "flag_type2_diabetes",
        "flag_type1_diabetes",
        "flag_diabetes",
        "flag_hypertension",
        "flag_chronic_kidney_disease",
        "flag_heart_failure",
        "flag_obesity",
        "flag_unknown_indication"

    ],

    "Feature Type":[

        "Numeric",
        "Numeric",

        "Binary",
        "Binary",
        "Binary",
        "Binary",
        "Binary",
        "Binary",
        "Binary",
        "Binary"

    ],

    "Source Variable":[
        "indi_pt",
        "indi_pt",
        "indi_pt",
        "indi_pt",
        "indi_pt",
        "indi_pt",
        "indi_pt",
        "indi_pt",
        "indi_pt",
        "indi_pt"
    ],

    "Aggregation Method":[

        "Count",
        "Unique Count",

        "Any Match",
        "Any Match",
        "Any Match",
        "Any Match",
        "Any Match",
        "Any Match",
        "Any Match",
        "Any Match"

    ],

    "Clinical Interpretation":[

        "Overall indication burden",
        "Indication diversity",

        "Type 2 diabetes",
        "Type 1 diabetes",
        "Diabetes",
        "Hypertension",
        "Chronic kidney disease",
        "Heart failure",
        "Obesity",
        "Unknown indication reported"

    ]

})

display(indi_feature_specification)

# ============================================================================
# Code Cell 63
# ============================================================================
# ==========================================================
# Investigate Disease-Specific Indication Terms
# ==========================================================

keywords = [
    "diabetes",
    "hypertension",
    "kidney",
    "renal",
    "heart",
    "failure",
    "obesity"
]

for keyword in keywords:

    print("=" * 70)
    print(f"Indications containing '{keyword}'")
    print("=" * 70)

    matches = (
        indi.loc[
            indi["indi_pt"].str.contains(
                keyword,
                case=False,
                na=False
            ),
            "indi_pt"
        ]
        .drop_duplicates()
        .sort_values()
    )

    display(
        pd.DataFrame({
            "Indication Term": matches.values
        })
    )

# ============================================================================
# Code Cell 64
# ============================================================================
# ==========================================================
# Disease-Specific Indication Dictionaries
# ==========================================================

TYPE2_DIABETES_TERMS = {
    "Type 2 diabetes mellitus"
}

TYPE1_DIABETES_TERMS = {
    "Type 1 diabetes mellitus"
}

ANY_DIABETES_TERMS = {
    "Diabetes mellitus",
    "Type 1 diabetes mellitus",
    "Type 2 diabetes mellitus"
}

HYPERTENSION_TERMS = {
    "Hypertension"
}

CKD_TERMS = {
    "Chronic kidney disease"
}

HEART_FAILURE_TERMS = {
    "Heart failure",
    "Cardiac failure",
    "Congestive heart failure",
    "Congestive cardiac failure",
    "Left ventricular failure",
    "Right ventricular failure"
}

OBESITY_TERMS = {
    "Obesity",
    "Central obesity"
}

UNKNOWN_INDICATION_TERMS = {
    "Product used for unknown indication"
}

print("=" * 70)
print("Disease Indication Dictionary Summary")
print("=" * 70)

print(f"Type 2 Diabetes terms      : {len(TYPE2_DIABETES_TERMS)}")
print(f"Type 1 Diabetes terms      : {len(TYPE1_DIABETES_TERMS)}")
print(f"Any Diabetes terms         : {len(ANY_DIABETES_TERMS)}")
print(f"Hypertension terms         : {len(HYPERTENSION_TERMS)}")
print(f"CKD terms                  : {len(CKD_TERMS)}")
print(f"Heart Failure terms        : {len(HEART_FAILURE_TERMS)}")
print(f"Obesity terms              : {len(OBESITY_TERMS)}")
print(f"Unknown Indication terms   : {len(UNKNOWN_INDICATION_TERMS)}")

# ============================================================================
# Code Cell 65
# ============================================================================
# ==========================================================
# Engineer General Indication Features
# ==========================================================

# Total indication count
total_indications = (
    indi
    .groupby("primaryid")
    .size()
    .rename("total_indications")
)

# Number of unique indication terms
unique_indications = (
    indi
    .groupby("primaryid")["indi_pt"]
    .nunique()
    .rename("num_unique_indications")
)

# Create report-level feature table
indi_features = pd.concat(
    [
        total_indications,
        unique_indications
    ],
    axis=1
).reset_index()

print("=" * 70)
print("General Indication Features")
print("=" * 70)

display(
    indi_features[
        [
            "total_indications",
            "num_unique_indications"
        ]
    ].describe()
)

print("\nMissing Values")

display(
    indi_features[
        [
            "total_indications",
            "num_unique_indications"
        ]
    ]
    .isna()
    .sum()
    .to_frame("Missing")
)

# ============================================================================
# Code Cell 66
# ============================================================================
# ==========================================================
# Validate Total Indication Count
# ==========================================================

print("=" * 70)
print("Total Indication Count Validation")
print("=" * 70)

print(f"Minimum indications per report : {indi_features['total_indications'].min()}")
print(f"Maximum indications per report : {indi_features['total_indications'].max()}")

print()

print(
    "Reports with >20 indications:",
    (indi_features["total_indications"] > 20).sum()
)

print(
    "Reports with >50 indications:",
    (indi_features["total_indications"] > 50).sum()
)

print(
    "Reports with >100 indications:",
    (indi_features["total_indications"] > 100).sum()
)

print("\nTop 20 reports with largest indication counts")

display(
    indi_features
    .sort_values(
        "total_indications",
        ascending=False
    )
    .head(20)
)

# ============================================================================
# Code Cell 67
# ============================================================================
# ==========================================================
# Compare Total vs Unique Indication Counts
# ==========================================================

indi_features["duplicate_indication_records"] = (
    indi_features["total_indications"] -
    indi_features["num_unique_indications"]
)

print("=" * 70)
print("Comparison of Indication Count Features")
print("=" * 70)

print(
    "Reports where total == unique:",
    (indi_features["duplicate_indication_records"] == 0).sum()
)

print(
    "Reports with repeated indication terms:",
    (indi_features["duplicate_indication_records"] > 0).sum()
)

print("\nSummary of repeated indication terms")

display(
    indi_features["duplicate_indication_records"].describe()
)

print("\nTop 20 reports with largest differences")

display(
    indi_features
    .sort_values(
        "duplicate_indication_records",
        ascending=False
    )
    .head(20)
)

# ============================================================================
# Code Cell 68
# ============================================================================
# ==========================================================
# Engineer Disease-Specific Indication Flags
# ==========================================================

# Aggregate indication terms by report
indication_lists = (
    indi
    .groupby("primaryid")["indi_pt"]
    .apply(set)
)

# Helper function
def contains_indication(indications, dictionary):
    """
    Return 1 if any indication term belongs
    to the specified disease dictionary.
    """
    return int(any(term in dictionary for term in indications))


# Create report-level indication flags
indication_flags = pd.DataFrame({

    "primaryid": indication_lists.index,

    "flag_type2_diabetes":
        indication_lists.apply(
            lambda x: contains_indication(
                x,
                TYPE2_DIABETES_TERMS
            )
        ),

    "flag_type1_diabetes":
        indication_lists.apply(
            lambda x: contains_indication(
                x,
                TYPE1_DIABETES_TERMS
            )
        ),

    "flag_any_diabetes":
        indication_lists.apply(
            lambda x: contains_indication(
                x,
                ANY_DIABETES_TERMS
            )
        ),

    "flag_hypertension":
        indication_lists.apply(
            lambda x: contains_indication(
                x,
                HYPERTENSION_TERMS
            )
        ),

    "flag_chronic_kidney_disease":
        indication_lists.apply(
            lambda x: contains_indication(
                x,
                CKD_TERMS
            )
        ),

    "flag_heart_failure":
        indication_lists.apply(
            lambda x: contains_indication(
                x,
                HEART_FAILURE_TERMS
            )
        ),

    "flag_obesity":
        indication_lists.apply(
            lambda x: contains_indication(
                x,
                OBESITY_TERMS
            )
        ),

    "flag_unknown_indication":
        indication_lists.apply(
            lambda x: contains_indication(
                x,
                UNKNOWN_INDICATION_TERMS
            )
        )

}).reset_index(drop=True)

# Merge into feature table
indi_features = indi_features.merge(
    indication_flags,
    on="primaryid",
    how="left"
)

print("=" * 70)
print("Disease-Specific Indication Flags")
print("=" * 70)

summary = (
    indi_features[
        [
            "flag_type2_diabetes",
            "flag_type1_diabetes",
            "flag_any_diabetes",
            "flag_hypertension",
            "flag_chronic_kidney_disease",
            "flag_heart_failure",
            "flag_obesity",
            "flag_unknown_indication"
        ]
    ]
    .sum()
    .to_frame("Reports")
)

summary["Percent"] = (
    summary["Reports"] /
    len(indi_features) * 100
).round(2)

display(summary)

print("\nMissing Values")

display(
    indi_features[
        [
            "flag_type2_diabetes",
            "flag_type1_diabetes",
            "flag_any_diabetes",
            "flag_hypertension",
            "flag_chronic_kidney_disease",
            "flag_heart_failure",
            "flag_obesity",
            "flag_unknown_indication"
        ]
    ]
    .isna()
    .sum()
    .to_frame("Missing")
)

# ============================================================================
# Code Cell 69
# ============================================================================
# ==========================================================
# Engineer Disease-Specific Indication Flags
# ==========================================================

# Aggregate indication terms by report
indication_lists = (
    indi
    .groupby("primaryid")["indi_pt"]
    .apply(set)
)

# Helper function
def contains_indication(indications, dictionary):
    """
    Return 1 if any indication term belongs
    to the specified disease dictionary.
    """
    return int(any(term in dictionary for term in indications))


# Create report-level disease flags
indication_flags = pd.DataFrame({

    "primaryid": indication_lists.index,

    "flag_type2_diabetes":
        indication_lists.apply(
            lambda x: contains_indication(
                x,
                TYPE2_DIABETES_TERMS
            )
        ),

    "flag_type1_diabetes":
        indication_lists.apply(
            lambda x: contains_indication(
                x,
                TYPE1_DIABETES_TERMS
            )
        ),

    "flag_any_diabetes":
        indication_lists.apply(
            lambda x: contains_indication(
                x,
                ANY_DIABETES_TERMS
            )
        ),

    "flag_hypertension":
        indication_lists.apply(
            lambda x: contains_indication(
                x,
                HYPERTENSION_TERMS
            )
        ),

    "flag_chronic_kidney_disease":
        indication_lists.apply(
            lambda x: contains_indication(
                x,
                CKD_TERMS
            )
        ),

    "flag_heart_failure":
        indication_lists.apply(
            lambda x: contains_indication(
                x,
                HEART_FAILURE_TERMS
            )
        ),

    "flag_obesity":
        indication_lists.apply(
            lambda x: contains_indication(
                x,
                OBESITY_TERMS
            )
        ),

    "flag_unknown_indication":
        indication_lists.apply(
            lambda x: contains_indication(
                x,
                UNKNOWN_INDICATION_TERMS
            )
        )

}).reset_index(drop=True)

# Merge into feature table
indi_features = indi_features.merge(
    indication_flags,
    on="primaryid",
    how="left"
)

print("=" * 70)
print("Disease-Specific Indication Flags")
print("=" * 70)

summary = (
    indi_features[
        [
            "flag_type2_diabetes",
            "flag_type1_diabetes",
            "flag_any_diabetes",
            "flag_hypertension",
            "flag_chronic_kidney_disease",
            "flag_heart_failure",
            "flag_obesity",
            "flag_unknown_indication"
        ]
    ]
    .sum()
    .to_frame("Reports")
)

summary["Percent"] = (
    summary["Reports"] /
    len(indi_features) * 100
).round(2)

display(summary)

print("\nMissing Values")

display(
    indi_features[
        [
            "flag_type2_diabetes",
            "flag_type1_diabetes",
            "flag_any_diabetes",
            "flag_hypertension",
            "flag_chronic_kidney_disease",
            "flag_heart_failure",
            "flag_obesity",
            "flag_unknown_indication"
        ]
    ]
    .isna()
    .sum()
    .to_frame("Missing")
)

# ============================================================================
# Code Cell 70
# ============================================================================
# ==========================================================
# Validate Disease-Specific Indication Dictionaries
# ==========================================================

indication_dictionaries = {
    "Type 2 Diabetes": TYPE2_DIABETES_TERMS,
    "Type 1 Diabetes": TYPE1_DIABETES_TERMS,
    "Any Diabetes": ANY_DIABETES_TERMS,
    "Hypertension": HYPERTENSION_TERMS,
    "Chronic Kidney Disease": CKD_TERMS,
    "Heart Failure": HEART_FAILURE_TERMS,
    "Obesity": OBESITY_TERMS,
    "Unknown Indication": UNKNOWN_INDICATION_TERMS
}

print("=" * 70)
print("Disease-Specific Indication Dictionary Validation")
print("=" * 70)

for name, dictionary in indication_dictionaries.items():

    matched = sorted(
        set(
            indi.loc[
                indi["indi_pt"].isin(dictionary),
                "indi_pt"
            ]
        )
    )

    print(f"\n{name}")
    print("-" * 50)
    print(f"Matched Terms: {len(matched)}")

    display(
        pd.DataFrame({
            "Matched MedDRA PT": matched
        })
    )

# ============================================================================
# Code Cell 71
# ============================================================================
# ==========================================================
# Validate Final INDI Feature Table
# ==========================================================

# Remove validation-only column if present
indi_features = indi_features.drop(
    columns=["duplicate_indication_records"],
    errors="ignore"
)

print("=" * 70)
print("INDI Feature Table Validation")
print("=" * 70)

print(f"Rows                 : {indi_features.shape[0]:,}")
print(f"Columns              : {indi_features.shape[1]}")
print(f"Unique PRIMARYIDs    : {indi_features['primaryid'].nunique():,}")
print(f"Duplicate PRIMARYIDs : {indi_features.duplicated('primaryid').sum():,}")

print("\nMissing Values")
print("-" * 70)

missing = (
    indi_features
    .isna()
    .sum()
    .to_frame("Missing")
)

missing["Percent"] = (
    missing["Missing"] /
    len(indi_features) * 100
).round(2)

display(missing)

print("\nData Types")
print("-" * 70)

display(
    indi_features.dtypes.to_frame("Data Type")
)

print("\nFirst Five Rows")
print("-" * 70)

display(
    indi_features.head()
)

# ============================================================================
# Code Cell 72
# ============================================================================
# ==========================================================
# Save Final INDI Feature Table (CSV + Parquet)
# ==========================================================

indi_features_final = indi_features.copy()

indi_csv_path = PROCESSED_DIR / "indi_features.csv"
indi_parquet_path = PROCESSED_DIR / "indi_features.parquet"

# Save CSV
indi_features_final.to_csv(
    indi_csv_path,
    index=False
)

# Save Parquet
indi_features_final.to_parquet(
    indi_parquet_path,
    index=False
)

print("=" * 70)
print("INDI Feature Table Saved Successfully")
print("=" * 70)

print(f"Rows    : {indi_features_final.shape[0]:,}")
print(f"Columns : {indi_features_final.shape[1]}")
print(f"CSV     : {indi_csv_path}")
print(f"Parquet : {indi_parquet_path}")

# ============================================================================
# Code Cell 73
# ============================================================================
# ==========================================================
# Rebuild INDI Feature Table
# ==========================================================

# General indication features
total_indications = (
    indi.groupby("primaryid")
        .size()
        .rename("total_indications")
)

unique_indications = (
    indi.groupby("primaryid")["indi_pt"]
        .nunique()
        .rename("num_unique_indications")
)

indi_features = (
    pd.concat(
        [total_indications, unique_indications],
        axis=1
    )
    .reset_index()
)

# Merge disease flags ONLY ONCE
indi_features = indi_features.merge(
    indication_flags,
    on="primaryid",
    how="left"
)

# ============================================================================
# Code Cell 74
# ============================================================================
# ==========================================================
# Rebuild and Validate Final INDI Feature Table
# ==========================================================

total_indications = (
    indi.groupby("primaryid")
        .size()
        .rename("total_indications")
)

unique_indications = (
    indi.groupby("primaryid")["indi_pt"]
        .nunique()
        .rename("num_unique_indications")
)

indi_features = (
    pd.concat([total_indications, unique_indications], axis=1)
      .reset_index()
)

indi_features = indi_features.merge(
    indication_flags,
    on="primaryid",
    how="left"
)

print("=" * 70)
print("INDI Feature Table Validation")
print("=" * 70)

print(f"Rows                 : {indi_features.shape[0]:,}")
print(f"Columns              : {indi_features.shape[1]}")
print(f"Unique PRIMARYIDs    : {indi_features['primaryid'].nunique():,}")
print(f"Duplicate PRIMARYIDs : {indi_features.duplicated('primaryid').sum():,}")

print("\nMissing Values")
print("-" * 70)

missing = indi_features.isna().sum().to_frame("Missing")
missing["Percent"] = (missing["Missing"] / len(indi_features) * 100).round(2)
display(missing)

print("\nData Types")
print("-" * 70)
display(indi_features.dtypes.to_frame("Data Type"))

print("\nFirst Five Rows")
print("-" * 70)
display(indi_features.head())

# ============================================================================
# Code Cell 75
# ============================================================================
# ==========================================================
# Save Final INDI Feature Table (CSV + Parquet)
# ==========================================================

indi_features_final = indi_features.copy()

indi_csv_path = PROCESSED_DIR / "indi_features.csv"
indi_parquet_path = PROCESSED_DIR / "indi_features.parquet"

indi_features_final.to_csv(indi_csv_path, index=False)

indi_features_final.to_parquet(indi_parquet_path, index=False)

print("=" * 70)
print("INDI Feature Table Saved Successfully")
print("=" * 70)
print(f"Rows    : {indi_features_final.shape[0]:,}")
print(f"Columns : {indi_features_final.shape[1]}")
print(f"CSV     : {indi_csv_path}")
print(f"Parquet : {indi_parquet_path}")

# ============================================================================
# Code Cell 76
# ============================================================================
# ==========================================================
# RPSR Data Quality Assessment
# ==========================================================

print("=" * 70)
print("RPSR Table Overview")
print("=" * 70)

print(f"Rows                    : {rpsr.shape[0]:,}")
print(f"Columns                 : {rpsr.shape[1]}")
print(f"Unique PRIMARYIDs       : {rpsr['primaryid'].nunique():,}")

duplicate_rows = rpsr.duplicated().sum()

print(f"Duplicate full rows     : {duplicate_rows:,}")

print("\nAverage reporter records per report:")

print(
    round(
        rpsr.shape[0] /
        rpsr["primaryid"].nunique(),
        2
    )
)

print("\n" + "=" * 70)
print("Missing Values")
print("=" * 70)

missing = (
    rpsr
    .isna()
    .sum()
    .to_frame("Missing")
)

missing["Percent"] = (
    missing["Missing"] /
    len(rpsr) * 100
).round(2)

display(
    missing.sort_values(
        "Missing",
        ascending=False
    )
)

print("\n" + "=" * 70)
print("Reporter Source Distribution")
print("=" * 70)

reporter_distribution = (
    rpsr["rpsr_cod"]
    .value_counts(dropna=False)
    .rename_axis("Reporter Code")
    .reset_index(name="Count")
)

reporter_distribution["Percentage"] = (
    reporter_distribution["Count"] /
    len(rpsr) * 100
).round(2)

display(reporter_distribution)

# ============================================================================
# Code Cell 77
# ============================================================================
# ==========================================================
# RPSR Variable Selection
# ==========================================================

rpsr_variable_selection = pd.DataFrame({

    "Original Variable":[
        "primaryid",
        "rpsr_cod",
        "caseid"
    ],

    "Decision":[
        "Keep",
        "Keep",
        "Drop"
    ],

    "Reason":[
        "Merge key",
        "Reporter source code",
        "Administrative identifier"
    ],

    "Planned Use":[
        "Merge",
        "Reporter feature engineering",
        "-"
    ]

})

display(rpsr_variable_selection)

# ============================================================================
# Code Cell 78
# ============================================================================
# ==========================================================
# RPSR Feature Specification
# ==========================================================

rpsr_feature_specification = pd.DataFrame({

    "Engineered Feature":[

        "num_reporter_sources",
        "has_health_professional_report",
        "has_consumer_report",
        "has_foreign_report"

    ],

    "Feature Type":[

        "Numeric",
        "Binary",
        "Binary",
        "Binary"

    ],

    "Source Variable":[

        "rpsr_cod",
        "rpsr_cod",
        "rpsr_cod",
        "rpsr_cod"

    ],

    "Aggregation Method":[

        "Count",
        "Any Match",
        "Any Match",
        "Any Match"

    ],

    "Clinical Interpretation":[

        "Number of reporter sources",
        "Health professional reporter",
        "Consumer reporter",
        "Foreign report"

    ]

})

display(rpsr_feature_specification)

# ============================================================================
# Code Cell 79
# ============================================================================
# ==========================================================
# Engineer Reporter Source Features
# ==========================================================

# Number of reporter sources
num_reporter_sources = (
    rpsr
    .groupby("primaryid")
    .size()
    .rename("num_reporter_sources")
)

# Aggregate reporter codes
reporter_lists = (
    rpsr
    .groupby("primaryid")["rpsr_cod"]
    .apply(set)
)

# Helper function
def has_reporter_source(reporter_codes, source):

    return int(source in reporter_codes)

# Create report-level features
rpsr_features = pd.DataFrame({

    "primaryid": reporter_lists.index,

    "has_health_professional_report":
        reporter_lists.apply(
            lambda x: has_reporter_source(x, "HP")
        ),

    "has_consumer_report":
        reporter_lists.apply(
            lambda x: has_reporter_source(x, "CSM")
        ),

    "has_foreign_report":
        reporter_lists.apply(
            lambda x: has_reporter_source(x, "FGN")
        )

}).reset_index(drop=True)

# Merge numeric feature
rpsr_features = (
    num_reporter_sources
    .reset_index()
    .merge(
        rpsr_features,
        on="primaryid",
        how="left"
    )
)

print("=" * 70)
print("Reporter Source Features")
print("=" * 70)

display(
    rpsr_features.drop(columns="primaryid").describe()
)

print("\nMissing Values")

display(
    rpsr_features.drop(columns="primaryid")
    .isna()
    .sum()
    .to_frame("Missing")
)

# ============================================================================
# Code Cell 80
# ============================================================================
# ==========================================================
# Validate Reporter Source Features
# ==========================================================

print("=" * 70)
print("Reporter Source Feature Validation")
print("=" * 70)

summary = (
    rpsr_features[
        [
            "num_reporter_sources",
            "has_health_professional_report",
            "has_consumer_report",
            "has_foreign_report"
        ]
    ]
    .agg(
        ["count", "mean", "std", "min", "max"]
    )
)

display(summary)

print("\nReporter Source Counts")

counts = (
    rpsr_features[
        [
            "has_health_professional_report",
            "has_consumer_report",
            "has_foreign_report"
        ]
    ]
    .sum()
    .to_frame("Reports")
)

counts["Percent"] = (
    counts["Reports"] /
    len(rpsr_features) * 100
).round(2)

display(counts)

# ============================================================================
# Code Cell 81
# ============================================================================
# ==========================================================
# Validate Final RPSR Feature Table
# ==========================================================

print("=" * 70)
print("RPSR Feature Table Validation")
print("=" * 70)

print(f"Rows                 : {rpsr_features.shape[0]:,}")
print(f"Columns              : {rpsr_features.shape[1]}")
print(f"Unique PRIMARYIDs    : {rpsr_features['primaryid'].nunique():,}")
print(f"Duplicate PRIMARYIDs : {rpsr_features.duplicated('primaryid').sum():,}")

print("\nMissing Values")
print("-" * 70)

missing = (
    rpsr_features
    .isna()
    .sum()
    .to_frame("Missing")
)

missing["Percent"] = (
    missing["Missing"] /
    len(rpsr_features) * 100
).round(2)

display(missing)

print("\nData Types")
print("-" * 70)

display(
    rpsr_features.dtypes.to_frame("Data Type")
)

print("\nFirst Five Rows")
print("-" * 70)

display(
    rpsr_features.head()
)

# ============================================================================
# Code Cell 82
# ============================================================================
# ==========================================================
# Save Final RPSR Feature Table (CSV + Parquet)
# ==========================================================

rpsr_features_final = rpsr_features.copy()

rpsr_csv_path = PROCESSED_DIR / "rpsr_features.csv"
rpsr_parquet_path = PROCESSED_DIR / "rpsr_features.parquet"

# Save CSV
rpsr_features_final.to_csv(
    rpsr_csv_path,
    index=False
)

# Save Parquet
rpsr_features_final.to_parquet(
    rpsr_parquet_path,
    index=False
)

print("=" * 70)
print("RPSR Feature Table Saved Successfully")
print("=" * 70)

print(f"Rows    : {rpsr_features_final.shape[0]:,}")
print(f"Columns : {rpsr_features_final.shape[1]}")
print(f"CSV     : {rpsr_csv_path}")
print(f"Parquet : {rpsr_parquet_path}")

# ============================================================================
# Code Cell 83
# ============================================================================
# ==========================================================
# THER Data Quality Assessment
# ==========================================================

print("=" * 70)
print("THER Table Overview")
print("=" * 70)

print(f"Rows                    : {ther.shape[0]:,}")
print(f"Columns                 : {ther.shape[1]}")
print(f"Unique PRIMARYIDs       : {ther['primaryid'].nunique():,}")

duplicate_rows = ther.duplicated().sum()

print(f"Duplicate full rows     : {duplicate_rows:,}")

print("\nAverage therapy records per report:")

print(
    round(
        ther.shape[0] /
        ther["primaryid"].nunique(),
        2
    )
)

print("\n" + "=" * 70)
print("Missing Values")
print("=" * 70)

missing = (
    ther
    .isna()
    .sum()
    .to_frame("Missing")
)

missing["Percent"] = (
    missing["Missing"] /
    len(ther) * 100
).round(2)

display(
    missing.sort_values(
        "Missing",
        ascending=False
    )
)

print("\n" + "=" * 70)
print("Therapy Duration Units")
print("=" * 70)

display(
    ther["dur_cod"]
    .value_counts(dropna=False)
    .rename_axis("Duration Unit")
    .reset_index(name="Count")
)

# ============================================================================
# Code Cell 84
# ============================================================================
# ==========================================================
# Remove Duplicate THER Records
# ==========================================================

print("=" * 70)
print("Removing Duplicate THER Records")
print("=" * 70)

rows_before = len(ther)

ther = ther.drop_duplicates()

rows_after = len(ther)

print(f"Rows before cleaning : {rows_before:,}")
print(f"Rows after cleaning  : {rows_after:,}")
print(f"Duplicates removed   : {rows_before - rows_after:,}")

print(f"\nUnique PRIMARYIDs: {ther['primaryid'].nunique():,}")

# ============================================================================
# Code Cell 85
# ============================================================================
# ==========================================================
# THER Variable Selection
# ==========================================================

ther_variable_selection = pd.DataFrame({

    "Original Variable":[
        "primaryid",
        "dur",
        "dur_cod",
        "start_dt",
        "end_dt",
        "dsg_drug_seq",
        "caseid"
    ],

    "Decision":[
        "Keep",
        "Keep",
        "Keep",
        "Review",
        "Review",
        "Drop",
        "Drop"
    ],

    "Reason":[
        "Merge key",
        "Therapy duration",
        "Duration unit",
        "Potential future feature",
        "Potential future feature",
        "Administrative sequence",
        "Administrative identifier"
    ],

    "Planned Use":[
        "Merge",
        "Duration standardization",
        "Unit conversion",
        "Future enhancement",
        "Future enhancement",
        "-",
        "-"
    ]

})

display(ther_variable_selection)

# ============================================================================
# Code Cell 86
# ============================================================================
# ==========================================================
# Standardize Therapy Duration to Days
# ==========================================================

# Convert duration to numeric
ther["dur"] = pd.to_numeric(
    ther["dur"],
    errors="coerce"
)

# Unit conversion factors
duration_conversion = {
    "DAY": 1,
    "WK": 7,
    "MON": 30.44,
    "YR": 365.25,
    "HR": 1 / 24,
    "MIN": 1 / (24 * 60),
    "SEC": 1 / (24 * 60 * 60)
}

# Convert to days
ther["therapy_duration_days"] = (
    ther["dur"] *
    ther["dur_cod"].map(duration_conversion)
)

print("=" * 70)
print("Therapy Duration Standardization")
print("=" * 70)

print("Summary Statistics")

display(
    ther["therapy_duration_days"].describe()
)

print("\nMissing Values")

print(
    ther["therapy_duration_days"].isna().sum()
)

print("\nDuration Units Used")

display(
    ther["dur_cod"]
    .value_counts(dropna=False)
)

# ============================================================================
# Code Cell 87
# ============================================================================
# ==========================================================
# Validate Therapy Duration
# ==========================================================

print("=" * 70)
print("Therapy Duration Validation")
print("=" * 70)

print(
    f"Minimum duration : {ther['therapy_duration_days'].min():.2f} days"
)

print(
    f"Maximum duration : {ther['therapy_duration_days'].max():.2f} days"
)

print()

print(
    "Therapies >365 days:",
    (ther["therapy_duration_days"] > 365).sum()
)

print(
    "Therapies >5 years:",
    (ther["therapy_duration_days"] > 365 * 5).sum()
)

print(
    "Therapies >10 years:",
    (ther["therapy_duration_days"] > 365 * 10).sum()
)

print("\nTop 20 longest therapies")

display(

    ther[
        [
            "primaryid",
            "dur",
            "dur_cod",
            "therapy_duration_days"
        ]
    ]
    .sort_values(
        "therapy_duration_days",
        ascending=False
    )
    .head(20)

)

# ============================================================================
# Code Cell 88
# ============================================================================
# ==========================================================
# Clean Therapy Duration
# ==========================================================

MAX_DURATION_DAYS = 3650   # 10 years

ther["therapy_duration_days_clean"] = ther[
    "therapy_duration_days"
].clip(
    upper=MAX_DURATION_DAYS
)

ther["therapy_duration_outlier"] = (
    ther["therapy_duration_days"] >
    MAX_DURATION_DAYS
).astype(int)

print("=" * 70)
print("Therapy Duration Cleaning")
print("=" * 70)

print("Original maximum duration :",
      ther["therapy_duration_days"].max())

print("Cleaned maximum duration :",
      ther["therapy_duration_days_clean"].max())

print()

print(
    "Outlier records:",
    ther["therapy_duration_outlier"].sum()
)

print()

display(
    ther[
        [
            "therapy_duration_days",
            "therapy_duration_days_clean"
        ]
    ].describe()
)

# ============================================================================
# Code Cell 89
# ============================================================================
# ==========================================================
# THER Feature Specification
# ==========================================================

ther_feature_specification = pd.DataFrame({

    "Engineered Feature":[
        "total_therapies",
        "num_therapies_with_duration",
        "mean_therapy_duration_days",
        "median_therapy_duration_days",
        "min_therapy_duration_days",
        "max_therapy_duration_days",
        "has_missing_duration",
        "long_term_therapy",
        "short_term_therapy",
        "therapy_duration_outlier_present"
    ],

    "Feature Type":[
        "Numeric",
        "Numeric",
        "Numeric",
        "Numeric",
        "Numeric",
        "Numeric",
        "Binary",
        "Binary",
        "Binary",
        "Binary"
    ],

    "Source Variable":[
        "therapy records",
        "therapy_duration_days_clean",
        "therapy_duration_days_clean",
        "therapy_duration_days_clean",
        "therapy_duration_days_clean",
        "therapy_duration_days_clean",
        "therapy_duration_days",
        "therapy_duration_days_clean",
        "therapy_duration_days_clean",
        "therapy_duration_outlier"
    ],

    "Aggregation Method":[
        "Count",
        "Count",
        "Mean",
        "Median",
        "Minimum",
        "Maximum",
        "Any Missing",
        "Any ≥180 days",
        "Any ≤30 days",
        "Any Outlier"
    ],

    "Clinical Interpretation":[
        "Total therapy records",
        "Therapies with duration information",
        "Average therapy duration",
        "Typical therapy duration",
        "Shortest therapy duration",
        "Longest therapy duration",
        "Incomplete duration reporting",
        "Long-term therapy exposure",
        "Short-term therapy exposure",
        "Contains implausible duration"
    ]

})

display(ther_feature_specification)

# ============================================================================
# Code Cell 90
# ============================================================================
# ==========================================================
# Engineer THER Features
# ==========================================================

# Total therapy records
total_therapies = (
    ther.groupby("primaryid")
        .size()
        .rename("total_therapies")
)

# Aggregate report-level features
ther_features = (
    ther.groupby("primaryid")
        .agg(
            num_therapies_with_duration=(
                "therapy_duration_days_clean",
                lambda x: x.notna().sum()
            ),

            mean_therapy_duration_days=(
                "therapy_duration_days_clean",
                "mean"
            ),

            median_therapy_duration_days=(
                "therapy_duration_days_clean",
                "median"
            ),

            min_therapy_duration_days=(
                "therapy_duration_days_clean",
                "min"
            ),

            max_therapy_duration_days=(
                "therapy_duration_days_clean",
                "max"
            ),

            has_missing_duration=(
                "therapy_duration_days_clean",
                lambda x: int(x.isna().any())
            ),

            long_term_therapy=(
                "therapy_duration_days_clean",
                lambda x: int((x >= 180).any())
            ),

            short_term_therapy=(
                "therapy_duration_days_clean",
                lambda x: int((x <= 30).any())
            ),

            therapy_duration_outlier_present=(
                "therapy_duration_outlier",
                "max"
            )
        )
        .reset_index()
)

# Merge total therapies
ther_features = (
    total_therapies
    .reset_index()
    .merge(
        ther_features,
        on="primaryid",
        how="left"
    )
)

print("=" * 70)
print("THER Features Created")
print("=" * 70)

display(
    ther_features.describe()
)

print("\nMissing Values")

display(
    ther_features.isna().sum().to_frame("Missing")
)

# ============================================================================
# Code Cell 91
# ============================================================================
# ==========================================================
# Replace Missing Duration Statistics with Zero
# ==========================================================

duration_cols = [
    "mean_therapy_duration_days",
    "median_therapy_duration_days",
    "min_therapy_duration_days",
    "max_therapy_duration_days"
]

ther_features[duration_cols] = (
    ther_features[duration_cols]
    .fillna(0)
)

print("=" * 70)
print("Duration Summary Features After Missing Value Handling")
print("=" * 70)

display(
    ther_features[duration_cols]
    .isna()
    .sum()
    .to_frame("Missing")
)

# ============================================================================
# Code Cell 92
# ============================================================================
# ==========================================================
# Therapy Duration Availability
# ==========================================================

ther_features["therapy_duration_available"] = (
    ther_features["num_therapies_with_duration"] > 0
).astype(int)

print("=" * 70)
print("Therapy Duration Availability")
print("=" * 70)

display(
    ther_features["therapy_duration_available"]
    .value_counts()
    .to_frame("Reports")
)

# ============================================================================
# Code Cell 93
# ============================================================================
# ==========================================================
# Validate THER Feature Table
# ==========================================================

print("=" * 70)
print("THER Feature Table Validation")
print("=" * 70)

print(f"Rows                 : {ther_features.shape[0]:,}")
print(f"Columns              : {ther_features.shape[1]}")
print(f"Unique PRIMARYIDs    : {ther_features['primaryid'].nunique():,}")
print(f"Duplicate PRIMARYIDs : {ther_features.duplicated('primaryid').sum():,}")

print("\nMissing Values")
print("-" * 70)

missing = (
    ther_features
    .isna()
    .sum()
    .to_frame("Missing")
)

missing["Percent"] = (
    missing["Missing"] /
    len(ther_features) * 100
).round(2)

display(missing)

print("\nData Types")
print("-" * 70)

display(
    ther_features.dtypes.to_frame("Data Type")
)

print("\nFirst Five Rows")
print("-" * 70)

display(
    ther_features.head()
)

# ============================================================================
# Code Cell 94
# ============================================================================
# ==========================================================
# Save Final THER Feature Table (CSV + Parquet)
# ==========================================================

ther_features_final = ther_features.copy()

ther_csv_path = PROCESSED_DIR / "ther_features.csv"
ther_parquet_path = PROCESSED_DIR / "ther_features.parquet"

# Save CSV
ther_features_final.to_csv(
    ther_csv_path,
    index=False
)

# Save Parquet
ther_features_final.to_parquet(
    ther_parquet_path,
    index=False
)

print("=" * 70)
print("THER Feature Table Saved Successfully")
print("=" * 70)

print(f"Rows    : {ther_features_final.shape[0]:,}")
print(f"Columns : {ther_features_final.shape[1]}")
print(f"CSV     : {ther_csv_path}")
print(f"Parquet : {ther_parquet_path}")

# ============================================================================
# Code Cell 95
# ============================================================================
# ==========================================================
# Merge All Feature Tables
# ==========================================================

final_df = (
    demo_features

    .merge(
        drug_features,
        on="primaryid",
        how="left"
    )

    .merge(
        reac_features,
        on="primaryid",
        how="left"
    )

    .merge(
        indi_features,
        on="primaryid",
        how="left"
    )

    .merge(
        rpsr_features,
        on="primaryid",
        how="left"
    )

    .merge(
        ther_features,
        on="primaryid",
        how="left"
    )
)

print("=" * 70)
print("Feature Tables Successfully Merged")
print("=" * 70)

print(f"Rows    : {final_df.shape[0]:,}")
print(f"Columns : {final_df.shape[1]}")

# ============================================================================
# Code Cell 96
# ============================================================================
# ==========================================================
# Merge Validation
# ==========================================================

print("=" * 70)
print("Merge Validation")
print("=" * 70)

print(f"Rows                 : {len(final_df):,}")
print(f"Columns              : {final_df.shape[1]}")

print(f"Unique PRIMARYIDs    : {final_df['primaryid'].nunique():,}")

print(
    f"Duplicate PRIMARYIDs : {final_df.duplicated('primaryid').sum():,}"
)

# ============================================================================
# Code Cell 97
# ============================================================================
# ==========================================================
# Fill Missing Engineered Features
# ==========================================================

# Numeric engineered features
numeric_fill_zero = [

    # INDI
    "total_indications",
    "num_unique_indications",

    # RPSR
    "num_reporter_sources",

    # THER
    "total_therapies",
    "num_therapies_with_duration",
    "mean_therapy_duration_days",
    "median_therapy_duration_days",
    "min_therapy_duration_days",
    "max_therapy_duration_days"

]

# Binary engineered features
binary_fill_zero = [

    # DRUG
    "contains_insulin",
    "contains_sglt2",
    "contains_sulfonylurea",

    # REAC
    "flag_AKI",
    "flag_DKA",
    "flag_hypoglycemia",
    "flag_lactic_acidosis",
    "flag_amputation",
    "flag_genital_infection",

    # INDI
    "flag_type2_diabetes",
    "flag_type1_diabetes",
    "flag_any_diabetes",
    "flag_hypertension",
    "flag_chronic_kidney_disease",
    "flag_heart_failure",
    "flag_obesity",
    "flag_unknown_indication",

    # RPSR
    "has_health_professional_report",
    "has_consumer_report",
    "has_foreign_report",

    # THER
    "has_missing_duration",
    "therapy_duration_available",
    "long_term_therapy",
    "short_term_therapy",
    "therapy_duration_outlier_present"

]

final_df[numeric_fill_zero] = (
    final_df[numeric_fill_zero]
    .fillna(0)
)

final_df[binary_fill_zero] = (
    final_df[binary_fill_zero]
    .fillna(0)
    .astype(int)
)

# ============================================================================
# Code Cell 98
# ============================================================================
# ==========================================================
# Check for Duplicate Column Names
# ==========================================================

duplicate_cols = final_df.columns[final_df.columns.duplicated()]

print("=" * 70)
print("Duplicate Column Check")
print("=" * 70)

if len(duplicate_cols) == 0:
    print("✓ No duplicate column names found.")
else:
    print("Duplicate columns:")
    print(list(duplicate_cols))

# ============================================================================
# Code Cell 99
# ============================================================================
# ==========================================================
# Missing Value Summary
# ==========================================================

missing = (
    final_df
    .isna()
    .sum()
    .to_frame("Missing")
)

missing["Percent"] = (
    missing["Missing"] /
    len(final_df) * 100
).round(2)

missing = (
    missing
    .sort_values(
        "Missing",
        ascending=False
    )
)

print("=" * 70)
print("Final Dataset Missing Values")
print("=" * 70)

display(missing)

# ============================================================================
# Code Cell 100
# ============================================================================
# ==========================================================
# Create Feature Dictionary
# ==========================================================

feature_dictionary = pd.DataFrame({

    "Feature": final_df.columns,

    "Data Type": final_df.dtypes.astype(str).values

})

feature_dictionary["Source Table"] = ""

feature_dictionary["Category"] = ""

feature_dictionary["Description"] = ""

display(feature_dictionary.head())

# ============================================================================
# Code Cell 101
# ============================================================================
# ==========================================================
# Create Clean DEMO Feature Table
# ==========================================================

demo_model_features = demo_features_final[
    [
        "primaryid",
        "age_years",
        "weight_kg",
        "sex",
        "reporter_country",
        "occur_country",
        "reporter_type",
        "report_type"
    ]
].copy()

print("=" * 70)
print("Clean DEMO Feature Table")
print("=" * 70)

print(f"Rows    : {demo_model_features.shape[0]:,}")
print(f"Columns : {demo_model_features.shape[1]}")

display(demo_model_features.head())

# ============================================================================
# Code Cell 102
# ============================================================================
# ==========================================================
# Validate Clean DEMO Feature Table
# ==========================================================

print("=" * 70)
print("Clean DEMO Feature Table Validation")
print("=" * 70)

print(f"Rows                 : {demo_model_features.shape[0]:,}")
print(f"Columns              : {demo_model_features.shape[1]}")
print(f"Unique PRIMARYIDs    : {demo_model_features['primaryid'].nunique():,}")
print(f"Duplicate PRIMARYIDs : {demo_model_features.duplicated('primaryid').sum():,}")

print("\nMissing Values")
print("-" * 70)

missing = (
    demo_model_features
    .isna()
    .sum()
    .to_frame("Missing")
)

missing["Percent"] = (
    missing["Missing"] /
    len(demo_model_features) * 100
).round(2)

display(missing)

print("\nData Types")
print("-" * 70)

display(
    demo_model_features.dtypes.to_frame("Data Type")
)

# ============================================================================
# Code Cell 103
# ============================================================================
# ==========================================================
# Create Final Modeling Dataset
# ==========================================================

model_df = (
    demo_model_features

    .merge(
        drug_features_final,
        on="primaryid",
        how="left"
    )

    .merge(
        reac_features_final,
        on="primaryid",
        how="left"
    )

    .merge(
        indi_features_final,
        on="primaryid",
        how="left"
    )

    .merge(
        rpsr_features_final,
        on="primaryid",
        how="left"
    )

    .merge(
        ther_features_final,
        on="primaryid",
        how="left"
    )
)

print("=" * 70)
print("Final Modeling Dataset Created")
print("=" * 70)

print(f"Rows    : {model_df.shape[0]:,}")
print(f"Columns : {model_df.shape[1]}")

# ============================================================================
# Code Cell 104
# ============================================================================
# ==========================================================
# Handle Missing Values in Modeling Dataset
# ==========================================================

# Features that should be zero when no related records exist
fill_zero_columns = [

    # INDI
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

    # RPSR
    "num_reporter_sources",
    "has_health_professional_report",
    "has_consumer_report",
    "has_foreign_report",

    # THER
    "total_therapies",
    "num_therapies_with_duration",
    "mean_therapy_duration_days",
    "median_therapy_duration_days",
    "min_therapy_duration_days",
    "max_therapy_duration_days",
    "therapy_duration_available",
    "has_missing_duration",
    "long_term_therapy",
    "short_term_therapy",
    "therapy_duration_outlier_present"
]

model_df[fill_zero_columns] = (
    model_df[fill_zero_columns]
    .fillna(0)
)

print("=" * 70)
print("Missing Values Successfully Handled")
print("=" * 70)

print(f"Columns Updated : {len(fill_zero_columns)}")

# ============================================================================
# Code Cell 105
# ============================================================================
# ==========================================================
# Validate Final Modeling Dataset
# ==========================================================

print("=" * 70)
print("Final Modeling Dataset Validation")
print("=" * 70)

print(f"Rows                 : {model_df.shape[0]:,}")
print(f"Columns              : {model_df.shape[1]}")
print(f"Unique PRIMARYIDs    : {model_df['primaryid'].nunique():,}")
print(f"Duplicate PRIMARYIDs : {model_df.duplicated('primaryid').sum():,}")

print("\nMissing Values")
print("-" * 70)

missing = (
    model_df
    .isna()
    .sum()
    .to_frame("Missing")
)

missing["Percent"] = (
    missing["Missing"] /
    len(model_df) * 100
).round(2)

display(
    missing.sort_values(
        "Missing",
        ascending=False
    )
)

print("\nData Types")
print("-" * 70)

display(
    model_df.dtypes.to_frame("Data Type")
)

print("\nFirst Five Rows")
print("-" * 70)

display(
    model_df.head()
)

# ============================================================================
# Code Cell 106
# ============================================================================
# ==========================================================
# Restore Integer Data Types
# ==========================================================

integer_columns = [

    # INDI
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

    # RPSR
    "num_reporter_sources",
    "has_health_professional_report",
    "has_consumer_report",
    "has_foreign_report",

    # THER
    "total_therapies",
    "num_therapies_with_duration",
    "has_missing_duration",
    "long_term_therapy",
    "short_term_therapy",
    "therapy_duration_outlier_present",
    "therapy_duration_available"

]

model_df[integer_columns] = model_df[integer_columns].astype("int64")

print("=" * 70)
print("Integer Data Types Restored")
print("=" * 70)

display(
    model_df[integer_columns].dtypes.to_frame("Data Type")
)

# ============================================================================
# Code Cell 107
# ============================================================================
# ==========================================================
# Save Final Modeling Dataset
# ==========================================================

model_csv_path = PROCESSED_DIR / "faers_modeling_dataset.csv"
model_parquet_path = PROCESSED_DIR / "faers_modeling_dataset.parquet"

# Save CSV
model_df.to_csv(
    model_csv_path,
    index=False
)

# Save Parquet
model_df.to_parquet(
    model_parquet_path,
    index=False
)

print("=" * 70)
print("Final Modeling Dataset Saved Successfully")
print("=" * 70)

print(f"Rows    : {model_df.shape[0]:,}")
print(f"Columns : {model_df.shape[1]}")

print(f"\nCSV     : {model_csv_path}")
print(f"Parquet : {model_parquet_path}")

# ============================================================================
# Code Cell 108
# ============================================================================
# ==========================================================
# Save Master Gold Dataset
# ==========================================================

gold_csv_path = PROCESSED_DIR / "faers_serious_event_modeling.csv"
gold_parquet_path = PROCESSED_DIR / "faers_serious_event_modeling.parquet"

# Save CSV
final_df.to_csv(
    gold_csv_path,
    index=False
)

# Save Parquet
final_df.to_parquet(
    gold_parquet_path,
    index=False
)

print("=" * 70)
print("Master Gold Dataset Saved Successfully")
print("=" * 70)

print(f"Rows    : {final_df.shape[0]:,}")
print(f"Columns : {final_df.shape[1]}")

print(f"\nCSV     : {gold_csv_path}")
print(f"Parquet : {gold_parquet_path}")

# ============================================================================
# Code Cell 109
# ============================================================================
# ==========================================================
# Add Target Variable to Modeling Dataset
# ==========================================================

target_path = PROCESSED_DIR / "target_is_serious.parquet"

# If target was not saved separately, recreate from OUTC
if target_path.exists():
    target_df = pd.read_parquet(target_path)
else:
    target_df = (
        demo_model_features[["primaryid"]]
        .merge(
            outc[["primaryid"]].drop_duplicates().assign(is_serious=1),
            on="primaryid",
            how="left"
        )
    )

    target_df["is_serious"] = (
        target_df["is_serious"]
        .fillna(0)
        .astype(int)
    )

# Merge target into modeling dataset if missing
if "is_serious" not in model_df.columns:
    model_df = model_df.merge(
        target_df[["primaryid", "is_serious"]],
        on="primaryid",
        how="left"
    )

model_df["is_serious"] = model_df["is_serious"].astype(int)

print("=" * 70)
print("Target Variable Added")
print("=" * 70)

print(f"Rows    : {model_df.shape[0]:,}")
print(f"Columns : {model_df.shape[1]}")

display(
    model_df["is_serious"]
    .value_counts()
    .rename_axis("is_serious")
    .reset_index(name="Count")
)

# ============================================================================
# Code Cell 110
# ============================================================================
# ==========================================================
# Save Corrected Modeling Dataset
# ==========================================================

model_csv_path = PROCESSED_DIR / "faers_modeling_dataset.csv"
model_parquet_path = PROCESSED_DIR / "faers_modeling_dataset.parquet"

model_df.to_csv(
    model_csv_path,
    index=False
)

model_df.to_parquet(
    model_parquet_path,
    index=False
)

print("=" * 70)
print("Corrected Modeling Dataset Saved")
print("=" * 70)
print(f"Rows    : {model_df.shape[0]:,}")
print(f"Columns : {model_df.shape[1]}")
print(f"CSV     : {model_csv_path}")
print(f"Parquet : {model_parquet_path}")

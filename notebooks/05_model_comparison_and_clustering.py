# Extracted from: 05_notebook5.ipynb
# Complete executable code cells in notebook order.


# ============================================================================
# Code Cell 1
# ============================================================================
# ==========================================================
# Notebook 05: Final Report, Results & Interpretation
# ISYE 7406 Final Project
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display, Image, Markdown

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 120)

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_DIR = Path("/Users/sumi/Documents/GEORGIA TECH/ISYE 7406/project/Project_Solution")

RESULTS_DIR = PROJECT_DIR / "results"

EDA_DIR = RESULTS_DIR / "eda"
EDA_TABLES_DIR = EDA_DIR / "tables"
EDA_STATS_DIR = EDA_DIR / "statistics"

MODELING_DIR = RESULTS_DIR / "modeling"
PERFORMANCE_DIR = MODELING_DIR / "performance"
COMPARISON_DIR = MODELING_DIR / "comparison"
PREDICTIONS_DIR = MODELING_DIR / "predictions"
ROC_DIR = MODELING_DIR / "roc_curves"
PR_DIR = MODELING_DIR / "precision_recall_curves"
REPORTS_DIR = MODELING_DIR / "classification_reports"
CM_DIR = MODELING_DIR / "confusion_matrices"
FI_DIR = MODELING_DIR / "feature_importance"
CLUSTERING_DIR = MODELING_DIR / "clustering"

FINAL_DIR = RESULTS_DIR / "final_report"
FINAL_TABLES_DIR = FINAL_DIR / "tables"
FINAL_FIGURES_DIR = FINAL_DIR / "figures"

FINAL_TABLES_DIR.mkdir(parents=True, exist_ok=True)
FINAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Notebook 05 Output Directories Ready")
print("=" * 80)
print(FINAL_TABLES_DIR)
print(FINAL_FIGURES_DIR)

# ============================================================================
# Code Cell 2
# ============================================================================
# ==========================================================
# Section 1: Load EDA Summary Tables
# ==========================================================

eda_table_files = {
    "target_distribution": "target_distribution.csv",
    "missing_values": "missing_values.csv",
    "numeric_summary": "numeric_summary.csv",
    "demographic_summary": "demographic_summary.csv",
    "therapy_summary": "therapy_summary.csv",
    "drug_burden_summary": "drug_burden_summary.csv",
    "reaction_burden_summary": "reaction_burden_summary.csv",
    "indication_burden_summary": "indication_burden_summary.csv",
    "reporter_summary": "reporter_summary.csv",
    "reaction_category_summary": "reaction_category_summary.csv",
    "indication_category_summary": "indication_category_summary.csv"
}

eda_tables = {}

for name, file in eda_table_files.items():
    path = EDA_TABLES_DIR / file
    if path.exists():
        eda_tables[name] = pd.read_csv(path)
        print(f"✅ Loaded {file}")
    else:
        print(f"⚠️ Missing {file}")

print("\nEDA tables loaded:", len(eda_tables))

# ============================================================================
# Code Cell 3
# ============================================================================
# ==========================================================
# Section 1: Load Available EDA Summary Tables
# ==========================================================

eda_table_files = {
    "target_distribution": "target_distribution.csv",
    "missing_values": "missing_values.csv",
    "numeric_summary": "numeric_summary.csv",
    "demographic_summary": "demographic_summary.csv",
    "therapy_summary": "therapy_summary.csv",
    "drug_burden_summary": "drug_burden_summary.csv",
    "reaction_burden_summary": "reaction_burden_summary.csv",
    "indication_burden_summary": "indication_burden_summary.csv",
    "reporter_summary": "reporter_summary.csv",
    "reaction_category_summary": "reaction_category_summary.csv",
    "indication_category_summary": "indication_category_summary.csv"
}

eda_tables = {}
missing_eda_tables = []

for name, file in eda_table_files.items():
    path = EDA_TABLES_DIR / file
    
    if path.exists():
        eda_tables[name] = pd.read_csv(path)
        print(f"✅ Loaded {file}")
    else:
        missing_eda_tables.append(file)
        print(f"⚠️ Missing {file}")

missing_eda_tables_df = pd.DataFrame({
    "Missing EDA Table": missing_eda_tables
})

missing_eda_tables_df.to_csv(
    FINAL_TABLES_DIR / "missing_eda_tables_log.csv",
    index=False
)

print("\nEDA tables loaded:", len(eda_tables))
print("Missing EDA tables:", len(missing_eda_tables))

# ============================================================================
# Code Cell 4
# ============================================================================
# ==========================================================
# Section 1: Load Available EDA Summary Tables
# ==========================================================

eda_table_files = {
    "target_distribution": "target_distribution.csv",
    "missing_values": "missing_values.csv",
    "numeric_summary": "numeric_summary.csv",
    "summary_statistics": "summary_statistics.csv",
    "demographic_summary": "demographic_summary.csv",
    "drug_burden_summary": "drug_burden_summary.csv",
    "reaction_burden_summary": "reaction_burden_summary.csv",
    "modeling_dataset_summary": "modeling_dataset_summary.csv",
    "feature_dictionary": "feature_dictionary.csv",
    "feature_list": "feature_list.csv",
    "column_types": "column_types.csv",
    "correlation_matrix": "correlation_matrix.csv",
    "high_correlations": "high_correlations.csv",
    "vif_results": "vif_results.csv"
}

eda_tables = {}
missing_eda_tables = []

for name, file in eda_table_files.items():
    path = EDA_TABLES_DIR / file
    
    if path.exists():
        eda_tables[name] = pd.read_csv(path)
        print(f"✅ Loaded {file}")
    else:
        missing_eda_tables.append(file)
        print(f"⚠️ Missing {file}")

missing_eda_tables_df = pd.DataFrame({
    "Missing EDA Table": missing_eda_tables
})

missing_eda_tables_df.to_csv(
    FINAL_TABLES_DIR / "missing_eda_tables_log.csv",
    index=False
)

print("\nEDA tables loaded:", len(eda_tables))
print("Missing EDA tables:", len(missing_eda_tables))

# ============================================================================
# Code Cell 5
# ============================================================================
from pathlib import Path

print("EDA Tables Directory:")
print(EDA_TABLES_DIR)

print("\nFiles found:")

for f in sorted(EDA_TABLES_DIR.glob("*.csv")):
    print(f.name)

# ============================================================================
# Code Cell 6
# ============================================================================
# ==========================================================
# Section 1: Project Summary
# ==========================================================

project_summary = pd.DataFrame({
    "Item": [
        "Project Title",
        "Dataset",
        "Study Period",
        "Rows",
        "Columns",
        "Target Variable",
        "Number of Supervised Models",
        "Unsupervised Method",
        "Selected Final Model"
    ],
    "Value": [
        "Predicting Serious Adverse Drug Events Using Statistical Learning Methods on FDA FAERS Data",
        "FDA FAERS",
        "2024 Q1",
        "406,184",
        "51",
        "is_serious",
        "9",
        "K-Means Clustering",
        "Random Forest"
    ]
})

display(project_summary)

project_summary.to_csv(
    FINAL_TABLES_DIR / "project_summary.csv",
    index=False
)

print("=" * 80)
print("Project summary exported.")
print("=" * 80)

# ============================================================================
# Code Cell 7
# ============================================================================
# ==========================================================
# Section 2.1: Verify EDA Outputs
# ==========================================================

eda_verification = pd.DataFrame({
    "EDA Table": list(eda_tables.keys()),
    "Rows": [df.shape[0] for df in eda_tables.values()],
    "Columns": [df.shape[1] for df in eda_tables.values()],
    "Status": "Loaded"
})

display(eda_verification)

eda_verification.to_csv(
    FINAL_TABLES_DIR / "eda_output_verification.csv",
    index=False
)

print(f"EDA Tables Verified: {len(eda_verification)}")

# ============================================================================
# Code Cell 8
# ============================================================================
print("Statistics Folder:")
print(EDA_STATS_DIR)

print("\nFiles found:\n")

for f in sorted(EDA_STATS_DIR.glob("*.csv")):
    print(f.name)

# ============================================================================
# Code Cell 9
# ============================================================================
# ==========================================================
# Load Statistical Test Tables
# ==========================================================

stat_test_files = {
    "demographic_tests": "demographic_tests.csv",
    "drug_tests": "drug_tests.csv",
    "reaction_tests": "reaction_tests.csv",
    "indication_tests": "indication_tests.csv",
    "therapy_tests": "therapy_tests.csv",
    "reporter_tests": "reporter_tests.csv",
    "reaction_category_tests": "reaction_category_tests.csv",
    "indication_category_tests": "indication_category_tests.csv"
}

stat_tests = {}

for name, file in stat_test_files.items():

    path = EDA_STATS_DIR / file

    if path.exists():
        stat_tests[name] = pd.read_csv(path)
        print(f"✅ Loaded {file}")
    else:
        print(f"⚠️ Missing {file}")

print(f"\nStatistical tables loaded: {len(stat_tests)}")

# ============================================================================
# Code Cell 10
# ============================================================================
# ==========================================================
# Section 2.2: Verify Statistical Test Outputs
# ==========================================================

stats_verification = pd.DataFrame({
    "Statistical Test": list(stat_tests.keys()),
    "Rows": [df.shape[0] for df in stat_tests.values()],
    "Columns": [df.shape[1] for df in stat_tests.values()],
    "Status": "Loaded"
})

display(stats_verification)

stats_verification.to_csv(
    FINAL_TABLES_DIR / "statistical_output_verification.csv",
    index=False
)

print(f"Statistical Test Tables Verified: {len(stats_verification)}")

# ============================================================================
# Code Cell 11
# ============================================================================
# ==========================================================
# Section 2.3: Verify Modeling Output Directories
# ==========================================================

directories = {
    "Performance": PERFORMANCE_DIR,
    "Comparison": COMPARISON_DIR,
    "Predictions": PREDICTIONS_DIR,
    "ROC Curves": ROC_DIR,
    "Precision-Recall Curves": PR_DIR,
    "Classification Reports": REPORTS_DIR,
    "Confusion Matrices": CM_DIR,
    "Feature Importance": FI_DIR,
    "Clustering": CLUSTERING_DIR
}

verification = []

for name, folder in directories.items():

    if folder.exists():
        files = len(list(folder.glob("*")))
    else:
        files = 0

    verification.append({
        "Directory": name,
        "Files Found": files,
        "Status": "Available" if files > 0 else "Empty"
    })

verification = pd.DataFrame(verification)

display(verification)

verification.to_csv(
    FINAL_TABLES_DIR / "model_output_verification.csv",
    index=False
)

print(f"Directories verified: {len(verification)}")

# ============================================================================
# Code Cell 12
# ============================================================================
# ==========================================================
# Section 2.4: Overall Project Verification
# ==========================================================

summary = pd.DataFrame({
    "Component": [
        "EDA Tables",
        "Statistical Test Tables",
        "Model Output Directories"
    ],
    "Available": [
        len(eda_verification),
        len(stats_verification),
        (verification["Status"] == "Available").sum()
    ]
})

display(summary)

summary.to_csv(
    FINAL_TABLES_DIR / "overall_project_verification.csv",
    index=False
)

print("=" * 80)
print("PROJECT OUTPUT VERIFICATION COMPLETED")
print("=" * 80)

print(f"EDA Tables              : {len(eda_verification)}")
print(f"Statistical Tests       : {len(stats_verification)}")
print(f"Model Output Directories: {(verification['Status']=='Available').sum()}")

print("=" * 80)

# ============================================================================
# Code Cell 13
# ============================================================================
# ==========================================================
# Section 3: Load Dataset Summary
# ==========================================================

dataset_summary = eda_tables["modeling_dataset_summary"]

display(dataset_summary)

dataset_summary.to_csv(
    FINAL_TABLES_DIR / "dataset_summary_report.csv",
    index=False
)

print("=" * 80)
print("Dataset summary loaded and exported.")
print("=" * 80)

# ============================================================================
# Code Cell 14
# ============================================================================
# ==========================================================
# Publication-Ready Dataset Summary
# ==========================================================

study_summary = pd.DataFrame({
    "Characteristic": [
        "Data Source",
        "Study Period",
        "Number of Reports",
        "Number of Features",
        "Target Variable",
        "Serious Reports",
        "Non-Serious Reports"
    ],
    "Value": [
        "FDA FAERS",
        "2024 Quarter 1",
        "406,184",
        "51",
        "is_serious",
        "222,364 (54.74%)",
        "183,820 (45.26%)"
    ]
})

display(study_summary)

study_summary.to_csv(
    FINAL_TABLES_DIR / "study_population_summary.csv",
    index=False
)

# ============================================================================
# Code Cell 15
# ============================================================================
# ==========================================================
# List Model Comparison Files
# ==========================================================

print(COMPARISON_DIR)

print("\nFiles found:\n")

for f in sorted(COMPARISON_DIR.glob("*")):
    print(f.name)

# ============================================================================
# Code Cell 16
# ============================================================================
print(PERFORMANCE_DIR)

print("\nFiles found:\n")

for f in sorted(PERFORMANCE_DIR.glob("*")):
    print(f.name)

# ============================================================================
# Code Cell 17
# ============================================================================
# ==========================================================
# Section 4.1: Load Final Model Performance Summary
# ==========================================================

performance_summary_path = PERFORMANCE_DIR / "model_performance_summary.csv"

rq1_results = pd.read_csv(performance_summary_path)

print("=" * 80)
print("Model Performance Summary Loaded")
print("=" * 80)
print(performance_summary_path)

display(rq1_results)

rq1_results.to_csv(
    FINAL_TABLES_DIR / "rq1_model_comparison.csv",
    index=False
)

# ============================================================================
# Code Cell 18
# ============================================================================
print(rq1_results.columns.tolist())

# ============================================================================
# Code Cell 19
# ============================================================================
# ==========================================================
# Section 4.1: Load Final Model Performance Summary
# ==========================================================

performance_summary_path = PERFORMANCE_DIR / "model_performance_summary.csv"

rq1_results = pd.read_csv(performance_summary_path)

print("=" * 80)
print("Model Performance Summary Loaded")
print("=" * 80)
print(performance_summary_path)

display(
    rq1_results[
        [
            "Overall Rank",
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "Specificity",
            "F1",
            "ROC AUC",
            "CV Accuracy"
        ]
    ]
)

rq1_results.to_csv(
    FINAL_TABLES_DIR / "rq1_model_comparison.csv",
    index=False
)

print("=" * 80)
print("RQ1 model comparison table exported.")
print("=" * 80)

# ============================================================================
# Code Cell 20
# ============================================================================
# ==========================================================
# Section 4.2: Accuracy Comparison Across Models
# ==========================================================

plot_df = rq1_results.sort_values(
    by="Overall Rank",
    ascending=False
)

plt.figure(figsize=(10, 6))

bars = plt.barh(
    plot_df["Model"],
    plot_df["Accuracy"]
)

# Add value labels
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.002,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.3f}",
        va="center",
        fontsize=9
    )

plt.xlabel("Accuracy")
plt.ylabel("Model")
plt.title("Comparison of Classification Accuracy Across Supervised Models")

plt.xlim(0.65, 1.00)

plt.grid(axis="x", alpha=0.3)

plt.tight_layout()

plt.savefig(
    FINAL_FIGURES_DIR / "rq1_accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("=" * 80)
print("Accuracy comparison figure exported.")
print("=" * 80)

# ============================================================================
# Code Cell 21
# ============================================================================
# ==========================================================
# Section 4.3: ROC AUC Comparison Across Models
# ==========================================================

plot_df = rq1_results.sort_values(
    by="Overall Rank",
    ascending=False
)

plt.figure(figsize=(10, 6))

bars = plt.barh(
    plot_df["Model"],
    plot_df["ROC AUC"]
)

# Add value labels
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.001,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.3f}",
        va="center",
        fontsize=9
    )

plt.xlabel("ROC AUC")
plt.ylabel("Model")
plt.title("Comparison of ROC AUC Across Supervised Models")

plt.xlim(0.90, 1.00)

plt.grid(axis="x", alpha=0.3)

plt.tight_layout()

plt.savefig(
    FINAL_FIGURES_DIR / "rq1_roc_auc_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("=" * 80)
print("ROC AUC comparison figure exported.")
print("=" * 80)

# ============================================================================
# Code Cell 22
# ============================================================================
# ==========================================================
# Section 5.1: Inspect Available Tables for RQ2
# ==========================================================

print("=" * 80)
print("Available EDA Tables")
print("=" * 80)

for key in eda_tables.keys():
    print(key)

print("\n" + "=" * 80)
print("Available Statistical Test Tables")
print("=" * 80)

for key in stat_tests.keys():
    print(key)

# ============================================================================
# Code Cell 23
# ============================================================================
# ==========================================================
# Section 5.2: Demographic Summary
# ==========================================================

demographic_summary = eda_tables["demographic_summary"]

print("=" * 80)
print("Demographic Summary")
print("=" * 80)

display(demographic_summary)

demographic_summary.to_csv(
    FINAL_TABLES_DIR / "rq2_demographic_summary.csv",
    index=False
)

print("=" * 80)
print("RQ2 demographic summary exported.")
print("=" * 80)

# ============================================================================
# Code Cell 24
# ============================================================================
# ==========================================================
# Section 5.3: Demographic Statistical Tests
# ==========================================================

demographic_tests = stat_tests["demographic_tests"]

print("=" * 80)
print("Demographic Statistical Tests")
print("=" * 80)

display(demographic_tests)

demographic_tests.to_csv(
    FINAL_TABLES_DIR / "rq2_demographic_tests.csv",
    index=False
)

print("=" * 80)
print("RQ2 demographic statistical tests exported.")
print("=" * 80)

# ============================================================================
# Code Cell 25
# ============================================================================
# ==========================================================
# Section 5.4: Drug Burden Summary
# ==========================================================

drug_summary = eda_tables["drug_burden_summary"]

print("=" * 80)
print("Drug Burden Summary")
print("=" * 80)

display(drug_summary)

drug_summary.to_csv(
    FINAL_TABLES_DIR / "rq2_drug_burden_summary.csv",
    index=False
)

print("=" * 80)
print("RQ2 drug burden summary exported.")
print("=" * 80)

# ============================================================================
# Code Cell 26
# ============================================================================
# ==========================================================
# Section 5.5: Drug Burden Statistical Tests
# ==========================================================

drug_tests = stat_tests["drug_tests"]

print("=" * 80)
print("Drug Burden Statistical Tests")
print("=" * 80)

display(drug_tests)

drug_tests.to_csv(
    FINAL_TABLES_DIR / "rq2_drug_burden_tests.csv",
    index=False
)

print("=" * 80)
print("RQ2 drug burden statistical tests exported.")
print("=" * 80)

# ============================================================================
# Code Cell 27
# ============================================================================
# ==========================================================
# Section 5.6: Reaction Burden Summary
# ==========================================================

reaction_summary = eda_tables["reaction_burden_summary"]

print("=" * 80)
print("Reaction Burden Summary")
print("=" * 80)

display(reaction_summary)

reaction_summary.to_csv(
    FINAL_TABLES_DIR / "rq2_reaction_burden_summary.csv",
    index=False
)

print("=" * 80)
print("RQ2 reaction burden summary exported.")
print("=" * 80)

# ============================================================================
# Code Cell 28
# ============================================================================
# ==========================================================
# Section 5.7: Reaction Burden Statistical Tests
# ==========================================================

reaction_tests = stat_tests["reaction_tests"]

print("=" * 80)
print("Reaction Burden Statistical Tests")
print("=" * 80)

display(reaction_tests)

reaction_tests.to_csv(
    FINAL_TABLES_DIR / "rq2_reaction_burden_tests.csv",
    index=False
)

print("=" * 80)
print("RQ2 reaction burden statistical tests exported.")
print("=" * 80)

# ============================================================================
# Code Cell 29
# ============================================================================
# ==========================================================
# Section 6.1: Load Weighted Model Rankings
# ==========================================================

ranking_path = PERFORMANCE_DIR / "weighted_model_rankings.csv"

weighted_rankings = pd.read_csv(ranking_path)

print("=" * 80)
print("Weighted Model Rankings")
print("=" * 80)

display(weighted_rankings)

weighted_rankings.to_csv(
    FINAL_TABLES_DIR / "rq3_weighted_model_rankings.csv",
    index=False
)

print("=" * 80)
print("RQ3 weighted model rankings exported.")
print("=" * 80)

# ============================================================================
# Code Cell 30
# ============================================================================
# ==========================================================
# Section 6.2: Final Model Ranking
# ==========================================================

ranking_display = weighted_rankings[
    [
        "Overall Rank",
        "Model",
        "Accuracy",
        "F1",
        "ROC AUC",
        "CV Accuracy",
        "Weighted Score"
    ]
]

display(ranking_display)

ranking_display.to_csv(
    FINAL_TABLES_DIR / "rq3_final_model_ranking.csv",
    index=False
)

print("=" * 80)
print("Final ranking table exported.")
print("=" * 80)

# ============================================================================
# Code Cell 31
# ============================================================================
# ==========================================================
# Section 6.3: Final Weighted Model Ranking
# ==========================================================

plot_df = weighted_rankings.sort_values(
    by="Overall Rank",
    ascending=False
)

plt.figure(figsize=(10, 6))

bars = plt.barh(
    plot_df["Model"],
    plot_df["Weighted Score"]
)

# Add score labels
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.08,
        bar.get_y() + bar.get_height()/2,
        f"{width:.1f}",
        va="center",
        fontsize=9
    )

plt.xlabel("Weighted Score (Lower = Better)")
plt.ylabel("Model")
plt.title("Overall Model Ranking Based on Weighted Performance Metrics")

plt.grid(axis="x", alpha=0.3)

plt.tight_layout()

plt.savefig(
    FINAL_FIGURES_DIR / "rq3_weighted_model_ranking.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("=" * 80)
print("Weighted model ranking figure exported.")
print("=" * 80)

# ============================================================================
# Code Cell 32
# ============================================================================
# ==========================================================
# Section 7.1: Load Cluster Summary
# ==========================================================

cluster_summary_path = CLUSTERING_DIR / "kmeans_cluster_summary.csv"

cluster_summary = pd.read_csv(cluster_summary_path)

print("=" * 80)
print("Cluster Summary")
print("=" * 80)

display(cluster_summary)

cluster_summary.to_csv(
    FINAL_TABLES_DIR / "rq4_cluster_summary.csv",
    index=False
)

print("=" * 80)
print("Cluster summary exported.")
print("=" * 80)

# ============================================================================
# Code Cell 33
# ============================================================================
# ==========================================================
# Section 7.2: Load Serious Event Distribution by Cluster
# ==========================================================

cluster_serious_path = CLUSTERING_DIR / "kmeans_cluster_serious_distribution.csv"

cluster_serious = pd.read_csv(cluster_serious_path)

print("=" * 80)
print("Serious Event Distribution by Cluster")
print("=" * 80)

display(cluster_serious)

cluster_serious.to_csv(
    FINAL_TABLES_DIR / "rq4_cluster_serious_distribution.csv",
    index=False
)

print("=" * 80)
print("Cluster serious distribution exported.")
print("=" * 80)

# ============================================================================
# Code Cell 34
# ============================================================================
# ==========================================================
# Final Notebook Verification
# ==========================================================

print("=" * 80)
print("NOTEBOOK 05 COMPLETED")
print("=" * 80)

print(f"EDA Tables Loaded                 : {len(eda_tables)}")
print(f"Statistical Test Tables Loaded    : {len(stat_tests)}")
print(f"Final Model                       : Random Forest")
print(f"Number of Supervised Models       : 9")
print(f"Number of Clusters                : 7")
print(f"Final Reports                     : 406,184")

print("=" * 80)
print("Notebook 05 completed successfully.")
print("=" * 80)

# ============================================================================
# Code Cell 35
# ============================================================================
# ==========================================================
# Create Midterm Report Resource Folder
# ==========================================================

from pathlib import Path
import shutil
import pandas as pd

# ----------------------------------------------------------
# Project Directories
# ----------------------------------------------------------

PROJECT_DIR = Path("/Users/sumi/Documents/GEORGIA TECH/ISYE 7406/project/Project_Solution")

MIDTERM_DIR = PROJECT_DIR / "mid_report_doc"
MIDTERM_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------
# Source Directories
# ----------------------------------------------------------

EDA_FIG_DIR = PROJECT_DIR / "results" / "eda" / "figures"
EDA_TABLE_DIR = PROJECT_DIR / "results" / "eda" / "tables"

# ----------------------------------------------------------
# Images Needed
# ----------------------------------------------------------

image_files = [
    "target_distribution.png",
    "missing_values.png",
    "drug_burden_summary.png",
    "correlation_heatmap.png"
]

print("="*70)
print("Copying Images")
print("="*70)

for img in image_files:

    src = EDA_FIG_DIR / img
    dst = MIDTERM_DIR / img

    if src.exists():
        shutil.copy2(src, dst)
        print(f"✅ {img}")
    else:
        print(f"❌ Missing: {img}")

# ----------------------------------------------------------
# Tables Needed
# ----------------------------------------------------------

table_files = [
    "summary_statistics.csv"
]

print("\n" + "="*70)
print("Copying Tables")
print("="*70)

for tbl in table_files:

    src = EDA_TABLE_DIR / tbl
    dst = MIDTERM_DIR / tbl

    if src.exists():
        shutil.copy2(src, dst)
        print(f"✅ {tbl}")
    else:
        print(f"❌ Missing: {tbl}")

print("\nDone.")
print("Folder:", MIDTERM_DIR)

# ============================================================================
# Code Cell 36
# ============================================================================
from pathlib import Path

PROJECT_DIR = Path("/Users/sumi/Documents/GEORGIA TECH/ISYE 7406/project/Project_Solution")

print("="*70)
print("Searching for PNG files...")
print("="*70)

for f in PROJECT_DIR.rglob("*.png"):
    print(f)

# ============================================================================
# Code Cell 37
# ============================================================================
# ==========================================================
# Create Midterm Report Figure Folder
# ==========================================================

from pathlib import Path
import shutil
import pandas as pd

PROJECT_DIR = Path("/Users/sumi/Documents/GEORGIA TECH/ISYE 7406/project/Project_Solution")

MIDTERM_DIR = PROJECT_DIR / "mid_report_doc"
MIDTERM_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------
# Files to Copy
# ----------------------------------------------------------

files_to_copy = {

    # Figures
    PROJECT_DIR / "figures" / "project_workflow.png":
        "project_workflow.png",

    PROJECT_DIR / "figures" / "notebook03" / "section13_target_distribution.png":
        "target_distribution.png",

    PROJECT_DIR / "figures" / "notebook03" / "demo_missing_values.png":
        "missing_values.png",

    PROJECT_DIR / "figures" / "notebook03" / "section13B_drug_burden_by_target.png":
        "drug_burden_summary.png",

    PROJECT_DIR / "figures" / "notebook03" / "section14A_correlation_heatmap.png":
        "correlation_heatmap.png",

    # Table
    PROJECT_DIR / "results" / "eda" / "tables" / "summary_statistics.csv":
        "summary_statistics.csv"

}

print("="*70)
print("Copying Midterm Report Resources")
print("="*70)

for src, new_name in files_to_copy.items():

    dst = MIDTERM_DIR / new_name

    if src.exists():
        shutil.copy2(src, dst)
        print(f"✅ {new_name}")
    else:
        print(f"❌ Missing: {src}")

print("\nFolder Created:")
print(MIDTERM_DIR)

# ============================================================================
# Code Cell 38
# ============================================================================
# ==========================================================
# Create Table 3
# ==========================================================

summary = pd.read_csv(MIDTERM_DIR / "summary_statistics.csv")

table3 = summary.iloc[:8].copy()

table3.to_csv(
    MIDTERM_DIR / "Table_3_Selected_Summary_Statistics.csv",
    index=False
)

print("="*70)
print("Table 3 Created")
print("="*70)

display(table3)

# ============================================================================
# Code Cell 39
# ============================================================================
# ==========================================================
# Create Improved Table 3: Selected Summary Statistics
# ==========================================================

import pandas as pd
from pathlib import Path

MIDTERM_DIR = Path("/Users/sumi/Documents/GEORGIA TECH/ISYE 7406/project/Project_Solution/mid_report_doc")

summary_path = MIDTERM_DIR / "summary_statistics.csv"

summary = pd.read_csv(summary_path)

# If variable names were saved as an unnamed first column, rename it
if "Unnamed: 0" in summary.columns:
    summary = summary.rename(columns={"Unnamed: 0": "Variable"})

# If no Variable column exists, try to recover from index
if "Variable" not in summary.columns:
    summary.insert(0, "Variable", summary.index)

# Select report-friendly columns
table3 = summary[
    [
        "Variable",
        "count",
        "Missing",
        "Missing (%)",
        "mean",
        "std",
        "50%",
        "min",
        "max"
    ]
].copy()

# Rename columns for report readability
table3 = table3.rename(columns={
    "count": "Non-Missing Count",
    "mean": "Mean",
    "std": "Std. Dev.",
    "50%": "Median",
    "min": "Min",
    "max": "Max"
})

# Keep selected important variables only
selected_variables = [
    "age_years",
    "weight_kg",
    "total_drugs",
    "num_unique_drugs",
    "num_primary_suspect",
    "num_secondary_suspect",
    "num_concomitant",
    "num_interacting",
    "total_reactions",
    "num_unique_reactions"
]

table3 = table3[table3["Variable"].isin(selected_variables)]

# Round numeric columns
numeric_cols = table3.select_dtypes(include="number").columns
table3[numeric_cols] = table3[numeric_cols].round(2)

# Save improved table
output_path = MIDTERM_DIR / "Table_3_Selected_Summary_Statistics_Improved.csv"

table3.to_csv(output_path, index=False)

print("=" * 80)
print("Improved Table 3 Saved")
print("=" * 80)
print(output_path)

display(table3)

# ============================================================================
# Code Cell 40
# ============================================================================
# =============================================================================
# GENERATE ALL EDA FIGURES FOR THE FINAL REPORT
# =============================================================================
#
# Output folder:
# Project_Solution/Final_report_all/figures/
#
# Figures generated:
# 1. target_distribution.png
# 2. missing_values.png
# 3. demographic_summary.png
# 4. drug_burden_summary.png
# 5. reaction_burden_summary.png
# 6. indication_burden_summary.png
# 7. therapy_summary.png
# 8. reporter_summary.png
# 9. correlation_heatmap.png
#
# =============================================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


# =============================================================================
# 1. DEFINE PROJECT PATHS
# =============================================================================

# Your notebook is located inside:
# Project_Solution/notebooks/05_notebook5.ipynb
#
# Path.cwd() may therefore point either to:
# - Project_Solution/
# - Project_Solution/notebooks/
#
# This code detects the correct project root automatically.

CURRENT_DIR = Path.cwd()

if CURRENT_DIR.name.lower() == "notebooks":
    PROJECT_ROOT = CURRENT_DIR.parent
else:
    PROJECT_ROOT = CURRENT_DIR

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "Final_report_all" / "figures"

# Create the final report figure directory if it does not already exist.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PROJECT PATHS")
print("=" * 80)
print(f"Current directory : {CURRENT_DIR}")
print(f"Project root      : {PROJECT_ROOT}")
print(f"Processed data    : {PROCESSED_DIR}")
print(f"Figure output     : {OUTPUT_DIR}")


# =============================================================================
# 2. FIND AND LOAD THE FINAL MODELING DATASET
# =============================================================================

def find_modeling_dataset(processed_dir: Path) -> Path:
    """
    Search the processed-data directory for a file containing the final
    modeling dataset.

    The correct file should contain:
    - the target column: is_serious
    - approximately 51 columns
    - one row per FAERS report
    """

    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed-data directory was not found:\n{processed_dir}"
        )

    # Search common data formats.
    candidate_files = (
        list(processed_dir.glob("*.parquet"))
        + list(processed_dir.glob("*.csv"))
        + list(processed_dir.glob("*.pkl"))
        + list(processed_dir.glob("*.pickle"))
    )

    if not candidate_files:
        raise FileNotFoundError(
            f"No parquet, CSV, or pickle files were found in:\n{processed_dir}"
        )

    print("\nSearching for the dataset containing 'is_serious'...")

    for file_path in candidate_files:
        try:
            # Read only enough information to inspect columns.
            if file_path.suffix.lower() == ".parquet":
                preview = pd.read_parquet(file_path)

            elif file_path.suffix.lower() == ".csv":
                preview = pd.read_csv(file_path, nrows=5)

            else:
                preview = pd.read_pickle(file_path)

            if "is_serious" in preview.columns:
                print(f"✅ Modeling dataset found: {file_path.name}")
                return file_path

        except Exception as error:
            print(f"Skipped {file_path.name}: {error}")

    raise FileNotFoundError(
        "A processed dataset containing the target column "
        "'is_serious' could not be found."
    )


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the final modeling dataset based on its file extension."""

    suffix = file_path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(file_path)

    if suffix == ".csv":
        return pd.read_csv(file_path)

    if suffix in [".pkl", ".pickle"]:
        return pd.read_pickle(file_path)

    raise ValueError(f"Unsupported file type: {suffix}")


MODELING_FILE = find_modeling_dataset(PROCESSED_DIR)
df = load_dataset(MODELING_FILE)

print("\n" + "=" * 80)
print("FINAL MODELING DATASET")
print("=" * 80)
print(f"File    : {MODELING_FILE.name}")
print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]:,}")
print(f"Target  : {'is_serious' if 'is_serious' in df.columns else 'Not found'}")


# =============================================================================
# 3. PREPARE LABELS AND PLOTTING SAMPLE
# =============================================================================

# Convert the numeric target into readable labels for chart axes and legends.
df["serious_status"] = df["is_serious"].map(
    {
        0: "Non-serious",
        1: "Serious"
    }
)

# Use a reproducible sample for visually dense plots.
# This keeps plotting fast while preserving the target proportions.
PLOT_SAMPLE_SIZE = min(50_000, len(df))

plot_df = df.sample(
    n=PLOT_SAMPLE_SIZE,
    random_state=42
).copy()

print(f"Plotting sample: {len(plot_df):,} observations")


# =============================================================================
# 4. GLOBAL FIGURE SETTINGS
# =============================================================================

sns.set_theme(
    style="whitegrid",
    context="talk"
)

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 18
    }
)


def save_figure(filename: str) -> None:
    """
    Save the current Matplotlib figure as a publication-quality PNG.
    """

    output_path = OUTPUT_DIR / filename

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.show()
    plt.close()

    print(f"✅ Saved: {output_path}")


def existing_columns(dataframe: pd.DataFrame, requested_columns: list[str]) -> list[str]:
    """
    Return only requested columns that exist in the dataset.
    """

    available = [column for column in requested_columns if column in dataframe.columns]

    missing = [column for column in requested_columns if column not in dataframe.columns]

    if missing:
        print(f"Columns not found and skipped: {missing}")

    return available


# =============================================================================
# FIGURE 1: TARGET DISTRIBUTION
# =============================================================================

target_counts = (
    df["serious_status"]
    .value_counts()
    .reindex(["Non-serious", "Serious"])
)

target_percentages = target_counts / target_counts.sum() * 100

fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(
    target_counts.index,
    target_counts.values
)

ax.set_title("Distribution of Serious and Non-serious Adverse Event Reports")
ax.set_xlabel("Adverse Event Classification")
ax.set_ylabel("Number of Reports")

# Add exact count and percentage above each bar.
for bar, count, percentage in zip(
    bars,
    target_counts.values,
    target_percentages.values
):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{count:,.0f}\n({percentage:.2f}%)",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )

ax.ticklabel_format(style="plain", axis="y")
sns.despine()

save_figure("target_distribution.png")


# =============================================================================
# FIGURE 2: MISSING VALUES
# =============================================================================

missing_percentage = (
    df.drop(columns=["serious_status"], errors="ignore")
    .isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

# Retain variables with missingness greater than zero.
missing_plot = missing_percentage[missing_percentage > 0]

fig_height = max(6, len(missing_plot) * 0.32)

fig, ax = plt.subplots(figsize=(12, fig_height))

sns.barplot(
    x=missing_plot.values,
    y=missing_plot.index,
    ax=ax
)

ax.set_title("Percentage of Missing Values Across Analytical Variables")
ax.set_xlabel("Missing Values (%)")
ax.set_ylabel("Variable")

# Add missing percentage labels to the bars.
for index, value in enumerate(missing_plot.values):
    ax.text(
        value + 0.5,
        index,
        f"{value:.1f}%",
        va="center",
        fontsize=9
    )

ax.set_xlim(0, max(100, missing_plot.max() + 10))
sns.despine()

save_figure("missing_values.png")


# =============================================================================
# FIGURE 3: DEMOGRAPHIC SUMMARY
# =============================================================================

demographic_columns = existing_columns(
    plot_df,
    [
        "age_years",
        "weight_kg"
    ]
)

if demographic_columns:

    demographic_long = plot_df[
        demographic_columns + ["serious_status"]
    ].melt(
        id_vars="serious_status",
        value_vars=demographic_columns,
        var_name="Variable",
        value_name="Value"
    )

    demographic_label_map = {
        "age_years": "Age (years)",
        "weight_kg": "Weight (kg)"
    }

    demographic_long["Variable"] = (
        demographic_long["Variable"]
        .map(demographic_label_map)
    )

    fig, axes = plt.subplots(
        1,
        len(demographic_columns),
        figsize=(14, 6)
    )

    # Ensure axes is iterable even when only one column exists.
    axes = np.atleast_1d(axes)

    for axis, variable in zip(
        axes,
        demographic_long["Variable"].dropna().unique()
    ):
        subset = demographic_long[
            demographic_long["Variable"] == variable
        ]

        sns.boxplot(
            data=subset,
            x="serious_status",
            y="Value",
            showfliers=False,
            ax=axis
        )

        axis.set_title(variable)
        axis.set_xlabel("")
        axis.set_ylabel(variable)

    fig.suptitle(
        "Demographic Characteristics by Adverse Event Classification",
        y=1.02
    )

    save_figure("demographic_summary.png")


# =============================================================================
# FIGURE 4: DRUG BURDEN SUMMARY
# =============================================================================

drug_burden_columns = existing_columns(
    plot_df,
    [
        "total_drugs",
        "num_unique_drugs",
        "num_primary_suspect",
        "num_secondary_suspect",
        "num_concomitant",
        "num_interacting"
    ]
)

if drug_burden_columns:

    drug_summary = (
        plot_df
        .groupby("serious_status")[drug_burden_columns]
        .median()
        .T
    )

    drug_label_map = {
        "total_drugs": "Total Drugs",
        "num_unique_drugs": "Unique Drugs",
        "num_primary_suspect": "Primary Suspect",
        "num_secondary_suspect": "Secondary Suspect",
        "num_concomitant": "Concomitant",
        "num_interacting": "Interacting"
    }

    drug_summary.index = [
        drug_label_map.get(column, column)
        for column in drug_summary.index
    ]

    drug_summary.plot(
        kind="barh",
        figsize=(12, 8)
    )

    plt.title("Median Drug Burden by Adverse Event Classification")
    plt.xlabel("Median Number of Drugs")
    plt.ylabel("Drug Burden Variable")
    plt.legend(
        title="Classification",
        loc="best"
    )

    save_figure("drug_burden_summary.png")


# =============================================================================
# FIGURE 5: REACTION BURDEN SUMMARY
# =============================================================================

reaction_count_columns = existing_columns(
    plot_df,
    [
        "total_reactions",
        "num_unique_reactions"
    ]
)

reaction_flag_columns = existing_columns(
    df,
    [
        "has_aki",
        "has_dka",
        "has_hypoglycemia",
        "has_lactic_acidosis",
        "has_amputation",
        "has_genital_infection"
    ]
)

# Support alternative names if the original features do not use "has_".
alternative_reaction_flags = {
    "aki": "AKI",
    "dka": "DKA",
    "hypoglycemia": "Hypoglycemia",
    "lactic_acidosis": "Lactic Acidosis",
    "amputation": "Amputation",
    "genital_infection": "Genital Infection"
}

for alternative_name in alternative_reaction_flags:
    if (
        alternative_name in df.columns
        and alternative_name not in reaction_flag_columns
    ):
        reaction_flag_columns.append(alternative_name)

if reaction_count_columns or reaction_flag_columns:

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7)
    )

    # Left panel: median reaction counts.
    if reaction_count_columns:
        reaction_count_summary = (
            plot_df
            .groupby("serious_status")[reaction_count_columns]
            .median()
            .T
        )

        reaction_count_summary.index = [
            "Total Reactions"
            if column == "total_reactions"
            else "Unique Reactions"
            for column in reaction_count_summary.index
        ]

        reaction_count_summary.plot(
            kind="bar",
            ax=axes[0]
        )

        axes[0].set_title("Median Reaction Burden")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Median Number of Reactions")
        axes[0].tick_params(axis="x", rotation=0)
        axes[0].legend(title="Classification")

    else:
        axes[0].axis("off")

    # Right panel: prevalence of clinically important reaction categories.
    if reaction_flag_columns:
        reaction_prevalence = (
            df
            .groupby("serious_status")[reaction_flag_columns]
            .mean()
            .mul(100)
            .T
        )

        reaction_labels = {
            "has_aki": "AKI",
            "aki": "AKI",
            "has_dka": "DKA",
            "dka": "DKA",
            "has_hypoglycemia": "Hypoglycemia",
            "hypoglycemia": "Hypoglycemia",
            "has_lactic_acidosis": "Lactic Acidosis",
            "lactic_acidosis": "Lactic Acidosis",
            "has_amputation": "Amputation",
            "amputation": "Amputation",
            "has_genital_infection": "Genital Infection",
            "genital_infection": "Genital Infection"
        }

        reaction_prevalence.index = [
            reaction_labels.get(column, column)
            for column in reaction_prevalence.index
        ]

        reaction_prevalence.plot(
            kind="barh",
            ax=axes[1]
        )

        axes[1].set_title("Prevalence of Clinically Important Reactions")
        axes[1].set_xlabel("Reports Containing Reaction (%)")
        axes[1].set_ylabel("")
        axes[1].legend(title="Classification")

    else:
        axes[1].axis("off")

    fig.suptitle(
        "Reaction Burden by Adverse Event Classification",
        y=1.02
    )

    save_figure("reaction_burden_summary.png")


# =============================================================================
# FIGURE 6: INDICATION BURDEN SUMMARY
# =============================================================================

indication_count_columns = existing_columns(
    plot_df,
    [
        "total_indications",
        "num_unique_indications"
    ]
)

requested_indication_flags = [
    "has_type2_diabetes",
    "has_type1_diabetes",
    "has_any_diabetes",
    "has_hypertension",
    "has_ckd",
    "has_heart_failure",
    "has_obesity",
    "has_unknown_indication",
    "type2_diabetes",
    "type1_diabetes",
    "any_diabetes",
    "hypertension",
    "ckd",
    "heart_failure",
    "obesity",
    "unknown_indication"
]

indication_flag_columns = existing_columns(
    df,
    requested_indication_flags
)

# Avoid duplicate clinical categories when both naming versions exist.
preferred_indication_columns = []

indication_pairs = [
    ("has_type2_diabetes", "type2_diabetes"),
    ("has_type1_diabetes", "type1_diabetes"),
    ("has_any_diabetes", "any_diabetes"),
    ("has_hypertension", "hypertension"),
    ("has_ckd", "ckd"),
    ("has_heart_failure", "heart_failure"),
    ("has_obesity", "obesity"),
    ("has_unknown_indication", "unknown_indication")
]

for preferred, alternative in indication_pairs:
    if preferred in indication_flag_columns:
        preferred_indication_columns.append(preferred)
    elif alternative in indication_flag_columns:
        preferred_indication_columns.append(alternative)

indication_flag_columns = preferred_indication_columns

if indication_count_columns or indication_flag_columns:

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(17, 8)
    )

    if indication_count_columns:
        indication_count_summary = (
            plot_df
            .groupby("serious_status")[indication_count_columns]
            .median()
            .T
        )

        indication_count_summary.index = [
            "Total Indications"
            if column == "total_indications"
            else "Unique Indications"
            for column in indication_count_summary.index
        ]

        indication_count_summary.plot(
            kind="bar",
            ax=axes[0]
        )

        axes[0].set_title("Median Indication Burden")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Median Number of Indications")
        axes[0].tick_params(axis="x", rotation=0)
        axes[0].legend(title="Classification")

    else:
        axes[0].axis("off")

    if indication_flag_columns:
        indication_prevalence = (
            df
            .groupby("serious_status")[indication_flag_columns]
            .mean()
            .mul(100)
            .T
        )

        indication_labels = {
            "has_type2_diabetes": "Type 2 Diabetes",
            "type2_diabetes": "Type 2 Diabetes",
            "has_type1_diabetes": "Type 1 Diabetes",
            "type1_diabetes": "Type 1 Diabetes",
            "has_any_diabetes": "Any Diabetes",
            "any_diabetes": "Any Diabetes",
            "has_hypertension": "Hypertension",
            "hypertension": "Hypertension",
            "has_ckd": "Chronic Kidney Disease",
            "ckd": "Chronic Kidney Disease",
            "has_heart_failure": "Heart Failure",
            "heart_failure": "Heart Failure",
            "has_obesity": "Obesity",
            "obesity": "Obesity",
            "has_unknown_indication": "Unknown Indication",
            "unknown_indication": "Unknown Indication"
        }

        indication_prevalence.index = [
            indication_labels.get(column, column)
            for column in indication_prevalence.index
        ]

        indication_prevalence.plot(
            kind="barh",
            ax=axes[1]
        )

        axes[1].set_title("Prevalence of Selected Disease Indications")
        axes[1].set_xlabel("Reports Containing Indication (%)")
        axes[1].set_ylabel("")
        axes[1].legend(title="Classification")

    else:
        axes[1].axis("off")

    fig.suptitle(
        "Indication Burden by Adverse Event Classification",
        y=1.02
    )

    save_figure("indication_burden_summary.png")


# =============================================================================
# FIGURE 7: THERAPY CHARACTERISTICS
# =============================================================================

therapy_columns = existing_columns(
    plot_df,
    [
        "total_therapies",
        "num_therapies_with_duration",
        "mean_therapy_duration_days",
        "median_therapy_duration_days",
        "min_therapy_duration_days",
        "max_therapy_duration_days"
    ]
)

if therapy_columns:

    therapy_summary = (
        plot_df
        .groupby("serious_status")[therapy_columns]
        .median()
        .T
    )

    therapy_label_map = {
        "total_therapies": "Total Therapies",
        "num_therapies_with_duration": "Therapies with Duration",
        "mean_therapy_duration_days": "Mean Duration",
        "median_therapy_duration_days": "Median Duration",
        "min_therapy_duration_days": "Minimum Duration",
        "max_therapy_duration_days": "Maximum Duration"
    }

    therapy_summary.index = [
        therapy_label_map.get(column, column)
        for column in therapy_summary.index
    ]

    therapy_summary.plot(
        kind="barh",
        figsize=(13, 8)
    )

    plt.title("Median Therapy Characteristics by Adverse Event Classification")
    plt.xlabel("Median Value")
    plt.ylabel("Therapy Variable")
    plt.legend(title="Classification")

    save_figure("therapy_summary.png")


# =============================================================================
# FIGURE 8: REPORTER CHARACTERISTICS
# =============================================================================

reporter_count_columns = existing_columns(
    df,
    [
        "num_reporter_sources"
    ]
)

requested_reporter_flags = [
    "has_health_professional_report",
    "has_consumer_report",
    "has_foreign_report"
]

reporter_flag_columns = existing_columns(
    df,
    requested_reporter_flags
)

if reporter_count_columns or reporter_flag_columns:

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 7)
    )

    if reporter_count_columns:
        reporter_count_summary = (
            df
            .groupby("serious_status")[reporter_count_columns]
            .mean()
            .T
        )

        reporter_count_summary.index = ["Reporter Sources"]

        reporter_count_summary.plot(
            kind="bar",
            ax=axes[0]
        )

        axes[0].set_title("Average Number of Reporter Sources")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Mean Number of Sources")
        axes[0].tick_params(axis="x", rotation=0)
        axes[0].legend(title="Classification")

    else:
        axes[0].axis("off")

    if reporter_flag_columns:
        reporter_prevalence = (
            df
            .groupby("serious_status")[reporter_flag_columns]
            .mean()
            .mul(100)
            .T
        )

        reporter_labels = {
            "has_health_professional_report": "Healthcare Professional",
            "has_consumer_report": "Consumer",
            "has_foreign_report": "Foreign Report"
        }

        reporter_prevalence.index = [
            reporter_labels.get(column, column)
            for column in reporter_prevalence.index
        ]

        reporter_prevalence.plot(
            kind="barh",
            ax=axes[1]
        )

        axes[1].set_title("Reporter-Type Prevalence")
        axes[1].set_xlabel("Reports Containing Reporter Type (%)")
        axes[1].set_ylabel("")
        axes[1].legend(title="Classification")

    else:
        axes[1].axis("off")

    fig.suptitle(
        "Reporter Characteristics by Adverse Event Classification",
        y=1.02
    )

    save_figure("reporter_summary.png")


# =============================================================================
# FIGURE 9: CORRELATION HEATMAP
# =============================================================================

# Select numeric predictor variables.
numeric_df = df.select_dtypes(include=np.number).copy()

# Remove identifiers and the target because the purpose is to evaluate
# relationships among predictors.
columns_to_remove = [
    "primaryid",
    "caseid",
    "is_serious"
]

numeric_df = numeric_df.drop(
    columns=columns_to_remove,
    errors="ignore"
)

# Retain features with meaningful variation.
numeric_df = numeric_df.loc[
    :,
    numeric_df.nunique(dropna=True) > 1
]

# Limit the heatmap to the most analytically important numeric features.
preferred_correlation_columns = [
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
    "total_therapies",
    "num_therapies_with_duration",
    "mean_therapy_duration_days",
    "median_therapy_duration_days",
    "min_therapy_duration_days",
    "max_therapy_duration_days",
    "num_reporter_sources"
]

correlation_columns = [
    column
    for column in preferred_correlation_columns
    if column in numeric_df.columns
]

# Fall back to the first 25 numeric columns if preferred variables are unavailable.
if len(correlation_columns) < 3:
    correlation_columns = list(numeric_df.columns[:25])

correlation_matrix = (
    numeric_df[correlation_columns]
    .sample(
        n=min(75_000, len(numeric_df)),
        random_state=42
    )
    .corr()
)

# Improve axis labels for the final report.
correlation_labels = {
    "age_years": "Age",
    "weight_kg": "Weight",
    "total_drugs": "Total Drugs",
    "num_unique_drugs": "Unique Drugs",
    "num_primary_suspect": "Primary Suspect",
    "num_secondary_suspect": "Secondary Suspect",
    "num_concomitant": "Concomitant",
    "num_interacting": "Interacting",
    "total_reactions": "Total Reactions",
    "num_unique_reactions": "Unique Reactions",
    "total_indications": "Total Indications",
    "num_unique_indications": "Unique Indications",
    "total_therapies": "Total Therapies",
    "num_therapies_with_duration": "Therapies with Duration",
    "mean_therapy_duration_days": "Mean Duration",
    "median_therapy_duration_days": "Median Duration",
    "min_therapy_duration_days": "Minimum Duration",
    "max_therapy_duration_days": "Maximum Duration",
    "num_reporter_sources": "Reporter Sources"
}

correlation_matrix = correlation_matrix.rename(
    index=correlation_labels,
    columns=correlation_labels
)

fig, ax = plt.subplots(figsize=(17, 14))

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 7},
    linewidths=0.4,
    square=True,
    cbar_kws={
        "label": "Pearson Correlation"
    },
    ax=ax
)

ax.set_title("Correlation Matrix of Engineered Numerical Features", pad=20)
ax.tick_params(axis="x", rotation=55)
ax.tick_params(axis="y", rotation=0)

save_figure("correlation_heatmap.png")


# =============================================================================
# 5. VERIFY ALL GENERATED FIGURES
# =============================================================================

generated_figures = sorted(
    OUTPUT_DIR.glob("*.png")
)

print("\n" + "=" * 80)
print("EDA FIGURE EXPORT COMPLETED")
print("=" * 80)

for figure_path in generated_figures:
    print(f"✅ {figure_path.name}")

print(f"\nTotal PNG figures in output folder: {len(generated_figures)}")
print(f"\nSaved to:\n{OUTPUT_DIR}")

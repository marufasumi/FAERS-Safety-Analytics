"""
Shared constants and filesystem paths for the FAERS Streamlit application.

All paths are derived from this file's location so the application works
both locally and on Streamlit Community Cloud.
"""

from pathlib import Path


# ---------------------------------------------------------------------
# Project directories
# ---------------------------------------------------------------------

# constants.py is located inside:
# FAERS_Streamlit_App/utils/constants.py
#
# Therefore, parent.parent points to:
# FAERS_Streamlit_App/
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"
PAGES_DIR = BASE_DIR / "pages"
STREAMLIT_DIR = BASE_DIR / ".streamlit"


# ---------------------------------------------------------------------
# Model files
# ---------------------------------------------------------------------

MODEL_PATH = MODELS_DIR / "random_forest_pipeline.joblib"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"


# ---------------------------------------------------------------------
# Core data files
# ---------------------------------------------------------------------

TARGET_DISTRIBUTION_PATH = DATA_DIR / "target_distribution.csv"
DEMOGRAPHIC_SUMMARY_PATH = DATA_DIR / "demographic_summary.csv"
DRUG_BURDEN_SUMMARY_PATH = DATA_DIR / "drug_burden_summary.csv"
REACTION_BURDEN_SUMMARY_PATH = DATA_DIR / "reaction_burden_summary.csv"

MISSING_VALUES_PATH = DATA_DIR / "missing_values.csv"
NUMERIC_SUMMARY_PATH = DATA_DIR / "numeric_summary.csv"
THERAPY_SUMMARY_PATH = DATA_DIR / "therapy_summary.csv"
INDICATION_BURDEN_SUMMARY_PATH = DATA_DIR / "indication_burden_summary.csv"
REPORTER_SUMMARY_PATH = DATA_DIR / "reporter_summary.csv"

MODEL_PERFORMANCE_PATH = DATA_DIR / "model_performance_summary.csv"
WEIGHTED_MODEL_RANKINGS_PATH = DATA_DIR / "weighted_model_rankings.csv"
CLUSTER_SUMMARY_PATH = DATA_DIR / "cluster_summary.csv"


# ---------------------------------------------------------------------
# Statistical test files
# ---------------------------------------------------------------------

STATISTICAL_TEST_FILES = {
    "Demographic tests": DATA_DIR / "demographic_tests.csv",
    "Drug burden tests": DATA_DIR / "drug_tests.csv",
    "Reaction burden tests": DATA_DIR / "reaction_tests.csv",
    "Indication burden tests": DATA_DIR / "indication_tests.csv",
    "Therapy tests": DATA_DIR / "therapy_tests.csv",
    "Reporter tests": DATA_DIR / "reporter_tests.csv",
    "Reaction category tests": DATA_DIR / "reaction_category_tests.csv",
    "Indication category tests": DATA_DIR / "indication_category_tests.csv",
}


# ---------------------------------------------------------------------
# Key project metrics
# ---------------------------------------------------------------------

TOTAL_REPORTS = 406_184
TOTAL_PREDICTORS = 49
TOTAL_MODELS = 9

SERIOUS_REPORTS = 222_364
NON_SERIOUS_REPORTS = 183_820
SERIOUS_PERCENT = 54.74
NON_SERIOUS_PERCENT = 45.26

RANDOM_FOREST_ACCURACY = 0.925760
RANDOM_FOREST_ROC_AUC = 0.968455
RANDOM_FOREST_F1 = 0.932483
RANDOM_FOREST_CV_ACCURACY = 0.924406

KMEANS_K = 7


# ---------------------------------------------------------------------
# Scientific disclaimer
# ---------------------------------------------------------------------

DISCLAIMER = (
    "This application is an educational pharmacovigilance analytics tool "
    "and must not be used for clinical diagnosis or treatment decisions. "
    "The model classifies FAERS-style reports and does not estimate an "
    "individual patient's clinical risk."
)
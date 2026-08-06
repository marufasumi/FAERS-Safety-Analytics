# Extracted from: 04_Statistical_Learning_Models.ipynb .ipynb
# Complete executable code cells in notebook order.


# ============================================================================
# Code Cell 1
# ============================================================================
# ============================================================
# Section 1 — Project Setup
# ============================================================

# -------------------------
# Standard Library Imports
# -------------------------
from pathlib import Path
import warnings
import time
import joblib

# -------------------------
# Data Manipulation
# -------------------------
import numpy as np
import pandas as pd

# -------------------------
# Visualization
# -------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Scikit-Learn: Model Selection
# -------------------------
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_score,
    cross_validate
)

# -------------------------
# Scikit-Learn: Preprocessing
# -------------------------
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# -------------------------
# Scikit-Learn: Metrics
# -------------------------
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)

# -------------------------
# Scikit-Learn: Core Statistical Learning Models
# -------------------------
from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# -------------------------
# Scikit-Learn: Extended Comparative Models
# -------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# -------------------------
# Global Settings
# -------------------------
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -------------------------
# Plot Settings
# -------------------------
sns.set_theme(
    style="whitegrid",
    context="talk"
)

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

# -------------------------
# Project Directories
# -------------------------
PROJECT_ROOT = Path.cwd().parent

DATA_DIR = PROJECT_ROOT / "data"
MODELING_DATA_DIR = DATA_DIR / "modeling"

EDA_TABLE_DIR = PROJECT_ROOT / "results" / "eda" / "tables"
EDA_STAT_DIR = PROJECT_ROOT / "results" / "eda" / "statistics"

FIGURE_DIR = PROJECT_ROOT / "figures"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"
MODELING_RESULT_DIR = RESULT_DIR / "modeling"

CONFUSION_MATRIX_DIR = MODELING_RESULT_DIR / "confusion_matrices"
ROC_CURVE_DIR = MODELING_RESULT_DIR / "roc_curves"
PR_CURVE_DIR = MODELING_RESULT_DIR / "precision_recall_curves"

# -------------------------
# Create Modeling Output Directories
# -------------------------
output_dirs = [
    MODEL_DIR,
    MODELING_RESULT_DIR,
    CONFUSION_MATRIX_DIR,
    ROC_CURVE_DIR,
    PR_CURVE_DIR
]

for directory in output_dirs:
    directory.mkdir(parents=True, exist_ok=True)

# -------------------------
# Key File Paths
# -------------------------
MODELING_DATA_PATH = MODELING_DATA_DIR / "faers_modeling_dataset.parquet"

PERFORMANCE_SUMMARY_PATH = MODELING_RESULT_DIR / "performance_summary.csv"
CV_RESULTS_PATH = MODELING_RESULT_DIR / "cross_validation_results.csv"
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"

# -------------------------
# Setup Validation
# -------------------------
print("=" * 70)
print("SECTION 1: PROJECT SETUP COMPLETE")
print("=" * 70)

print(f"Project Root              : {PROJECT_ROOT}")
print(f"Modeling Dataset Path     : {MODELING_DATA_PATH}")
print(f"EDA Tables Directory      : {EDA_TABLE_DIR}")
print(f"EDA Statistics Directory  : {EDA_STAT_DIR}")
print(f"Figure Directory          : {FIGURE_DIR}")
print(f"Model Directory           : {MODEL_DIR}")
print(f"Modeling Results Directory: {MODELING_RESULT_DIR}")

print("=" * 70)
print(f"Random State              : {RANDOM_STATE}")
print("=" * 70)

print("Output directories created successfully:")
for directory in output_dirs:
    print(f"  - {directory}")

# ============================================================================
# Code Cell 2
# ============================================================================
# ============================================================
# Section 2 — Load Modeling Dataset
# ============================================================

# -------------------------
# Load Final Modeling Dataset
# -------------------------
df_model = pd.read_parquet(MODELING_DATA_PATH)

# -------------------------
# Basic Dataset Information
# -------------------------
n_rows, n_cols = df_model.shape

target_col = "is_serious"

print("=" * 70)
print("SECTION 2: MODELING DATASET LOADED")
print("=" * 70)

print(f"Dataset Path : {MODELING_DATA_PATH}")
print(f"Rows         : {n_rows:,}")
print(f"Columns      : {n_cols:,}")

print("=" * 70)
print("COLUMN PREVIEW")
print("=" * 70)
print(df_model.columns.tolist())

print("=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)
display(df_model.head())

# -------------------------
# Target Distribution
# -------------------------
target_distribution = (
    df_model[target_col]
    .value_counts()
    .sort_index()
    .rename_axis(target_col)
    .reset_index(name="count")
)

target_distribution["percent"] = (
    target_distribution["count"] / target_distribution["count"].sum() * 100
).round(2)

print("=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)
display(target_distribution)

# ============================================================================
# Code Cell 3
# ============================================================================
# ============================================================
# Validation
# ============================================================

print("=" * 70)
print("VALIDATION")
print("=" * 70)

# Expected dimensions
assert df_model.shape == (406184, 51), "Unexpected dataset dimensions."

# Target variable exists
assert target_col in df_model.columns, "Target variable is missing."

# Binary target validation
assert set(df_model[target_col].unique()) == {0, 1}, \
    "Target variable must contain only 0 and 1."

# Unique PRIMARYID validation
assert df_model["primaryid"].nunique() == len(df_model), \
    "PRIMARYID is not unique."

print("✓ Dataset dimensions validated.")
print("✓ Target variable validated.")
print("✓ Binary target validated.")
print("✓ One report per PRIMARYID validated.")

print("=" * 70)
print("All validation checks passed.")
print("=" * 70)

# ============================================================================
# Code Cell 4
# ============================================================================
# ============================================================
# Section 3 — Data Preparation for Machine Learning
# ============================================================

# -------------------------
# Remove Identifier
# -------------------------
X = df_model.drop(columns=["primaryid", "is_serious"])
y = df_model["is_serious"]

# -------------------------
# Identify Feature Types
# -------------------------
categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=[np.number]
).columns.tolist()

print("=" * 70)
print("FEATURE SUMMARY")
print("=" * 70)

print(f"Total Features      : {X.shape[1]}")
print(f"Numeric Features    : {len(numeric_features)}")
print(f"Categorical Features: {len(categorical_features)}")

print("\nCategorical Variables")
print("----------------------")
print(categorical_features)

# -------------------------
# Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(f"Training Samples : {X_train.shape[0]:,}")
print(f"Testing Samples  : {X_test.shape[0]:,}")

# -------------------------
# Target Distribution
# -------------------------
train_distribution = (
    y_train.value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

test_distribution = (
    y_test.value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

distribution_df = pd.DataFrame({
    "Train (%)": train_distribution,
    "Test (%)": test_distribution
})

print("\nTarget Distribution (%)")
display(distribution_df)

# -------------------------
# Preprocessing Pipelines
# -------------------------

# For scaled models:
# Logistic Regression
# KNN
# SVM
# Neural Network

scaled_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ]
)

# For non-scaled models:
# LDA
# QDA
# Naive Bayes
# Decision Tree
# Random Forest

unscaled_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            numeric_features
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ]
)

print("\n" + "=" * 70)
print("PREPROCESSING PIPELINES CREATED")
print("=" * 70)
print("✓ Scaled preprocessing pipeline")
print("✓ Unscaled preprocessing pipeline")

# ============================================================================
# Code Cell 5
# ============================================================================
# ============================================================
# Validation
# ============================================================

print("=" * 70)
print("VALIDATION")
print("=" * 70)

assert X.shape[0] == y.shape[0], "Predictor and target sizes do not match."

assert X_train.shape[0] + X_test.shape[0] == X.shape[0]

assert y_train.shape[0] + y_test.shape[0] == y.shape[0]

assert len(categorical_features) + len(numeric_features) == X.shape[1]

print("✓ Predictor matrix created.")
print("✓ Response vector created.")
print("✓ Stratified train/test split validated.")
print("✓ Feature types identified.")
print("✓ Preprocessing pipelines created.")

print("=" * 70)
print("All validation checks passed.")
print("=" * 70)

# ============================================================================
# Code Cell 6
# ============================================================================
# ============================================================
# Section 4 — Reusable Machine Learning Framework
# ============================================================

# -------------------------
# Storage Objects
# -------------------------
model_results = []
cv_results = []
trained_models = {}
model_predictions = {}
model_probabilities = {}
confusion_matrices = {}


# -------------------------
# Helper Function: Specificity
# -------------------------
def calculate_specificity(y_true, y_pred):
    """
    Calculate specificity from the confusion matrix.

    Specificity = TN / (TN + FP)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


# -------------------------
# Helper Function: Get Predicted Probabilities
# -------------------------
def get_positive_class_probability(model, X_data):
    """
    Return predicted probability for the positive class.

    If predict_proba is unavailable, use decision_function and rescale.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_data)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_data)
        return (scores - scores.min()) / (scores.max() - scores.min())

    return None


# -------------------------
# Helper Function: Train Model
# -------------------------
def train_model(model_name, estimator, preprocessor, X_train, y_train):
    """
    Train a model using a preprocessing pipeline.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ]
    )

    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time

    trained_models[model_name] = pipeline

    return pipeline, training_time


# -------------------------
# Helper Function: Evaluate Model
# -------------------------
def evaluate_model(model_name, model, X_test, y_test, training_time):
    """
    Evaluate a trained classification model on the testing set.
    """
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    y_prob = get_positive_class_probability(model, X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    specificity = calculate_specificity(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

    cm = confusion_matrix(y_test, y_pred)

    result = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "ROC AUC": roc_auc,
        "Training Time": training_time,
        "Prediction Time": prediction_time
    }

    model_results.append(result)
    model_predictions[model_name] = y_pred
    model_probabilities[model_name] = y_prob
    confusion_matrices[model_name] = cm

    print("=" * 70)
    print(f"MODEL EVALUATION: {model_name}")
    print("=" * 70)

    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"Specificity    : {specificity:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print(f"ROC AUC        : {roc_auc:.4f}")
    print(f"Training Time  : {training_time:.2f} seconds")
    print(f"Prediction Time: {prediction_time:.2f} seconds")

    print("\nClassification Report")
    print("-" * 70)
    print(classification_report(y_test, y_pred, digits=4))

    return result


# -------------------------
# Helper Function: Cross Validation
# -------------------------
def cross_validate_model(model_name, estimator, preprocessor, X_train, y_train, cv=5):
    """
    Perform stratified cross-validation on the training data.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ]
    )

    skf = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=skf,
        scoring="accuracy",
        n_jobs=-1
    )

    cv_result = {
        "Model": model_name,
        "CV Mean Accuracy": scores.mean(),
        "CV Std Accuracy": scores.std()
    }

    cv_results.append(cv_result)

    print("=" * 70)
    print(f"CROSS VALIDATION: {model_name}")
    print("=" * 70)
    print(f"CV Accuracy Scores: {np.round(scores, 4)}")
    print(f"Mean CV Accuracy  : {scores.mean():.4f}")
    print(f"Std CV Accuracy   : {scores.std():.4f}")

    return scores


# -------------------------
# Helper Function: Plot Confusion Matrix
# -------------------------
def plot_confusion_matrix_custom(model_name, y_test, y_pred):
    """
    Plot and save a confusion matrix.
    """
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Non-Serious", "Serious"],
        yticklabels=["Non-Serious", "Serious"],
        ax=ax
    )

    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    filename = CONFUSION_MATRIX_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(filename)
    plt.show()


# -------------------------
# Helper Function: Plot ROC Curve
# -------------------------
def plot_roc_curve_custom(model_name, y_test, y_prob):
    """
    Plot and save a ROC curve.
    """
    if y_prob is None:
        print(f"ROC curve skipped for {model_name}: probability scores unavailable.")
        return

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_value = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

    ax.set_title(f"ROC Curve — {model_name}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    filename = ROC_CURVE_DIR / f"{model_name.lower().replace(' ', '_')}_roc_curve.png"
    plt.savefig(filename)
    plt.show()


# -------------------------
# Helper Function: Plot Precision-Recall Curve
# -------------------------
def plot_precision_recall_curve_custom(model_name, y_test, y_prob):
    """
    Plot and save a precision-recall curve.
    """
    if y_prob is None:
        print(f"Precision-recall curve skipped for {model_name}: probability scores unavailable.")
        return

    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(recall, precision, label=model_name)

    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")

    filename = PR_CURVE_DIR / f"{model_name.lower().replace(' ', '_')}_precision_recall_curve.png"
    plt.savefig(filename)
    plt.show()


# -------------------------
# Helper Function: Full Model Workflow
# -------------------------
def run_model_workflow(
    model_name,
    estimator,
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test,
    cv=5
):
    """
    Train, evaluate, cross-validate, and visualize a model.
    """
    model, training_time = train_model(
        model_name,
        estimator,
        preprocessor,
        X_train,
        y_train
    )

    result = evaluate_model(
        model_name,
        model,
        X_test,
        y_test,
        training_time
    )

    cv_scores = cross_validate_model(
        model_name,
        estimator,
        preprocessor,
        X_train,
        y_train,
        cv=cv
    )

    y_pred = model_predictions[model_name]
    y_prob = model_probabilities[model_name]

    plot_confusion_matrix_custom(model_name, y_test, y_pred)
    plot_roc_curve_custom(model_name, y_test, y_prob)
    plot_precision_recall_curve_custom(model_name, y_test, y_prob)

    return model, result, cv_scores


print("=" * 70)
print("REUSABLE MACHINE LEARNING FRAMEWORK CREATED")
print("=" * 70)

print("Helper functions available:")
print("✓ calculate_specificity()")
print("✓ get_positive_class_probability()")
print("✓ train_model()")
print("✓ evaluate_model()")
print("✓ cross_validate_model()")
print("✓ plot_confusion_matrix_custom()")
print("✓ plot_roc_curve_custom()")
print("✓ plot_precision_recall_curve_custom()")
print("✓ run_model_workflow()")

# ============================================================================
# Code Cell 7
# ============================================================================
# ============================================================
# Validation
# ============================================================

print("=" * 70)
print("VALIDATION")
print("=" * 70)

required_functions = [
    calculate_specificity,
    get_positive_class_probability,
    train_model,
    evaluate_model,
    cross_validate_model,
    plot_confusion_matrix_custom,
    plot_roc_curve_custom,
    plot_precision_recall_curve_custom,
    run_model_workflow
]

for function in required_functions:
    assert callable(function), f"{function.__name__} is not callable."

assert isinstance(model_results, list), "model_results must be a list."
assert isinstance(cv_results, list), "cv_results must be a list."
assert isinstance(trained_models, dict), "trained_models must be a dictionary."
assert isinstance(model_predictions, dict), "model_predictions must be a dictionary."
assert isinstance(model_probabilities, dict), "model_probabilities must be a dictionary."
assert isinstance(confusion_matrices, dict), "confusion_matrices must be a dictionary."

print("✓ All helper functions are callable.")
print("✓ Model result storage initialized.")
print("✓ Cross-validation result storage initialized.")
print("✓ Prediction storage initialized.")
print("✓ Probability storage initialized.")
print("✓ Confusion matrix storage initialized.")

print("=" * 70)
print("All validation checks passed.")
print("=" * 70)

# ============================================================================
# Code Cell 8
# ============================================================================
# ============================================================
# Modeling Figure Directories
# ============================================================

MODELING_FIGURE_DIR = FIGURE_DIR / "modeling"

ROC_FIGURE_DIR = MODELING_FIGURE_DIR / "roc_curves"
CM_FIGURE_DIR = MODELING_FIGURE_DIR / "confusion_matrices"
PR_FIGURE_DIR = MODELING_FIGURE_DIR / "precision_recall_curves"
COEFFICIENT_FIGURE_DIR = MODELING_FIGURE_DIR / "coefficients"
FEATURE_IMPORTANCE_FIGURE_DIR = MODELING_FIGURE_DIR / "feature_importance"
DASHBOARD_FIGURE_DIR = MODELING_FIGURE_DIR / "dashboards"

figure_dirs = [
    MODELING_FIGURE_DIR,
    ROC_FIGURE_DIR,
    CM_FIGURE_DIR,
    PR_FIGURE_DIR,
    COEFFICIENT_FIGURE_DIR,
    FEATURE_IMPORTANCE_FIGURE_DIR,
    DASHBOARD_FIGURE_DIR
]

for directory in figure_dirs:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Code Cell 9
# ============================================================================
# ============================================================
# Section 5 — Logistic Regression
# ============================================================

# -------------------------
# Initialize Model
# -------------------------
logistic_model = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=1000,
    solver="lbfgs"
)

# -------------------------
# Train, Evaluate, and Validate
# -------------------------
logistic_pipeline, logistic_result, logistic_cv = run_model_workflow(
    model_name="Logistic Regression",
    estimator=logistic_model,
    preprocessor=scaled_preprocessor,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    cv=5
)

# -------------------------
# Extract Feature Names
# -------------------------
feature_names = logistic_pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

# -------------------------
# Logistic Coefficients
# -------------------------
coefficients = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": logistic_pipeline.named_steps[
        "model"
    ].coef_[0]
})

coefficients["Abs_Coefficient"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "Abs_Coefficient",
    ascending=False
)

display(coefficients.head(20))

# -------------------------
# Save Coefficients
# -------------------------
coefficients.to_csv(
    MODELING_RESULT_DIR / "logistic_coefficients.csv",
    index=False
)

# -------------------------
# Plot Top 20 Coefficients
# -------------------------
top20 = coefficients.head(20)

plt.figure(figsize=(10,8))

sns.barplot(
    data=top20,
    x="Coefficient",
    y="Feature"
)

plt.title("Top 20 Logistic Regression Coefficients")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR /
    "logistic_regression_coefficients.png"
)

plt.show()

# ============================================================================
# Code Cell 10
# ============================================================================
# ============================================================
# Validation
# ============================================================

print("=" * 70)
print("VALIDATION")
print("=" * 70)

assert logistic_pipeline is not None

assert len(coefficients) == len(feature_names)

assert (
    MODELING_RESULT_DIR /
    "logistic_coefficients.csv"
).exists()

print("✓ Logistic Regression trained successfully.")
print("✓ Coefficients extracted.")
print("✓ Coefficients exported.")
print("✓ Feature importance figure created.")

print("=" * 70)
print("All validation checks passed.")
print("=" * 70)

# ============================================================================
# Code Cell 11
# ============================================================================
# ============================================================
# Save Logistic Regression Tables
# ============================================================

# Predictions and probabilities
logistic_predictions_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": model_predictions["Logistic Regression"],
    "y_prob": model_probabilities["Logistic Regression"]
})

logistic_predictions_df.to_csv(
    MODELING_RESULT_DIR / "logistic_regression_predictions.csv",
    index=False
)

# Coefficients table
coefficients.to_csv(
    MODELING_RESULT_DIR / "logistic_coefficients.csv",
    index=False
)

# Performance table for Logistic Regression
logistic_performance_df = pd.DataFrame([logistic_result])

logistic_performance_df.to_csv(
    MODELING_RESULT_DIR / "logistic_regression_performance.csv",
    index=False
)

print("Saved Logistic Regression tables.")

# ============================================================================
# Code Cell 12
# ============================================================================
# ============================================================
# Save Confusion Matrix
# ============================================================

cm = confusion_matrices["Logistic Regression"]

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Non-Serious", "Serious"],
    yticklabels=["Non-Serious", "Serious"]
)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    CM_FIGURE_DIR / "logistic_regression_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ Confusion matrix saved.")

# ============================================================================
# Code Cell 13
# ============================================================================
# ============================================================
# Save ROC Curve
# ============================================================

y_prob_logistic = model_probabilities["Logistic Regression"]

fpr, tpr, thresholds = roc_curve(y_test, y_prob_logistic)
roc_auc = roc_auc_score(y_test, y_prob_logistic)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve - Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(
    ROC_FIGURE_DIR / "logistic_regression_roc_curve.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ ROC curve saved.")

# ============================================================================
# Code Cell 14
# ============================================================================
# ============================================================
# Save Precision-Recall Curve
# ============================================================

precision_vals, recall_vals, pr_thresholds = precision_recall_curve(
    y_test,
    y_prob_logistic
)

plt.figure(figsize=(8, 6))

plt.plot(
    recall_vals,
    precision_vals,
    label="Logistic Regression"
)

plt.title("Precision-Recall Curve - Logistic Regression")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")

plt.tight_layout()

plt.savefig(
    PR_FIGURE_DIR / "logistic_regression_precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ Precision-recall curve saved.")

# ============================================================================
# Code Cell 15
# ============================================================================
# ============================================================
# Save Top 20 Logistic Regression Coefficients Figure
# ============================================================

top20 = coefficients.head(20)

plt.figure(figsize=(10, 8))

sns.barplot(
    data=top20,
    x="Coefficient",
    y="Feature"
)

plt.title("Top 20 Logistic Regression Coefficients")
plt.xlabel("Coefficient")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    COEFFICIENT_FIGURE_DIR / "logistic_regression_top20_coefficients.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ Top 20 logistic regression coefficients figure saved.")

# ============================================================================
# Code Cell 16
# ============================================================================
# ============================================================
# Check run_model_workflow Function Signature
# ============================================================

import inspect

print(inspect.signature(run_model_workflow))

# ============================================================================
# Code Cell 17
# ============================================================================
# ============================================================
# Dense Unscaled Preprocessor for LDA, QDA, and GaussianNB
# ============================================================

dense_unscaled_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            numeric_features
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ))
            ]),
            categorical_features
        )
    ]
)

print("✓ Dense unscaled preprocessor created for LDA, QDA, and GaussianNB.")

# ============================================================================
# Code Cell 18
# ============================================================================
# ============================================================
# Section 6 — Linear Discriminant Analysis
# Corrected Dense Preprocessor Version
# ============================================================

# -------------------------
# Initialize Model
# -------------------------
lda_model = LinearDiscriminantAnalysis()

# -------------------------
# Train, Evaluate, Cross-Validate, and Plot
# -------------------------
lda_pipeline, lda_result, lda_cv = run_model_workflow(
    model_name="Linear Discriminant Analysis",
    estimator=lda_model,
    preprocessor=dense_unscaled_preprocessor,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    cv=5
)

# ============================================================================
# Code Cell 19
# ============================================================================
# ============================================================
# Fix Missing Modeling Result Folders
# ============================================================

CONFUSION_MATRIX_DIR = MODELING_RESULT_DIR / "confusion_matrices"
ROC_CURVE_DIR = MODELING_RESULT_DIR / "roc_curves"
PR_CURVE_DIR = MODELING_RESULT_DIR / "precision_recall_curves"
CLASSIFICATION_REPORT_DIR = MODELING_RESULT_DIR / "classification_reports"

result_dirs = [
    CONFUSION_MATRIX_DIR,
    ROC_CURVE_DIR,
    PR_CURVE_DIR,
    CLASSIFICATION_REPORT_DIR
]

for directory in result_dirs:
    directory.mkdir(parents=True, exist_ok=True)

print("✓ Modeling result folders created/confirmed:")
for directory in result_dirs:
    print(f"  - {directory}")

# ============================================================================
# Code Cell 20
# ============================================================================
# ============================================================
# Save LDA Tables and Classification Report
# ============================================================

lda_y_pred = model_predictions["Linear Discriminant Analysis"]
lda_y_prob = model_probabilities["Linear Discriminant Analysis"]

# -------------------------
# Save Predictions
# -------------------------
lda_predictions_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": lda_y_pred,
    "y_prob": lda_y_prob
})

lda_predictions_df.to_csv(
    MODELING_RESULT_DIR / "lda_predictions.csv",
    index=False
)

# -------------------------
# Save Performance
# -------------------------
lda_performance_df = pd.DataFrame([lda_result])

lda_performance_df.to_csv(
    MODELING_RESULT_DIR / "lda_performance.csv",
    index=False
)

# -------------------------
# Save Classification Report
# -------------------------
lda_report = classification_report(
    y_test,
    lda_y_pred,
    digits=4
)

with open(
    CLASSIFICATION_REPORT_DIR / "lda_classification_report.txt",
    "w"
) as file:
    file.write(lda_report)

print("✓ LDA predictions saved.")
print("✓ LDA performance table saved.")
print("✓ LDA classification report saved.")

# ============================================================================
# Code Cell 21
# ============================================================================
# ============================================================
# Recreate LDA Performance Table
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

lda_y_pred = model_predictions["Linear Discriminant Analysis"]
lda_y_prob = model_probabilities["Linear Discriminant Analysis"]

lda_result = {
    "Model": "Linear Discriminant Analysis",
    "Accuracy": accuracy_score(y_test, lda_y_pred),
    "Precision": precision_score(y_test, lda_y_pred),
    "Recall": recall_score(y_test, lda_y_pred),
    "Specificity": calculate_specificity(y_test, lda_y_pred),
    "F1": f1_score(y_test, lda_y_pred),
    "ROC AUC": roc_auc_score(y_test, lda_y_prob)
}

lda_performance_df = pd.DataFrame([lda_result])

lda_performance_df.to_csv(
    MODELING_RESULT_DIR / "lda_performance.csv",
    index=False
)

print("✓ LDA performance table saved.")

# ============================================================================
# Code Cell 22
# ============================================================================
# ============================================================
# Save LDA Predictions Table
# ============================================================

lda_predictions_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": lda_y_pred,
    "y_prob": lda_y_prob
})

lda_predictions_df.to_csv(
    MODELING_RESULT_DIR / "lda_predictions.csv",
    index=False
)

print("✓ LDA predictions table saved.")

# ============================================================================
# Code Cell 23
# ============================================================================
# ============================================================
# Save LDA Classification Report
# ============================================================

lda_report = classification_report(
    y_test,
    lda_y_pred,
    digits=4
)

report_path = (
    CLASSIFICATION_REPORT_DIR /
    "lda_classification_report.txt"
)

with open(report_path, "w") as f:
    f.write("Linear Discriminant Analysis\n")
    f.write("=" * 70 + "\n\n")
    f.write(lda_report)

print(f"✓ Classification report saved:\n{report_path}")

# ============================================================================
# Code Cell 24
# ============================================================================
# ============================================================
# Save LDA Confusion Matrix
# ============================================================

lda_cm = confusion_matrix(y_test, lda_y_pred)

plt.figure(figsize=(7, 6))

sns.heatmap(
    lda_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Non-Serious", "Serious"],
    yticklabels=["Non-Serious", "Serious"]
)

plt.title("Confusion Matrix - Linear Discriminant Analysis")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    CM_FIGURE_DIR / "lda_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ LDA confusion matrix saved.")

# ============================================================================
# Code Cell 25
# ============================================================================
# ============================================================
# Save LDA Confusion Matrix
# ============================================================

lda_cm = confusion_matrix(y_test, lda_y_pred)

plt.figure(figsize=(7, 6))

sns.heatmap(
    lda_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Non-Serious", "Serious"],
    yticklabels=["Non-Serious", "Serious"]
)

plt.title("Confusion Matrix - Linear Discriminant Analysis")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    CM_FIGURE_DIR / "lda_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ LDA confusion matrix saved.")

# ============================================================================
# Code Cell 26
# ============================================================================
# ============================================================
# Save LDA ROC Curve
# ============================================================

lda_fpr, lda_tpr, lda_thresholds = roc_curve(
    y_test,
    lda_y_prob
)

lda_auc = roc_auc_score(
    y_test,
    lda_y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    lda_fpr,
    lda_tpr,
    label=f"LDA (AUC = {lda_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve - Linear Discriminant Analysis")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(
    ROC_FIGURE_DIR / "lda_roc_curve.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ LDA ROC curve saved.")

# ============================================================================
# Code Cell 27
# ============================================================================
# ============================================================
# Save LDA Precision-Recall Curve
# ============================================================

lda_precision_vals, lda_recall_vals, lda_pr_thresholds = precision_recall_curve(
    y_test,
    lda_y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    lda_recall_vals,
    lda_precision_vals,
    label="LDA"
)

plt.title("Precision-Recall Curve - Linear Discriminant Analysis")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")

plt.tight_layout()

plt.savefig(
    PR_FIGURE_DIR / "lda_precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ LDA precision-recall curve saved.")

# ============================================================================
# Code Cell 28
# ============================================================================
# ============================================================
# Validate LDA Saved Outputs
# ============================================================

lda_files_to_check = [
    MODELING_RESULT_DIR / "lda_performance.csv",
    MODELING_RESULT_DIR / "lda_predictions.csv",
    CLASSIFICATION_REPORT_DIR / "lda_classification_report.txt",
    CM_FIGURE_DIR / "lda_confusion_matrix.png",
    ROC_FIGURE_DIR / "lda_roc_curve.png",
    PR_FIGURE_DIR / "lda_precision_recall_curve.png"
]

print("=" * 80)
print("LDA OUTPUT VERIFICATION")
print("=" * 80)

for file in lda_files_to_check:
    if file.exists():
        print(f"✓ {file.name}")
    else:
        print(f"✗ {file.name}")

print("=" * 80)
print(f"Total Files Checked: {len(lda_files_to_check)}")
print("=" * 80)

# ============================================================================
# Code Cell 29
# ============================================================================
# ============================================================
# Section 7 — Quadratic Discriminant Analysis
# ============================================================

# -------------------------
# Initialize Model
# -------------------------
qda_model = QuadraticDiscriminantAnalysis(
    reg_param=0.01
)

# -------------------------
# Train, Evaluate, Cross-Validate, and Plot
# -------------------------
qda_pipeline, qda_result, qda_cv = run_model_workflow(
    model_name="Quadratic Discriminant Analysis",
    estimator=qda_model,
    preprocessor=dense_unscaled_preprocessor,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    cv=5
)

# ============================================================================
# Code Cell 30
# ============================================================================
# ============================================================
# Save QDA Performance Table
# ============================================================

qda_y_pred = model_predictions["Quadratic Discriminant Analysis"]
qda_y_prob = model_probabilities["Quadratic Discriminant Analysis"]

qda_result = {
    "Model": "Quadratic Discriminant Analysis",
    "Accuracy": accuracy_score(y_test, qda_y_pred),
    "Precision": precision_score(y_test, qda_y_pred),
    "Recall": recall_score(y_test, qda_y_pred),
    "Specificity": calculate_specificity(y_test, qda_y_pred),
    "F1": f1_score(y_test, qda_y_pred),
    "ROC AUC": roc_auc_score(y_test, qda_y_prob)
}

qda_performance_df = pd.DataFrame([qda_result])

qda_performance_df.to_csv(
    MODELING_RESULT_DIR / "qda_performance.csv",
    index=False
)

print("✓ QDA performance table saved.")

# ============================================================================
# Code Cell 31
# ============================================================================
# ============================================================
# Save QDA Predictions
# ============================================================

qda_predictions_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": qda_y_pred,
    "y_prob": qda_y_prob
})

qda_predictions_df.to_csv(
    MODELING_RESULT_DIR / "qda_predictions.csv",
    index=False
)

print("✓ QDA predictions table saved.")

# ============================================================================
# Code Cell 32
# ============================================================================
# ============================================================
# Save QDA Classification Report
# ============================================================

qda_report = classification_report(
    y_test,
    qda_y_pred,
    digits=4
)

report_path = (
    CLASSIFICATION_REPORT_DIR /
    "qda_classification_report.txt"
)

with open(report_path, "w") as f:
    f.write("Quadratic Discriminant Analysis\n")
    f.write("=" * 70 + "\n\n")
    f.write(qda_report)

print(f"✓ QDA classification report saved:\n{report_path}")

# ============================================================================
# Code Cell 33
# ============================================================================
# ============================================================
# Save QDA Confusion Matrix
# ============================================================

qda_cm = confusion_matrix(y_test, qda_y_pred)

plt.figure(figsize=(7, 6))

sns.heatmap(
    qda_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Non-Serious", "Serious"],
    yticklabels=["Non-Serious", "Serious"]
)

plt.title("Confusion Matrix - Quadratic Discriminant Analysis")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    CM_FIGURE_DIR / "qda_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ QDA confusion matrix saved.")

# ============================================================================
# Code Cell 34
# ============================================================================
# ============================================================
# Save QDA ROC Curve
# ============================================================

qda_fpr, qda_tpr, qda_thresholds = roc_curve(
    y_test,
    qda_y_prob
)

qda_auc = roc_auc_score(
    y_test,
    qda_y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    qda_fpr,
    qda_tpr,
    label=f"QDA (AUC = {qda_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve - Quadratic Discriminant Analysis")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(
    ROC_FIGURE_DIR / "qda_roc_curve.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ QDA ROC curve saved.")

# ============================================================================
# Code Cell 35
# ============================================================================
# ============================================================
# Save QDA Precision-Recall Curve
# ============================================================

qda_precision_vals, qda_recall_vals, qda_pr_thresholds = precision_recall_curve(
    y_test,
    qda_y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    qda_recall_vals,
    qda_precision_vals,
    linewidth=2,
    label="QDA"
)

plt.title("Precision-Recall Curve - Quadratic Discriminant Analysis")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")

plt.tight_layout()

plt.savefig(
    PR_FIGURE_DIR / "qda_precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ QDA precision-recall curve saved.")

# ============================================================================
# Code Cell 36
# ============================================================================
# ============================================================
# Validate QDA Saved Outputs
# ============================================================

qda_files_to_check = [
    MODELING_RESULT_DIR / "qda_performance.csv",
    MODELING_RESULT_DIR / "qda_predictions.csv",
    CLASSIFICATION_REPORT_DIR / "qda_classification_report.txt",
    CM_FIGURE_DIR / "qda_confusion_matrix.png",
    ROC_FIGURE_DIR / "qda_roc_curve.png",
    PR_FIGURE_DIR / "qda_precision_recall_curve.png"
]

print("=" * 80)
print("QDA OUTPUT VERIFICATION")
print("=" * 80)

for file in qda_files_to_check:
    if file.exists():
        print(f"✓ {file.name}")
    else:
        print(f"✗ {file.name}")

print("=" * 80)
print(f"Total Files Checked: {len(qda_files_to_check)}")
print("=" * 80)

# ============================================================================
# Code Cell 37
# ============================================================================
# ============================================================
# Section 8 — Gaussian Naïve Bayes
# ============================================================

# -------------------------
# Initialize Model
# -------------------------
gnb_model = GaussianNB()

# -------------------------
# Train, Evaluate, Cross-Validate, and Plot
# -------------------------
gnb_pipeline, gnb_result, gnb_cv = run_model_workflow(
    model_name="Gaussian Naive Bayes",
    estimator=gnb_model,
    preprocessor=dense_unscaled_preprocessor,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    cv=5
)

# ============================================================================
# Code Cell 38
# ============================================================================
# ============================================================
# Save Gaussian Naive Bayes Performance Table
# ============================================================

gnb_y_pred = model_predictions["Gaussian Naive Bayes"]
gnb_y_prob = model_probabilities["Gaussian Naive Bayes"]

gnb_result = {
    "Model": "Gaussian Naive Bayes",
    "Accuracy": accuracy_score(y_test, gnb_y_pred),
    "Precision": precision_score(y_test, gnb_y_pred),
    "Recall": recall_score(y_test, gnb_y_pred),
    "Specificity": calculate_specificity(y_test, gnb_y_pred),
    "F1": f1_score(y_test, gnb_y_pred),
    "ROC AUC": roc_auc_score(y_test, gnb_y_prob)
}

gnb_performance_df = pd.DataFrame([gnb_result])

gnb_performance_df.to_csv(
    MODELING_RESULT_DIR / "gaussian_naive_bayes_performance.csv",
    index=False
)

print("✓ Gaussian Naive Bayes performance table saved.")

# ============================================================================
# Code Cell 39
# ============================================================================
# ============================================================
# Save Gaussian Naive Bayes Predictions
# ============================================================

gnb_predictions_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": gnb_y_pred,
    "y_prob": gnb_y_prob
})

gnb_predictions_df.to_csv(
    MODELING_RESULT_DIR / "gaussian_naive_bayes_predictions.csv",
    index=False
)

print("✓ Gaussian Naive Bayes predictions table saved.")

# ============================================================================
# Code Cell 40
# ============================================================================
# ============================================================
# Save Gaussian Naive Bayes Classification Report
# ============================================================

gnb_report = classification_report(
    y_test,
    gnb_y_pred,
    digits=4
)

report_path = (
    CLASSIFICATION_REPORT_DIR /
    "gaussian_naive_bayes_classification_report.txt"
)

with open(report_path, "w") as f:
    f.write("Gaussian Naive Bayes\n")
    f.write("=" * 70 + "\n\n")
    f.write(gnb_report)

print(f"✓ Gaussian Naive Bayes classification report saved:\n{report_path}")

# ============================================================================
# Code Cell 41
# ============================================================================
# ============================================================
# Save Gaussian Naive Bayes Confusion Matrix
# ============================================================

gnb_cm = confusion_matrix(y_test, gnb_y_pred)

plt.figure(figsize=(7, 6))

sns.heatmap(
    gnb_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Non-Serious", "Serious"],
    yticklabels=["Non-Serious", "Serious"]
)

plt.title("Confusion Matrix - Gaussian Naive Bayes")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    CM_FIGURE_DIR / "gaussian_naive_bayes_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ Gaussian Naive Bayes confusion matrix saved.")

# ============================================================================
# Code Cell 42
# ============================================================================
# ============================================================
# Save Gaussian Naive Bayes ROC Curve
# ============================================================

gnb_fpr, gnb_tpr, gnb_thresholds = roc_curve(
    y_test,
    gnb_y_prob
)

gnb_auc = roc_auc_score(
    y_test,
    gnb_y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    gnb_fpr,
    gnb_tpr,
    label=f"Gaussian Naive Bayes (AUC = {gnb_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve - Gaussian Naive Bayes")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(
    ROC_FIGURE_DIR / "gaussian_naive_bayes_roc_curve.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ Gaussian Naive Bayes ROC curve saved.")

# ============================================================================
# Code Cell 43
# ============================================================================
# ============================================================
# Save Gaussian Naive Bayes Precision-Recall Curve
# ============================================================

gnb_precision_vals, gnb_recall_vals, gnb_pr_thresholds = precision_recall_curve(
    y_test,
    gnb_y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    gnb_recall_vals,
    gnb_precision_vals,
    linewidth=2,
    label="Gaussian Naive Bayes"
)

plt.title("Precision-Recall Curve - Gaussian Naive Bayes")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")

plt.tight_layout()

plt.savefig(
    PR_FIGURE_DIR / "gaussian_naive_bayes_precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ Gaussian Naive Bayes precision-recall curve saved.")

# ============================================================================
# Code Cell 44
# ============================================================================
# ============================================================
# Validate Gaussian Naive Bayes Saved Outputs
# ============================================================

gnb_files_to_check = [
    MODELING_RESULT_DIR / "gaussian_naive_bayes_performance.csv",
    MODELING_RESULT_DIR / "gaussian_naive_bayes_predictions.csv",
    CLASSIFICATION_REPORT_DIR / "gaussian_naive_bayes_classification_report.txt",
    CM_FIGURE_DIR / "gaussian_naive_bayes_confusion_matrix.png",
    ROC_FIGURE_DIR / "gaussian_naive_bayes_roc_curve.png",
    PR_FIGURE_DIR / "gaussian_naive_bayes_precision_recall_curve.png"
]

print("=" * 80)
print("GAUSSIAN NAIVE BAYES OUTPUT VERIFICATION")
print("=" * 80)

for file in gnb_files_to_check:
    if file.exists():
        print(f"✓ {file.name}")
    else:
        print(f"✗ {file.name}")

print("=" * 80)
print(f"Total Files Checked: {len(gnb_files_to_check)}")
print("=" * 80)

# ============================================================================
# Code Cell 45
# ============================================================================
# ============================================================
# Create Stratified Sample for KNN
# ============================================================

from sklearn.model_selection import train_test_split

# Approximately 40,000 observations from the training set
sample_fraction = 40000 / len(X_train)

X_train_knn, _, y_train_knn, _ = train_test_split(
    X_train,
    y_train,
    train_size=sample_fraction,
    stratify=y_train,
    random_state=RANDOM_STATE
)

print("=" * 70)
print("KNN TRAINING SAMPLE CREATED")
print("=" * 70)

print(f"Original Training Size : {len(X_train):,}")
print(f"KNN Training Size      : {len(X_train_knn):,}")

print("\nClass Distribution")

print(
    pd.DataFrame({
        "Original (%)": y_train.value_counts(normalize=True).sort_index() * 100,
        "Sample (%)": y_train_knn.value_counts(normalize=True).sort_index() * 100
    }).round(2)
)

# ============================================================================
# Code Cell 46
# ============================================================================
# ============================================================
# KNN Hyperparameter Tuning (Sampled Training Data)
# ============================================================

knn_pipeline = Pipeline(
    steps=[
        ("preprocessor", scaled_preprocessor),
        ("model", KNeighborsClassifier())
    ]
)

param_grid = {
    "model__n_neighbors": [3, 5, 7, 9, 11]
}

knn_grid = GridSearchCV(
    estimator=knn_pipeline,
    param_grid=param_grid,
    scoring="accuracy",
    cv=3,
    n_jobs=-1,
    verbose=2
)

knn_grid.fit(
    X_train_knn,
    y_train_knn
)

best_k = knn_grid.best_params_["model__n_neighbors"]

print("=" * 70)
print("KNN GRID SEARCH COMPLETED")
print("=" * 70)

print(f"Best K          : {best_k}")
print(f"Best CV Accuracy: {knn_grid.best_score_:.4f}")

# ============================================================================
# Code Cell 47
# ============================================================================
# Train
knn_pipeline, training_time = train_model(
    "K-Nearest Neighbors",
    knn_final_model,
    scaled_preprocessor,
    X_train_knn,
    y_train_knn
)

# Evaluate
knn_result = evaluate_model(
    "K-Nearest Neighbors",
    knn_pipeline,
    X_test,
    y_test,
    training_time
)

# ============================================================================
# Code Cell 48
# ============================================================================
# ============================================================
# Save KNN Performance Table
# ============================================================

knn_y_pred = model_predictions["K-Nearest Neighbors"]
knn_y_prob = model_probabilities["K-Nearest Neighbors"]

knn_result = {
    "Model": "K-Nearest Neighbors",
    "Accuracy": accuracy_score(y_test, knn_y_pred),
    "Precision": precision_score(y_test, knn_y_pred),
    "Recall": recall_score(y_test, knn_y_pred),
    "Specificity": calculate_specificity(y_test, knn_y_pred),
    "F1": f1_score(y_test, knn_y_pred),
    "ROC AUC": roc_auc_score(y_test, knn_y_prob),
    "Best K": best_k,
    "CV Accuracy": knn_grid.best_score_
}

knn_performance_df = pd.DataFrame([knn_result])

knn_performance_df.to_csv(
    MODELING_RESULT_DIR / "knn_performance.csv",
    index=False
)

print("✓ KNN performance table saved.")

# ============================================================================
# Code Cell 49
# ============================================================================
# ============================================================
# Save KNN Predictions
# ============================================================

knn_predictions_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": knn_y_pred,
    "y_prob": knn_y_prob
})

knn_predictions_df.to_csv(
    MODELING_RESULT_DIR / "knn_predictions.csv",
    index=False
)

print("✓ KNN predictions table saved.")

# ============================================================================
# Code Cell 50
# ============================================================================
# ============================================================
# Classification Report Directory
# ============================================================

CLASSIFICATION_REPORT_DIR = (
    MODELING_RESULT_DIR / "classification_reports"
)

CLASSIFICATION_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print(CLASSIFICATION_REPORT_DIR)

# ============================================================================
# Code Cell 51
# ============================================================================
# ============================================================
# Save KNN Classification Report
# ============================================================

knn_report = classification_report(
    y_test,
    knn_y_pred,
    digits=4
)

report_path = (
    CLASSIFICATION_REPORT_DIR /
    "knn_classification_report.txt"
)

with open(report_path, "w") as f:
    f.write("K-Nearest Neighbors\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Best K: {best_k}\n")
    f.write(f"Grid Search CV Accuracy: {knn_grid.best_score_:.4f}\n\n")
    f.write(knn_report)

print(f"✓ KNN classification report saved:\n{report_path}")

# ============================================================================
# Code Cell 52
# ============================================================================
# ============================================================
# Restore Modeling Figure Directories After Kernel Restart
# ============================================================

MODELING_FIGURE_DIR = FIGURE_DIR / "modeling"

ROC_FIGURE_DIR = MODELING_FIGURE_DIR / "roc_curves"
CM_FIGURE_DIR = MODELING_FIGURE_DIR / "confusion_matrices"
PR_FIGURE_DIR = MODELING_FIGURE_DIR / "precision_recall_curves"
COEFFICIENT_FIGURE_DIR = MODELING_FIGURE_DIR / "coefficients"
FEATURE_IMPORTANCE_FIGURE_DIR = MODELING_FIGURE_DIR / "feature_importance"
DASHBOARD_FIGURE_DIR = MODELING_FIGURE_DIR / "dashboards"

for directory in [
    ROC_FIGURE_DIR,
    CM_FIGURE_DIR,
    PR_FIGURE_DIR,
    COEFFICIENT_FIGURE_DIR,
    FEATURE_IMPORTANCE_FIGURE_DIR,
    DASHBOARD_FIGURE_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

print("✓ Modeling figure directories restored.")

# ============================================================================
# Code Cell 53
# ============================================================================
# ============================================================
# Save KNN Confusion Matrix
# ============================================================

knn_cm = confusion_matrix(y_test, knn_y_pred)

plt.figure(figsize=(7, 6))

sns.heatmap(
    knn_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Non-Serious", "Serious"],
    yticklabels=["Non-Serious", "Serious"]
)

plt.title("Confusion Matrix - K-Nearest Neighbors")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    CM_FIGURE_DIR / "knn_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ KNN confusion matrix saved.")

# ============================================================================
# Code Cell 54
# ============================================================================
# ============================================================
# Save KNN ROC Curve
# ============================================================

knn_fpr, knn_tpr, _ = roc_curve(
    y_test,
    knn_y_prob
)

knn_auc = roc_auc_score(
    y_test,
    knn_y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    knn_fpr,
    knn_tpr,
    linewidth=2,
    label=f"KNN (AUC = {knn_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Random Classifier"
)

plt.title("ROC Curve - K-Nearest Neighbors")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(
    ROC_FIGURE_DIR / "knn_roc_curve.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ KNN ROC curve saved.")

# ============================================================================
# Code Cell 55
# ============================================================================
# ============================================================
# Save KNN Precision-Recall Curve
# ============================================================

knn_precision_vals, knn_recall_vals, _ = precision_recall_curve(
    y_test,
    knn_y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    knn_recall_vals,
    knn_precision_vals,
    linewidth=2,
    label="KNN"
)

plt.title("Precision-Recall Curve - K-Nearest Neighbors")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")

plt.tight_layout()

plt.savefig(
    PR_FIGURE_DIR / "knn_precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ KNN precision-recall curve saved.")

# ============================================================================
# Code Cell 56
# ============================================================================
# ============================================================
# Save KNN Tuning Results and K-Selection Plot
# ============================================================

knn_tuning_df = pd.DataFrame(knn_grid.cv_results_)

knn_tuning_summary = knn_tuning_df[
    [
        "param_model__n_neighbors",
        "mean_test_score",
        "std_test_score",
        "rank_test_score"
    ]
].rename(
    columns={
        "param_model__n_neighbors": "K",
        "mean_test_score": "Mean CV Accuracy",
        "std_test_score": "Std CV Accuracy",
        "rank_test_score": "Rank"
    }
)

knn_tuning_summary.to_csv(
    MODELING_RESULT_DIR / "knn_tuning_results.csv",
    index=False
)

plt.figure(figsize=(8, 6))

sns.lineplot(
    data=knn_tuning_summary,
    x="K",
    y="Mean CV Accuracy",
    marker="o"
)

plt.title("KNN Hyperparameter Tuning")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Mean CV Accuracy")

plt.tight_layout()

plt.savefig(
    DASHBOARD_FIGURE_DIR / "knn_k_selection_plot.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ KNN tuning results saved.")
print("✓ KNN K-selection plot saved.")

# ============================================================================
# Code Cell 57
# ============================================================================
# ============================================================
# Validate KNN Saved Outputs
# ============================================================

knn_files_to_check = [
    MODELING_RESULT_DIR / "knn_performance.csv",
    MODELING_RESULT_DIR / "knn_predictions.csv",
    MODELING_RESULT_DIR / "knn_tuning_results.csv",
    CLASSIFICATION_REPORT_DIR / "knn_classification_report.txt",
    CM_FIGURE_DIR / "knn_confusion_matrix.png",
    ROC_FIGURE_DIR / "knn_roc_curve.png",
    PR_FIGURE_DIR / "knn_precision_recall_curve.png",
    DASHBOARD_FIGURE_DIR / "knn_k_selection_plot.png"
]

print("=" * 80)
print("KNN OUTPUT VERIFICATION")
print("=" * 80)

for file in knn_files_to_check:
    if file.exists():
        print(f"✓ {file.name}")
    else:
        print(f"✗ {file.name}")

print("=" * 80)
print(f"Total Files Checked: {len(knn_files_to_check)}")
print("=" * 80)

# ============================================================================
# Code Cell 58
# ============================================================================
# ============================================================
# Decision Tree Hyperparameter Tuning
# ============================================================

dt_pipeline = Pipeline(
    steps=[
        ("preprocessor", unscaled_preprocessor),
        (
            "model",
            DecisionTreeClassifier(
                random_state=RANDOM_STATE
            )
        )
    ]
)

dt_param_grid = {
    "model__max_depth": [5, 10, 15, 20, None],
    "model__min_samples_leaf": [1, 5, 10]
}

dt_grid = GridSearchCV(
    estimator=dt_pipeline,
    param_grid=dt_param_grid,
    scoring="accuracy",
    cv=3,
    n_jobs=-1,
    verbose=2
)

dt_grid.fit(
    X_train,
    y_train
)

print("=" * 70)
print("DECISION TREE GRID SEARCH COMPLETE")
print("=" * 70)

print(f"Best Parameters : {dt_grid.best_params_}")
print(f"Best CV Accuracy: {dt_grid.best_score_:.4f}")

# ============================================================================
# Code Cell 59
# ============================================================================
# ============================================================
# Train Final Decision Tree
# ============================================================

dt_final_model = DecisionTreeClassifier(
    max_depth=15,
    min_samples_leaf=10,
    random_state=RANDOM_STATE
)

dt_pipeline, dt_result, dt_cv = run_model_workflow(
    model_name="Decision Tree",
    estimator=dt_final_model,
    preprocessor=unscaled_preprocessor,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    cv=3
)

# ============================================================================
# Code Cell 60
# ============================================================================
# ============================================================
# Save Decision Tree Performance Table
# ============================================================

dt_y_pred = model_predictions["Decision Tree"]
dt_y_prob = model_probabilities["Decision Tree"]

dt_result = {
    "Model": "Decision Tree",
    "Accuracy": accuracy_score(y_test, dt_y_pred),
    "Precision": precision_score(y_test, dt_y_pred),
    "Recall": recall_score(y_test, dt_y_pred),
    "Specificity": calculate_specificity(y_test, dt_y_pred),
    "F1": f1_score(y_test, dt_y_pred),
    "ROC AUC": roc_auc_score(y_test, dt_y_prob),
    "CV Accuracy": dt_grid.best_score_,
    "Best Parameters": str(dt_grid.best_params_)
}

dt_performance_df = pd.DataFrame([dt_result])

dt_performance_df.to_csv(
    MODELING_RESULT_DIR / "decision_tree_performance.csv",
    index=False
)

print("✓ Decision Tree performance table saved.")

# ============================================================================
# Code Cell 61
# ============================================================================
# ============================================================
# Save Decision Tree Predictions
# ============================================================

dt_predictions_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": dt_y_pred,
    "y_prob": dt_y_prob
})

dt_predictions_df.to_csv(
    MODELING_RESULT_DIR / "decision_tree_predictions.csv",
    index=False
)

print("✓ Decision Tree predictions table saved.")

# ============================================================================
# Code Cell 62
# ============================================================================
# ============================================================
# Save Decision Tree Classification Report
# ============================================================

dt_report = classification_report(
    y_test,
    dt_y_pred,
    digits=4
)

report_path = (
    CLASSIFICATION_REPORT_DIR /
    "decision_tree_classification_report.txt"
)

with open(report_path, "w") as f:
    f.write("Decision Tree\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Best Parameters: {dt_grid.best_params_}\n")
    f.write(f"Grid Search CV Accuracy: {dt_grid.best_score_:.4f}\n\n")
    f.write(dt_report)

print(f"✓ Decision Tree classification report saved:\n{report_path}")

# ============================================================================
# Code Cell 63
# ============================================================================
# ============================================================
# Save Decision Tree Confusion Matrix
# ============================================================

dt_cm = confusion_matrix(y_test, dt_y_pred)

plt.figure(figsize=(7, 6))

sns.heatmap(
    dt_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Non-Serious", "Serious"],
    yticklabels=["Non-Serious", "Serious"]
)

plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    CM_FIGURE_DIR / "decision_tree_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ Decision Tree confusion matrix saved.")

# ============================================================================
# Code Cell 64
# ============================================================================
# ============================================================
# Save Decision Tree ROC Curve
# ============================================================

dt_fpr, dt_tpr, _ = roc_curve(
    y_test,
    dt_y_prob
)

dt_auc = roc_auc_score(
    y_test,
    dt_y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    dt_fpr,
    dt_tpr,
    linewidth=2,
    label=f"Decision Tree (AUC = {dt_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve - Decision Tree")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(
    ROC_FIGURE_DIR / "decision_tree_roc_curve.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ Decision Tree ROC curve saved.")

# ============================================================================
# Code Cell 65
# ============================================================================
# ============================================================
# Save Decision Tree Precision-Recall Curve
# ============================================================

dt_precision_vals, dt_recall_vals, _ = precision_recall_curve(
    y_test,
    dt_y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    dt_recall_vals,
    dt_precision_vals,
    linewidth=2,
    label="Decision Tree"
)

plt.title("Precision-Recall Curve - Decision Tree")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")

plt.tight_layout()

plt.savefig(
    PR_FIGURE_DIR / "decision_tree_precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ Decision Tree precision-recall curve saved.")

# ============================================================================
# Code Cell 66
# ============================================================================
# ============================================================
# Save Decision Tree Feature Importance
# ============================================================

# Get feature names after preprocessing
dt_feature_names = dt_pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

# Get feature importance values
dt_importance = dt_pipeline.named_steps[
    "model"
].feature_importances_

# Create feature importance table
dt_feature_importance_df = pd.DataFrame({
    "Feature": dt_feature_names,
    "Importance": dt_importance
}).sort_values(
    "Importance",
    ascending=False
)

# Save full feature importance table
dt_feature_importance_df.to_csv(
    MODELING_RESULT_DIR / "decision_tree_feature_importance.csv",
    index=False
)

# Display top 20
display(dt_feature_importance_df.head(20))

print("✓ Decision Tree feature importance table saved.")

# ============================================================================
# Code Cell 67
# ============================================================================
# ============================================================
# Save Decision Tree Top 20 Feature Importance Plot
# ============================================================

dt_top20_importance = dt_feature_importance_df.head(20)

plt.figure(figsize=(10, 8))

sns.barplot(
    data=dt_top20_importance,
    x="Importance",
    y="Feature"
)

plt.title("Top 20 Decision Tree Feature Importances")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    FEATURE_IMPORTANCE_FIGURE_DIR / "decision_tree_top20_feature_importance.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ Decision Tree feature importance plot saved.")

# ============================================================================
# Code Cell 68
# ============================================================================
# ============================================================
# Validate Decision Tree Saved Outputs
# ============================================================

dt_files_to_check = [
    MODELING_RESULT_DIR / "decision_tree_performance.csv",
    MODELING_RESULT_DIR / "decision_tree_predictions.csv",
    MODELING_RESULT_DIR / "decision_tree_feature_importance.csv",
    CLASSIFICATION_REPORT_DIR / "decision_tree_classification_report.txt",
    CM_FIGURE_DIR / "decision_tree_confusion_matrix.png",
    ROC_FIGURE_DIR / "decision_tree_roc_curve.png",
    PR_FIGURE_DIR / "decision_tree_precision_recall_curve.png",
    FEATURE_IMPORTANCE_FIGURE_DIR / "decision_tree_top20_feature_importance.png"
]

print("=" * 80)
print("DECISION TREE OUTPUT VERIFICATION")
print("=" * 80)

for file in dt_files_to_check:
    if file.exists():
        print(f"✓ {file.name}")
    else:
        print(f"✗ {file.name}")

print("=" * 80)
print(f"Total Files Checked: {len(dt_files_to_check)}")
print("=" * 80)

# ============================================================================
# Code Cell 69
# ============================================================================
# ============================================================
# Random Forest Hyperparameter Tuning
# ============================================================

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", unscaled_preprocessor),
        (
            "model",
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)

rf_param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [15, 20],
    "model__min_samples_leaf": [1, 5]
}

rf_grid = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_param_grid,
    scoring="accuracy",
    cv=3,
    n_jobs=-1,
    verbose=2
)

rf_grid.fit(
    X_train,
    y_train
)

print("=" * 70)
print("RANDOM FOREST GRID SEARCH COMPLETE")
print("=" * 70)

print(f"Best Parameters : {rf_grid.best_params_}")
print(f"Best CV Accuracy: {rf_grid.best_score_:.4f}")

# ============================================================================
# Code Cell 70
# ============================================================================
# ============================================================
# Train Final Random Forest
# ============================================================

rf_final_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_pipeline, rf_result, rf_cv = run_model_workflow(
    model_name="Random Forest",
    estimator=rf_final_model,
    preprocessor=unscaled_preprocessor,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    cv=3
)

# ============================================================================
# Code Cell 71
# ============================================================================
# ============================================================
# Save Random Forest Performance Table
# ============================================================

rf_y_pred = model_predictions["Random Forest"]
rf_y_prob = model_probabilities["Random Forest"]

rf_result = {
    "Model": "Random Forest",
    "Accuracy": accuracy_score(y_test, rf_y_pred),
    "Precision": precision_score(y_test, rf_y_pred),
    "Recall": recall_score(y_test, rf_y_pred),
    "Specificity": calculate_specificity(y_test, rf_y_pred),
    "F1": f1_score(y_test, rf_y_pred),
    "ROC AUC": roc_auc_score(y_test, rf_y_prob),
    "CV Accuracy": rf_grid.best_score_,
    "Best Parameters": str(rf_grid.best_params_)
}

rf_performance_df = pd.DataFrame([rf_result])

rf_performance_df.to_csv(
    MODELING_RESULT_DIR / "random_forest_performance.csv",
    index=False
)

print("✓ Random Forest performance table saved.")

# ============================================================================
# Code Cell 72
# ============================================================================
# ============================================================
# Save Random Forest Predictions
# ============================================================

rf_predictions_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": rf_y_pred,
    "y_prob": rf_y_prob
})

rf_predictions_df.to_csv(
    MODELING_RESULT_DIR / "random_forest_predictions.csv",
    index=False
)

print("✓ Random Forest predictions table saved.")

# ============================================================================
# Code Cell 73
# ============================================================================
# ============================================================
# Save Random Forest Classification Report
# ============================================================

rf_report = classification_report(
    y_test,
    rf_y_pred,
    digits=4
)

report_path = (
    CLASSIFICATION_REPORT_DIR /
    "random_forest_classification_report.txt"
)

with open(report_path, "w") as f:
    f.write("Random Forest\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Best Parameters: {rf_grid.best_params_}\n")
    f.write(f"Grid Search CV Accuracy: {rf_grid.best_score_:.4f}\n\n")
    f.write(rf_report)

print(f"✓ Random Forest classification report saved:\n{report_path}")

# ============================================================================
# Code Cell 74
# ============================================================================
# ============================================================
# Save Random Forest Confusion Matrix
# ============================================================

rf_cm = confusion_matrix(y_test, rf_y_pred)

plt.figure(figsize=(7, 6))

sns.heatmap(
    rf_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Non-Serious", "Serious"],
    yticklabels=["Non-Serious", "Serious"]
)

plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    CM_FIGURE_DIR / "random_forest_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ Random Forest confusion matrix saved.")

# ============================================================================
# Code Cell 75
# ============================================================================
# ============================================================
# Save Random Forest ROC Curve
# ============================================================

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_y_prob
)

rf_auc = roc_auc_score(
    y_test,
    rf_y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    rf_fpr,
    rf_tpr,
    linewidth=2,
    label=f"Random Forest (AUC = {rf_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(
    ROC_FIGURE_DIR / "random_forest_roc_curve.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ Random Forest ROC curve saved.")

# ============================================================================
# Code Cell 76
# ============================================================================
# ============================================================
# Save Random Forest Precision-Recall Curve
# ============================================================

rf_precision_vals, rf_recall_vals, _ = precision_recall_curve(
    y_test,
    rf_y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    rf_recall_vals,
    rf_precision_vals,
    linewidth=2,
    label="Random Forest"
)

plt.title("Precision-Recall Curve - Random Forest")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")

plt.tight_layout()

plt.savefig(
    PR_FIGURE_DIR / "random_forest_precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ Random Forest precision-recall curve saved.")

# ============================================================================
# Code Cell 77
# ============================================================================
# ============================================================
# Save Random Forest Feature Importance
# ============================================================

# Get feature names after preprocessing
rf_feature_names = rf_pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

# Get feature importance values
rf_importance = rf_pipeline.named_steps[
    "model"
].feature_importances_

# Create feature importance table
rf_feature_importance_df = pd.DataFrame({
    "Feature": rf_feature_names,
    "Importance": rf_importance
}).sort_values(
    "Importance",
    ascending=False
)

# Save full feature importance table
rf_feature_importance_df.to_csv(
    MODELING_RESULT_DIR / "random_forest_feature_importance.csv",
    index=False
)

# Display top 20
display(rf_feature_importance_df.head(20))

print("✓ Random Forest feature importance table saved.")

# ============================================================================
# Code Cell 78
# ============================================================================
# ============================================================
# Save Random Forest Top 20 Feature Importance Plot
# ============================================================

rf_top20_importance = rf_feature_importance_df.head(20)

plt.figure(figsize=(10, 8))

sns.barplot(
    data=rf_top20_importance,
    x="Importance",
    y="Feature"
)

plt.title("Top 20 Random Forest Feature Importances")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    FEATURE_IMPORTANCE_FIGURE_DIR /
    "random_forest_top20_feature_importance.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ Random Forest feature importance plot saved.")

# ============================================================================
# Code Cell 79
# ============================================================================
# ============================================================
# Validate Random Forest Saved Outputs
# ============================================================

rf_files_to_check = [
    MODELING_RESULT_DIR / "random_forest_performance.csv",
    MODELING_RESULT_DIR / "random_forest_predictions.csv",
    MODELING_RESULT_DIR / "random_forest_feature_importance.csv",
    CLASSIFICATION_REPORT_DIR / "random_forest_classification_report.txt",
    CM_FIGURE_DIR / "random_forest_confusion_matrix.png",
    ROC_FIGURE_DIR / "random_forest_roc_curve.png",
    PR_FIGURE_DIR / "random_forest_precision_recall_curve.png",
    FEATURE_IMPORTANCE_FIGURE_DIR / "random_forest_top20_feature_importance.png"
]

print("=" * 80)
print("RANDOM FOREST OUTPUT VERIFICATION")
print("=" * 80)

for file in rf_files_to_check:
    if file.exists():
        print(f"✓ {file.name}")
    else:
        print(f"✗ {file.name}")

print("=" * 80)
print(f"Total Files Checked: {len(rf_files_to_check)}")
print("=" * 80)

# ============================================================================
# Code Cell 80
# ============================================================================
# ============================================================
# Create Stratified Sample for SVM
# ============================================================

# SVM is computationally expensive on very large datasets.
# A stratified sample is used to preserve class proportions.

svm_sample_fraction = 20000 / len(X_train)

X_train_svm, _, y_train_svm, _ = train_test_split(
    X_train,
    y_train,
    train_size=svm_sample_fraction,
    stratify=y_train,
    random_state=RANDOM_STATE
)

print("=" * 70)
print("SVM TRAINING SAMPLE CREATED")
print("=" * 70)

print(f"Original Training Size : {len(X_train):,}")
print(f"SVM Training Size      : {len(X_train_svm):,}")

print("\nClass Distribution")

print(
    pd.DataFrame({
        "Original (%)": y_train.value_counts(normalize=True).sort_index() * 100,
        "SVM Sample (%)": y_train_svm.value_counts(normalize=True).sort_index() * 100
    }).round(2)
)

# ============================================================================
# Code Cell 81
# ============================================================================
# ============================================================
# SVM Hyperparameter Tuning
# ============================================================

svm_pipeline = Pipeline(
    steps=[
        ("preprocessor", scaled_preprocessor),
        (
            "model",
            SVC(
                probability=True,
                random_state=RANDOM_STATE
            )
        )
    ]
)

svm_param_grid = {
    "model__C": [0.1, 1, 10],
    "model__kernel": ["rbf"]
}

svm_grid = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=svm_param_grid,
    scoring="accuracy",
    cv=3,
    n_jobs=-1,
    verbose=2
)

svm_grid.fit(
    X_train_svm,
    y_train_svm
)

print("=" * 70)
print("SVM GRID SEARCH COMPLETE")
print("=" * 70)

print(f"Best Parameters : {svm_grid.best_params_}")
print(f"Best CV Accuracy: {svm_grid.best_score_:.4f}")

# ============================================================================
# Code Cell 82
# ============================================================================
# ============================================================
# Train Final Support Vector Machine
# ============================================================

svm_final_model = SVC(
    C=10,
    kernel="rbf",
    probability=True,
    random_state=RANDOM_STATE
)

svm_pipeline, svm_result, svm_cv = run_model_workflow(
    model_name="Support Vector Machine",
    estimator=svm_final_model,
    preprocessor=scaled_preprocessor,
    X_train=X_train_svm,
    X_test=X_test,
    y_train=y_train_svm,
    y_test=y_test,
    cv=3
)

# ============================================================================
# Code Cell 83
# ============================================================================
# ============================================================
# Save SVM Performance Table
# ============================================================

svm_y_pred = model_predictions["Support Vector Machine"]
svm_y_prob = model_probabilities["Support Vector Machine"]

svm_result = {
    "Model": "Support Vector Machine",
    "Accuracy": accuracy_score(y_test, svm_y_pred),
    "Precision": precision_score(y_test, svm_y_pred),
    "Recall": recall_score(y_test, svm_y_pred),
    "Specificity": calculate_specificity(y_test, svm_y_pred),
    "F1": f1_score(y_test, svm_y_pred),
    "ROC AUC": roc_auc_score(y_test, svm_y_prob),
    "CV Accuracy": svm_grid.best_score_,
    "Best Parameters": str(svm_grid.best_params_)
}

svm_performance_df = pd.DataFrame([svm_result])

svm_performance_df.to_csv(
    MODELING_RESULT_DIR / "svm_performance.csv",
    index=False
)

print("✓ SVM performance table saved.")

# ============================================================================
# Code Cell 84
# ============================================================================
# ============================================================
# Save SVM Classification Report
# ============================================================

svm_report = classification_report(
    y_test,
    svm_y_pred,
    digits=4
)

report_path = (
    CLASSIFICATION_REPORT_DIR /
    "svm_classification_report.txt"
)

with open(report_path, "w") as f:
    f.write("Support Vector Machine\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Best Parameters: {svm_grid.best_params_}\n")
    f.write(f"Grid Search CV Accuracy: {svm_grid.best_score_:.4f}\n\n")
    f.write(svm_report)

print(f"✓ SVM classification report saved:\n{report_path}")

# ============================================================================
# Code Cell 85
# ============================================================================
# ============================================================
# Save SVM Confusion Matrix
# ============================================================

svm_cm = confusion_matrix(y_test, svm_y_pred)

plt.figure(figsize=(7, 6))

sns.heatmap(
    svm_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Non-Serious", "Serious"],
    yticklabels=["Non-Serious", "Serious"]
)

plt.title("Confusion Matrix - Support Vector Machine")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    CM_FIGURE_DIR / "svm_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ SVM confusion matrix saved.")

# ============================================================================
# Code Cell 86
# ============================================================================
# ============================================================
# Save SVM ROC Curve
# ============================================================

svm_fpr, svm_tpr, _ = roc_curve(
    y_test,
    svm_y_prob
)

svm_auc = roc_auc_score(
    y_test,
    svm_y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    svm_fpr,
    svm_tpr,
    linewidth=2,
    label=f"SVM (AUC = {svm_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve - Support Vector Machine")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(
    ROC_FIGURE_DIR / "svm_roc_curve.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ SVM ROC curve saved.")

# ============================================================================
# Code Cell 87
# ============================================================================
# ============================================================
# Save SVM Precision-Recall Curve
# ============================================================

svm_precision_vals, svm_recall_vals, _ = precision_recall_curve(
    y_test,
    svm_y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    svm_recall_vals,
    svm_precision_vals,
    linewidth=2,
    label="SVM"
)

plt.title("Precision-Recall Curve - Support Vector Machine")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")

plt.tight_layout()

plt.savefig(
    PR_FIGURE_DIR / "svm_precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ SVM precision-recall curve saved.")

# ============================================================================
# Code Cell 88
# ============================================================================
# ============================================================
# Save SVM Predictions
# ============================================================

svm_predictions_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": svm_y_pred,
    "y_prob": svm_y_prob
})

svm_predictions_df.to_csv(
    MODELING_RESULT_DIR / "svm_predictions.csv",
    index=False
)

print("✓ svm_predictions.csv saved.")
print(svm_predictions_df.head())

# ============================================================================
# Code Cell 89
# ============================================================================
# ============================================================
# Validate SVM Saved Outputs
# ============================================================

svm_files_to_check = [
    MODELING_RESULT_DIR / "svm_performance.csv",
    MODELING_RESULT_DIR / "svm_predictions.csv",
    CLASSIFICATION_REPORT_DIR / "svm_classification_report.txt",
    CM_FIGURE_DIR / "svm_confusion_matrix.png",
    ROC_FIGURE_DIR / "svm_roc_curve.png",
    PR_FIGURE_DIR / "svm_precision_recall_curve.png"
]

print("=" * 80)
print("SUPPORT VECTOR MACHINE OUTPUT VERIFICATION")
print("=" * 80)

for file in svm_files_to_check:
    if file.exists():
        print(f"✓ {file.name}")
    else:
        print(f"✗ {file.name}")

print("=" * 80)
print(f"Total Files Checked: {len(svm_files_to_check)}")
print("=" * 80)

# ============================================================================
# Code Cell 90
# ============================================================================
# ============================================================
# Create Stratified Sample for Neural Network
# ============================================================

# MLP training can be computationally expensive on a local machine.
# A stratified sample is used to preserve class proportions.

mlp_sample_fraction = 20000 / len(X_train)

X_train_mlp, _, y_train_mlp, _ = train_test_split(
    X_train,
    y_train,
    train_size=mlp_sample_fraction,
    stratify=y_train,
    random_state=RANDOM_STATE
)

print("=" * 70)
print("MLP TRAINING SAMPLE CREATED")
print("=" * 70)

print(f"Original Training Size : {len(X_train):,}")
print(f"MLP Training Size      : {len(X_train_mlp):,}")

print("\nClass Distribution")

print(
    pd.DataFrame({
        "Original (%)": y_train.value_counts(normalize=True).sort_index() * 100,
        "MLP Sample (%)": y_train_mlp.value_counts(normalize=True).sort_index() * 100
    }).round(2)
)

# ============================================================================
# Code Cell 91
# ============================================================================
# ============================================================
# MLP Hyperparameter Tuning
# ============================================================

mlp_pipeline = Pipeline(
    steps=[
        ("preprocessor", scaled_preprocessor),
        (
            "model",
            MLPClassifier(
                random_state=RANDOM_STATE,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.10
            )
        )
    ]
)

mlp_param_grid = {
    "model__hidden_layer_sizes": [(50,), (100,), (50, 25)],
    "model__alpha": [0.0001, 0.001],
    "model__activation": ["relu"]
}

mlp_grid = GridSearchCV(
    estimator=mlp_pipeline,
    param_grid=mlp_param_grid,
    scoring="accuracy",
    cv=3,
    n_jobs=-1,
    verbose=2
)

mlp_grid.fit(
    X_train_mlp,
    y_train_mlp
)

print("=" * 70)
print("MLP GRID SEARCH COMPLETE")
print("=" * 70)

print(f"Best Parameters : {mlp_grid.best_params_}")
print(f"Best CV Accuracy: {mlp_grid.best_score_:.4f}")

# ============================================================================
# Code Cell 92
# ============================================================================
# ============================================================
# Train Final Neural Network Model
# ============================================================

mlp_final_model = MLPClassifier(
    hidden_layer_sizes=(50,),
    activation="relu",
    alpha=0.001,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.10,
    random_state=RANDOM_STATE
)

mlp_pipeline, mlp_result, mlp_cv = run_model_workflow(
    model_name="Neural Network",
    estimator=mlp_final_model,
    preprocessor=scaled_preprocessor,
    X_train=X_train_mlp,
    X_test=X_test,
    y_train=y_train_mlp,
    y_test=y_test,
    cv=3
)

# ============================================================================
# Code Cell 93
# ============================================================================
# ============================================================
# Save Neural Network Performance Table
# ============================================================

mlp_y_pred = model_predictions["Neural Network"]
mlp_y_prob = model_probabilities["Neural Network"]

mlp_result = {
    "Model": "Neural Network",
    "Accuracy": accuracy_score(y_test, mlp_y_pred),
    "Precision": precision_score(y_test, mlp_y_pred),
    "Recall": recall_score(y_test, mlp_y_pred),
    "Specificity": calculate_specificity(y_test, mlp_y_pred),
    "F1": f1_score(y_test, mlp_y_pred),
    "ROC AUC": roc_auc_score(y_test, mlp_y_prob),
    "CV Accuracy": mlp_grid.best_score_,
    "Best Parameters": str(mlp_grid.best_params_)
}

mlp_performance_df = pd.DataFrame([mlp_result])

mlp_performance_df.to_csv(
    MODELING_RESULT_DIR / "neural_network_performance.csv",
    index=False
)

print("✓ Neural Network performance table saved.")

# ============================================================================
# Code Cell 94
# ============================================================================
# ============================================================
# Save Neural Network Predictions
# ============================================================

mlp_predictions_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": mlp_y_pred,
    "y_prob": mlp_y_prob
})

mlp_predictions_df.to_csv(
    MODELING_RESULT_DIR / "neural_network_predictions.csv",
    index=False
)

print("✓ Neural Network predictions table saved.")

# ============================================================================
# Code Cell 95
# ============================================================================
# ============================================================
# Save Neural Network Classification Report
# ============================================================

mlp_report = classification_report(
    y_test,
    mlp_y_pred,
    digits=4
)

report_path = (
    CLASSIFICATION_REPORT_DIR /
    "neural_network_classification_report.txt"
)

with open(report_path, "w") as f:
    f.write("Neural Network / MLPClassifier\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Best Parameters: {mlp_grid.best_params_}\n")
    f.write(f"Grid Search CV Accuracy: {mlp_grid.best_score_:.4f}\n\n")
    f.write(mlp_report)

print(f"✓ Neural Network classification report saved:\n{report_path}")

# ============================================================================
# Code Cell 96
# ============================================================================
# ============================================================
# Save Neural Network Confusion Matrix
# ============================================================

mlp_cm = confusion_matrix(y_test, mlp_y_pred)

plt.figure(figsize=(7, 6))

sns.heatmap(
    mlp_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Non-Serious", "Serious"],
    yticklabels=["Non-Serious", "Serious"]
)

plt.title("Confusion Matrix - Neural Network")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    CM_FIGURE_DIR / "neural_network_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ Neural Network confusion matrix saved.")

# ============================================================================
# Code Cell 97
# ============================================================================
# ============================================================
# Save Neural Network ROC Curve
# ============================================================

mlp_fpr, mlp_tpr, _ = roc_curve(
    y_test,
    mlp_y_prob
)

mlp_auc = roc_auc_score(
    y_test,
    mlp_y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    mlp_fpr,
    mlp_tpr,
    linewidth=2,
    label=f"Neural Network (AUC = {mlp_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve - Neural Network")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(
    ROC_FIGURE_DIR / "neural_network_roc_curve.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ Neural Network ROC curve saved.")

# ============================================================================
# Code Cell 98
# ============================================================================
# ============================================================
# Save Neural Network Precision-Recall Curve
# ============================================================

mlp_precision_vals, mlp_recall_vals, _ = precision_recall_curve(
    y_test,
    mlp_y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    mlp_recall_vals,
    mlp_precision_vals,
    linewidth=2,
    label="Neural Network"
)

plt.title("Precision-Recall Curve - Neural Network")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")

plt.tight_layout()

plt.savefig(
    PR_FIGURE_DIR / "neural_network_precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ Neural Network precision-recall curve saved.")

# ============================================================================
# Code Cell 99
# ============================================================================
# ============================================================
# Validate Neural Network Saved Outputs
# ============================================================

mlp_files_to_check = [
    MODELING_RESULT_DIR / "neural_network_performance.csv",
    MODELING_RESULT_DIR / "neural_network_predictions.csv",
    CLASSIFICATION_REPORT_DIR / "neural_network_classification_report.txt",
    CM_FIGURE_DIR / "neural_network_confusion_matrix.png",
    ROC_FIGURE_DIR / "neural_network_roc_curve.png",
    PR_FIGURE_DIR / "neural_network_precision_recall_curve.png"
]

print("=" * 80)
print("NEURAL NETWORK OUTPUT VERIFICATION")
print("=" * 80)

for file in mlp_files_to_check:
    if file.exists():
        print(f"✓ {file.name}")
    else:
        print(f"✗ {file.name}")

print("=" * 80)
print(f"Total Files Checked: {len(mlp_files_to_check)}")
print("=" * 80)

# ============================================================================
# Code Cell 100
# ============================================================================
# ============================================================
# Create Clustering Dataset
# ============================================================

# Remove identifier and target variable
clustering_df = df_model.drop(
    columns=[
        "primaryid",
        "is_serious"
    ]
)

# Keep only numeric features
clustering_df = clustering_df.select_dtypes(
    include=["number"]
).copy()

print("=" * 70)
print("CLUSTERING DATASET")
print("=" * 70)

print(f"Rows    : {clustering_df.shape[0]:,}")
print(f"Columns : {clustering_df.shape[1]}")

display(clustering_df.head())

# ============================================================================
# Code Cell 101
# ============================================================================
# ============================================================
# Impute Missing Values and Standardize Features
# ============================================================

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Median imputation
imputer = SimpleImputer(strategy="median")

clustering_imputed = imputer.fit_transform(clustering_df)

# Standardization
scaler = StandardScaler()

clustering_scaled = scaler.fit_transform(clustering_imputed)

print("=" * 70)
print("CLUSTERING DATA PREPARED")
print("=" * 70)

print("Missing values after imputation:",
      np.isnan(clustering_scaled).sum())

print("Scaled dataset shape:", clustering_scaled.shape)

# ============================================================================
# Code Cell 102
# ============================================================================
# ============================================================
# Create Sample for Elbow and Silhouette Analysis
# ============================================================

from sklearn.model_selection import train_test_split

# Random sample of 100,000 observations
cluster_sample_size = 100000

_, clustering_sample = train_test_split(
    clustering_scaled,
    test_size=cluster_sample_size,
    random_state=RANDOM_STATE
)

print("=" * 70)
print("CLUSTERING SAMPLE CREATED")
print("=" * 70)

print(f"Full Dataset   : {clustering_scaled.shape[0]:,}")
print(f"Sample Dataset : {clustering_sample.shape[0]:,}")
print(f"Features       : {clustering_sample.shape[1]}")

# ============================================================================
# Code Cell 103
# ============================================================================
# ============================================================
# Elbow Method
# ============================================================

from sklearn.cluster import KMeans

k_values = range(2, 11)
inertias = []

for k in k_values:

    km = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=20
    )

    km.fit(clustering_sample)

    inertias.append(km.inertia_)

plt.figure(figsize=(8,6))

plt.plot(
    k_values,
    inertias,
    marker="o",
    linewidth=2
)

plt.xticks(k_values)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Within-Cluster Sum of Squares (Inertia)")
plt.title("Elbow Method for K-Means")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    MODELING_RESULT_DIR / "elbow_method.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ Elbow method figure saved.")

# ============================================================================
# Code Cell 104
# ============================================================================
# ============================================================
# Silhouette Analysis
# ============================================================

from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in range(2, 11):

    km = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=20
    )

    labels = km.fit_predict(clustering_sample)

    score = silhouette_score(
        clustering_sample,
        labels
    )

    silhouette_scores.append(score)

    print(f"K = {k}: Silhouette Score = {score:.4f}")

# Plot
plt.figure(figsize=(8,6))

plt.plot(
    range(2,11),
    silhouette_scores,
    marker="o",
    linewidth=2
)

plt.xticks(range(2,11))

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis for K-Means")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    MODELING_RESULT_DIR / "silhouette_scores.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ Silhouette analysis figure saved.")

# ============================================================================
# Code Cell 105
# ============================================================================
# ============================================================
# Train Final K-Means Model
# ============================================================

FINAL_K = 7

kmeans = KMeans(
    n_clusters=FINAL_K,
    random_state=RANDOM_STATE,
    n_init=20
)

cluster_labels = kmeans.fit_predict(clustering_scaled)

cluster_summary = (
    pd.Series(cluster_labels)
      .value_counts()
      .sort_index()
      .reset_index()
)

cluster_summary.columns = [
    "Cluster",
    "Count"
]

cluster_summary["Percent"] = (
    cluster_summary["Count"]
    / len(cluster_labels)
    * 100
).round(2)

print("=" * 70)
print("FINAL K-MEANS MODEL")
print("=" * 70)

display(cluster_summary)

# ============================================================================
# Code Cell 106
# ============================================================================
# ============================================================
# Clustering Output Directory
# ============================================================

from pathlib import Path

CLUSTERING_RESULT_DIR = (
    PROJECT_ROOT /
    "results" /
    "modeling" /
    "clustering"
)

CLUSTERING_RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print(CLUSTERING_RESULT_DIR)

# ============================================================================
# Code Cell 107
# ============================================================================
# ============================================================
# Save Cluster Assignments
# ============================================================

cluster_assignments = df_model[
    ["primaryid", "is_serious"]
].copy()

cluster_assignments["cluster"] = cluster_labels

cluster_assignments.to_csv(
    CLUSTERING_RESULT_DIR / "kmeans_cluster_assignments.csv",
    index=False
)

print("✓ Cluster assignments saved.")

# ============================================================================
# Code Cell 108
# ============================================================================
# ============================================================
# Save Cluster Centroids
# ============================================================

cluster_centroids = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=clustering_df.columns
)

cluster_centroids.index.name = "Cluster"

cluster_centroids.to_csv(
    CLUSTERING_RESULT_DIR / "kmeans_cluster_centroids.csv"
)

print("✓ Cluster centroids saved.")

display(cluster_centroids.head())

# ============================================================================
# Code Cell 109
# ============================================================================
# ============================================================
# PCA Visualization of Clusters
# ============================================================

from sklearn.decomposition import PCA

# Reduce to two principal components
pca = PCA(
    n_components=2,
    random_state=RANDOM_STATE
)

pca_components = pca.fit_transform(clustering_scaled)

pca_df = pd.DataFrame({
    "PC1": pca_components[:, 0],
    "PC2": pca_components[:, 1],
    "Cluster": cluster_labels.astype(str)
})

plt.figure(figsize=(10, 8))

sns.scatterplot(
    data=pca_df.sample(
        n=30000,
        random_state=RANDOM_STATE
    ),
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="tab10",
    s=12,
    alpha=0.6
)

plt.title("K-Means Clusters (PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.legend(
    title="Cluster",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    CLUSTERING_RESULT_DIR / "pca_clusters.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ PCA cluster visualization saved.")

# ============================================================================
# Code Cell 110
# ============================================================================
# ============================================================
# Cluster Size Visualization
# ============================================================

plt.figure(figsize=(8,6))

sns.barplot(
    data=cluster_summary,
    x="Cluster",
    y="Count",
    palette="viridis",
    hue="Cluster",
    legend=False
)

for i, row in cluster_summary.iterrows():
    plt.text(
        i,
        row["Count"] + 2000,
        f'{row["Percent"]:.1f}%',
        ha="center",
        fontsize=10
    )

plt.title("Number of Reports in Each K-Means Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Reports")

plt.tight_layout()

plt.savefig(
    CLUSTERING_RESULT_DIR / "cluster_sizes.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ Cluster size figure saved.")

# ============================================================================
# Code Cell 111
# ============================================================================
# ============================================================
# Cluster vs Serious Adverse Event Distribution
# ============================================================

cluster_results = pd.DataFrame({
    "cluster": cluster_labels,
    "is_serious": df_model["is_serious"].values
})

cluster_serious = (
    pd.crosstab(
        cluster_results["cluster"],
        cluster_results["is_serious"]
    )
    .rename(columns={
        0: "Non-Serious",
        1: "Serious"
    })
)

# Add totals and percentages
cluster_serious["Total"] = (
    cluster_serious["Non-Serious"] +
    cluster_serious["Serious"]
)

cluster_serious["Serious (%)"] = (
    cluster_serious["Serious"] /
    cluster_serious["Total"] * 100
).round(2)

cluster_serious["Non-Serious (%)"] = (
    cluster_serious["Non-Serious"] /
    cluster_serious["Total"] * 100
).round(2)

display(cluster_serious)

# Save table
cluster_serious.to_csv(
    CLUSTERING_RESULT_DIR /
    "kmeans_cluster_serious_distribution.csv"
)

print("✓ Cluster serious distribution saved.")

# ============================================================================
# Code Cell 112
# ============================================================================
# ============================================================
# Serious vs Non-Serious by Cluster
# ============================================================

plot_df = cluster_serious.reset_index()

plot_df = plot_df[
    ["cluster", "Non-Serious", "Serious"]
]

plot_df = plot_df.set_index("cluster")

ax = plot_df.plot(
    kind="bar",
    stacked=True,
    figsize=(10,6),
    colormap="Set2"
)

plt.title("Serious vs Non-Serious Reports by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Reports")

plt.legend(
    title="Outcome",
    bbox_to_anchor=(1.02,1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    CLUSTERING_RESULT_DIR /
    "cluster_serious_distribution.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ Cluster serious distribution figure saved.")

# ============================================================================
# Code Cell 113
# ============================================================================
# ============================================================
# Cluster Profile Summary
# ============================================================

# Create dataframe with cluster labels
cluster_profile_df = df_model.copy()

cluster_profile_df["cluster"] = cluster_labels

# Variables to summarize
profile_features = [
    "age_years",
    "weight_kg",
    "total_drugs",
    "num_unique_drugs",
    "num_reactions",
    "contains_insulin",
    "contains_sglt2",
    "contains_sulfonylurea",
    "mean_therapy_duration_days"
]

# Keep only variables that exist
profile_features = [
    col for col in profile_features
    if col in cluster_profile_df.columns
]

cluster_profile = (
    cluster_profile_df
    .groupby("cluster")[profile_features]
    .mean()
    .round(2)
)

display(cluster_profile)

cluster_profile.to_csv(
    CLUSTERING_RESULT_DIR /
    "kmeans_cluster_summary.csv"
)

print("✓ Cluster profile summary saved.")

# ============================================================================
# Code Cell 114
# ============================================================================
# ============================================================
# Cluster Centroid Heatmap
# ============================================================

plt.figure(figsize=(18, 8))

sns.heatmap(
    cluster_centroids,
    cmap="coolwarm",
    center=0,
    cbar_kws={"label": "Standardized Feature Value"}
)

plt.title("Standardized Cluster Centroids")
plt.xlabel("Features")
plt.ylabel("Cluster")

plt.tight_layout()

plt.savefig(
    CLUSTERING_RESULT_DIR /
    "cluster_centroid_heatmap.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ Cluster centroid heatmap saved.")

# ============================================================================
# Code Cell 115
# ============================================================================
# ============================================================
# Validate Clustering Outputs
# ============================================================

clustering_files = [
    "kmeans_cluster_assignments.csv",
    "kmeans_cluster_centroids.csv",
    "kmeans_cluster_summary.csv",
    "kmeans_cluster_serious_distribution.csv",
    "elbow_method.png",
    "silhouette_scores.png",
    "pca_clusters.png",
    "cluster_sizes.png",
    "cluster_serious_distribution.png",
    "cluster_centroid_heatmap.png"
]

print("=" * 80)
print("K-MEANS CLUSTERING OUTPUT VERIFICATION")
print("=" * 80)

missing = 0

for file in clustering_files:
    path = CLUSTERING_RESULT_DIR / file
    if path.exists():
        print(f"✓ {file}")
    else:
        print(f"✗ {file}")
        missing += 1

print("=" * 80)
print(f"Files Checked : {len(clustering_files)}")
print(f"Missing Files : {missing}")
print("=" * 80)

# ============================================================================
# Code Cell 116
# ============================================================================
# ============================================================
# Load All Model Performance Tables
# ============================================================

PERFORMANCE_DIR = MODELING_RESULT_DIR / "performance"

performance_files = [
    "logistic_regression_performance.csv",
    "lda_performance.csv",
    "qda_performance.csv",
    "gaussian_naive_bayes_performance.csv",
    "knn_performance.csv",
    "decision_tree_performance.csv",
    "random_forest_performance.csv",
    "svm_performance.csv",
    "neural_network_performance.csv"
]

performance_tables = []

for file in performance_files:

    path = PERFORMANCE_DIR / file

    df_perf = pd.read_csv(path)

    performance_tables.append(df_perf)

performance_summary = pd.concat(
    performance_tables,
    ignore_index=True
)

print("=" * 70)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 70)

display(performance_summary)

# ============================================================================
# Code Cell 117
# ============================================================================
# ============================================================
# Save Master Performance Summary
# ============================================================

# Sort by ROC AUC first, then Accuracy
performance_summary_ranked = performance_summary.sort_values(
    by=["ROC AUC", "Accuracy"],
    ascending=False
).reset_index(drop=True)

performance_summary_ranked["Overall Rank"] = (
    performance_summary_ranked.index + 1
)

# Reorder columns
cols = ["Overall Rank"] + [
    col for col in performance_summary_ranked.columns
    if col != "Overall Rank"
]

performance_summary_ranked = performance_summary_ranked[cols]

# Save table
performance_summary_ranked.to_csv(
    PERFORMANCE_DIR / "model_performance_summary.csv",
    index=False
)

print("✓ Master performance summary saved.")

display(performance_summary_ranked)

# ============================================================================
# Code Cell 118
# ============================================================================
# ============================================================
# Create Model Rankings
# ============================================================

ranking_metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
    "ROC AUC"
]

model_rankings = performance_summary[
    ["Model"] + ranking_metrics
].copy()

for metric in ranking_metrics:
    model_rankings[f"{metric} Rank"] = (
        model_rankings[metric]
        .rank(ascending=False, method="min")
        .astype(int)
    )

model_rankings["Average Rank"] = (
    model_rankings[
        [f"{metric} Rank" for metric in ranking_metrics]
    ]
    .mean(axis=1)
    .round(2)
)

model_rankings = model_rankings.sort_values(
    "Average Rank"
).reset_index(drop=True)

model_rankings.to_csv(
    PERFORMANCE_DIR / "model_rankings.csv",
    index=False
)

print("✓ Model rankings saved.")

display(model_rankings)

# ============================================================================
# Code Cell 119
# ============================================================================
# ============================================================
# Weighted Model Ranking
# ============================================================

ranking = performance_summary.copy()

# Fill missing CV values with 0 so all models can be ranked
ranking["CV Accuracy"] = ranking["CV Accuracy"].fillna(0)

# Rank (1 = best)
ranking["Accuracy Rank"] = ranking["Accuracy"].rank(
    ascending=False,
    method="min"
)

ranking["ROC Rank"] = ranking["ROC AUC"].rank(
    ascending=False,
    method="min"
)

ranking["F1 Rank"] = ranking["F1"].rank(
    ascending=False,
    method="min"
)

ranking["CV Rank"] = ranking["CV Accuracy"].rank(
    ascending=False,
    method="min"
)

# Weighted score (lower is better)
ranking["Weighted Score"] = (
    ranking["Accuracy Rank"] * 0.30 +
    ranking["ROC Rank"] * 0.40 +
    ranking["F1 Rank"] * 0.20 +
    ranking["CV Rank"] * 0.10
)

ranking = ranking.sort_values(
    "Weighted Score"
).reset_index(drop=True)

ranking["Overall Rank"] = ranking.index + 1

display(
    ranking[
        [
            "Overall Rank",
            "Model",
            "Accuracy",
            "ROC AUC",
            "F1",
            "CV Accuracy",
            "Weighted Score"
        ]
    ]
)

ranking.to_csv(
    PERFORMANCE_DIR / "weighted_model_rankings.csv",
    index=False
)

print("✓ Weighted model rankings saved.")

# ============================================================================
# Code Cell 120
# ============================================================================
# ============================================================
# Comparison Output Directory
# ============================================================

COMPARISON_DIR = MODELING_RESULT_DIR / "comparison"

COMPARISON_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print(COMPARISON_DIR)

# ============================================================================
# Code Cell 121
# ============================================================================
# ============================================================
# Accuracy Comparison
# ============================================================

plot_df = performance_summary.sort_values(
    "Accuracy",
    ascending=False
)

plt.figure(figsize=(10,6))

sns.barplot(
    data=plot_df,
    x="Accuracy",
    y="Model",
    hue="Model",
    palette="viridis",
    legend=False
)

for i, value in enumerate(plot_df["Accuracy"]):
    plt.text(
        value + 0.001,
        i,
        f"{value:.3f}",
        va="center"
    )

plt.title("Model Comparison: Accuracy")
plt.xlim(0.65, 1.0)

plt.tight_layout()

plt.savefig(
    COMPARISON_DIR /
    "accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ Accuracy comparison saved.")

# ============================================================================
# Code Cell 122
# ============================================================================
# ============================================================
# Generate All Model Comparison Figures
# ============================================================

comparison_metrics = [
    ("ROC AUC", "roc_auc_comparison.png"),
    ("F1", "f1_comparison.png"),
    ("Precision", "precision_comparison.png"),
    ("Recall", "recall_comparison.png"),
    ("Specificity", "specificity_comparison.png")
]

for metric, filename in comparison_metrics:

    plot_df = performance_summary.sort_values(
        metric,
        ascending=False
    )

    plt.figure(figsize=(10,6))

    sns.barplot(
        data=plot_df,
        x=metric,
        y="Model",
        hue="Model",
        palette="viridis",
        legend=False
    )

    for i, value in enumerate(plot_df[metric]):
        plt.text(
            value + 0.001,
            i,
            f"{value:.3f}",
            va="center",
            fontsize=10
        )

    plt.title(f"Model Comparison: {metric}")

    plt.tight_layout()

    plt.savefig(
        COMPARISON_DIR / filename,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()
    plt.close()

print("✓ All comparison figures saved.")

# ============================================================================
# Code Cell 123
# ============================================================================
# ============================================================
# Cross-Validation Comparison
# ============================================================

cv_df = performance_summary[
    performance_summary["CV Accuracy"].notna()
].copy()

cv_df = cv_df.sort_values(
    "CV Accuracy",
    ascending=False
)

plt.figure(figsize=(8,5))

sns.barplot(
    data=cv_df,
    x="CV Accuracy",
    y="Model",
    hue="Model",
    palette="viridis",
    legend=False
)

for i, value in enumerate(cv_df["CV Accuracy"]):
    plt.text(
        value + 0.0005,
        i,
        f"{value:.3f}",
        va="center"
    )

plt.title("Cross-Validation Accuracy Comparison")

plt.tight_layout()

plt.savefig(
    COMPARISON_DIR / "cv_accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()

print("✓ CV comparison saved.")

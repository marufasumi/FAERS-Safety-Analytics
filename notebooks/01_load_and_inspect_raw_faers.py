# Extracted from: 01_load_and_inspect_raw_faers.ipynb.ipynb
# Complete executable code cells in notebook order.


# ============================================================================
# Code Cell 1
# ============================================================================
# =====================================================
# Import Required Libraries
# =====================================================

import os
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Suppress unnecessary warning messages
warnings.filterwarnings("ignore")

# Set plotting style
plt.style.use("ggplot")
sns.set_theme(style="whitegrid")

# Configure Pandas display options
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 120)

# ============================================================================
# Code Cell 2
# ============================================================================
# =====================================================
# Define Project Directory
# =====================================================

DATA_PATH = "../data/raw/faers_ascii_2024q1"

print("Raw data directory:")
print(DATA_PATH)

# ============================================================================
# Code Cell 3
# ============================================================================
# =====================================================
# Verify Raw Data Files
# =====================================================

# List all files in the raw data directory
files = sorted(os.listdir(DATA_PATH))

print(f"Number of files found: {len(files)}\n")

for file in files:
    print(file)

# ============================================================================
# Code Cell 4
# ============================================================================
# =====================================================
# Define Project Directory
# =====================================================

DATA_PATH = "../data/raw/faers_ascii_2024q1"

print("Raw data directory:")
print(DATA_PATH)

print("\nDoes this folder exist?")
print(os.path.exists(DATA_PATH))

# ============================================================================
# Code Cell 5
# ============================================================================
# Check what is inside the data folders

print("Project root:")
print(os.listdir(".."))

print("\nInside data:")
print(os.listdir("../data"))

print("\nInside data/raw, if it exists:")
if os.path.exists("../data/raw"):
    print(os.listdir("../data/raw"))
else:
    print("data/raw does not exist")

# ============================================================================
# Code Cell 6
# ============================================================================
# Check the contents of the extracted FAERS folder

print(os.listdir("../data/raw/faers_ascii_2024q1"))

# ============================================================================
# Code Cell 7
# ============================================================================
# =====================================================
# Define Project Directory
# =====================================================

DATA_PATH = "../data/raw/faers_ascii_2024q1/ASCII"

print("Raw data directory:")
print(DATA_PATH)

print("\nDoes this folder exist?")
print(os.path.exists(DATA_PATH))

# ============================================================================
# Code Cell 8
# ============================================================================
# =====================================================
# Verify Raw Data Files
# =====================================================

files = sorted(os.listdir(DATA_PATH))

print(f"Number of files found: {len(files)}\n")

for file in files:
    print(file)

# ============================================================================
# Code Cell 9
# ============================================================================
# =====================================================
# Define Raw Data File Names
# =====================================================

DEMO_FILE = "DEMO24Q1.txt"
DRUG_FILE = "DRUG24Q1.txt"
REAC_FILE = "REAC24Q1.txt"
OUTC_FILE = "OUTC24Q1.txt"
RPSR_FILE = "RPSR24Q1.txt"
THER_FILE = "THER24Q1.txt"
INDI_FILE = "INDI24Q1.txt"

print("Raw FAERS data files:")

for file in [
    DEMO_FILE,
    DRUG_FILE,
    REAC_FILE,
    OUTC_FILE,
    RPSR_FILE,
    THER_FILE,
    INDI_FILE
]:
    print(f"• {file}")

# ============================================================================
# Code Cell 10
# ============================================================================
# =====================================================
# Load DEMO Table
# =====================================================

demo = pd.read_csv(
    os.path.join(DATA_PATH, DEMO_FILE),
    sep="$",
    engine="python",
    encoding="latin1"
)

print("DEMO table loaded successfully.")

# ============================================================================
# Code Cell 11
# ============================================================================
# =====================================================
# Initial Inspection of the DEMO Table
# =====================================================

# Display dataset dimensions
print("=" * 60)
print("DEMO Table Dimensions")
print("=" * 60)
print(f"Number of rows    : {demo.shape[0]:,}")
print(f"Number of columns : {demo.shape[1]}")

# Display the first five records
print("\n" + "=" * 60)
print("First Five Records")
print("=" * 60)
display(demo.head())

# Display the last five records
print("\n" + "=" * 60)
print("Last Five Records")
print("=" * 60)
display(demo.tail())

# ============================================================================
# Code Cell 12
# ============================================================================
# =====================================================
# Create Variable Inventory for DEMO Table
# =====================================================

# Create a summary table of variables
demo_variable_inventory = pd.DataFrame({
    "Column No.": range(1, len(demo.columns) + 1),
    "Variable Name": demo.columns,
    "Data Type": demo.dtypes.astype(str)
})

# Display the inventory
display(demo_variable_inventory)

# ============================================================================
# Code Cell 13
# ============================================================================
# =====================================================
# Review Data Types
# =====================================================

# Display the data type of each variable
demo.dtypes.to_frame(name="Data Type")

# ============================================================================
# Code Cell 14
# ============================================================================
# =====================================================
# Missing Value Assessment
# =====================================================

# Calculate missing values
missing_summary = pd.DataFrame({
    "Variable": demo.columns,
    "Missing Count": demo.isnull().sum().values,
})

# Calculate missing percentage
missing_summary["Missing Percentage"] = (
    missing_summary["Missing Count"] / len(demo) * 100
).round(2)

# Sort by highest percentage of missing values
missing_summary = missing_summary.sort_values(
    by="Missing Percentage",
    ascending=False
).reset_index(drop=True)

# Display results
display(missing_summary)

# ============================================================================
# Code Cell 15
# ============================================================================
# =====================================================
# Visualize Missing Values
# =====================================================

# Sort variables by missing percentage
plot_data = missing_summary.sort_values(
    by="Missing Percentage",
    ascending=True
)

# Create figure
plt.figure(figsize=(10, 8))

bars = plt.barh(
    plot_data["Variable"],
    plot_data["Missing Percentage"]
)

# Add percentage labels
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 1,
        bar.get_y() + bar.get_height()/2,
        f"{width:.1f}%",
        va="center",
        fontsize=9
    )

plt.xlabel("Missing Percentage (%)", fontsize=12)
plt.ylabel("Variable", fontsize=12)
plt.title("Percentage of Missing Values in the DEMO Table", fontsize=14)

plt.xlim(0, 100)

plt.tight_layout()

# Save figure
plt.savefig(
    "../figures/demo_missing_values.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================================
# Code Cell 16
# ============================================================================
# =====================================================
# Verify Primary Keys
# =====================================================

print("=" * 60)
print("Primary Key Verification")
print("=" * 60)

# Total observations
print(f"Total rows              : {len(demo):,}")

# PRIMARYID
print(f"Unique PRIMARYID values : {demo['primaryid'].nunique():,}")

# CASEID
print(f"Unique CASEID values    : {demo['caseid'].nunique():,}")

# Duplicate counts
print(f"Duplicate PRIMARYID     : {demo['primaryid'].duplicated().sum():,}")
print(f"Duplicate CASEID        : {demo['caseid'].duplicated().sum():,}")

# ============================================================================
# Code Cell 17
# ============================================================================
# =====================================================
# Load and Inspect DRUG Table
# =====================================================

drug = pd.read_csv(
    os.path.join(DATA_PATH, DRUG_FILE),
    sep="$",
    engine="python",
    encoding="latin1"
)

print("=" * 60)
print("DRUG Table")
print("=" * 60)

print(f"Rows    : {drug.shape[0]:,}")
print(f"Columns : {drug.shape[1]}")

display(drug.head())

# ============================================================================
# Code Cell 18
# ============================================================================
# =====================================================
# Load and Inspect REAC Table
# =====================================================

reac = pd.read_csv(
    os.path.join(DATA_PATH, REAC_FILE),
    sep="$",
    engine="python",
    encoding="latin1"
)

print("=" * 60)
print("REAC Table")
print("=" * 60)

print(f"Rows    : {reac.shape[0]:,}")
print(f"Columns : {reac.shape[1]}")

display(reac.head())

# ============================================================================
# Code Cell 19
# ============================================================================
# =====================================================
# Load and Inspect OUTC Table
# =====================================================

outc = pd.read_csv(
    os.path.join(DATA_PATH, OUTC_FILE),
    sep="$",
    engine="python",
    encoding="latin1"
)

print("=" * 60)
print("OUTC Table")
print("=" * 60)

print(f"Rows    : {outc.shape[0]:,}")
print(f"Columns : {outc.shape[1]}")

display(outc.head())

# ============================================================================
# Code Cell 20
# ============================================================================
# =====================================================
# Load and Inspect RPSR Table
# =====================================================

rpsr = pd.read_csv(
    os.path.join(DATA_PATH, RPSR_FILE),
    sep="$",
    engine="python",
    encoding="latin1"
)

print("=" * 60)
print("RPSR Table")
print("=" * 60)

print(f"Rows    : {rpsr.shape[0]:,}")
print(f"Columns : {rpsr.shape[1]}")

display(rpsr.head())

# ============================================================================
# Code Cell 21
# ============================================================================
# =====================================================
# Load and Inspect THER Table
# =====================================================

ther = pd.read_csv(
    os.path.join(DATA_PATH, THER_FILE),
    sep="$",
    engine="python",
    encoding="latin1"
)

print("=" * 60)
print("THER Table")
print("=" * 60)

print(f"Rows    : {ther.shape[0]:,}")
print(f"Columns : {ther.shape[1]}")

display(ther.head())

# ============================================================================
# Code Cell 22
# ============================================================================
# =====================================================
# Load and Inspect INDI Table
# =====================================================

indi = pd.read_csv(
    os.path.join(DATA_PATH, INDI_FILE),
    sep="$",
    engine="python",
    encoding="latin1"
)

print("=" * 60)
print("INDI Table")
print("=" * 60)

print(f"Rows    : {indi.shape[0]:,}")
print(f"Columns : {indi.shape[1]}")

display(indi.head())

# ============================================================================
# Code Cell 23
# ============================================================================
# Uncomment and run once if graphviz is not installed
!pip install graphviz

# ============================================================================
# Code Cell 24
# ============================================================================
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

# ---------- Master Table ----------
ax.text(
    5, 9,
    "DEMO\n(Master Table)\n406,184 Reports",
    ha="center",
    va="center",
    bbox=dict(boxstyle="round", facecolor="lightblue", edgecolor="black")
)

# ---------- Child Tables ----------
tables = {
    "DRUG\n1,909,327": (1.5, 6.5),
    "REAC\n1,445,416": (3.5, 6.5),
    "OUTC\n295,044": (5.5, 6.5),
    "RPSR\n12,381": (7.5, 6.5),
    "THER\n594,449": (2.8, 3.8),
    "INDI\n1,186,115": (6.8, 3.8),
}

for label, (x, y) in tables.items():
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
    )

# ---------- Connections ----------
connections = [
    (5, 8.5, 1.5, 6.8),
    (5, 8.5, 3.5, 6.8),
    (5, 8.5, 5.5, 6.8),
    (5, 8.5, 7.5, 6.8),
    (5, 8.5, 2.8, 4.1),
    (5, 8.5, 6.8, 4.1),
]

for x1, y1, x2, y2 in connections:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->")
    )

plt.title("FAERS Database Relationship Diagram", fontsize=15)

plt.savefig(
    "../figures/notebook01/faers_database_relationship.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================================
# Code Cell 25
# ============================================================================
# Core data manipulation libraries
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# File system utilities
from pathlib import Path

# Display settings
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 120)

# Plot settings
sns.set_theme(style="whitegrid")

print("Libraries imported successfully.")

# ============================================================================
# Code Cell 26
# ============================================================================
# ---------------------------------------------------
# Define project directories
# ---------------------------------------------------

PROJECT_DIR = Path.cwd().parent

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

FIGURE_DIR = PROJECT_DIR / "figures" / "notebook02"
DOCS_DIR = PROJECT_DIR / "docs"

# Create Notebook 02 figure directory if it does not exist
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("Project directory:", PROJECT_DIR)
print("Processed data directory:", PROCESSED_DIR)
print("Notebook 02 figure directory:", FIGURE_DIR)

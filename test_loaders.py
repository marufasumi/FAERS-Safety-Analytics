from utils.constants import (
    DEMOGRAPHIC_SUMMARY_PATH,
    MODEL_PATH,
    FEATURE_COLUMNS_PATH,
)

from utils.loaders import (
    load_csv,
    load_model,
    load_feature_columns,
)

print("=" * 60)
print("LOADERS TEST")
print("=" * 60)

# CSV
df = load_csv(DEMOGRAPHIC_SUMMARY_PATH)
print(f"CSV loaded          : {df is not None}")
if df is not None:
    print(f"Rows x Columns      : {df.shape}")

# Feature schema
features = load_feature_columns(FEATURE_COLUMNS_PATH)
print(f"Number of features  : {len(features)}")

# Model
model = load_model(MODEL_PATH)
print(f"Model type          : {type(model)}")

print("\nAll loader tests passed.")
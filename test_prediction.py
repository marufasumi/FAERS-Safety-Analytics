"""
Structural test for the FAERS prediction utility.

The row below uses synthetic placeholder values only to verify that:
- all 49 expected features are accepted;
- feature ordering is preserved;
- the saved model can produce predictions and probabilities.
"""

import pandas as pd

from utils.constants import FEATURE_COLUMNS_PATH, MODEL_PATH
from utils.loaders import load_feature_columns, load_model
from utils.prediction import get_model_classes, predict_serious_report


feature_columns = load_feature_columns(FEATURE_COLUMNS_PATH)
model = load_model(MODEL_PATH)

# Start every expected feature at zero.
test_values = {
    feature: 0
    for feature in feature_columns
}

# Provide valid categorical values and a few basic numeric values.
test_values.update(
    {
        "age_years": 55.0,
        "weight_kg": 75.0,
        "sex": "F",
        "reporter_country": "US",
        "occur_country": "US",
        "reporter_type": "MD",
        "report_type": "EXP",
        "total_drugs": 2,
        "num_unique_drugs": 2,
        "num_primary_suspect": 1,
        "total_reactions": 1,
        "num_unique_reactions": 1,
        "total_indications": 1,
        "num_unique_indications": 1,
        "num_reporter_sources": 1,
    }
)

input_df = pd.DataFrame([test_values])

print("=" * 60)
print("PREDICTION UTILITY TEST")
print("=" * 60)

print(f"Input rows           : {input_df.shape[0]}")
print(f"Input columns        : {input_df.shape[1]}")
print(f"Expected features    : {len(feature_columns)}")
print(f"Model classes        : {get_model_classes(model)}")

result = predict_serious_report(
    model=model,
    input_df=input_df,
    feature_columns=feature_columns,
)

print()
print(f"Predicted class      : {result['predicted_class']}")
print(f"Predicted label      : {result['predicted_label']}")
print(
    "Serious probability : "
    f"{result['serious_probability']:.4f}"
)
print(
    "Non-serious prob.    : "
    f"{result['non_serious_probability']:.4f}"
)
print(f"Model confidence     : {result['confidence']:.4f}")

probability_total = (
    result["serious_probability"]
    + result["non_serious_probability"]
)

print(f"Probability total    : {probability_total:.4f}")

assert input_df.shape[1] == len(feature_columns)
assert abs(probability_total - 1.0) < 1e-9

print("\nPrediction utility test passed.")
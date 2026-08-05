"""
Prediction utilities for the FAERS serious-report classification model.

This module validates the submitted report schema, preserves the expected
feature order, and runs prediction with the saved scikit-learn pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def validate_prediction_input(
    input_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Validate and reorder a single-row prediction DataFrame.

    Parameters
    ----------
    input_df:
        DataFrame containing one FAERS-style report.
    feature_columns:
        Exact model feature names in their expected order.

    Returns
    -------
    pandas.DataFrame
        A validated one-row DataFrame with columns in the expected order.

    Raises
    ------
    TypeError
        If input_df is not a pandas DataFrame.
    ValueError
        If the DataFrame does not contain exactly one row or if its
        feature schema does not match the saved model schema.
    """
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("Prediction input must be a pandas DataFrame.")

    if len(input_df) != 1:
        raise ValueError(
            "Prediction input must contain exactly one report row."
        )

    if not feature_columns:
        raise ValueError("The model feature schema is empty.")

    missing_columns = [
        column for column in feature_columns
        if column not in input_df.columns
    ]

    unexpected_columns = [
        column for column in input_df.columns
        if column not in feature_columns
    ]

    if missing_columns:
        raise ValueError(
            "Prediction input is missing required features: "
            + ", ".join(missing_columns)
        )

    if unexpected_columns:
        raise ValueError(
            "Prediction input contains unexpected features: "
            + ", ".join(unexpected_columns)
        )

    # Preserve the exact order used by the saved model pipeline.
    return input_df.loc[:, feature_columns].copy()


def get_model_classes(model: Any) -> np.ndarray:
    """
    Return the fitted classifier's class labels.

    The saved object may be either:
    - a fitted sklearn Pipeline containing a final model step; or
    - a fitted classifier with a classes_ attribute.
    """
    if hasattr(model, "classes_"):
        classes = model.classes_

    elif hasattr(model, "named_steps"):
        final_estimator = list(model.named_steps.values())[-1]

        if not hasattr(final_estimator, "classes_"):
            raise AttributeError(
                "The final pipeline estimator does not expose classes_."
            )

        classes = final_estimator.classes_

    else:
        raise AttributeError(
            "Unable to identify model class labels."
        )

    classes_array = np.asarray(classes)

    if classes_array.size != 2:
        raise ValueError(
            "The deployed prediction utility expects a binary classifier."
        )

    return classes_array


def predict_serious_report(
    model: Any,
    input_df: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, Any]:
    """
    Classify one FAERS-style report and return class probabilities.

    Serious is identified by class label 1.
    Non-serious is identified by class label 0.

    Returns
    -------
    dict
        Dictionary containing:
        - predicted_class
        - predicted_label
        - serious_probability
        - non_serious_probability
        - confidence
    """
    validated_df = validate_prediction_input(
        input_df=input_df,
        feature_columns=feature_columns,
    )

    classes = get_model_classes(model)

    if 0 not in classes or 1 not in classes:
        raise ValueError(
            "Expected class labels 0 and 1 were not found in model.classes_."
        )

    predicted_class = int(model.predict(validated_df)[0])
    probability_array = model.predict_proba(validated_df)[0]

    class_probability_map = {
        int(class_label): float(probability)
        for class_label, probability in zip(
            classes,
            probability_array,
            strict=True,
        )
    }

    serious_probability = class_probability_map[1]
    non_serious_probability = class_probability_map[0]

    predicted_label = (
        "Serious report classification"
        if predicted_class == 1
        else "Non-serious report classification"
    )

    confidence = max(
        serious_probability,
        non_serious_probability,
    )

    return {
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "serious_probability": serious_probability,
        "non_serious_probability": non_serious_probability,
        "confidence": confidence,
    }
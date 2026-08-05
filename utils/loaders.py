"""
Reusable loading utilities for the FAERS Streamlit application.

This module centralizes loading of:
- CSV files
- Images
- Trained model
- Feature schema

All functions handle missing files gracefully.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from PIL import Image


# ==========================================================
# Generic CSV Loader
# ==========================================================

@st.cache_data
def load_csv(file_path: Path) -> pd.DataFrame | None:
    """
    Load a CSV file.

    Returns None if the file does not exist.
    """
    if not file_path.exists():
        st.warning(f"Missing CSV file:\n{file_path.name}")
        return None

    return pd.read_csv(file_path)


# ==========================================================
# Image Loader
# ==========================================================

@st.cache_data
def load_image(image_path: Path):
    """
    Load an image.

    Returns None if unavailable.
    """
    if not image_path.exists():
        st.warning(f"Missing image:\n{image_path.name}")
        return None

    return Image.open(image_path)


# ==========================================================
# Model Loader
# ==========================================================

@st.cache_resource
def load_model(model_path: Path):
    """
    Load the trained Random Forest pipeline.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found:\n{model_path}")

    return joblib.load(model_path)


# ==========================================================
# Feature Schema Loader
# ==========================================================

@st.cache_data
def load_feature_columns(json_path: Path) -> list[str]:
    """
    Load the expected feature order from JSON.
    """
    if not json_path.exists():
        raise FileNotFoundError(
            f"Feature schema not found:\n{json_path}"
        )

    with open(json_path, "r") as f:
        return json.load(f)
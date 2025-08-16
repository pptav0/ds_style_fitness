import pandas as pd
import numpy as np

# Helper function to categorize columns
def get_column_groups(df: pd.DataFrame, low_card_threshold=20, target_col=None):
    """
    Automatically detect column groups for preprocessing:
    - binary_cols: columns with <= binary_threshold unique values (excluding target)
    - numeric_cols: numeric columns (excluding target)
    - low_card_cols: categorical with <= low_card_threshold unique values
    - high_card_cols: categorical with > low_card_threshold unique values
    """

    # Parameters
    binary_threshold=2

    # Drop target col for processing
    features = df.drop(columns=[target_col]) if target_col and target_col in df.columns else df.copy()

    # Numeric & Categorical columns
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    category_cols = features.select_dtypes(include=['object']).columns.tolist()

    # Binary columns (numeric or object/bool with small unique values)
    binary_cols = [
        c for c in features.columns
        if features[c].nunique(dropna=False) <= binary_threshold and \
        c != target_col and c not in category_cols
    ]

    # Categorical candidates = non-numeric
    categorical_cols = features.select_dtypes(exclude=[np.number]).columns.tolist()

    # Low-card categorical
    low_card_cols = [
        c for c in categorical_cols
        if features[c].nunique(dropna=False) <= low_card_threshold and c not in binary_cols
    ]

    # High-card categorical
    high_card_cols = [
        c for c in categorical_cols
        if c not in binary_cols + low_card_cols
    ]

    return {
        "binary_cols": binary_cols,
        "numeric_cols": numeric_cols,
        "low_card_cols": low_card_cols,
        "high_card_cols": high_card_cols
    }

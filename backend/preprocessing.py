import pandas as pd
import numpy as np
import joblib
import os

def load_dataset(file_path):
    """Safely loads the Excel dataset from the specified path.""" 
    try:
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return None
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

def validate_required_columns(df, required_cols):
    """Checks if a list of required columns exists in the DataFrame."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Warning: Missing required columns: {missing}")
        return False
    return True

def get_numeric_columns(df):
    """Filters and returns numeric columns from the DataFrame."""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def safe_scaler_load(scaler_path):
    """Safely attempts to load a scaler object if it exists."""
    if os.path.exists(scaler_path):
        try:
            return joblib.load(scaler_path)
        except Exception as e:
            print(f"Error loading scaler: {e}")
    return None

def align_features(df, feature_order):
    """Aligns DataFrame columns to match a specific feature order, filling missing with 0."""
    # Create a copy to avoid SettingWithCopy warnings
    aligned_df = df.copy()
    
    # Add missing columns
    for col in feature_order:
        if col not in aligned_df.columns:
            aligned_df[col] = 0
            
    # Select and order columns, ignoring extras
    return aligned_df[feature_order]
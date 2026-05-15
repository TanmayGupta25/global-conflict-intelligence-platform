import joblib
import os
import pandas as pd
import numpy as np

def load_model(model_path):
    """Safely loads the trained ML model artifact."""
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def load_feature_order(features_path):
    """Safely loads the list of features used during training."""
    if not os.path.exists(features_path):
        print(f"Error: Feature list not found at {features_path}")
        return None
    try:
        return joblib.load(features_path)
    except Exception as e:
        print(f"Failed to load feature list: {e}")
        return None

def prepare_prediction_input(df, feature_order, scaler=None):
    """Aligns features and applies scaling if a scaler is provided."""
    # Create copy to avoid side-effects
    X = df.copy()
    
    # Add missing columns with 0
    for col in feature_order:
        if col not in X.columns:
            X[col] = 0
            
    # Select and order columns exactly as trained
    X = X[feature_order]
    
    # Apply scaling if available
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
        
    return X

def predict_conflict(model, X):
    """Generates prediction probability and class."""
    # Handle case for classifiers that support predict_proba
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X) # Fallback
    
    preds = model.predict(X)
    return probs, preds

def classify_risk(probability, low_thresh=0.35, high_thresh=0.65):
    """Converts probability into a qualitative risk category."""
    if probability < low_thresh:
        return "Low Risk"
    elif probability <= high_thresh:
        return "Moderate Risk"
    else:
        return "High Risk"
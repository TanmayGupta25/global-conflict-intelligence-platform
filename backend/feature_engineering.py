import pandas as pd
import numpy as np

def get_country_history(df, country, group_col="Country", time_col="Year"):
    """Safely retrieves and sorts historical records for a specific country."""
    country_df = df[df[group_col] == country].copy()
    return country_df.sort_values(by=time_col)

def create_lag_features(df, lag_columns, lag=1):
    """Generates lag variables while preserving chronological order."""
    df_lag = df.copy()
    for col in lag_columns:
        if col in df_lag.columns:
            df_lag[f"{col}_lag{lag}"] = df_lag[col].shift(lag)
    return df_lag

def create_rolling_features(df, rolling_columns, window=3):
    """Generates rolling mean features for specified columns."""
    df_roll = df.copy()
    for col in rolling_columns:
        if col in df_roll.columns:
            df_roll[f"{col}_rolling_mean{window}"] = df_roll[col].rolling(window=window, min_periods=1).mean()
    return df_roll

def create_change_features(df, change_columns):
    """Generates year-over-year change (difference) features."""
    df_change = df.copy()
    for col in change_columns:
        if col in df_change.columns:
            df_change[f"{col}_change"] = df_change[col].diff()
    return df_change

def generate_features(country_df, lag_cols, roll_cols, current_inputs=None, mode="historical"):
    """
    CENTRAL MASTER FUNCTION: Combines temporal logic for deployment.
    
    mode="historical": Processes existing sequence.
    mode="forecast": Appends current_inputs to history then calculates features for the last row.
    """
    work_df = country_df.copy()
    
    if mode == "forecast" and current_inputs is not None:
        # Convert inputs to DataFrame row and append
        input_row = pd.DataFrame([current_inputs])
        work_df = pd.concat([work_df, input_row], ignore_index=True)
    
    # Apply temporal transformations
    work_df = create_lag_features(work_df, lag_cols, lag=1)
    work_df = create_rolling_features(work_df, roll_cols, window=3)
    work_df = create_change_features(work_df, roll_cols)
    
    # Handle NaNs (common in first years of history)
    work_df.fillna(0, inplace=True)
    
    return work_df
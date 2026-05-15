import pandas as pd
import numpy as np
import feature_engineering

def get_latest_country_state(country_df):
    """Retrieves the most recent verified record for a country."""
    if country_df.empty:
        return None
    return country_df.sort_values(by='Year').tail(1).copy()

def apply_scenario_adjustments(latest_state, scenario_inputs):
    """
    Safely applies user-defined hypothetical changes to the latest state.
    scenario_inputs: dict of {feature_name: value}
    """
    simulated_state = latest_state.copy()
    for feature, value in scenario_inputs.items():
        if feature in simulated_state.columns:
            simulated_state[feature] = value
    return simulated_state

def generate_future_state(country_df, scenario_inputs, forecast_year):
    """Creates a simulated future row appended to the country history."""
    latest = get_latest_country_state(country_df)
    if latest is None:
        return country_df
    
    future_row = apply_scenario_adjustments(latest, scenario_inputs)
    future_row['Year'] = forecast_year
    
    return pd.concat([country_df, future_row], ignore_index=True)

def simulate_future_risk(country_df, scenario_inputs, forecast_year, lag_cols, roll_cols):
    """
    CENTRAL FORECAST FUNCTION: Merges history with simulation and engineers features.
    """
    # 1. Create the chronological sequence including the simulation
    sim_sequence = generate_future_state(country_df, scenario_inputs, forecast_year)
    
    # 2. Use feature_engineering module to calculate temporal features
    # This ensures the 2024 forecast uses the real 2023 values for lags
    forecast_df = feature_engineering.generate_features(
        sim_sequence,
        lag_cols=lag_cols,
        roll_cols=roll_cols,
        mode='historical' # Using historical mode because we manually appended the row
    )
    
    return forecast_df
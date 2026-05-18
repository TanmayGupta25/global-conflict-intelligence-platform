import shap
import pandas as pd
import numpy as np

def create_shap_explainer(model):
    """Initializes SHAP explainer safely."""

    try:
        print("Initializing SHAP TreeExplainer...")

        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent"
        )

        print("SHAP explainer initialized successfully.")

        return explainer

    except Exception as e:

        print(f"Error initializing SHAP explainer: {e}")

        return None

def generate_shap_values(explainer, X):
    """Generates SHAP values for the input features."""
    try:
        # check_additivity=False for XGBoost compatibility in some environments
        return explainer.shap_values(X, check_additivity=False)
    except Exception as e:
        print(f"Error generating SHAP values: {e}")
        return None

def get_top_feature_impacts(shap_values, feature_names, top_n=10):
    """Ranks features by absolute SHAP impact for a single prediction."""
    # Handle case where shap_values is a list (multiclass) or array (binary)
    if isinstance(shap_values, list):
        impacts = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        impacts = shap_values

    # If multiple rows, take the first (designed for deployment single-inference)
    if len(impacts.shape) > 1:
        impacts = impacts[0]

    # Create dataframe for sorting
    impact_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': impacts,
        'abs_impact': np.abs(impacts)
    })

    return impact_df.sort_values(by='abs_impact', ascending=False).head(top_n)

def generate_explanation_summary(top_features):
    """Generates a geopolitical intelligence style summary."""
    summary_lines = []
    
    for _, row in top_features.iterrows():
        direction = "increased" if row['shap_value'] > 0 else "reduced"
        impact_type = "escalation probability" if row['shap_value'] > 0 else "instability risk"
        
        # Standardizing feature names for cleaner summary
        clean_name = row['feature'].replace('_', ' ')
        
        line = f"- '{clean_name}' {direction} predicted {impact_type}."
        summary_lines.append(line)

    return "\n".join(summary_lines)

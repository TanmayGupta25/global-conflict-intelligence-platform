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

        # SAFETY CHECK
        if explainer is None:
            return None

        # check_additivity=False for XGBoost compatibility
        return explainer.shap_values(
            X,
            check_additivity=False
        )

    except Exception as e:

        print(f"Error generating SHAP values: {e}")

        return None


def get_top_feature_impacts(
    shap_values,
    feature_names,
    top_n=10
):
    """Ranks features by absolute SHAP impact."""

    try:

        # SAFETY CHECK
        if shap_values is None:
            return []

        # Handle case where shap_values is a list
        if isinstance(shap_values, list):

            impacts = (
                shap_values[1]
                if len(shap_values) > 1
                else shap_values[0]
            )

        else:

            impacts = shap_values

        # If multiple rows, take first row
        if len(impacts.shape) > 1:
            impacts = impacts[0]

        # Create dataframe
        impact_df = pd.DataFrame({

            'feature': feature_names,

            'shap_value': impacts,

            'abs_impact': np.abs(impacts)
        })

        return impact_df.sort_values(
            by='abs_impact',
            ascending=False
        ).head(top_n)

    except Exception as e:

        print(f"Error extracting top SHAP features: {e}")

        return []


def generate_explanation_summary(top_features):
    """Generates geopolitical intelligence style summary."""

    # SAFETY CHECKS
    if top_features is None:
        return "No SHAP feature explanations available."

    if isinstance(top_features, list):
        return "No SHAP feature explanations available."

    summary_lines = []

    try:

        for _, row in top_features.iterrows():

            direction = (
                "increased"
                if row['shap_value'] > 0
                else "reduced"
            )

            impact_type = (
                "escalation probability"
                if row['shap_value'] > 0
                else "instability risk"
            )

            # Cleaner feature names
            clean_name = row['feature'].replace('_', ' ')

            line = (
                f"- '{clean_name}' "
                f"{direction} predicted {impact_type}."
            )

            summary_lines.append(line)

        return "\n".join(summary_lines)

    except Exception as e:

        print(f"Error generating explanation summary: {e}")

        return "SHAP explanation generation unavailable."

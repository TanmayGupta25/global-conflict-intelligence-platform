import shap
import pandas as pd
import numpy as np


def create_shap_explainer(model):
    """Initializes SHAP explainer safely."""

    try:

        # ============================================================
        # DEPLOYMENT DEBUG MARKER
        # ============================================================

        print("================================================")
        print("=== NEW CLEAN MODEL LOADED ===")
        print("Initializing SHAP TreeExplainer...")
        print("================================================")

        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent"
        )

        print("SHAP explainer initialized successfully.")

        return explainer

    except Exception as e:

        print("================================================")
        print("SHAP INITIALIZATION FAILED")
        print(f"Error initializing SHAP explainer: {e}")
        print("================================================")

        return None


def generate_shap_values(explainer, X):
    """Generates SHAP values for the input features."""

    try:

        # ============================================================
        # SAFETY CHECK
        # ============================================================

        if explainer is None:

            print("SHAP explainer is None.")
            return None

        # ============================================================
        # FORCE NUMERIC SAFETY
        # ============================================================

        X = X.copy()

        X = X.apply(pd.to_numeric, errors='coerce')

        X = X.fillna(0)

        X = X.astype(np.float64)

        print("SHAP INPUT TYPES VERIFIED:")
        print(X.dtypes)

        # ============================================================
        # GENERATE SHAP VALUES
        # ============================================================

        shap_values = explainer.shap_values(
            X,
            check_additivity=False
        )

        print("SHAP values generated successfully.")

        return shap_values

    except Exception as e:

        print("================================================")
        print("SHAP VALUE GENERATION FAILED")
        print(f"Error generating SHAP values: {e}")
        print("================================================")

        return None


def get_top_feature_impacts(
    shap_values,
    feature_names,
    top_n=10
):
    """Ranks features by absolute SHAP impact."""

    try:

        # ============================================================
        # SAFETY CHECK
        # ============================================================

        if shap_values is None:

            print("No SHAP values available.")
            return []

        # ============================================================
        # HANDLE MULTICLASS / BINARY OUTPUT
        # ============================================================

        if isinstance(shap_values, list):

            impacts = (
                shap_values[1]
                if len(shap_values) > 1
                else shap_values[0]
            )

        else:

            impacts = shap_values

        # ============================================================
        # HANDLE MULTI-ROW INPUT
        # ============================================================

        if len(impacts.shape) > 1:

            impacts = impacts[0]

        # ============================================================
        # CREATE IMPACT DATAFRAME
        # ============================================================

        impact_df = pd.DataFrame({

            'feature': feature_names,

            'shap_value': impacts,

            'abs_impact': np.abs(impacts)
        })

        impact_df = impact_df.sort_values(
            by='abs_impact',
            ascending=False
        ).head(top_n)

        print("Top SHAP features extracted successfully.")

        return impact_df

    except Exception as e:

        print("================================================")
        print("TOP FEATURE EXTRACTION FAILED")
        print(f"Error extracting top SHAP features: {e}")
        print("================================================")

        return []


def generate_explanation_summary(top_features):
    """Generates geopolitical intelligence style summary."""

    # ============================================================
    # SAFETY CHECKS
    # ============================================================

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

        print("SHAP explanation summary generated successfully.")

        return "\n".join(summary_lines)

    except Exception as e:

        print("================================================")
        print("SUMMARY GENERATION FAILED")
        print(f"Error generating explanation summary: {e}")
        print("================================================")

        return "SHAP explanation generation unavailable."

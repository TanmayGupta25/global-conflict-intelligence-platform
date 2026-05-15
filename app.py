import os
import sys
import logging
import pandas as pd
from flask import Flask, jsonify, render_template, request

# ============================================================
# 1. Path Setup
# ============================================================

project_root = os.getcwd()

if project_root not in sys.path:
    sys.path.append(project_root)

# ============================================================
# 2. Import Config
# ============================================================

import config

if config.BACKEND_DIR not in sys.path:
    sys.path.append(config.BACKEND_DIR)

import environment_validator

# ============================================================
# 3. Logging Setup
# ============================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("ConflictApp")

# ============================================================
# 4. Flask App Initialization
# ============================================================

app = Flask(
    __name__,
    template_folder=config.TEMPLATES_DIR,
    static_folder=config.STATIC_DIR
)

app.config['SECRET_KEY'] = config.SECRET_KEY

# ============================================================
# 5. Startup Diagnostics
# ============================================================

logger.info("=== DEPLOYMENT STARTUP INITIATED ===")

health_check = environment_validator.validate_system_readiness(config)

# ============================================================
# 6. Global Resource Initialization
# ============================================================

try:

    if not health_check['ready']:

        logger.error(f"Pre-flight failure: {health_check}")

        system_operational = False

        countries = []

    else:

        import prediction_engine
        import simulation_engine
        import shap_engine
        import report_generator
        import visualization_engine

        model = prediction_engine.load_model(config.MODEL_PATH)

        feature_order = prediction_engine.load_feature_order(
            config.FEATURES_PATH
        )

        explainer = shap_engine.create_shap_explainer(model)

        df_base = pd.read_excel(config.DATASET_PATH)

        countries = sorted(df_base['Country'].unique().tolist())

        system_operational = True

        logger.info(
            "SYSTEM STATUS: OPERATIONAL - Production configuration loaded."
        )

except Exception as e:

    logger.critical(
        f"DEPLOYMENT STARTUP FAILURE: {e}",
        exc_info=True
    )

    system_operational = False

    countries = []

# ============================================================
# 7. Home Route
# ============================================================

@app.route("/")
def home():

    return render_template(
        "home.html",
        error=None if system_operational else "Deployment Environment Degraded."
    )

# ============================================================
# 8. Global Map Route
# ============================================================

@app.route("/map")
def global_map():

    if not system_operational:
        return jsonify({"error": "Service Unavailable"}), 503

    try:

        map_json = visualization_engine.generate_global_map_json(
            df_base,
            feature_order,
            model,
            prediction_engine
        )

        return render_template(
            "global_map.html",
            map_json=map_json
        )

    except Exception as e:

        logger.error(f"Map route error: {e}")

        return "Internal Server Error", 500

# ============================================================
# 9. Historical Route
# ============================================================

@app.route("/historical", methods=['GET', 'POST'])
def historical():

    if not system_operational:

        return render_template(
            "historical_validation.html",
            error="Environment Error"
        )

    results = None

    if request.method == 'POST':

        try:

            country = request.form.get('country')

            year = int(request.form.get('year', 2022))

            hist_data = df_base[
                (df_base['Country'] == country) &
                (df_base['Year'] <= year)
            ].sort_values('Year')

            if not hist_data.empty:

                X_input = prediction_engine.prepare_prediction_input(
                    hist_data.tail(1),
                    feature_order
                )

                prob, _ = prediction_engine.predict_conflict(
                    model,
                    X_input
                )

                risk = prediction_engine.classify_risk(
                    prob[0],
                    config.LOW_RISK_THRESHOLD,
                    config.HIGH_RISK_THRESHOLD
                )

                results = {
                    "country": country,
                    "gauge": visualization_engine.create_risk_gauge(
                        prob[0],
                        risk
                    )
                }

        except Exception as e:

            logger.error(f"Historical Analysis Error: {e}")

    return render_template(
        "historical_validation.html",
        countries=countries,
        results=results
    )

# ============================================================
# 10. Forecast Route
# ============================================================

@app.route("/forecast", methods=['GET', 'POST'])
def forecast():

    if not system_operational:

        return render_template(
            "future_forecasting.html",
            error="Environment Error"
        )

    results = None

    if request.method == 'POST':

        try:

            country = request.form.get('country')

            scenario = {

                'GDP': float(request.form.get('gdp', 100)),

                'Stability': float(
                    request.form.get('stability', 0)
                ),

                'MilExp': float(
                    request.form.get('milexp', 1000)
                )
            }

            country_history = df_base[
                df_base['Country'] == country
            ]

            sim_df = simulation_engine.simulate_future_risk(
                country_history,
                scenario,
                2024,
                ['GDP', 'Stability', 'MilExp'],
                ['GDP', 'Stability', 'MilExp']
            )

            X_input = prediction_engine.prepare_prediction_input(
                sim_df.tail(1),
                feature_order
            )

            prob, _ = prediction_engine.predict_conflict(
                model,
                X_input
            )

            risk = prediction_engine.classify_risk(
                prob[0],
                config.LOW_RISK_THRESHOLD,
                config.HIGH_RISK_THRESHOLD
            )

            sv = shap_engine.generate_shap_values(
                explainer,
                X_input
            )

            top_f = shap_engine.get_top_feature_impacts(
                sv,
                feature_order,
                top_n=config.TOP_N_FEATURES
            )

            report = report_generator.generate_full_report(
                country,
                2024,
                float(prob[0]),
                risk,
                top_f
            )

            results = {

                "report": report['full_text'],

                "gauge": visualization_engine.create_risk_gauge(
                    prob[0],
                    risk
                ),

                "shap_chart": visualization_engine.create_shap_chart(
                    top_f
                )
            }

        except Exception as e:

            logger.error(f"Forecasting Simulation Error: {e}")

    return render_template(
        "future_forecasting.html",
        countries=countries,
        results=results
    )

# ============================================================
# 11. Health Route
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status":
            "operational"
            if system_operational
            else "degraded",

        "diagnostics": health_check,

        "hardened": True
    })

# ============================================================
# 12. Main Runner
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=config.PORT,
        debug=config.DEBUG_MODE,
        use_reloader=False
    )

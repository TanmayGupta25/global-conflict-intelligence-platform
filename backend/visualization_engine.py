import json
import plotly
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def generate_global_map_json(df_base, feature_order, model, prediction_engine):
    """
    Aggregates latest predictions for all countries
    and builds a Plotly Choropleth.
    """

    try:

        latest_records = (
            df_base
            .sort_values('Year')
            .groupby('Country')
            .tail(1)
            .copy()
        )

        # =====================================================
        # GENERATE PREDICTIONS
        # =====================================================

        X_map = prediction_engine.prepare_prediction_input(
            latest_records,
            feature_order
        )

        probs, _ = prediction_engine.predict_conflict(
            model,
            X_map
        )

        # =====================================================
        # SAFETY FIXES
        # =====================================================

        cleaned_probs = []

        for p in probs:

            try:

                value = float(p)

                if pd.isna(value):
                    value = 0.0

                if value < 0:
                    value = 0.0

                if value > 1:
                    value = 1.0

            except:
                value = 0.0

            cleaned_probs.append(value)

        risk_scores = []

        for p in cleaned_probs:

            try:
                value = float(p) * 100

                if pd.isna(value):
                     value = 0.0

            except:
                value = 0.0

            risk_scores.append(round(value, 2))

        latest_records['Risk_Score'] = risk_scores

        print(latest_records[['Country', 'Risk_Score']].head())

        latest_records['Risk_Level'] = [
            prediction_engine.classify_risk(p)
            for p in cleaned_probs
        ]

        # =====================================================
        # REMOVE BAD COUNTRY VALUES
        # =====================================================

        latest_records = latest_records.dropna(
            subset=['Country', 'Risk_Score']
        )

        # =====================================================
        # BUILD MAP
        # =====================================================

        print(latest_records[["Country", "Risk_Score"]].head(50))
        fig = px.choropleth(
            latest_records,
            locations="Country",
            locationmode="country names",
            color="Risk_Score",
            hover_name="Country",
            hover_data={
                "Risk_Score": True,
                "Risk_Level": True,
                "Year": True
            },
            color_continuous_scale=[
                "#064e3b",
                "#78350f",
                "#ef4444"
            ],
            range_color=(0, 1),
            labels={
                'Risk_Score': 'Conflict Probability %'
            }
        )

        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular',
                bgcolor='#0f172a',
                lakecolor='#1e293b',
                landcolor='#1e293b',
                subunitcolor='#334155'
            ),
            paper_bgcolor='#0f172a',
            plot_bgcolor='#0f172a',
            font={'color': "#f1f5f9"},
            margin=dict(l=0, r=0, t=0, b=0)
        )

        return json.dumps(
            fig,
            cls=plotly.utils.PlotlyJSONEncoder
        )

    except Exception as e:

        print(f"MAP GENERATION ERROR: {e}")

        fallback_fig = go.Figure()

        fallback_fig.update_layout(
            paper_bgcolor='#0f172a',
            plot_bgcolor='#0f172a',
            font={'color': "#f1f5f9"},
            annotations=[
                dict(
                    text="Global map temporarily unavailable.",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=18)
                )
            ],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )

        return json.dumps(
            fallback_fig,
            cls=plotly.utils.PlotlyJSONEncoder
        )


def create_risk_gauge(prob, risk_level):
    """Generates a Plotly gauge chart for conflict probability."""

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={
            'text': f"Risk Level: {risk_level}",
            'font': {'size': 18}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': "#94a3b8"
            },
            'bar': {
                'color': "#ef4444" if prob > 0.5 else "#38bdf8"
            },
            'bgcolor': "#1e293b",
            'steps': [
                {'range': [0, 35], 'color': '#064e3b'},
                {'range': [35, 65], 'color': '#78350f'},
                {'range': [65, 100], 'color': '#7f1d1d'}
            ],
        }
    ))

    fig.update_layout(
        paper_bgcolor='#1e293b',
        font={'color': "#f1f5f9"},
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return json.dumps(
        fig,
        cls=plotly.utils.PlotlyJSONEncoder
    )


def create_shap_chart(top_f):
    """
    Generates a horizontal bar chart for feature impacts.
    Supports:
    - SHAP dataframe
    - fallback list
    - empty/None values
    """

    # =========================================================
    # 1. EMPTY SAFETY
    # =========================================================

    if top_f is None:

        fig = go.Figure()

        fig.update_layout(
            paper_bgcolor='#1e293b',
            plot_bgcolor='#1e293b',
            font={'color': "#f1f5f9"},
            annotations=[
                dict(
                    text="No instability driver data available.",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=16, color="#f1f5f9")
                )
            ],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=300
        )

        return json.dumps(
            fig,
            cls=plotly.utils.PlotlyJSONEncoder
        )

    # =========================================================
    # 2. FALLBACK LIST SUPPORT
    # =========================================================

    if isinstance(top_f, list):

        if len(top_f) == 0:

            fig = go.Figure()

            fig.update_layout(
                paper_bgcolor='#1e293b',
                plot_bgcolor='#1e293b',
                font={'color': "#f1f5f9"},
                annotations=[
                    dict(
                        text="No instability driver data available.",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=16, color="#f1f5f9")
                    )
                ],
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=300
            )

            return json.dumps(
                fig,
                cls=plotly.utils.PlotlyJSONEncoder
            )

        features = [x['feature'] for x in top_f]
        values = [x['impact'] for x in top_f]

    # =========================================================
    # 3. DATAFRAME SUPPORT
    # =========================================================

    else:

        if top_f.empty:

            fig = go.Figure()

            fig.update_layout(
                paper_bgcolor='#1e293b',
                plot_bgcolor='#1e293b',
                font={'color': "#f1f5f9"},
                annotations=[
                    dict(
                        text="No instability driver data available.",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=16, color="#f1f5f9")
                    )
                ],
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=300
            )

            return json.dumps(
                fig,
                cls=plotly.utils.PlotlyJSONEncoder
            )

        features = top_f['feature'].tolist()
        values = top_f['shap_value'].tolist()

    # =========================================================
    # 4. BUILD CHART
    # =========================================================

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker=dict(
            color=[
                '#ef4444' if x > 0 else '#38bdf8'
                for x in values
            ]
        )
    ))

    fig.update_layout(
        paper_bgcolor='#1e293b',
        plot_bgcolor='#1e293b',
        font={'color': "#f1f5f9"},
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(autorange="reversed")
    )

    return json.dumps(
        fig,
        cls=plotly.utils.PlotlyJSONEncoder
    )

import json
import plotly
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def generate_global_map_json(df_base, feature_order, model, prediction_engine):
    """Aggregates latest predictions for all countries and builds a Plotly Choropleth."""
    latest_records = df_base.sort_values('Year').groupby('Country').tail(1).copy()
    
    # Generate predictions for the map
    X_map = prediction_engine.prepare_prediction_input(latest_records, feature_order)
    probs, _ = prediction_engine.predict_conflict(model, X_map)
    latest_records['Risk_Score'] = [round(p * 100, 2) for p in probs]
    latest_records['Risk_Level'] = [prediction_engine.classify_risk(p) for p in probs]

    fig = px.choropleth(
        latest_records,
        locations="Country",
        locationmode="country names",
        color="Risk_Score",
        hover_name="Country",
        hover_data={"Risk_Score": True, "Risk_Level": True, "Year": True},
        color_continuous_scale=["#064e3b", "#78350f", "#ef4444"],
        labels={'Risk_Score': 'Conflict Probability %'}
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
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_risk_gauge(prob, risk_level):
    """Generates a Plotly gauge chart for conflict probability."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        title = {'text': f"Risk Level: {risk_level}", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
            'bar': {'color': "#ef4444" if prob > 0.5 else "#38bdf8"},
            'bgcolor': "#1e293b",
            'steps': [
                {'range': [0, 35], 'color': '#064e3b'},
                {'range': [35, 65], 'color': '#78350f'},
                {'range': [65, 100], 'color': '#7f1d1d'}
            ],
        }
    ))
    fig.update_layout(paper_bgcolor='#1e293b', font={'color': "#f1f5f9"}, height=300, margin=dict(l=20, r=20, t=50, b=20))
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_shap_chart(top_f):
    """Generates a horizontal bar chart for feature impacts."""
    fig = go.Figure(go.Bar(
        x=top_f['shap_value'],
        y=top_f['feature'],
        orientation='h',
        marker=dict(color=['#ef4444' if x > 0 else '#38bdf8' for x in top_f['shap_value']])
    ))
    fig.update_layout(
        paper_bgcolor='#1e293b', plot_bgcolor='#1e293b',
        font={'color': "#f1f5f9"}, height=300, margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(autorange="reversed")
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
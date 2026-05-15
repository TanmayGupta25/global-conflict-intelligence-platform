import datetime

def generate_risk_summary(country, year, probability, risk_level):
    """Generates a concise executive-level risk summary."""
    summary = "EXECUTIVE SUMMARY: " + str(country).upper() + " (" + str(year) + ")\n"
    summary += "Assessed Conflict Probability: " + "{:.2%}".format(probability) + "\n"
    summary += "Risk Classification: " + str(risk_level).upper() + "\n"
    summary += "Strategic Outlook: This assessment identifies primary drivers of instability using machine learning inference."
    return summary

def filter_features(top_features, keywords):
    """Helper to filter SHAP impacts by category keywords."""
    return top_features[top_features['feature'].str.contains('|'.join(keywords), case=False, na=False)]

def generate_governance_analysis(top_features):
    """Summarizes governance-related instability/stability drivers."""
    gov_keys = ['PV', 'GE', 'RL', 'CC', 'VA', 'RQ', 'FreedomScore']
    gov_data = filter_features(top_features, gov_keys)

    if gov_data.empty:
        return "Governance Factors: No significant governance drivers identified."

    narrative = "GOVERNANCE & STABILITY ANALYSIS:\n"
    for _, row in gov_data.iterrows():
        impact = "contributing to instability" if row['shap_value'] > 0 else "acting as a stabilizing buffer"
        narrative += "- " + str(row['feature']).replace('_', ' ') + " is " + impact + ".\n"
    return narrative

def generate_conflict_analysis(top_features):
    """Summarizes conflict persistence and escalation indicators."""
    con_keys = ['Conflict', 'Internal', 'MilExp', 'lag', 'rolling']
    con_data = filter_features(top_features, con_keys)

    if con_data.empty:
        return "Conflict Dynamics: Historical patterns showed marginal influence in this specific forecast."

    narrative = "CONFLICT DYNAMICS & PERSISTENCE:\n"
    for _, row in con_data.iterrows():
        impact = "elevating escalation risk" if row['shap_value'] > 0 else "suggesting de-escalation potential"
        narrative += "- " + str(row['feature']).replace('_', ' ') + " is " + impact + ".\n"
    return narrative

def generate_economic_analysis(top_features):
    """Summarizes economic stability/fragility indicators."""
    eco_keys = ['GDP', 'Inflation', 'Unemployment', 'Pop_Growth', 'Population']
    eco_data = filter_features(top_features, eco_keys)

    if eco_data.empty:
        return "Economic Indicators: Economic variables showed marginal impact."

    narrative = "ECONOMIC FRAGILITY & EXPOSURE:\n"
    for _, row in eco_data.iterrows():
        impact = "exacerbating fragility" if row['shap_value'] > 0 else "providing economic resilience"
        narrative += "- " + str(row['feature']).replace('_', ' ') + " is " + impact + ".\n"
    return narrative

def generate_full_report(country, year, probability, risk_level, top_features):
    """CENTRAL REPORT FUNCTION: Compiles all analytical components."""
    report = {
        "metadata": {
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": "V6-Alpha"
        },
        "executive_summary": generate_risk_summary(country, year, probability, risk_level),
        "governance_analysis": generate_governance_analysis(top_features),
        "conflict_analysis": generate_conflict_analysis(top_features),
        "economic_analysis": generate_economic_analysis(top_features)
    }

    report["full_text"] = (
        report['executive_summary'] + "\n\n" +
        report['governance_analysis'] + "\n" +
        report['conflict_analysis'] + "\n" +
        report['economic_analysis']
    )

    return report
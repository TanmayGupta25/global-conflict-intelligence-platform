# 🌍 Geopolitical Intelligence Dashboard (V6)

An AI-powered decision support system that predicts geopolitical conflict risk using advanced machine learning (XGBoost) and explainable AI (SHAP). This platform provides intelligence-grade visualizations for strategic risk assessment.

## 🚀 Key Features
- **Interactive Global Risk Map**: Real-time choropleth visualization of conflict probabilities across all countries.
- **Scenario Simulation Engine**: 'What-if' analysis to test the impact of GDP fluctuations, military spending, and stability shifts.
- **Explainable AI (XAI)**: SHAP-driven narratives that decompose model predictions into actionable intelligence briefs.
- **Historical Validation**: Compare model outputs against verified historical datasets for accuracy testing.
- **Modular Backend**: Decoupled architecture for preprocessing, prediction, and reporting.

## 🏗️ Architecture & Technology Stack
- **Frontend**: HTML5/CSS3 (Custom Hardened UI), Plotly.js for interactive visualizations.
- **Backend**: Flask (Python) with a modular engine design.
- **Machine Learning**: XGBoost Classifier with SMOTE for handling geopolitical data imbalances.
- **Explainability**: SHAP (SHapley Additive exPlanations) for model transparency.
- **Deployment**: Configured for Render/Gunicorn and local tunneling via pyngrok.

## 🛠️ Installation & Setup
1. **Clone the Repo**:
   ```bash
   git clone https://github.com/your-username/geopolitical-intelligence.git
   cd geopolitical-intelligence
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Environment Setup**:
   Copy `.env.example` to `.env` and configure your `FLASK_SECRET_KEY`.
4. **Run Locally**:
   ```bash
   python app.py
   ```

## 🛰️ Deployment Strategy
This project is **Render-ready**. To deploy:
- Connect your GitHub repository to [Render](https://render.com).
- Set the build command to `pip install -r requirements.txt`.
- Set the start command to `gunicorn app:app`.

## 🗺️ Future Roadmap
- [ ] Integration of real-time news API sentiment analysis.
- [ ] Multi-year forecasting trajectories (2025-2030).
- [ ] Exportable PDF Strategic Intelligence Briefs.
- [ ] User authentication for secure dashboard access.

---
**Geopolitical Intelligence Unit &copy; 2024**
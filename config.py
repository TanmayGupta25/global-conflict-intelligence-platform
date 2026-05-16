import os
import secrets

# --- PROJECT PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# --- ARTIFACTS ---
DATASET_PATH = os.path.join(DATA_DIR, "BASE_DATASET_V6.xlsx")
MODEL_PATH = os.path.join(MODELS_DIR, "best_conflict_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "model_features.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# --- SECURITY & ENVIRONMENT ---
# Use environment variable for Secret Key or generate a safe fallback
SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))

# Default to production settings (False) unless explicitly set to True
DEBUG_MODE = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

# --- ANALYTICAL SETTINGS ---
LOW_RISK_THRESHOLD = 0.35
HIGH_RISK_THRESHOLD = 0.65
TOP_N_FEATURES = 8

# --- FLASK SETTINGS ---
PORT = int(os.environ.get('PORT', 5000))

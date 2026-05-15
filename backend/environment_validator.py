import os
import sys
import importlib

def check_dependencies(required_packages):
    """Checks if required Python packages are installed."""
    missing = []
    for pkg in required_packages:
        try:
            importlib.import_module(pkg.replace('-', '_'))
        except ImportError:
            missing.append(pkg)
    return missing

def check_artifacts(required_files):
    """Checks if critical model and data artifacts exist."""
    missing = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing.append(file_path)
    return missing

def validate_system_readiness(config):
    """Performs a full system health check before app startup."""
    packages = ['flask', 'pandas', 'numpy', 'sklearn', 'shap', 'plotly', 'joblib']
    artifacts = [config.DATASET_PATH, config.MODEL_PATH, config.FEATURES_PATH]
    
    pkg_errors = check_dependencies(packages)
    art_errors = check_artifacts(artifacts)
    
    return {
        "ready": len(pkg_errors) == 0 and len(art_errors) == 0,
        "missing_packages": pkg_errors,
        "missing_artifacts": art_errors
    }
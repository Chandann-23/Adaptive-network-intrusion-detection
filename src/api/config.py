"""
REST API Serving Configuration
==============================
Declares registry paths for serialized preprocessors and hybrid models,
server environment variables, and metadata info.
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PREPROCESSOR_PATH = os.path.join(BASE_DIR, "data", "processed", "preprocessing_pipeline.joblib")
MODEL_PATH = os.path.join(BASE_DIR, "models", "multiclass", "hybrid_pipeline", "model.joblib")

# Server settings — PORT defaults to 7860 for Hugging Face Spaces
# Override locally with: PORT=8000 python src/api/main.py
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 7860))

# Metadata
APP_NAME = "Adaptive NIDS REST API"
VERSION = "1.0.0"
DESCRIPTION = (
    "Production-grade Network Intrusion Detection serving platform. "
    "Features a two-stage zero-day hybrid pipeline combining unsupervised novelty filtering "
    "and high-precision multiclass ensemble routing."
)

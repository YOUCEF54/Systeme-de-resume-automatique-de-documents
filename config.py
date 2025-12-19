"""
Backend Configuration
"""
import os
from pathlib import Path


class Settings:
    """Application settings"""
    
    # Paths
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent
    MODELS_DIR = BASE_DIR / "models"
    ARABART_MODEL_PATH = MODELS_DIR / "arabart_finetuned_final"
    
    # API Settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    
    # CORS
    CORS_ORIGINS = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    
    # Model Settings
    ARABERT_MODEL_NAME = "aubmindlab/bert-base-arabertv02"
    MIN_SENTENCES_FOR_ABSTRACTIVE = 3
    
    # Summarization defaults
    DEFAULT_EXTRACT_TOP_N = 5
    DEFAULT_METHOD = "mmr"


settings = Settings()

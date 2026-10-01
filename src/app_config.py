"""
app_config.py - Application Configuration

Central location for all configuration settings.
Load from environment variables if available, otherwise use defaults.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Base configuration"""

    # Application
    APP_NAME = "Sentiment Analysis System"
    APP_VERSION = "2.0.0"
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

    # Database (SQLite - your current setup)
    DB_PATH = os.getenv('DB_PATH', 'sentiment_analysis.db')

    # API Server (Flask)
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 5000))
    API_DEBUG = os.getenv('API_DEBUG', 'True').lower() == 'true'
    API_THREADED = os.getenv('API_THREADED', 'True').lower() == 'true'

    # Streamlit Dashboard
    STREAMLIT_PORT = int(os.getenv('STREAMLIT_PORT', 8501))
    STREAMLIT_HOST = os.getenv('STREAMLIT_HOST', 'localhost')

    # Sentiment Analysis
    DEFAULT_DOMAIN = os.getenv('DEFAULT_DOMAIN', 'general')
    ENABLE_DEBUG = os.getenv('ENABLE_DEBUG', 'False').lower() == 'true'
    COLLECT_TIMING_STATS = os.getenv('COLLECT_TIMING_STATS', 'False').lower() == 'true'

    # Lexicons
    LEXICON_DIR = os.getenv('LEXICON_DIR', 'lexicons')

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'sentiment_analysis.log')

    # Feature Flags
    ENABLE_SARCASM_DETECTION = os.getenv('ENABLE_SARCASM_DETECTION', 'True').lower() == 'true'
    ENABLE_ASPECT_ANALYSIS = os.getenv('ENABLE_ASPECT_ANALYSIS', 'True').lower() == 'true'
    ENABLE_TEMPORAL_TRACKING = os.getenv('ENABLE_TEMPORAL_TRACKING', 'True').lower() == 'true'

    # Performance
    MAX_BATCH_SIZE = int(os.getenv('MAX_BATCH_SIZE', 10000))
    ANALYSIS_TIMEOUT = int(os.getenv('ANALYSIS_TIMEOUT', 300))  # seconds

    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {
            'app_name': cls.APP_NAME,
            'app_version': cls.APP_VERSION,
            'debug': cls.DEBUG,
            'db_path': cls.DB_PATH,
            'api_host': cls.API_HOST,
            'api_port': cls.API_PORT,
            'default_domain': cls.DEFAULT_DOMAIN,
            'lexicon_dir': cls.LEXICON_DIR
        }


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    API_DEBUG = True
    ENABLE_DEBUG = True
    COLLECT_TIMING_STATS = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    API_DEBUG = False
    ENABLE_DEBUG = False
    LOG_LEVEL = 'INFO'


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    DB_PATH = ':memory:'  # Use in-memory database for tests
    ENABLE_DEBUG = True


# Select configuration based on environment
ENV = os.getenv('ENVIRONMENT', 'development').lower()

if ENV == 'production':
    config = ProductionConfig()
elif ENV == 'testing':
    config = TestingConfig()
else:
    config = DevelopmentConfig()


# ============================================================================
# QUICK ACCESS
# ============================================================================

def get_config():
    """Get current configuration"""
    return config


def is_debug():
    """Check if debug mode is enabled"""
    return config.DEBUG


def is_production():
    """Check if running in production"""
    return ENV == 'production'


def get_database_path():
    """Get database path"""
    return config.DB_PATH


def get_api_url():
    """Get API URL"""
    return f"http://{config.API_HOST}:{config.API_PORT}"


# ============================================================================
# EXAMPLE .env FILE
# ============================================================================

"""
Create a .env file in your project root with:

# Environment
ENVIRONMENT=development

# Database
DB_PATH=sentiment_analysis.db

# API
API_HOST=0.0.0.0
API_PORT=5000
API_DEBUG=True
API_THREADED=True

# Streamlit
STREAMLIT_PORT=8501
STREAMLIT_HOST=localhost

# Sentiment Analysis
DEFAULT_DOMAIN=general
ENABLE_DEBUG=False
COLLECT_TIMING_STATS=False

# Logging
LOG_LEVEL=INFO
LOG_FILE=sentiment_analysis.log

# Features
ENABLE_SARCASM_DETECTION=True
ENABLE_ASPECT_ANALYSIS=True
ENABLE_TEMPORAL_TRACKING=True

# Performance
MAX_BATCH_SIZE=10000
ANALYSIS_TIMEOUT=300
"""

if __name__ == "__main__":
    print("Current Configuration:")
    print("=" * 50)
    for key, value in config.to_dict().items():
        print(f"{key:30} = {value}")

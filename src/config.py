import os
import logging
from dotenv import load_dotenv
import streamlit as st

# Centralized logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def _get_env_or_secret(key: str, default: str = None) -> str:
    """
    Retrieves a configuration value from environment variables or Streamlit secrets.
    Prioritizes environment variables.
    """
    # 1. Check environment variables (populated by .env or system env)
    value = os.getenv(key)
    if value:
        return value

    # 2. Fallback to Streamlit secrets
    try:
        if key in st.secrets:
            return st.secrets[key]
    except (FileNotFoundError, KeyError, RuntimeError):
        # RuntimeError can occur if accessed outside a Streamlit context
        pass

    return default

# Ensure .env variables are loaded into environment
load_dotenv()

class Config:
    """
    Centralized configuration hardening layer for Cinema-Sense.
    Handles environment variable loading, validation, and defaults.
    """

    # TMDb API Key for fetching movie posters and details
    TMDB_API_KEY = _get_env_or_secret("TMDB_API_KEY")

    # Data File Paths
    MOVIES_PICKLE_PATH = _get_env_or_secret("MOVIES_PICKLE_PATH", "data/processed/movies_with_tags.pkl")
    FAISS_INDEX_PATH = _get_env_or_secret("FAISS_INDEX_PATH", "data/vector_db/movies.faiss")

    @classmethod
    def validate(cls):
        """
        Validates that required configuration is present.
        Logs warnings for missing optional but recommended settings.
        """
        if not cls.TMDB_API_KEY:
            logger.warning("MISSING CONFIG: TMDB_API_KEY is not set. Movie metadata fetching will be disabled.")
        else:
            logger.info("CONFIG: TMDB_API_KEY successfully loaded.")

        # Check if data files exist (minimal logging)
        if not os.path.exists(cls.MOVIES_PICKLE_PATH):
            logger.warning(f"MISSING DATA: {cls.MOVIES_PICKLE_PATH} not found.")
        if not os.path.exists(cls.FAISS_INDEX_PATH):
            logger.warning(f"MISSING DATA: {cls.FAISS_INDEX_PATH} not found.")

# Run validation once upon initialization
Config.validate()

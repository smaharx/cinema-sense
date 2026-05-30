import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data Paths
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

MOVIES_PKL_PATH = os.getenv("MOVIES_PKL_PATH", str(PROCESSED_DATA_DIR / "movies_with_tags.pkl"))
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", str(VECTOR_DB_DIR / "movies.faiss"))

# Model Settings
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")

# TMDB API
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Application Settings
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

def validate_config():
    """Basic validation of critical configuration."""
    if not TMDB_API_KEY:
        print("[WARNING] TMDB_API_KEY is not set. Poster fetching will fail.")

    critical_paths = [MOVIES_PKL_PATH, FAISS_INDEX_PATH]
    for path in critical_paths:
        if not os.path.exists(path):
            print(f"[WARNING] Critical data file not found at: {path}")

if __name__ == "__main__":
    validate_config()

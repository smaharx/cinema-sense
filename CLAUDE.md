# CLAUDE.md

This file provides guidance to Claude Code when working with the Cinema-Sense AI project.

## Common Development Commands

Build: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
Test: `pytest tests/` (or `pytest tests/test_models.py::test_tfidf_vectorizer` for specific tests)
Lint: `flake8 src/ tests/`

## Project Architecture

The Cinema-Sense application is structured as follows:

- **`src/`**: Main application directory containing:
  - `models/`: Holds the hybrid recommendation engine combining NLP routing and content-based recommendations
  - `utils/`: Contains utilities for data preprocessing and TMDb API integration
  - `app.py`: Main Streamlit application entry point
- **`tests/`**: Test suite for application modules
- **`requirements.txt`**: Project dependencies

## Key Components

1. **HybridEngine**: The core recommendation system that
   - Uses `NLPRouter` for query translation
   - Implements content-based ranking via TF-IDF and FAISS
   - Handles both title-based and pure semantic searches
2. **TMDb API Integration**: Movie poster fetching via `utils.tmdb_api`
3. **Streamlit Interface**: Provides the web UI with movie display

## Usage Tips
- For semantic searches, avoid including movie titles in queries to trigger pure semantic mode
- Configuration files (`movies_with_tags.pkl`, `tfidf_vectors.pkl`) are required for initial engine startup
- Test suite covers model functionality but may need expansion for edge cases
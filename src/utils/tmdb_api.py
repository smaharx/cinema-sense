import os
import re
import urllib.parse
import requests
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

def get_movie_poster(movie_title):
    """Fetches the movie poster, cleans the title, and provides a sleek fallback."""
    
    # 1. Clean the title: Remove years in parentheses like " (2014)"
    clean_title = re.sub(r'\(\d{4}\)', '', movie_title).strip()
    
    # Generate a sleek, dark-mode fallback URL with the movie title just in case
    encoded_title = urllib.parse.quote(clean_title)
    fallback_url = f"https://placehold.co/500x750/111418/ffffff?text={encoded_title}\nPoster+Unavailable"

    if not TMDB_API_KEY:
        return "https://placehold.co/500x750/111418/ff4444?text=Missing+API+Key"

    base_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": clean_title # Use the cleaned title for better search results!
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("results"):
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
            
    except Exception as e:
        print(f"[ERROR] TMDb failed for '{clean_title}': {e}")
        
    return fallback_url
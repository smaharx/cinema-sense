import os
import requests
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

def get_movie_poster(movie_title):
    """Fetches the movie poster URL from TMDb based on the movie title."""
    # Fallback image if no API key is found
    if not TMDB_API_KEY:
        print("[WARNING] TMDB_API_KEY is missing!")
        return "https://via.placeholder.com/500x750?text=No+API+Key"

    base_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": movie_title
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status() # Raises an error for bad HTTP status codes
        data = response.json()

        if data.get("results"):
            # Get the poster path of the top search result
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                # TMDb requires this specific base URL for images
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
            
    except Exception as e:
        print(f"[ERROR] Failed to fetch poster for '{movie_title}': {e}")
        
    # Fallback image if the movie isn't found or an error occurs
    return "https://via.placeholder.com/500x750?text=No+Poster+Found"
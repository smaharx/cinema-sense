import streamlit as st
import os
import re
import urllib.parse
import requests
from dotenv import load_dotenv


load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

import logging

# Just placing a camera for this file
logger = logging.getLogger(__name__)

@st.cache_data
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
        logger.info(f"[ERROR] TMDb failed for '{clean_title}': {e}")
        
    return fallback_url



@st.cache_data
def get_movie_details(movie_title):
    """Fetches both the poster URL and the plot synopsis from TMDb."""
    api_key = os.getenv("TMDB_API_KEY")
    # If the key is in Streamlit secrets, it will also fall back to this:
    if not api_key:
        try:
            api_key = st.secrets["TMDB_API_KEY"]
        except:
            pass

    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"
    
    try:
        response = requests.get(search_url)
        data = response.json()
        
        if data['results']:
            # Grab the first search result
            movie = data['results'][0]
            
            # Get the poster
            poster_path = movie.get('poster_path')
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/500x750?text=No+Poster+Found"
            
            # Get the plot synopsis
            overview = movie.get('overview', 'No synopsis available for this movie.')
            
            return poster_url, overview
            
    except Exception as e:
        print(f"Error fetching details for {movie_title}: {e}")
        
    # Fallback if nothing is found
    return "https://via.placeholder.com/500x750?text=No+Poster+Found", "No synopsis available."
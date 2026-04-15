"""
Cinema-Sense: AI-Powered Semantic Movie Search
"""

import streamlit as st
from models.hybrid_engine import HybridEngine
from utils.tmdb_api import get_movie_poster  

from dotenv import load_dotenv
import os

load_dotenv() 
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

st.set_page_config(
    page_title="Cinema-Sense",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="auto"
)

st.title("Cinema-Sense: AI-Powered Semantic Movie Search")

# Add a text input for the user to enter a movie plot
plot_text = st.text_input("Enter a movie plot:", placeholder="What's the movie about?")

# Add a button to trigger the movie recommendation
if st.button("Find Movies"):
    # Get the top 3 movie recommendations based on the plot text
    engine = HybridEngine("data/processed/movies_with_tags.pkl", "data/processed/tfidf_vectors.pkl", "data/vector_db/movies.faiss")
    recommended_movies = engine.get_recommendations(plot_text, top_n=3)

    # Fetch the poster URLs
    poster_urls = [get_movie_poster(movie) for movie in recommended_movies]

    # Display the movie posters in a responsive grid
    cols = st.columns(3)
    
    # We use zip() to pair up the movie title with its poster URL!
    for i, (movie_title, poster_url) in enumerate(zip(recommended_movies, poster_urls)):
        with cols[i % 3]: 
            st.image(poster_url, width="stretch") 
            # Write the actual movie title under the poster as a subheader
            st.subheader(movie_title)
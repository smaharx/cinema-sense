"""
Cinema-Sense: AI-Powered Semantic Movie Search
"""

import streamlit as st
from models.hybrid_engine import HybridEngine
from utils.tmdb_api import get_movie_poster  

from dotenv import load_dotenv
import os

load_dotenv() 

st.set_page_config(
    page_title="Cinema-Sense",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded" # We open the sidebar so they see the filters!
)

# --- SIDEBAR: DETERMINISTIC FILTERS ---
with st.sidebar:
    st.markdown("### ⚙️ Search Filters")
    st.caption("Fine-tune your recommendations")
    
    # Slider for Minimum Rating
    min_rating = st.slider("Minimum IMDb Rating", min_value=0.0, max_value=10.0, value=6.0, step=0.5)
    
    # Dual-Slider for Release Year
    year_range = st.slider("Release Year", min_value=1920, max_value=2024, value=(1990, 2024))
    
    st.divider()
    st.markdown("💡 *Pro tip: Use higher ratings to filter out B-movies.*")

# --- MAIN UI ---
st.markdown("<h1 style='text-align: center;'>🎬 Cinema-Sense</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>AI-Powered Semantic Movie Search Engine</p>", unsafe_allow_html=True)
st.divider()

st.markdown("### What kind of movie are you looking for?")
plot_text = st.text_input("", placeholder="e.g., A team of astronauts travel through a wormhole...", label_visibility="collapsed")

if st.button("Search Movies", type="primary"):
    with st.spinner("Analyzing semantics and applying filters..."):
        engine = HybridEngine("data/processed/movies_with_tags.pkl", "data/processed/tfidf_vectors.pkl", "data/vector_db/movies.faiss")
        
        # We are now passing the UI variables into the engine!
        recommended_movies = engine.get_recommendations(
            plot_text, 
            top_n=3,
            min_rating=min_rating, 
            year_range=year_range
        )
        
        poster_urls = [get_movie_poster(movie) for movie in recommended_movies]

    st.markdown("### Top Matches")
    cols = st.columns(3)
    
    for i, (movie_title, poster_url) in enumerate(zip(recommended_movies, poster_urls)):
        with cols[i % 3]: 
            with st.container(border=True):
                st.image(poster_url, width="stretch") 
                st.subheader(movie_title)
                st.caption(f"Recommendation #{i+1}")
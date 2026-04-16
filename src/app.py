"""
Cinema-Sense: AI-Powered Semantic Movie Search
"""

import streamlit as st
from models.hybrid_engine import HybridEngine
from utils.tmdb_api import get_movie_poster  

from dotenv import load_dotenv
import os
    
load_dotenv() 

# 1. Professional Page Configuration
st.set_page_config(
    page_title="Cinema-Sense",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed" # Hides the sidebar for a cleaner look
)

# Custom header styling
st.markdown("<h1 style='text-align: center;'>🎬 Cinema-Sense</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>AI-Powered Semantic Movie Search Engine</p>", unsafe_allow_html=True)
st.divider() # Adds a clean horizontal line

# Search Interface
st.markdown("### What kind of movie are you looking for?")
plot_text = st.text_input("", placeholder="e.g., A team of astronauts travel through a wormhole to save humanity...", label_visibility="collapsed")

if st.button("Search Movies", type="primary"): # Makes the button visually pop
    with st.spinner("Analyzing semantics and fetching recommendations..."):
        engine = HybridEngine("data/processed/movies_with_tags.pkl", "data/processed/tfidf_vectors.pkl", "data/vector_db/movies.faiss")
        recommended_movies = engine.get_recommendations(plot_text, top_n=3)
        poster_urls = [get_movie_poster(movie) for movie in recommended_movies]

    st.markdown("### Top Matches")
    cols = st.columns(3)
    
    for i, (movie_title, poster_url) in enumerate(zip(recommended_movies, poster_urls)):
        with cols[i % 3]: 
            # 2. Wrap the movie in a stylized card container
            with st.container(border=True):
                st.image(poster_url, width="stretch") 
                st.subheader(movie_title)
                st.caption(f"Recommendation #{i+1}")
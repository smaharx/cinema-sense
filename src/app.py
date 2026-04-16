"""
Cinema-Sense: AI-Powered Semantic Movie Search
"""
import concurrent.futures
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
    initial_sidebar_state="expanded" 
)

# --- THE FIX: CACHING THE HEAVY AI ENGINE ---
@st.cache_resource(show_spinner="Booting up AI Engine (This only happens once)...")
def load_engine():
    """Loads the heavy pickle files and FAISS index into RAM permanently."""
    return HybridEngine(
        "data/processed/movies_with_tags.pkl", 
        "data/processed/tfidf_vectors.pkl", 
        "data/vector_db/movies.faiss"
    )

# Instantiate the engine using the cached function
engine = load_engine()

# --- SIDEBAR: DETERMINISTIC FILTERS ---
with st.sidebar:
    st.markdown("### ⚙️ Search Filters")
    st.caption("Fine-tune your recommendations")
    
    min_rating = st.slider("Minimum IMDb Rating", min_value=0.0, max_value=10.0, value=6.0, step=0.5)
    year_range = st.slider("Release Year", min_value=1920, max_value=2024, value=(1990, 2024))
    
    st.divider()
    st.markdown("💡 *Pro tip: Use higher ratings to filter out B-movies.*")

# --- MAIN UI ---
st.markdown("<h1 style='text-align: center;'>🎬 Cinema-Sense</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>AI-Powered Semantic Movie Search Engine</p>", unsafe_allow_html=True)
st.divider()

st.markdown("### What kind of movie are you looking for?")
plot_text = st.text_input("Search Plot", placeholder="e.g., A team of astronauts travel through a wormhole...", label_visibility="collapsed")

if st.button("Search Movies", type="primary"):
    with st.spinner("Analyzing semantics and fetching posters..."):
        recommended_movies = engine.get_recommendations(
            plot_text, 
            top_n=3,
            min_rating=min_rating, 
            year_range=year_range
        )
        
    # --- NEW LOGIC: THE EMPTY STATE HANDLER ---
    if not recommended_movies:
        st.warning("⚠️ **No movies found matching your exact criteria.**")
        st.info("💡 **Try this:** Lower your minimum IMDb rating, expand your Release Year range, or use a broader search phrase.")
    else:
        # --- THE NEW ASYNC POSTER FETCHING LOGIC ---
        with st.spinner("Fetching posters at lightspeed..."):
            # Open multiple 'lanes' to fetch images simultaneously
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Map the function to the list of movies and run them all at once
                poster_urls = list(executor.map(get_movie_poster, recommended_movies))

        st.markdown("### Top Matches")
        cols = st.columns(5)
        
        for i, (movie_title, poster_url) in enumerate(zip(recommended_movies, poster_urls)):
            with cols[i % 3]: 
                with st.container(border=True):
                    st.image(poster_url, width="stretch") 
                    st.subheader(movie_title)
                    st.caption(f"Recommendation #{i+1}")
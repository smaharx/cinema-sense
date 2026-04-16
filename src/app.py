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
    
    # NEW SLIDER: Let the user choose how many movies to see
    num_recs = st.slider("Number of Recommendations", min_value=3, max_value=15, value=6, step=3)
    
    min_rating = st.slider("Minimum IMDb Rating", min_value=0.0, max_value=10.0, value=6.0, step=0.5)
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
            top_n=num_recs, # <-- Pass the user's choice here!
            min_rating=min_rating, 
            year_range=year_range
        )
        
    if not recommended_movies:
        st.warning("⚠️ **No movies found matching your exact criteria.**")
        st.info("💡 **Try this:** Lower your minimum IMDb rating, expand your Release Year range, or use a broader search phrase.")
    else:
        with st.spinner("Fetching posters at lightspeed..."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                poster_urls = list(executor.map(get_movie_poster, recommended_movies))

        st.markdown("### Top Matches")
        
        # --- THE DYNAMIC GRID SYSTEM ---
        cols_per_row = 3
        
        # Loop through the movies in chunks of 3
        for i in range(0, len(recommended_movies), cols_per_row):
            cols = st.columns(cols_per_row)
            
            # Slice the current row's movies and posters
            row_movies = recommended_movies[i:i + cols_per_row]
            row_posters = poster_urls[i:i + cols_per_row]
            
            # Draw them in their respective columns
            for j, (movie_title, poster_url) in enumerate(zip(row_movies, row_posters)):
                with cols[j]: 
                    with st.container(border=True):
                        st.image(poster_url, width="stretch") 
                        st.subheader(movie_title)
                        st.caption(f"Recommendation #{i+j+1}")
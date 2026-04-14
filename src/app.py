"""
Cinema-Sense: AI-Powered Semantic Movie Search

This is the main entry point for the Cinema-Sense application, built using Streamlit.

The app provides a simple search interface where users can enter a movie plot, and the application will use the hybrid recommendation engine and TMDb API to display the top 3 movie posters that best match the input.

Key functionality:
1. Accept user input for a movie plot in a search bar
2. Pass the plot text to the HybridEngine to get the top 3 movie recommendations
3. Use the tmdb_api utility to fetch the poster URLs for the recommended movies
4. Display the movie posters in a Streamlit layout
"""


import streamlit as st
from models.hybrid_engine import HybridEngine
from utils.tmdb_api import fetch_poster

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

    # Fetch the poster URLs for the recommended movies using the TMDb API
    poster_urls = [fetch_poster(movie_id) for movie_id in recommended_movies]

    # Display the movie posters in a Streamlit layout
   # Display the movie posters in a responsive grid
    cols = st.columns(3)
    for i, poster_url in enumerate(poster_urls):
       # The % 3 math trick makes it loop back to the first column (0, 1, 2, 0, 1, 2...)
       with cols[i % 3]: 
           # We also fixed the deprecation warning here!
           st.image(poster_url, use_container_width=True) 
           st.write(f"Recommendation {i+1}")
            
            
            
            
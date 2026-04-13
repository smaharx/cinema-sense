 # api.py
import os
import streamlit as st
import hybrid_engine
import tmdb_api

  # Set up the Streamlit app
st.set_page_config(page_title="Cinema-Sense", layout="wide")

def search_movies():
      """
      Handles the movie search functionality.
      """
      st.subheader("Search for a movie")
      search_query = st.text_input("Enter a movie title or keywords:", "")

      if st.button("Search"):
          if search_query:
              try:
                  # Call the hybrid engine to get search results
                  search_results = hybrid_engine.search_movies(search_query)

                  # Display the search results
                  st.subheader("Search Results:")
                  for result in search_results:
                      st.write(f"- {result['title']} ({result['year']})")
                      st.write(f"  Genre: {', '.join(result['genres'])}")
                      st.write(f"  Overview: {result['overview']}")
                      st.write(f"  TMDB ID: {result['tmdb_id']}")
                      st.write("---")
              except Exception as e:
                  st.error(f"Error occurred while searching: {str(e)}")
          else:
              st.warning("Please enter a search query.")

def get_movie_details():
      """
      Handles the movie details functionality.
      """
      st.subheader("Movie Details")
      tmdb_id = st.text_input("Enter a TMDB movie ID:", "")

      if st.button("Get Movie Details"):
          if tmdb_id:
              try:
                  # Call the TMDB API to fetch movie details
                  movie_details = tmdb_api.get_movie_details(tmdb_id)

                  # Display the movie details
                  st.write(f"Title: {movie_details['title']}")
                  st.write(f"Release Date: {movie_details['release_date']}")
                  st.write(f"Overview: {movie_details['overview']}")
                  st.write(f"Genre(s): {', '.join(movie_details['genres'])}")
                  st.write(f"Runtime: {movie_details['runtime']} minutes")
                  st.write(f"TMDB ID: {movie_details['id']}")
              except Exception as e:
                  st.error(f"Error occurred while fetching movie details: {str(e)}")
          else:
              st.warning("Please enter a TMDB movie ID.")
def main():
      """
      Main function to run the Streamlit application.
      """
      st.title("Cinema-Sense")
      st.write("Welcome to the Cinema-Sense AI-powered movie search engine!")

      # Call the search_movies and get_movie_details functions
      search_movies()
      get_movie_details()
if __name__ == "__main__":
      main()

  # Create the api.py file
if not os.path.exists("api.py"):
      with open("api.py", "w") as f:
         f.write(inspect.getsource(sys.modules[__name__]))
      st.success("api.py file created successfully!")
else:
      st.info("api.py file already exists.")
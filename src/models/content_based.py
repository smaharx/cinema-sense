import pandas as pd
import numpy as np
import pickle
import faiss
import os
from .base_recommender import BaseRecommender

class ContentBasedRecommender(BaseRecommender):
    def __init__(self, df_path: str, vectors_path: str, faiss_index_path: str):
        print("[INFO] Initializing V2 Content-Based Recommender (with Deterministic Filters)...")
        self.movies_df = pickle.load(open(df_path, 'rb'))
        
        # Ensure we have a clean release_year column for our UI slider to use!
        if 'release_date' in self.movies_df.columns:
            # Convert string dates to datetime objects, extract the year, fill blanks with 0
            self.movies_df['release_year'] = pd.to_datetime(self.movies_df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
        else:
            print("[WARNING] 'release_date' column missing from dataframe. Year filtering will not work.")
            self.movies_df['release_year'] = 0

        self.vectors = pickle.load(open(vectors_path, 'rb')).astype('float32')
        self.faiss_index_path = faiss_index_path
        self.index = self._build_or_load_index()

    def _build_or_load_index(self):
        dimension = self.vectors.shape[1]
        if os.path.exists(self.faiss_index_path):
            return faiss.read_index(self.faiss_index_path)
        else:
            print("[INFO] Building new FAISS index...")
            index = faiss.IndexFlatIP(dimension)
            index.add(self.vectors)
            os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)
            faiss.write_index(index, self.faiss_index_path)
            return index

    def recommend(self, movie_title: str, top_n: int = 5, min_rating: float = 0.0, max_runtime: int = 999, year_range: tuple = None):
        """
        Qasimio Architecture: AI Similarity Search + Deterministic Hard Filters.
        """
        if movie_title not in self.movies_df['title'].values:
            return [] # Return empty list so Streamlit handles it cleanly
            
        movie_idx = self.movies_df[self.movies_df['title'] == movie_title].index[0]
        query_vector = self.vectors[movie_idx].reshape(1, -1)
        
        # 1. THE AI LAYER: Fetch a large pool of candidates (e.g., top 100)
        distances, indices = self.index.search(query_vector, 100)
        
        # Grab the candidate movies
        candidate_indices = indices[0]
        candidates = self.movies_df.iloc[candidate_indices].copy()
        candidates['similarity_score'] = np.round(distances[0], 3)
        
        # 2. THE DETERMINISTIC LAYER (The Guardrails)
        filtered_df = candidates[candidates['title'] != movie_title]
        
        # Base filters
        mask = (filtered_df['vote_average'] >= min_rating)
        if max_runtime:
            mask = mask & (filtered_df['runtime'] <= max_runtime)
            
        # The NEW UI Year Range Filter
        if year_range and 'release_year' in filtered_df.columns:
            min_year, max_year = year_range
            mask = mask & (filtered_df['release_year'] >= min_year) & (filtered_df['release_year'] <= max_year)

        filtered_df = filtered_df[mask]
        
        # 3. Return ONLY a list of titles for Streamlit to use for the API
        final_results = filtered_df['title'].head(top_n).tolist()
        return final_results

    # --- PURE SEMANTIC SEARCH ---
    def recommend_from_text(self, text_query: str, top_n: int = 5, min_rating: float = 0.0, max_runtime: int = 999, year_range: tuple = None):
        """
        Converts raw text into a vector on the fly and searches the database.
        """
        vectorizer_path = "data/processed/tfidf_vectorizer.pkl"
        
        if not os.path.exists(vectorizer_path):
            print("[ERROR] Missing tfidf_vectorizer.pkl!")
            return []
            
        vectorizer = pickle.load(open(vectorizer_path, 'rb'))
        query_vector = vectorizer.transform([text_query]).toarray().astype('float32')
        
        # 1. THE AI LAYER
        distances, indices = self.index.search(query_vector, 100)
        
        candidate_indices = indices[0]
        candidates = self.movies_df.iloc[candidate_indices].copy()
        candidates['similarity_score'] = np.round(distances[0], 3)
        
        # 2. THE DETERMINISTIC LAYER
        mask = (candidates['vote_average'] >= min_rating)
        if max_runtime:
            mask = mask & (candidates['runtime'] <= max_runtime)
            
        # The NEW UI Year Range Filter
        if year_range and 'release_year' in candidates.columns:
            min_year, max_year = year_range
            mask = mask & (candidates['release_year'] >= min_year) & (candidates['release_year'] <= max_year)

        filtered_df = candidates[mask]
        
        # 3. Return ONLY a list of titles for Streamlit
        final_results = filtered_df['title'].head(top_n).tolist()
        return final_results
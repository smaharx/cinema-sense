import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer

import logging

# Just placing a camera for this file
logger = logging.getLogger(__name__)

class HybridEngine:
    def __init__(self, df_path: str, faiss_path: str, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Boots up the engine by loading the freeze-dried database, 
        the ultra-fast FAISS map, and the Deep Learning language model.
        """
        if not os.path.exists(df_path):
            raise FileNotFoundError(f"Database file not found at: {df_path}")
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"FAISS index file not found at: {faiss_path}")

        logger.info(f"[INFO] Loading database from {df_path} and FAISS index from {faiss_path} into RAM...")
        try:
            self.df = pd.read_pickle(df_path)
            self.index = faiss.read_index(faiss_path)
        except Exception as e:
            logger.error(f"Error loading engine data: {e}")
            raise

        # Ensure release_date is a proper datetime object so we can filter by Year
        if 'release_date' in self.df.columns:
            self.df['release_date'] = pd.to_datetime(self.df['release_date'], errors='coerce')

        logger.info(f"[INFO] Booting Deep Learning Model ({model_name})...")
        try:
            # Load the exact same HuggingFace model we used in the preprocessor
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise

    def get_recommendations(self, query: str, top_n: int = 5, min_rating: float = 0.0, year_range: tuple = (1900, 2024)):
        """
        Translates a query into a vector, searches FAISS, and applies Pandas filters.
        """
        # --- 1. AI TRANSLATION (Text -> Dense Vector) ---
        # We pass the query as a list, and it returns a list of vectors. We grab the first one.
        query_vector = self.model.encode([query])
        
        # --- 2. THE MATH (Normalization) ---
        # Convert to float32 and normalize to length 1 so Cosine Similarity works perfectly
        query_vector = np.array(query_vector).astype('float32')
        faiss.normalize_L2(query_vector)

        # --- 3. THE SPEED SEARCH (Over-fetching) ---
        # We ask FAISS for 100 matches to ensure we survive the strict Pandas filters
        search_depth = max(top_n * 10, 100) 
        distances, indices = self.index.search(query_vector, k=search_depth)

        # Grab the row IDs from FAISS
        matched_indices = indices[0]
        
        # --- 4. THE PANDAS GAUNTLET (Filtering) ---
        # Pull the actual movie data for those 100 IDs from our DataFrame
        matched_movies = self.df.iloc[matched_indices].copy()

        # Filter 1: Minimum Rating
        filtered_movies = matched_movies[matched_movies['vote_average'] >= min_rating]

        # Filter 2: Year Range (Extracting the year from the datetime object)
        filtered_movies = filtered_movies[
            (filtered_movies['release_date'].dt.year >= year_range[0]) & 
            (filtered_movies['release_date'].dt.year <= year_range[1])
        ]

        # --- 5. THE OUTPUT ---
        # Return only the top_N titles that survived the gauntlet
        final_titles = filtered_movies['title'].head(top_n).tolist()
        
        return final_titles
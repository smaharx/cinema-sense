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

    def recommend(self, movie_title: str, top_n: int = 5, min_rating: float = 0.0, max_runtime: int = 999) -> pd.DataFrame:
        """
        Qasimio Architecture: AI Similarity Search + Deterministic Hard Filters.
        """
        if movie_title not in self.movies_df['title'].values:
            return pd.DataFrame({"Error": [f"Movie '{movie_title}' not found in database."]})
            
        movie_idx = self.movies_df[self.movies_df['title'] == movie_title].index[0]
        query_vector = self.vectors[movie_idx].reshape(1, -1)
        
        # 1. THE AI LAYER: Fetch a large pool of candidates (e.g., top 100)
        # We fetch more than we need because our hard filters will drop some!
        distances, indices = self.index.search(query_vector, 100)
        
        # Grab the candidate movies
        candidate_indices = indices[0]
        candidates = self.movies_df.iloc[candidate_indices].copy()
        candidates['similarity_score'] = np.round(distances[0], 3)
        
        # 2. THE DETERMINISTIC LAYER (The Guardrails)
        # Drop the movie the user actually searched for
        filtered_df = candidates[candidates['title'] != movie_title]
        
        # Apply strict time and quality filters
        filtered_df = filtered_df[
            (filtered_df['vote_average'] >= min_rating) & 
            (filtered_df['runtime'] <= max_runtime)
        ]
        
        # 3. Return only the top N that survived the gauntlet
        final_results = filtered_df[['title', 'vote_average', 'runtime', 'similarity_score']].head(top_n)
        
        if final_results.empty:
            return pd.DataFrame({"Message": ["No movies found matching both similarity and your strict filters."]})
            
        return final_results

# --- Quick Test Block ---
if __name__ == "__main__":
    DF_PATH = "data/processed/movies_with_tags.pkl"
    VECTORS_PATH = "data/processed/tfidf_vectors.pkl"
    FAISS_PATH = "data/vector_db/movies.faiss"

    try:
        recommender = ContentBasedRecommender(DF_PATH, VECTORS_PATH, FAISS_PATH)
        
        print("\n=== TEST 1: Standard AI Search ===")
        print("Movies similar to 'The Matrix':")
        print(recommender.recommend("The Matrix", top_n=5))
        
        print("\n=== TEST 2: The Qasimio Smart Filter ===")
        print("Movies similar to 'The Matrix' but MUST be under 110 mins and highly rated (>= 6.5):")
        print(recommender.recommend("The Matrix", top_n=5, min_rating=6.5, max_runtime=110))

    except Exception as e:
        print(f"[ERROR] {e}")
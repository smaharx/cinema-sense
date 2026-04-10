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
        distances, indices = self.index.search(query_vector, 100)
        
        # Grab the candidate movies
        candidate_indices = indices[0]
        candidates = self.movies_df.iloc[candidate_indices].copy()
        candidates['similarity_score'] = np.round(distances[0], 3)
        
        # 2. THE DETERMINISTIC LAYER (The Guardrails)
        filtered_df = candidates[candidates['title'] != movie_title]
        filtered_df = filtered_df[
            (filtered_df['vote_average'] >= min_rating) & 
            (filtered_df['runtime'] <= max_runtime)
        ]
        
        # 3. Return only the top N that survived the gauntlet
        final_results = filtered_df[['title', 'vote_average', 'runtime', 'similarity_score']].head(top_n)
        
        if final_results.empty:
            return pd.DataFrame({"Message": ["No movies found matching both similarity and your strict filters."]})
            
        return final_results

    # --- THE NEW FUNCTION: PURE SEMANTIC SEARCH ---
    def recommend_from_text(self, text_query: str, top_n: int = 5, min_rating: float = 0.0, max_runtime: int = 999) -> pd.DataFrame:
        """
        Converts raw text into a vector on the fly and searches the database.
        """
        # We need the vectorizer to translate the new text into math.
        vectorizer_path = "data/processed/tfidf_vectorizer.pkl"
        
        if not os.path.exists(vectorizer_path):
            return pd.DataFrame({"Error": ["Missing tfidf_vectorizer.pkl! You must save the vectorizer model in your data pipeline so we can translate raw text."]})
            
        vectorizer = pickle.load(open(vectorizer_path, 'rb'))
        
        # Translate the English sentence into a math vector
        query_vector = vectorizer.transform([text_query]).toarray().astype('float32')
        
        # 1. THE AI LAYER
        distances, indices = self.index.search(query_vector, 100)
        
        candidate_indices = indices[0]
        candidates = self.movies_df.iloc[candidate_indices].copy()
        candidates['similarity_score'] = np.round(distances[0], 3)
        
        # 2. THE DETERMINISTIC LAYER
        # Notice we don't drop any specific movie_title here because the user just typed a concept!
        filtered_df = candidates[
            (candidates['vote_average'] >= min_rating) & 
            (candidates['runtime'] <= max_runtime)
        ]
        
        # 3. Return results
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
        print(recommender.recommend("The Matrix", top_n=5))
        
        print("\n=== TEST 2: Pure Text Search (New!) ===")
        print(recommender.recommend_from_text("a guy with cancer", top_n=5))

    except Exception as e:
        print(f"[ERROR] {e}")
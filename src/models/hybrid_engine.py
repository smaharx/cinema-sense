import pandas as pd
from .nlp_router import NLPRouter
from .content_based import ContentBasedRecommender

class HybridEngine:
    def __init__(self, df_path: str, vectors_path: str, faiss_index_path: str):
        print("[INFO] Booting Hybrid Recommendation Engine...")
        # Initialize both sub-systems
        self.router = NLPRouter()
        self.content_recommender = ContentBasedRecommender(
            df_path, vectors_path, faiss_index_path
        )

    # Added explicit parameters to catch the UI sliders!
    def get_recommendations(self, user_query: str, top_n: int = 5, min_rating: float = None, year_range: tuple = None):
        """
        The Master Function. Translates English -> Code -> Movie Recommendations.
        """
        print(f"\n[USER SAYS]: '{user_query}'")
        
        # 1. Translate English to Dictionary using the NLP Router
        params = self.router.parse_query(user_query)
        print(f"[SYSTEM TRANSLATION]: {params}")
        
        # --- THE OVERRIDE LOGIC ---
        # If Streamlit passed a min_rating, use it. Otherwise, fallback to the NLP router's guess.
        final_min_rating = min_rating if min_rating is not None else params.get("min_rating")
        
        # --- Escaping the Title Trap ---
        if not params.get("movie_title"):
            print("[INFO] No movie title detected. Switching to Pure Semantic Search...")
            results = self.content_recommender.recommend_from_text(
                text_query=user_query,
                top_n=top_n,
                min_rating=final_min_rating,
                max_runtime=params.get("max_runtime"),
                year_range=year_range # Pass the UI tuple down to the recommender
            )
            # Ensure we return a clean list of strings for Streamlit, not a Pandas DataFrame
            return results.tolist() if isinstance(results, pd.Series) else results

        # 2. Feed the translated parameters into our V2 Engine (Title-based)
        results = self.content_recommender.recommend(
            movie_title=params["movie_title"],
            top_n=top_n,
            min_rating=final_min_rating,
            max_runtime=params.get("max_runtime"),
            year_range=year_range # Pass the UI tuple down to the recommender
        )
        
        return results.tolist() if isinstance(results, pd.Series) else results

# --- Quick Test Block ---
if __name__ == "__main__":
    DF_PATH = "data/processed/movies_with_tags.pkl"
    VECTORS_PATH = "data/processed/tfidf_vectors.pkl"
    FAISS_PATH = "data/vector_db/movies.faiss"

    try:
        engine = HybridEngine(DF_PATH, VECTORS_PATH, FAISS_PATH)
        
        # TEST with simulated UI slider values
        query2 = "a guy with cancer finding hope"
        print("\n=== TEST 2: PURE SEMANTIC SEARCH WITH UI FILTERS ===")
        print(engine.get_recommendations(query2, min_rating=8.0, year_range=(1990, 2024)))
        
    except Exception as e:
        print(f"[ERROR] {e}")
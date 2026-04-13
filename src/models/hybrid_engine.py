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

    def get_recommendations(self, user_query: str, top_n: int = 5) -> pd.DataFrame:
        """
        The Master Function. Translates English -> Code -> Movie Recommendations.
        """
        print(f"\n[USER SAYS]: '{user_query}'")
        
        # 1. Translate English to Dictionary using the NLP Router
        params = self.router.parse_query(user_query)
        print(f"[SYSTEM TRANSLATION]: {params}")
        
        # --- THE FIX: Escaping the Title Trap ---
        if not params.get("movie_title"):
            print("[INFO] No movie title detected. Switching to Pure Semantic Search...")
            # We pass the raw user_query directly to the AI to match against movie plots!
            # Note: Your ContentBasedRecommender needs this method to handle raw text.
            results = self.content_recommender.recommend_from_text(
                text_query=user_query,
                top_n=top_n,
                min_rating=params["min_rating"],
                max_runtime=params["max_runtime"]
            )
            return results

        # 2. Feed the translated parameters into our V2 Engine (Title-based)
        results = self.content_recommender.recommend(
            movie_title=params["movie_title"],
            top_n=top_n,
            min_rating=params["min_rating"],
            max_runtime=params["max_runtime"]
        )
        
        return results

# --- Quick Test Block ---
if __name__ == "__main__":
    DF_PATH = "data/processed/movies_with_tags.pkl"
    VECTORS_PATH = "data/processed/tfidf_vectors.pkl"
    FAISS_PATH = "data/vector_db/movies.faiss"

    try:
        # Boot up the master engine
        engine = HybridEngine(DF_PATH, VECTORS_PATH, FAISS_PATH)
        
        # TEST 1: The original title-based test
        query1 = "I want to watch something similar to The Matrix but under 110 minutes."
        print("\n=== TEST 1: TITLE SEARCH ===")
        print(engine.get_recommendations(query1))
        
        # TEST 2: The NEW pure semantic test (Escaping the trap!)
        query2 = "a guy with cancer finding hope"
        print("\n=== TEST 2: PURE SEMANTIC SEARCH ===")
        print(engine.get_recommendations(query2))
        
    except Exception as e:
        print(f"[ERROR] {e}")
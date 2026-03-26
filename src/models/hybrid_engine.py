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
        
        # Guardrail: If the user didn't mention a movie we can understand
        if not params.get("movie_title"):
            return pd.DataFrame({"Error": ["Could not detect a target movie. Try phrasing it like 'movies similar to [Movie Name]'."]})

        # 2. Feed the translated parameters directly into our V2 AI Engine
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
        
        # The Ultimate V3 Test: Full Natural Language to Filtered AI Output
        query = "I want to watch something similar to The Matrix but it needs to have a rating of at least 6.5 and be under 110 minutes."
        
        results = engine.get_recommendations(query)
        
        print("\n=== FINAL AI OUTPUT ===")
        print(results)
        
    except Exception as e:
        print(f"[ERROR] {e}")
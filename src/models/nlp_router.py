import re

class NLPRouter:
    def __init__(self):
        print("[INFO] Initializing Rule-Based NLP Router (with Mood Anchors)...")
        
        # The Qasimio Mood Matrix: Mapping abstract feelings to mathematical coordinates
        self.mood_anchors = {
            "laugh": "The Hangover",
            "funny": "The Hangover",
            "comedy": "Superbad",
            "sad": "Titanic",
            "cry": "The Notebook",
            "emotional": "The Pursuit of Happyness",
            "scary": "The Conjuring",
            "horror": "The Conjuring",
            "action": "Mad Max: Fury Road",
            "explosions": "The Terminator",
            "mind-blowing": "Inception",
            "space": "Interstellar",
            "romantic": "Titanic",
            "love": "The Notebook"
        }

    def parse_query(self, user_text: str) -> dict:
        """
        Scans a natural language sentence and extracts strict parameters.
        Includes fallback Anchor Mapping if no specific movie is named.
        """
        user_text = user_text.lower()
        
        params = {
            "movie_title": None,
            "max_runtime": 999,
            "min_rating": 0.0,
            "detected_mood": None
        }

        # 1. EXTRACT MOVIE TITLE (Explicit Request)
        title_match = re.search(r'(?:like|similar to) ([a-zA-Z0-9 ]+?)(?: under| with| but| and|$)', user_text)
        if title_match:
            params["movie_title"] = title_match.group(1).strip().title()

        # 2. EXTRACT MOOD (Implicit Request - Anchor Mapping)
        # If they didn't name a specific movie, hunt for mood words!
        if not params["movie_title"]:
            for keyword, anchor_movie in self.mood_anchors.items():
                if keyword in user_text:
                    params["movie_title"] = anchor_movie
                    params["detected_mood"] = keyword
                    break # Stop at the first mood found

        # 3. EXTRACT RUNTIME (Time-Based Filtering)
        # Qasimio V3 Regex: Grabs any number directly attached to min/mins/minutes
        runtime_match = re.search(r'(\d+)\s*(?:min|minutes|mins)', user_text)
        if runtime_match:
            params["max_runtime"] = int(runtime_match.group(1))

        # 4. EXTRACT RATING (Quality Filtering)
        rating_match = re.search(r'(?:rating|rated).*?(\d+\.?\d*)', user_text)
        if rating_match:
            params["min_rating"] = float(rating_match.group(1))

        return params

# --- Quick Test Block ---
if __name__ == "__main__":
    router = NLPRouter()
    print("\n=== Testing Mood Anchors ===")
    
    # Testing the exact phrases you just tried!
    test_queries = [
        "sad",
        "movies that i can laugh with",
        "I want a sad movie under 100 minutes" # Testing mood + time together!
    ]

    for query in test_queries:
        print(f"\nUser says: '{query}'")
        print(f"Engine translates to: {router.parse_query(query)}")